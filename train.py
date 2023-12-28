import os
import tqdm
import torch
import shutil
import argparse
import torch.nn as nn
from eval import eval
from torch import optim
from model.GPE import GPE, GPE3D
import torch.distributed as dist
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data import get_train_val_stage_dataset


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_ids

    if opt.local_rank == 0 and opt.build_tensorboard:
        shutil.rmtree(opt.logdir, True)
        writer = SummaryWriter(logdir=opt.logdir)
        opt.build_tensorboard = False
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method, world_size=opt.n_gpus)

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    train_set, val_set = get_train_val_stage_dataset(aug_ratio=opt.aug_ratio, stage=opt.stage)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=12)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=6)
        
    # model = GPE()
    model = GPE3D(depth=18)
    for name, p in model.backbone.named_parameters():
        if '.0.7.' not in name:
            p.requires_grad = False
    
    # loading checkpoint on GPU 0
    if opt.local_rank == 0:
        try:
            model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)
        except:
            print('No Checkpoint, training from scratch...')

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank],
                                                      output_device=opt.local_rank, broadcast_buffers=False,
                                                      find_unused_parameters=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)
    class_weights = torch.tensor([0.1, 0.9]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    for epoch in range(opt.epoch):
        train_loader.sampler.set_epoch(epoch)

        # only tqdm in rank 0
        if opt.local_rank == 0:
            data_loader = tqdm.tqdm(train_loader)
        else:
            data_loader = train_loader
        
        train_loss, val_loss = 0, 0
        train_pr_auc, train_roc_auc = 0, 0
        train_p, train_r, train_ap = 0, 0, 0

        val_pr_auc, val_roc_auc = 0, 0
        val_p, val_r, val_ap = 0, 0, 0

        model.train()
        # classification training
        for x, y in data_loader:
            x = x.float().squeeze(0).to(device)
            y = y.long().squeeze(0).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred_cls = torch.max(logits, dim=-1)[1]
            cls = torch.nn.Softmax(-1)(logits)[:, 1]
            pr_auc, roc_auc, p, r, ap = eval(cls, pred_cls, y)
            train_pr_auc += pr_auc
            train_roc_auc += roc_auc
            train_p += p
            train_r += r
            train_ap += ap

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # update learning rate
        scheduler.step()

        # evaluation
        if opt.local_rank == 0 and epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                for x, y in tqdm.tqdm(val_loader):
                    x = x.float().squeeze(0).to(device)
                    y = y.long().squeeze(0).to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
                    pred_cls = torch.max(logits, dim=-1)[1]
                    cls = torch.nn.Softmax(-1)(logits)[:, 1]
                    pr_auc, roc_auc, p, r, ap = eval(cls, pred_cls, y)
                    val_pr_auc += pr_auc
                    val_roc_auc += roc_auc
                    val_p += p
                    val_r += r
                    val_ap += ap

            train_loss = train_loss / len(train_loader)
            train_pr_auc = train_pr_auc / len(train_loader)
            train_roc_auc = train_roc_auc / len(train_loader)
            train_p = train_p / len(train_loader)
            train_r = train_r / len(train_loader)
            train_ap = train_ap / len(train_loader)

            val_loss = val_loss / len(val_loader)
            val_pr_auc = val_pr_auc / len(val_loader)
            val_roc_auc = val_roc_auc / len(val_loader)
            val_p= val_p / len(val_loader)
            val_r = val_r / len(val_loader)
            val_ap = val_ap / len(val_loader)

            print('EPOCH : %03d | Train Loss : %.3f | Train P : %.3f | Train R : %.3f | Train mAP : %.3f | Train PR-AUC : %.3f | Train ROC-AUC: %.3f |'
                  ' Val Loss : %.3f | Val P : %.3f | Val R : %.3f | Val mAP : %.3f | Val PR-AUC : %.3f | Val ROC-AUC : %.3f'
                % (epoch, train_loss, train_p, train_r, train_ap, train_pr_auc, train_roc_auc, val_loss, val_p, val_r, val_ap, val_pr_auc, val_roc_auc))

            if val_ap >= opt.best_mAP:
                opt.best_mAP = val_ap
                model_name = 'epoch_%d_P_%.3f_R_%.3f.pth' % (epoch, val_p, val_r)
                os.makedirs(opt.save_path, exist_ok=True)
                torch.save(model.module.state_dict(), '%s/%s' % (opt.save_path, model_name))

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/Precision', train_p, epoch)
            writer.add_scalar('train/Recall', train_r, epoch)
            writer.add_scalar('train/mAP', train_ap, epoch)
            writer.add_scalar('train/PR-AUC', train_pr_auc, epoch)
            writer.add_scalar('train/ROC-AUC', train_roc_auc, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/Precison', val_p, epoch)
            writer.add_scalar('val/Recall', val_r, epoch)
            writer.add_scalar('val/mAP', val_ap, epoch)
            writer.add_scalar('val/PR-AUC', val_pr_auc, epoch)
            writer.add_scalar('val/ROC-AUC', val_roc_auc, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VHD GPE')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--init_method', default='env://')

    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7')

    parser.add_argument('--build_tensorboard', type=bool, default=True)
    parser.add_argument('--best_mAP', type=float, default=0.1)
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument('--stage', type=int, default=1)

    parser.add_argument('--logdir', type=str, default='./tensorboard/gpe_stage1')
    parser.add_argument('--save_path', type=str, default='./saved_models/gpe_stage1')
    parser.add_argument('--checkpoint', type=str, default=None)

    opt = parser.parse_args()
    if opt.local_rank == 0:
        print('opt:', opt)

    main(opt)

# python -m torch.distributed.launch --nproc_per_node=8 train.py --n_gpus=8