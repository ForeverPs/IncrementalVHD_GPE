import os
import tqdm
import torch
import argparse
import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from model.backbone import ConvNeXtBackBone
from data import get_train_val_stage_dataset


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_ids
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method, world_size=opt.n_gpus)

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    train_set, val_set = get_train_val_stage_dataset(aug_ratio=opt.aug_ratio, stage=opt.stage, return_vid=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=12)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, sampler=val_sampler, num_workers=12)
        
    model = ConvNeXtBackBone()
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank],
                                                      output_device=opt.local_rank, broadcast_buffers=False,
                                                      find_unused_parameters=True)

    if opt.local_rank == 0:
        data_loader = tqdm.tqdm(train_loader)
        val_data_loader = tqdm.tqdm(val_loader)
    else:
        data_loader = train_loader
        val_data_loader = val_loader

    model.eval()
    with torch.no_grad():
        for x, y, vid in tqdm.tqdm(data_loader):
            npy_name = '/opt/tiger/debug_server/ByteFood_feat/%s.npy' % vid[0]
            if not os.path.exists(npy_name):
                x = x.squeeze(0).float()
                feats = list()
                for i in range(x.shape[0] // opt.batch_size):
                    temp_x = x[i * opt.batch_size: (i + 1) * opt.batch_size]
                    feat = model(temp_x.to(device)).detach().cpu().numpy()
                    feats.append(feat)
                if x.shape[0] % opt.batch_size:
                    temp_x = x[(i + 1) * opt.batch_size:]
                    feat = model(temp_x.to(device)).detach().cpu().numpy()
                    feats.append(feat)
                feats = np.concatenate(feats, axis=0)
                assert feats.shape[0] == x.shape[0]

                np.save(npy_name, feats)

        for x, y, vid in tqdm.tqdm(val_data_loader):
            npy_name = '/opt/tiger/debug_server/ByteFood_feat/%s.npy' % vid[0]
            if not os.path.exists(npy_name):
                x = x.squeeze(0).float()
                feats = list()
                for i in range(x.shape[0] // opt.batch_size):
                    temp_x = x[i * opt.batch_size: (i + 1) * opt.batch_size]
                    feat = model(temp_x.to(device)).detach().cpu().numpy()
                    feats.append(feat)
                if x.shape[0] % opt.batch_size:
                    temp_x = x[(i + 1) * opt.batch_size:]
                    feat = model(temp_x.to(device)).detach().cpu().numpy()
                    feats.append(feat)
                feats = np.concatenate(feats, axis=0)
                assert feats.shape[0] == x.shape[0]

                np.save(npy_name, feats)
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VHD GPE Feature Extractor')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--init_method', default='env://')

    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--aug_ratio', type=float, default=0.)
    parser.add_argument('--stage', type=int, default=10)

    opt = parser.parse_args()
    if opt.local_rank == 0:
        print('opt:', opt)

    main(opt)

# python -m torch.distributed.launch --nproc_per_node=8 feature_extraction.py --n_gpus=8