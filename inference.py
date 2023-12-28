import tqdm
import torch
import numpy as np
from model.GPE import GPET
import matplotlib.pyplot as plt
from feat_data import get_train_val_stage_dataset
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    # train_set, val_set = get_train_val_stage_dataset(stage=10, temporal=False, return_vid=True)

    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)

    model = GPET()
    # model_path = './saved_models/upper_bound/stage4/feat_gpe_stage4_dim_128_layer_3/epoch_20_P_0.43891_R_0.68276_mAP_0.36300.pth'
    model_path = './saved_models/lower_bound/stage4/feat_gpe_stage4_dim_128_layer_3/epoch_24_P_0.30223_R_0.63687_mAP_0.25984.pth'
    model_path = './saved_models/upper_bound/stage2/feat_gpe_stage2_dim_128_layer_3/epoch_30_P_0.46188_R_0.67637_mAP_0.37381.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = model.eval().to(device)

    # for x, label, vid in tqdm.tqdm(train_loader):
    #     x = x.squeeze(0).float().to(device)
    #     logits = model(x)
    #     conf = torch.nn.Softmax(-1)(logits)[:, 1]
    #     print(conf)

    data_path = '/opt/tiger/debug_server/ByteFood_feat/v0300fg10000c4terlrc77u2ee9n3qog.npy'
    data_path = '/opt/tiger/debug_server/ByteFood_feat/v0300fg10000c6njotbc77u6g30vqt00.npy'
    data_path = '/opt/tiger/debug_server/ByteFood_feat/v0200fg10000c6k4q0bc77u789h1858g.npy'
    x = np.load(data_path)
    x = torch.from_numpy(x).float().to(device)
    print(x.shape)

    with torch.no_grad():
        logits = model(x)
        conf = torch.nn.Softmax(-1)(logits)[:, 1]
        print(conf)
    
    conf = conf.detach().cpu().numpy()
    print(conf.shape)
    np.save('result_stage_1_v0200fg10000c6k4q0bc77u789h1858g.npy', conf)
    # np.save('result_v0300fg10000c6njotbc77u6g30vqt00.npy', conf)
    # np.save('result_v0300fg10000c4terlrc77u2ee9n3qog.npy', conf)

    plt.scatter(list(range(conf.shape[0])), conf, s=5)
    plt.savefig('conf.png', dpi=300)
    plt.close()

