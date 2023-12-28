import os
import tqdm
import torch
import numpy as np
from model.GPE import GPET
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



def show(feats, label):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(feats)
    embed = tsne.embedding_
    
    color = ['deeppink', 'yellow', 'blue', 'lime']

    # show stage cluster
    for i in range(8):
        index = np.where(label == i)[0]
        x = embed[index]
        if i % 2:
            plt.scatter(x[:, 0], x[:, 1], label='stage %d: highlights' % (i // 2 + 1), s=40, edgecolor='black', c=color[i // 2])
        else:
            plt.scatter(x[:, 0], x[:, 1], label='stage %d: negatives' % (i // 2 + 1), s=40, edgecolor='black', c=color[i // 2])
    plt.legend(markerscale=1.5)
    plt.savefig('./stage.png', dpi=300)
    plt.close()
        

if __name__ == '__main__':
    feats, labels = list(), list()

    model = GPET()

    # stage 1
    model_path = '/opt/tiger/debug_server/GPE/saved_models/GPE/stage1/feat_gpe_stage1_dim_128_layer_3/epoch_10_P_0.45162_R_0.64495_mAP_0.36094.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    feat = model.highlights.weight
    feats.append(feat.detach().numpy())
    labels.append(np.zeros(feat.shape[0]))

    feat = model.negatives.weight
    feats.append(feat.detach().numpy())
    labels.append(np.ones(feat.shape[0]))

    # stage 2
    model_path = '/opt/tiger/debug_server/GPE/saved_models/GPE/stage2/feat_gpe_stage2_dim_128_layer_3/epoch_18_P_0.42951_R_0.68083_mAP_0.35482.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    feat = model.highlights.weight
    feats.append(feat.detach().numpy())
    labels.append(2 * np.ones(feat.shape[0]))

    feat = model.negatives.weight
    feats.append(feat.detach().numpy())
    labels.append(3 * np.ones(feat.shape[0]))

    # stage 3
    model_path = '/opt/tiger/debug_server/GPE/saved_models/GPE/stage3/feat_gpe_stage2_dim_128_layer_3/epoch_22_P_0.34296_R_0.78910_mAP_0.30865.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    feat = model.highlights.weight
    feats.append(feat.detach().numpy())
    labels.append(4 * np.ones(feat.shape[0]))

    feat = model.negatives.weight
    feats.append(feat.detach().numpy())
    labels.append(5 * np.ones(feat.shape[0]))

    # stage 4
    model_path = '/opt/tiger/debug_server/GPE/saved_models/GPE/stage4/feat_gpe_stage2_dim_128_layer_3/epoch_432_P_0.31781_R_0.83024_mAP_0.29882.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    feat = model.highlights.weight
    feats.append(feat.detach().numpy())
    labels.append(6 * np.ones(feat.shape[0]))

    feat = model.negatives.weight
    feats.append(feat.detach().numpy())
    labels.append(7 * np.ones(feat.shape[0]))

    feats = np.concatenate(feats, axis=0)
    label = np.concatenate(labels)

    print(feats.shape, label.shape)

    show(feats, label)

