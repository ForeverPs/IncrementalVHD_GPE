import torch
import einops
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt


def data_transform(p):
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=p),
        # A.Blur(p=p),
        A.ChannelShuffle(p=0.1*p),
        A.Rotate(limit=5, p=p),
    ])
    return transform


def random_aug(x, transform):
    # x: batch, 3, 224, 224
    x = np.transpose(x, [0, 2, 3, 1])
    y = list()
    for i in range(x.shape[0]):
        y.append(transform(image=x[i])['image'])
    y = np.array(y).transpose([0, 3, 1, 2])
    return y


if __name__ == '__main__':
    transform = data_transform(p=0.1)
    data_path = '/opt/tiger/debug_server/douyin_frames/v0d00fg10000c2f59589b668hu50i4rg/x.npy'
    x = np.load(data_path)
    x = einops.rearrange(x, 't c fps h w -> (t fps) c h w')
    print(x.shape)
    aug_x = random_aug(x, transform)
    print(aug_x.shape)
    aug_x = einops.rearrange(aug_x, 't c h w -> t h w c')
    print(aug_x.shape)

    for i in range(aug_x.shape[0]):
        plt.imsave('./vis/%d.png' % i, aug_x[i])
