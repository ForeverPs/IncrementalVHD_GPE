import torch


def distance_loss(prototype_a, prototype_b):
    dist = torch.sqrt(1e-10 + torch.sum(torch.square(prototype_a - prototype_b), dim=-1)).mean()
    return dist


if __name__ == '__main__':
    a = torch.rand(10, 128)
    b = torch.rand(10, 128)
    print(distance_loss(a, b))