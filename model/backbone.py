from types import new_class
import torch
from torch import nn
from model.resnet import generate_model
from model.convnext import convnext_tiny


class ConvNeXtBackBone(nn.Module):
        def __init__(self, model_path=None):
            super(ConvNeXtBackBone, self).__init__()
            model = convnext_tiny()
            if model_path is None:
                model_path = '/opt/tiger/debug_server/GPE/pretrained_models/convnext_tiny.pth'
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.num_channels = 768
        
        def forward(self, x):
            return self.model(x).view(x.shape[0], -1)


class ResNet3DBackBone(nn.Module):
        def __init__(self, depth=50, model_path=None):
            super(ResNet3DBackBone, self).__init__()
            if depth == 50:
                n_classes = 439
                self.num_channels = 2048
                model_path = '/opt/tiger/debug_server/GPE/pretrained_models/R50_3D.pth'
            elif depth == 18:
                n_classes = 700
                self.num_channels = 512
                model_path = '/opt/tiger/debug_server/GPE/pretrained_models/R18_3D.pth'

            model = generate_model(model_depth=depth, n_classes=n_classes)
            model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
            self.model = nn.Sequential(*list(model.children())[:-1])
        
        def forward(self, x):
            return self.model(x).view(x.shape[0], -1)