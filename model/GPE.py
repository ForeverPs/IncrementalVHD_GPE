import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import ConvNeXtBackBone, ResNet3DBackBone


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GPE(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8,
                 num_encoder_layers=3, num_prototypes=20,
                 dropout=0.2, fps=3):
        super().__init__()

        # resnet backbone
        self.backbone = ConvNeXtBackBone()

        # reducing dimensionality of outputted feature
        self.reduce_dim = nn.Linear(self.backbone.num_channels, hidden_dim)

        # transformer block
        # use shallow layer for health gradient
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads,
                                                        dim_feedforward=128, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.linear = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # highlight prototypes
        self.highlights = nn.Embedding(num_prototypes, hidden_dim)
        self.negatives = nn.Embedding(num_prototypes, hidden_dim)

        # sampling rate
        self.fps = fps
    

    def forward(self, x):
        # x: T, 3, w, h
        feat = self.backbone(x)  # T, channel
        feat = self.reduce_dim(feat).unsqueeze(1)   # T, 1, hidden_dim
        feat = self.encoder(feat).squeeze(1) # T, hidden_dim
        feat = self.linear(feat)  # T, hidden_dim
        # calculate L2 distance
        positive_dist = torch.min(torch.cdist(feat, self.highlights.weight, p=2), dim=-1)[0]
        negative_dist = torch.min(torch.cdist(feat, self.negatives.weight, p=2), dim=-1)[0]
        logits = torch.cat([-positive_dist.unsqueeze(-1), -negative_dist.unsqueeze(-1)], dim=-1)
        return logits


class GPE3D(nn.Module):
    def __init__(self, depth=18, hidden_dim=256, nheads=8,
                 num_encoder_layers=3, num_prototypes=20,
                 dropout=0.2, fps=3):
        super().__init__()

        # resnet backbone
        self.backbone = ResNet3DBackBone(depth=depth)

        # reducing dimensionality of outputted feature
        self.reduce_dim = nn.Linear(self.backbone.num_channels, hidden_dim)

        # transformer block
        # use shallow layer for health gradient
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads,
                                                        dim_feedforward=128, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.linear = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # highlight prototypes
        self.highlights = nn.Embedding(num_prototypes, hidden_dim)
        self.negatives = nn.Embedding(num_prototypes, hidden_dim)

        # sampling rate
        self.fps = fps
    

    def forward(self, x):
        # x: T, 3, 3, w, h
        feat = self.backbone(x)  # T, channel
        feat = self.reduce_dim(feat).unsqueeze(1)   # T, 1, hidden_dim
        feat = self.encoder(feat).squeeze(1) # T, hidden_dim
        feat = self.linear(feat)  # T, hidden_dim
        # calculate L2 distance
        positive_dist = torch.min(torch.cdist(feat, self.highlights.weight, p=2), dim=-1)[0]
        negative_dist = torch.min(torch.cdist(feat, self.negatives.weight, p=2), dim=-1)[0]
        logits = torch.cat([-positive_dist.unsqueeze(-1), -negative_dist.unsqueeze(-1)], dim=-1)
        return logits


class GPET(nn.Module):
    def __init__(self, hidden_dim=512, nheads=8,
                 num_encoder_layers=3, num_prototypes=40,
                 dropout=0.2):
        super().__init__()
        # reducing dimensionality of outputted feature
        self.reduce_dim = nn.Linear(768, hidden_dim)

        # transformer block
        # use shallow layer for health gradient
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads,
                                                        dim_feedforward=512, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.linear = MLP(hidden_dim, hidden_dim, hidden_dim // 4, 3)

        # highlight prototypes
        self.highlights = nn.Embedding(num_prototypes, hidden_dim // 4)
        self.negatives = nn.Embedding(num_prototypes, hidden_dim // 4)

        self.norm = nn.InstanceNorm1d(hidden_dim // 4)
    

    def forward(self, x):
        # x: T, channel
        feat = self.norm(self.reduce_dim(x)).unsqueeze(1)   # T, 1, hidden_dim
        feat = feat.squeeze(1) + self.norm(self.encoder(feat).squeeze(1)) # T, hidden_dim
        feat = self.linear(feat)  # T, hidden_dim
        feat = self.norm(feat)    # T, hidden_dim
        # calculate L2 distance
        positive_dist = torch.min(torch.cdist(feat, self.highlights.weight, p=2), dim=-1)[0]
        negative_dist = torch.min(torch.cdist(feat, self.negatives.weight, p=2), dim=-1)[0]
        logits = torch.cat([-negative_dist.unsqueeze(-1), -positive_dist.unsqueeze(-1)], dim=-1)
        return logits


# class Incremental_GPET(nn.Module):
#     def __init__(self, model_path=None, new_prototypes=40, hidden_dim=512):
#         super().__init__()
#         self.old_model = GPET()
#         try:
#             self.old_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
#         except:
#             print('No Pretrained Model...')
#         self.reduce_dim = self.old_model.reduce_dim
#         self.norm = self.old_model.norm
#         self.encoder = self.old_model.encoder
#         self.old_highlights = self.old_model.highlights
#         self.new_highlights = nn.Embedding(new_prototypes, hidden_dim // 4)
#         self.negatives = self.old_model.negatives
#         self.linear = self.old_model.linear
    
#     def forward(self, x):
#         # x: T, channel
#         feat = self.norm(self.reduce_dim(x)).unsqueeze(1)   # T, 1, hidden_dim
#         feat = feat.squeeze(1) + self.norm(self.encoder(feat).squeeze(1)) # T, hidden_dim
#         feat = self.linear(feat)  # T, hidden_dim
#         feat = self.norm(feat)    # T, hidden_dim
#         # calculate L2 distance
#         positive_old_dist = torch.min(torch.cdist(feat, self.old_highlights.weight, p=2), dim=-1)[0].unsqueeze(-1)
#         positive_new_dist = torch.min(torch.cdist(feat, self.new_highlights.weight, p=2), dim=-1)[0].unsqueeze(-1)
#         positive_dist = torch.min(torch.cat([positive_new_dist, positive_old_dist], dim=-1), dim=-1)[0]
#         negative_dist = torch.min(torch.cdist(feat, self.negatives.weight, p=2), dim=-1)[0]
#         logits = torch.cat([-negative_dist.unsqueeze(-1), -positive_dist.unsqueeze(-1)], dim=-1)
#         return logits
    
    



