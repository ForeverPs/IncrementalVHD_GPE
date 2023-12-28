import torch
from loss import distance_loss
from model.GPE import GPE, GPE3D, GPET
from model.backbone import ResNet3DBackBone


if __name__ == '__main__':
    # # T, RGB, fps, h, w
    # x = torch.rand(100, 3, 3, 224, 224)
    # # model = GPE()
    # # model = ResNet3DBackBone(depth=18)
    # model = GPE3D(depth=18)
    # for name, p in model.backbone.named_parameters():
    #     print(name)
    # with torch.no_grad():
    #     cls = model(x)
    #     print(cls.shape)


    # # T, channel
    # x = torch.rand(100, 768)
    # model = GPET()
    # # for name, p in model.named_parameters():
    # #     print(name)
    # with torch.no_grad():
    #     cls = model(x)
    #     print(cls.shape)


    # model = Incremental_GPET()
    # for name, p in model.named_parameters():
    #     if 'old_model.highlights' not in name:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    #     print(name, p.requires_grad)
    # x = torch.rand(100, 768)
    # y = model(x)

    model = GPET()
    model_path = './saved_models/GPE/stage1/feat_gpe_stage1_dim_128_layer_3/epoch_10_P_0.45162_R_0.64495_mAP_0.36094.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

    new_model = GPET()
    # model_path = './saved_models/GPE/stage2/feat_gpe_stage2_dim_128_layer_3/epoch_18_P_0.42951_R_0.68083_mAP_0.35482.pth'
    model_path = './saved_models/GPE/stage3/feat_gpe_stage2_dim_128_layer_3/epoch_22_P_0.34296_R_0.78910_mAP_0.30865.pth'
    new_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

    print(distance_loss(model.highlights.weight, new_model.highlights.weight))
    print(distance_loss(model.negatives.weight, new_model.negatives.weight))