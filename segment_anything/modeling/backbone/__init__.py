import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from . import drn

class TimmBackboneWrapper(nn.Module):
    def __init__(self, model_name, img_ch):
        super().__init__()
        self.backbone = timm.create_model(model_name, in_chans=img_ch, 
                                          pretrained=False, 
                                          features_only=True)

        for m in self.backbone.modules():
            if hasattr(m, 'strict_img_size'):
                m.strict_img_size = False
                
        self.out_channels = self.backbone.feature_info.channels()[-1] 

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]  
        return x, None  

def build_backbone(backbone, img_ch):
    backbone_dict = {
        'resnet': 'resnet50',
        'xception': 'xception41',
        'mobilenet': 'mobilenetv2_100',
        'efficientnet': 'efficientnet_b0',
        'densenet': 'densenet121',
        'convnext': 'convnext_base',
    }
    
    if backbone in backbone_dict:
        model_name = backbone_dict[backbone]
        model = TimmBackboneWrapper(model_name, img_ch)
    elif backbone == 'drn':
        model = drn.drn_d_54(nn.BatchNorm2d, img_ch)
    else:
        raise NotImplementedError(f"Backbone {backbone} not implemented.")

    return model

class MaskEncoder(nn.Module):
    def __init__(self, in_chans, out_chans, backbone='resnet'):
        super().__init__()

        # Build backbone
        self.encoder = build_backbone(backbone, img_ch=in_chans)
        backbone_out_channels = self.encoder.out_channels

        # 1x1 conv to map backbone output to desired out_chans
        self.conv = nn.Conv2d(
            in_channels=backbone_out_channels, 
            out_channels=out_chans,
            kernel_size=1
        ) 
        
    def forward(self, x):
        x, _ = self.encoder(x)
        x = F.interpolate(x, size=(32, 32), 
                          mode='bilinear', 
                          align_corners=False)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(1)
    
    @torch.no_grad()   
    def freeze(self, x):
        x, _ = self.encoder(x)
        x = F.interpolate(x, size=(32, 32), 
                          mode='bilinear', 
                          align_corners=False)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(1)
