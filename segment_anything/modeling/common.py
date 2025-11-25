# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Type

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, 
                 kernel_size=3,
                 norm_func=nn.BatchNorm2d, 
                 actv_func=nn.GELU,
                 stride=1,
                 stack=1,
                ):
        super(ConvBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True),
                norm_func(ch_out),
                actv_func(),
            ))
        for _ in range(stack - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True),
                norm_func(ch_out),
                actv_func(),
            ))
    def forward(self,x,size=None):
        if size is not None:
            x = F.interpolate(x,size=size,mode='bilinear',align_corners=True)  
        for layer in self.layers:
            x = layer(x)
        return x   

class ResidualConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=nn.GELU, stack=1):
        super().__init__()
        self.conv_block = ConvBlock(ch_in, ch_out, kernel_size=kernel_size, norm_func=norm_func, actv_func=actv_func, stack=stack)
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv_block(x)
        return out + shortcut

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x 

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock1d(nn.Module):
    def __init__(self, ch_in, ch_out, 
                 kernel_size=3,
                 norm_func=nn.BatchNorm1d, 
                 actv_func=nn.GELU,
                 stride=1,
                 stack=1,
                ):
        super(ConvBlock1d, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True),
                norm_func(ch_out),
                actv_func(),
            ))
        for _ in range(stack - 1):
            self.layers.append(nn.Sequential(
                nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True),
                norm_func(ch_out),
                actv_func(),
            ))

    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode='linear', align_corners=True)  # Changed to 'linear' for 1D
        for layer in self.layers:
            x = layer(x)
        return x   


class EfficientSelfAttention1d(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Change to 1D pooling
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L) where L = H * W
        b, c, l = x.size()
        y = self.avg_pool(x).view(b, c)  # 1D avg pool
        y = self.fc(y).view(b, c, 1)  # Broadcasting over L
        return x * y

class ResidualConvBlock1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, norm_func=nn.BatchNorm1d, actv_func=nn.GELU, stack=1):
        super().__init__()
        self.conv_block = ConvBlock1d(ch_in, ch_out, kernel_size=kernel_size, norm_func=norm_func, actv_func=actv_func, stack=stack)
        if ch_in != ch_out:
            self.shortcut = nn.Conv1d(ch_in, ch_out, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv_block(x)
        return out + shortcut