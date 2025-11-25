# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import copy
from .common import LayerNorm2d, ConvBlock, ResidualConvBlock
from .lora import BlockLoRA
from typing import List, Tuple, Type, Optional


def load_weights_from_original(model, original_model):
    model_dict = model.state_dict()
    original_dict = original_model.state_dict()

    original_dict = {k: v for k, v in original_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

    model_dict.update(original_dict)
    model.load_state_dict(model_dict)  

class MaskEncoderLoRA(nn.Module):
    def __init__(
        self,
        in_chans: int,
        image_encoder: Type[nn.Module],
        vit_inner_chans: List[int],
        image_adapter: bool=True,
        num_lora: int=1,
    ) -> None:
        super().__init__()
        
        embed_dim = image_encoder.embed_dim
        patch_size = image_encoder.patch_size
        self.vit_inner_chans = vit_inner_chans

        adapt_out_chans = 32
        self.init_layer = ImageAdapter(
            in_chans=in_chans,
            out_chans=adapt_out_chans,
        ) if image_adapter else None
        
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=adapt_out_chans,
            embed_dim=embed_dim,
            porj_layer=None,
            # porj_layer=image_encoder.patch_embed.proj,
        )
        
        self.pos_embed = copy.deepcopy(image_encoder.pos_embed)
        
        self.blocks = nn.ModuleList()
        for i in range(image_encoder.depth):
            block = BlockLoRA(
                dim=image_encoder.embed_dim,
                num_heads=image_encoder.num_heads,
                mlp_ratio=image_encoder.mlp_ratio,
                qkv_bias=image_encoder.qkv_bias,
                norm_layer=image_encoder.norm_layer,
                act_layer=image_encoder.act_layer,
                use_rel_pos=image_encoder.use_rel_pos,
                rel_pos_zero_init=image_encoder.rel_pos_zero_init,
                window_size=image_encoder.window_size if i not in image_encoder.global_attn_indexes else 0,
                input_size=(image_encoder.img_size // image_encoder.patch_size, 
                            image_encoder.img_size // image_encoder.patch_size),
            )
            self.blocks.append(block)
            
        load_weights_from_original(self.blocks, copy.deepcopy(image_encoder.blocks))
        for name, param in self.blocks.named_parameters():
            param.requires_grad = False
            if 'lora' in name:  
                param.requires_grad = True

        for block in self.blocks[-num_lora:]:
            for param in block.parameters():
                param.requires_grad = True
            
    def optimize(self, x: torch.Tensor) -> torch.Tensor:

        if self.init_layer is not None:
            x = self.init_layer(x)
            
        x = self.patch_embed(x)
        
        if self.pos_embed is not None:
            pos_embed = F.interpolate(
                self.pos_embed.permute(0,3,1,2),
                x.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0,2,3,1)
            x = x + pos_embed   
        
        xs = []
        for blk in self.blocks:
            x = blk(x)
            xs.append(x) 
        # xs = []
        # for blk in self.blocks:
        #     x = torch.utils.checkpoint.checkpoint(blk, x)
        #     xs.append(x)
            
        inner_features = torch.stack(xs, dim=0).permute(1, 0, 2, 3, 4) 
        return inner_features[:, self.vit_inner_chans] 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.optimize(x) 

class MaskEncoderViT(nn.Module):
    def __init__(
        self,
        in_chans: int,
        image_encoder: Type[nn.Module],
        vit_inner_chans: List[int],
        image_adapter: bool= True,
        freeze_blocks: bool = False, 
    ) -> None:
        super().__init__()
        
        embed_dim = image_encoder.embed_dim
        patch_size = image_encoder.patch_size
        self.vit_inner_chans = vit_inner_chans

        adapt_out_chans = 32
        self.init_layer = ImageAdapter(
            in_chans=in_chans,
            out_chans=adapt_out_chans,
        ) if image_adapter else None
        
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=adapt_out_chans,
            embed_dim=embed_dim,
            porj_layer=None,
            # porj_layer=image_encoder.patch_embed.proj,
        )
        
        self.pos_embed = copy.deepcopy(image_encoder.pos_embed)
        self.blocks = copy.deepcopy(image_encoder.blocks)      

        if freeze_blocks:
            for blk in self.blocks:
                for param in blk.parameters():
                    param.requires_grad = False
    
    @torch.no_grad()   
    def freeze(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_layer is not None:
            x = self.init_layer(x)
            
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = F.interpolate(
                self.pos_embed.permute(0,3,1,2),
                x.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0,2,3,1)
            x = x + pos_embed   
        xs = []
        for blk in self.blocks:
            x = blk(x)
            xs.append(x) 

        inner_features = torch.stack(xs, dim=0).permute(1, 0, 2, 3, 4) 
        return inner_features[:, self.vit_inner_chans]    
    
    def optimize(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_layer is not None:
            x = self.init_layer(x)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = F.interpolate(
                self.pos_embed.permute(0,3,1,2),
                x.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0,2,3,1)
            x = x + pos_embed   

        xs = []
        for blk in self.blocks:
            x = blk(x)
            xs.append(x) 

        inner_features = torch.stack(xs, dim=0).permute(1, 0, 2, 3, 4) 
        return inner_features[:, self.vit_inner_chans] 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.optimize(x) 
    
    
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        porj_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding,
        )

        if porj_layer is not None:
            weights_conv = porj_layer.weight.data
            bias_conv = porj_layer.bias.data
            self.proj.weight.data = torch.cat([weights_conv] * (in_chans // 3), dim=1)
            self.proj.bias.data = bias_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
        
class ImageAdapter(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        norm_func = nn.BatchNorm2d
        actv_func = activation

        self.module = nn.Sequential(
            ConvBlock(in_chans, out_chans, kernel_size=3, norm_func=norm_func, actv_func=actv_func, stack=1),
            ResidualConvBlock(out_chans, out_chans, stack=2, norm_func=norm_func, actv_func=actv_func),
            ResidualConvBlock(out_chans, out_chans, stack=2, norm_func=norm_func, actv_func=actv_func),
        )

    def forward(self, x):
        return self.module(x)
