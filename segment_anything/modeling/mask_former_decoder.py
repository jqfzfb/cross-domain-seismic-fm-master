# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Optional
import einops
from .common import LayerNorm2d, ConvBlock, ResidualConvBlock, EfficientSelfAttention, MLP
from .common import ConvBlock1d, EfficientSelfAttention1d
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv

class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int, 
        transformer_dim: int,
        transformer: nn.Module, 
        num_multimask_outputs: int,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        **kwargs,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens+1, transformer_dim)
        
        concatenate_dim = sum(in_channels)
        self.fusion_layer = nn.Sequential(
            ConvBlock(concatenate_dim, concatenate_dim//2, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
            ConvBlock(concatenate_dim//2, transformer_dim, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
        )

        self.output_upscaling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(transformer_dim, transformer_dim // 2, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
            EfficientSelfAttention(transformer_dim // 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(transformer_dim // 2, transformer_dim // 4, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(transformer_dim // 4, transformer_dim // 8, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
        )
                
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
        )

        self.class_prediction_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Linear(transformer_dim // 2, num_classes + 1),
        )   
        
        self.iou_prediction_head = MLP(
            transformer_dim, 
            iou_head_hidden_dim, 
            self.num_mask_tokens, 
            iou_head_depth,
            sigmoid_output=True,
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        prompt_weight: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        b, _, h, w, _ = image_embeddings.shape
        x = einops.rearrange(image_embeddings, 'b l h w c -> b l c h w')
        x = x.reshape(b,-1,h,w)
        image_embeddings = self.fusion_layer(x)       

        masks, class_pred, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            prompt_weight=prompt_weight,
        )

        # Prepare output
        return masks, class_pred, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        prompt_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight

        output_tokens = output_tokens.unsqueeze(0).expand(dense_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        src0 = torch.sigmoid(src) * (1. - prompt_weight)
        src1 = torch.sigmoid(dense_prompt_embeddings) * prompt_weight
        src = src0 + src1
        
        b, c, h, w = src.shape
        
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        class_token_out = hs[:, 1:self.num_mask_tokens+1, :]
        mask_tokens_out = hs[:, 1:self.num_mask_tokens+1, :]
        
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.output_upscaling(src)
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1) 

        b, c, h, w = upscaled_embedding.shape
        masks_pred = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        # Generate mask quality predictions
        class_pred = self.class_prediction_head(class_token_out)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks_pred, class_pred, iou_pred


### Graph Neural Network

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_geometric.utils import coalesce as pyg_coalesce
try:
    from torch_geometric.nn import GraphNorm
    _HAS_GRAPHNORM = True
except Exception:
    _HAS_GRAPHNORM = False


@torch.no_grad()
def discretize_rgt_map_bins(rgt_map: torch.Tensor, n_bins: int) -> torch.Tensor:
    """把RGT归一化到[0,1]并量化到离散bin。"""
    B = rgt_map.size(0)
    x = rgt_map.view(B, -1)
    rmin = x.min(dim=1, keepdim=True).values
    rmax = x.max(dim=1, keepdim=True).values
    x = (x - rmin) / (rmax - rmin + 1e-8)
    x = x.view_as(rgt_map)
    bins = torch.clamp((x * (n_bins - 1)).round().long(), 0, n_bins - 1)
    return bins


def build_edge_index_from_discrete_rgt_weighted(
    rgt: torch.Tensor, sigma: float = 1.0, use_4nbr: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    给定2D RGT图（H,W），构建带权图：
    - 节点：像素
    - 边：4/8邻接
    - 权重：根据RGT差的高斯相似度 exp(-Δ^2 / 2σ^2)
    返回：
        edge_index: (2, E) long
        edge_weight: (E,) float
    """
    H, W = rgt.shape
    device = rgt.device

    rgt = rgt.float()
    rmin = torch.min(rgt)
    rmax = torch.max(rgt)
    denom = (rmax - rmin).clamp_min(1e-12)
    rgt01 = (rgt - rmin) / denom  # 归一化到[0,1]

    i = torch.arange(H, device=device, dtype=torch.long)
    j = torch.arange(W, device=device, dtype=torch.long)
    ii, jj = torch.meshgrid(i, j, indexing='ij')

    if use_4nbr:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

    node_idx = torch.arange(H * W, device=device).view(H, W)

    e_list = []
    w_list = []
    for dx, dy in offsets:
        ii_n = ii + dx
        jj_n = jj + dy
        valid = (ii_n >= 0) & (ii_n < H) & (jj_n >= 0) & (jj_n < W)

        ii_v = ii[valid]
        jj_v = jj[valid]
        ii_nv = ii_n[valid]
        jj_nv = jj_n[valid]

        src = node_idx[ii_v, jj_v]
        dst = node_idx[ii_nv, jj_nv]

        diff = (rgt01[ii_v, jj_v] - rgt01[ii_nv, jj_nv])
        w = torch.exp(- (diff * diff) / (2.0 * (sigma ** 2) + 1e-12))

        e = torch.stack([src, dst], dim=0)
        e_list.append(e)
        w_list.append(w)

    if not e_list:
        return (
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    edge_index = torch.cat(e_list, dim=1)      # (2, E)
    edge_weight = torch.cat(w_list, dim=0)     # (E,)

    # 双向 + 去重平均
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = pyg_coalesce(edge_index, edge_weight, H * W, reduce='mean')
    return edge_index, edge_weight


# ===== 稀疏归一化与稀疏乘（不依赖 torch_sparse/torch_scatter） =====
def _gcn_norm(edge_index: torch.Tensor,
              edge_weight: torch.Tensor,
              num_nodes: int,
              eps: float = 1e-12) -> torch.Tensor:
    """
    计算对称归一化系数：w_norm = w * D^{-1/2}[src] * D^{-1/2}[dst]
    其中 D[i] = sum_{src=i} w
    """
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    deg.index_add_(0, src, edge_weight)  # 出度和（双向图等价于度）
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    w_norm = edge_weight * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    return w_norm


def _spmm(edge_index: torch.Tensor,
          edge_weight_norm: torch.Tensor,
          x: torch.Tensor,
          num_nodes: int) -> torch.Tensor:
    """
    稀疏矩阵乘 y = A_norm @ x
    A_norm 由 (edge_index, edge_weight_norm) 定义
    """
    src, dst = edge_index
    # 消息：m_e = w_norm * x[src]
    m = x[src] * edge_weight_norm.unsqueeze(-1)  # (E, C)
    out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
    out.index_add_(0, dst, m)
    return out


@torch.no_grad()
def diffuse_features(x_all: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_weight: torch.Tensor,
                     K: int = 2) -> torch.Tensor:
    """
    预计算K阶扩散特征：[X, A_norm X, A_norm^2 X, ...]
    训练/推理时几乎不加额外算力（只在进入GCN前做K次稀疏乘）
    """
    if K <= 0:
        return x_all

    num_nodes = x_all.size(0)
    w_norm = _gcn_norm(edge_index, edge_weight, num_nodes)
    outs = [x_all]
    cur = x_all
    for _ in range(K):
        cur = _spmm(edge_index, w_norm, cur, num_nodes)
        outs.append(cur)
    return torch.cat(outs, dim=1)  # (N, C*(K+1))


# ---------- 轻量平滑正则（可选） ----------
def laplacian_smooth_loss(y: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_weight: torch.Tensor) -> torch.Tensor:
    """
    简单的图平滑正则： sum_w || y_i - y_j ||^2 / E
    y: (N, C)
    """
    src, dst = edge_index
    diff = y[src] - y[dst]
    loss = (edge_weight.unsqueeze(-1) * (diff * diff)).mean()
    return loss


# ---------- 模型 ----------
class RGTGraphRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        sigma: float = 0.08,
        use_4nbr: bool = True,
        gated_residual_init: float = 0.2,
        add_norm: bool = True,
        use_diffuse_features: bool = True,  # 启用扩散特征
        diffuse_K: int = 2,                 # 扩散阶数K
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.use_4nbr = bool(use_4nbr)
        self.gate = nn.Parameter(torch.tensor(float(gated_residual_init)))
        self.add_norm = bool(add_norm)
        self.use_diffuse_features = bool(use_diffuse_features)
        self.diffuse_K = int(diffuse_K)

        in_feat = in_channels * (self.diffuse_K + 1) if self.use_diffuse_features else in_channels

        self.gcn1 = GCNConv(in_feat, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn2 = GCNConv(hidden_dim, out_channels, add_self_loops=True, normalize=True)

        if self.add_norm and _HAS_GRAPHNORM:
            self.norm = GraphNorm(hidden_dim)
        elif self.add_norm and not _HAS_GRAPHNORM:
            # 退化为LayerNorm（按特征维），避免依赖失败
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = None

    def _maybe_norm(self, y: torch.Tensor, batch_vec: Optional[torch.Tensor]) -> torch.Tensor:
        if self.norm is None:
            return y
        if _HAS_GRAPHNORM:
            return self.norm(y, batch=batch_vec)
        # LayerNorm 情况：逐节点的特征归一化
        return self.norm(y)

    def forward(self, feat: torch.Tensor, rgt_map: Optional[torch.Tensor]) -> torch.Tensor:
        """
        feat: (B, C, H, W)
        rgt_map: (B, 1, H', W') 或 None
        """
        if rgt_map is None:
            return feat

        B, C, H, W = feat.shape
        device = feat.device
        num_nodes = B * H * W

        # 调整RGT到当前特征大小
        rgt_resized = F.interpolate(rgt_map, size=(H, W), mode="nearest")

        # 构建批图（block-diag）
        edge_chunks = []
        weight_chunks = []
        base = 0
        with torch.no_grad():
            for b in range(B):
                e, w = build_edge_index_from_discrete_rgt_weighted(
                    rgt_resized[b, 0],
                    sigma=self.sigma,
                    use_4nbr=self.use_4nbr
                )
                if e.numel() == 0:
                    base += H * W
                    continue
                e = e + base  # 节点索引偏移
                edge_chunks.append(e)
                weight_chunks.append(w)
                base += H * W

            if not edge_chunks:
                return feat

            edge_index_all = torch.cat(edge_chunks, dim=1)   # (2, E_tot)
            edge_weight_all = torch.cat(weight_chunks, dim=0)  # (E_tot,)

            # 再次coalesce，确保完全去重
            edge_index_all, edge_weight_all = pyg_coalesce(
                edge_index_all, edge_weight_all, num_nodes=B * H * W, reduce='mean'
            )

        # 展平节点特征
        x_all = feat.permute(0, 2, 3, 1).contiguous().view(num_nodes, C)  # (N, C)

        # 预计算扩散特征：[X, AX, A^2X]
        if self.use_diffuse_features and self.diffuse_K > 0:
            with torch.no_grad():
                x_all = diffuse_features(x_all, edge_index_all, edge_weight_all, K=self.diffuse_K)

        # 构建batch向量用于GraphNorm
        batch_vec = None
        if self.norm is not None and _HAS_GRAPHNORM:
            batch_vec = torch.arange(B, device=device).repeat_interleave(H * W)  # (N,)

        # 两层GCN
        y = self.gcn1(x_all, edge_index_all, edge_weight=edge_weight_all)
        y = self._maybe_norm(y, batch_vec)
        y = F.relu(y, inplace=True)
        y = self.gcn2(y, edge_index_all, edge_weight=edge_weight_all)

        # 还原形状并做门控残差
        y = y.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        gate = torch.clamp(self.gate, 0.0, 1.0)
        return feat + gate * (y - feat)

class ApplyRGTRefiner(nn.Module):
    def __init__(self, refiner: RGTGraphRefiner):
        super().__init__()
        self.refiner = refiner

    def forward(self, x: torch.Tensor, rgt_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rgt_map is None:
            return x
        return self.refiner(x, rgt_map)

class MaskFormerDecoderGraph(nn.Module):
    def __init__(
        self,
        *,
        in_channels: List[int],
        num_classes: int, 
        transformer_dim: int,
        transformer: nn.Module, 
        num_multimask_outputs: int,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_graph: bool = True,
        **kwargs,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens+1, transformer_dim)
        
        concatenate_dim = sum(in_channels)
        self.fusion_layer = nn.Sequential(
            ConvBlock(concatenate_dim, concatenate_dim//2, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
            ConvBlock(concatenate_dim//2, transformer_dim, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
        )
        if use_graph:
            
            rgt_refiner1 = RGTGraphRefiner(transformer_dim // 1, transformer_dim // 1, transformer_dim // 1)
            rgt_refiner2 = RGTGraphRefiner(transformer_dim // 2, transformer_dim // 1, transformer_dim // 2)
            rgt_refiner3 = RGTGraphRefiner(transformer_dim // 4, transformer_dim // 2, transformer_dim // 4)
    
            self.output_upscaling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ApplyRGTRefiner(rgt_refiner1),
                ConvBlock(transformer_dim, transformer_dim // 2, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
                EfficientSelfAttention(transformer_dim // 2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ApplyRGTRefiner(rgt_refiner2),
                ConvBlock(transformer_dim // 2, transformer_dim // 4, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ApplyRGTRefiner(rgt_refiner3),
                ConvBlock(transformer_dim // 4, transformer_dim // 8, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
            )
        else:
            self.output_upscaling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvBlock(transformer_dim, transformer_dim // 2, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
                EfficientSelfAttention(transformer_dim // 2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvBlock(transformer_dim // 2, transformer_dim // 4, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvBlock(transformer_dim // 4, transformer_dim // 8, kernel_size=3, norm_func=nn.BatchNorm2d, actv_func=activation),
            )
                
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
        )

        self.class_prediction_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Linear(transformer_dim // 2, num_classes + 1),
        )   
        
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth,
            sigmoid_output=True,
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        prompt_weight: float,
        mask_inputs: torch.Tensor,
        original_size: Tuple[int, ...],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        b, _, h, w, _ = image_embeddings.shape
        x = einops.rearrange(image_embeddings, 'b l h w c -> b l c h w')
        x = x.reshape(b,-1,h,w)
        image_embeddings = self.fusion_layer(x)       

        masks, class_pred, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            prompt_weight=prompt_weight,
            mask_inputs=mask_inputs,
            original_size=original_size,
        )

        # Prepare output
        return masks, class_pred, iou_pred

    def output_upscaling_with_rgt(self, x, rgt_map):
        for layer in self.output_upscaling:
            if isinstance(layer, ApplyRGTRefiner):
                x = layer(x, rgt_map)
            else:
                x = layer(x)
        return x
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        prompt_weight: float,
        mask_inputs: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight
        
        output_tokens = output_tokens.unsqueeze(0).expand(dense_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # src0 = torch.sigmoid(src) * (1. - prompt_weight)
        # src1 = torch.sigmoid(dense_prompt_embeddings) * prompt_weight
        # src = src0 + src1
        
        b, c, h, w = src.shape
        
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        class_token_out = hs[:, 1:self.num_mask_tokens+1, :]
        mask_tokens_out = hs[:, 1:self.num_mask_tokens+1, :]
        
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling_with_rgt(src, mask_inputs)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1) 

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        # Generate mask quality predictions
        class_pred = self.class_prediction_head(class_token_out)
        iou_pred = self.iou_prediction_head(iou_token_out)
            
        return masks, class_pred, iou_pred
        
