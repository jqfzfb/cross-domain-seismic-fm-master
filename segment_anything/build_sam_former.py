import os
import sys
import random
import numpy as np
import einops
import torch
from torch import nn
from torch.nn import functional as F

from sklearn.decomposition import PCA

from typing import Any, Dict, List, Tuple
from typing import Optional, Type  

from . import sam_model_registry
from .modeling import PromptEncoder, TwoWayTransformer, MaskDecoder
from .modeling import MaskFormerDecoder, MaskFormerDecoderGraph, MaskEncoderViT, MaskEncoderLoRA, MaskEncoder, PromptEncoder
from .utils.transforms import ResizeLongestSide
from .modeling.common import ConvBlock1d, ResidualConvBlock1d

class SAMFormer(nn.Module):
    def __init__(
        self, 
        param: Dict[str, Any],
    ) -> None:
        
        super().__init__()
        
        pixel_std=np.array(param.get('pixel_std', []),
                           dtype=np.single)
        pixel_mean=np.array(param.get('pixel_mean', []),
                            dtype=np.single)
        num_classes=len(param.get('classes', []))
        
        sam_checkpoint=param['sam_checkpoint']
        vit_type = param['model_type']
        image_type = param['image_type']
        num_multimask_outputs = param['num_multimask_outputs']
        
        assert vit_type in ['vit_h', 'vit_l', 'vit_b'], f"Unexpected model type: {model_type}!"
        vit_dict = {
            'vit_h': {
                'feat_inner_chan': 1280,
                'selected_inner_chans': [7, 15, 23, 31],
                'checkpoint_file': "sam_vit_h_4b8939.pth",
            },
            'vit_l':{
                'feat_inner_chan': 1024,
                'selected_inner_chans': [5, 11, 17, 23],
                'checkpoint_file': "sam_vit_l_0b3195.pth",
            },
            'vit_b':{
                'feat_inner_chan': 768,
                'selected_inner_chans': [2, 5, 8, 11],
                'checkpoint_file': "sam_vit_b_01ec64.pth",
            }
        }   

        mask_in_chans = param.get("mask_in_chans", 16)
        mask_decoder_depth = param.get("mask_decoder_depth", 2)
        mask_decoder_heads = param.get("mask_decoder_heads", 8)
        image_encode_size = param.get('image_encode_size', 1024)
        image_encode_dim = param.get('image_encode_dim', 1024) 
        mask_mlp_hidden_dim = param.get("mask_mlp_hidden_dim", 2048)
        num_lora = param.get("num_lora", 1)
        use_graph = param.get('use_graph', False)
        self.prompt_weight = param.get('prompt_weight', 0.5)
        self.image_encode_size = image_encode_size
        self.transfer = param["transfer"]
        self.task = param["task_mode"]
        self.encoder_type = param['encoder_type']
        self.selected_inner_chans = vit_dict[vit_type]['selected_inner_chans']
        self.low_res_size = image_encode_size // 4
        
        patch_size = 16 
        self.image_embed_size = image_encode_size // patch_size
        self.resize_transform = ResizeLongestSide(image_encode_size) 
        
        feat_inner_chan = vit_dict[vit_type]['feat_inner_chan']
        
        if len(pixel_mean.shape) > 1:
            image_adapter = False 
        else:
            image_adapter = True
            
        # Encoder
        if self.encoder_type == 'vit':
            if sam_checkpoint:
                checkpoint_file = vit_dict[vit_type]['checkpoint_file']
                checkpoint_path = os.path.join(
                    'segment_anything', 
                    'checkpoints', 
                    checkpoint_file,
                )
            else:
                checkpoint_path = None 
                
            sam = sam_model_registry[vit_type](
                image_size=1024, 
                checkpoint=checkpoint_path,
            )
            
            in_chans = 0
            for x in image_type:
                if x.endswith('image'):
                    in_chans += 3
                else:
                    in_chans += 1
            self.in_chans = in_chans
            
            if self.transfer == "refine":
                self.image_encoder = MaskEncoderViT(
                    in_chans=in_chans,
                    image_encoder=sam.image_encoder,
                    vit_inner_chans=vit_dict[vit_type]['selected_inner_chans'],
                    image_adapter=image_adapter,
                )  

            elif self.transfer == "freeze":
                self.image_encoder = MaskEncoderViT(
                    in_chans=in_chans,
                    image_encoder=sam.image_encoder,
                    vit_inner_chans=vit_dict[vit_type]['selected_inner_chans'],
                    image_adapter=image_adapter,
                    freeze_blocks=True,
                )  
            
            elif self.transfer == "lora":
                self.image_encoder = MaskEncoderLoRA(
                    in_chans=len(image_type),
                    image_encoder=sam.image_encoder,
                    vit_inner_chans=vit_dict[vit_type]['selected_inner_chans'],
                    image_adapter=image_adapter,
                    num_lora=num_lora,
                )
                
        else:  
            self.image_encoder = MaskEncoder(
                in_chans=len(image_type) if image_adapter else len(image_type) * 3,
                out_chans=feat_inner_chan * 4,
                backbone=self.encoder_type,
            )
            
        # Prompter
        self.prompt_encoder = PromptEncoder(
            embed_dim=image_encode_dim,
            image_embedding_size=(self.image_embed_size, 
                                  self.image_embed_size),
            input_image_size=(image_encode_size, image_encode_size),
            mask_in_chans=mask_in_chans,
        )
            
        # Decoder
        decoder_in_channels = [feat_inner_chan] * 4
        if self.task in ['match', 'segment', 'regress']:
            if use_graph:
                self.mask_decoder = MaskFormerDecoderGraph(
                    in_channels=decoder_in_channels,
                    num_classes=num_classes,
                    transformer_dim=image_encode_dim,
                    transformer=TwoWayTransformer(
                        depth=mask_decoder_depth,
                        embedding_dim=image_encode_dim,
                        mlp_dim=mask_mlp_hidden_dim,
                        num_heads=mask_decoder_heads,
                    ),
                    num_multimask_outputs=num_multimask_outputs, 
                    use_graph=use_graph,
                )
            else:
                self.mask_decoder = MaskFormerDecoder(
                    in_channels=decoder_in_channels,
                    num_classes=num_classes,
                    transformer_dim=image_encode_dim,
                    transformer=TwoWayTransformer(
                        depth=mask_decoder_depth,
                        embedding_dim=image_encode_dim,
                        mlp_dim=mask_mlp_hidden_dim,
                        num_heads=mask_decoder_heads,
                    ),
                    num_multimask_outputs=num_multimask_outputs, 
                )
 
        if len(pixel_mean):
            self.register_buffer(
                "pixel_mean", 
                torch.stack([torch.tensor(x).view(-1, 1, 1) for x in pixel_mean]),
                False,
            )
            
        if len(pixel_std):
            self.register_buffer(
                "pixel_std", 
                torch.stack([torch.tensor(x).view(-1, 1, 1) for x in pixel_std]),
                False,
            )     
        
    @property
    def device(self) -> Any:
        return self.pixel_mean.device 
        
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]: 

         # load image embeddings      
        if self.transfer in ["refine", "lora", "freeze"]:
            input_images = torch.cat([x["image"] for x in batched_input], dim=0)
            features = self.image_encoder(input_images)
        else:
            assert False, f"Unexpected {self.transfer}!"
        
        outputs = []
        for k, (image_record, curr_embeddings) in enumerate(zip(batched_input, features)):

            curr_embeddings = curr_embeddings.unsqueeze(0) # 1, t, h, w, c
            masks, points, boxes = None, None, None
            if "point_coords" in image_record:
                points = (
                    image_record["point_coords"],
                    image_record["point_labels"],
                )
                
            if "boxes" in image_record:
                boxes = image_record["boxes"]

            mask_inputs = None
            if "mask_inputs" in image_record:
                mask_inputs = image_record['mask_inputs']
                if len(mask_inputs.shape) < 4:
                    mask_inputs = mask_inputs[:,None,:,:]
                masks = F.interpolate(
                    mask_inputs,
                    [self.low_res_size]*2,
                    mode = 'bilinear',
                    align_corners=True,
                )

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points, 
                boxes=boxes,
                masks=masks,
            )      

            original_size = image_record["original_size"]
            low_res_masks, class_predictions, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                prompt_weight=self.prompt_weight,
                mask_inputs=mask_inputs,
                original_size=original_size,
            )

            mask_predictions = self.postprocess_masks(
                low_res_masks,
                original_size=original_size,
            )
            
            outputs.append(
                {
                    "pred_masks": mask_predictions,
                    "pred_logits": class_predictions,
                    "pred_ious": iou_predictions,
                    "UID": image_record['UID'].tile(mask_predictions.shape[0]),
                }
            )
        return outputs                      

    def postprocess_sequence(
        self,
        masks: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        masks = F.interpolate(masks, target_length, 
                              mode="linear", 
                              align_corners=False,
                             )
        return masks
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
        mode: str="bilinear",
    ) -> torch.Tensor:

        rescale_size = self.resize_transform.get_preprocess_shape(
            original_size[0], 
            original_size[1], 
            self.resize_transform.target_length,
        )
        
        masks = F.interpolate(masks, self.resize_transform.target_length, mode=mode, align_corners=False)
        masks = masks[..., : rescale_size[0], : rescale_size[1]]
        masks = F.interpolate(masks, original_size.tolist(), mode=mode, align_corners=False)
        return masks
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def pca_reduce(self, feats, n_components=12):
        w, h, c = feats.shape
        X = feats.reshape(w*h,c)
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        feats_reduced = X_reduced.reshape(w,h,-1)
        feats_reduced = feats_reduced[:,:,:3]
        return feats_reduced
    
    def encode(
        self,
        batched_input: List[Dict[str, Any]],
        n_components: int = 4,
    ) -> torch.Tensor: 
        
        input_images = torch.cat([x["image"] for x in batched_input], dim=0)
         # load image embeddings      
        if self.transfer in ["refine", "lora", "freeze"]:
            inner_features = self.image_encoder(input_images)
        else:
            assert False, f"Unexpected {self.transfer}!"
            
        image_record = batched_input[0]
        original_size = image_record["original_size"]
        b, _, w, h, c = inner_features.shape
        
        inner_feats = inner_features.detach().cpu()
        inner_feats = inner_feats.permute(0,2,3,1,4).reshape(b, w, h, -1)

        rescale_size = self.resize_transform.get_preprocess_shape(
            original_size[0], 
            original_size[1], 
            inner_feats.shape[1],
        )

        inner_feats = inner_feats[:, : rescale_size[0], : rescale_size[1]]
        
        feats_reduce = []
        for inner_feat in inner_feats:
            inner_feat = self.pca_reduce(inner_feat, n_components=n_components)
            inner_feat = torch.tensor(inner_feat).permute(2,0,1)
            feats_reduce.append(inner_feat)
        feats_reduce = torch.stack(feats_reduce)

        feats_reduce = F.interpolate(
            feats_reduce,
            [int(x) for x in original_size],
            mode='nearest',
        )
        return feats_reduce



