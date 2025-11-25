import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

from dataset import prepare_image, preprocess_image, normalize_image

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class ImageEncodeEmbeddings():
    def __init__(self,
        pixel_mean,
        pixel_std,
        image_encode_size=1024,    
        model_type="vit_h",
        sam_checkpoint = "sam_vit_h_4b8939.pth",    
        device='cuda', 
    ):
        sam = sam_model_registry[model_type](
            image_size=image_encode_size, 
            checkpoint=os.path.join(
                'segment_anything', 
                'checkpoints', 
                sam_checkpoint,
            ),
        )
        self.image_encode_size = image_encode_size
        self.image_encoder = sam.image_encoder.to(device) 
        self.pixel_mean = [torch.tensor(x).view(-1, 1, 1) for x in pixel_mean]
        self.pixel_std = [torch.tensor(x).view(-1, 1, 1) for x in pixel_std]
        self.device = device
        
    def __call__(self, images):
        tensors = self.preprocess(images)
        embeddings, inner_features = [], []
        for tensor in tensors:
            embedding, inner_feature = self.encode(tensor)
            embeddings.append(embedding)
            inner_features.append(inner_feature[:, self.image_encoder.global_attn_indexes])
        return torch.cat(embeddings, dim=1), torch.cat(inner_features, dim=1)
    
    def encode(self, tensors):
        embeddings, inner_features = self.image_encoder(tensors.to(self.device))
        return embeddings.cpu(), inner_features.cpu()
    
    def preprocess(self, images):
        tensors = list()
        for idx, image in enumerate(images):
            tensor = prepare_image(
                image, 
                ResizeLongestSide(self.image_encode_size),
            ).float()
            tensor = normalize_image(tensor,                
                                     self.pixel_mean[idx],
                                     self.pixel_std[idx])
            tensor = preprocess_image(
                tensor, 
                self.image_encode_size,
            ).unsqueeze(0).float()
            tensors.append(tensor)
        return tensors

from segment_anything.build_sam_former import SAMFormer
class SDSAM(nn.Module):
    def __init__(self, param):
        super(SDSAM, self).__init__()  
        self.sam = SAMFormer(param)

    def encode(self, batched_dict, 
                **kwargs):
        batched_input = batched_dict
        return self.sam.encode(batched_input)

    def forward(self, 
                batched_dict, 
                **kwargs):
        batched_input = batched_dict
        return self.sam(batched_input)


    
    
    
    
    
    
    
    
    