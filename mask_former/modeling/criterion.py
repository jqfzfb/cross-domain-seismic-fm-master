# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from ..utils.misc import nested_tensor_from_tensor_list

from .ssim import MultiScaleSSIMLoss

def mask_to_box(masks):
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    return out

def compute_iou_group(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection_area = torch.max(torch.tensor(0), x2 - x1 + 1) * torch.max(torch.tensor(0), y2 - y1 + 1)

    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


class SetMaskFormerCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, mask_threshold=0.0):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.mask_threshold = mask_threshold
        self.register_buffer("empty_weight", empty_weight)
    
    def metric_ious(self, src_mask, tgt_mask):
        src_box = mask_to_box(src_mask.clone().detach())
        tgt_box = mask_to_box(tgt_mask.clone().detach())
        ious = compute_iou_group(src_box, tgt_box)
        return ious
    
    def loss_ious(self, outputs, targets, num_masks, indices):
        """IOU loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_ious" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        masks = [t["masks"].cpu() for t in targets]
        tgt_masks_, valid = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks = tgt_masks_[tgt_idx].clone().detach()

        src_masks = outputs["pred_masks"][src_idx].clone().detach()
        src_masks = F.interpolate(
            src_masks[:, None], 
            size=tgt_masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False,
        )[:, 0]
        src_masks = src_masks > self.mask_threshold
        tgt_masks = tgt_masks.to(src_masks)

        src_iou = outputs["pred_ious"][src_idx]
        tgt_iou = self.metric_ious(src_masks, tgt_masks).to(src_iou)
        
        loss_iou = F.mse_loss(src_iou, tgt_iou)
        losses = {"iou": loss_iou}
        return losses  
        
    def loss_labels(self, outputs, targets, num_masks, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)  
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        losses = {"ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, num_masks, indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        
        # TODO use valid to mask invalid areas due to padding in loss
        masks = [t["masks"] for t in targets]
        tgt_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks = tgt_masks.to(src_masks)
        tgt_masks = tgt_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=tgt_masks.shape[1:], mode="bilinear", align_corners=False
        )[:, 0]
        
        src_masks = src_masks.flatten(1)
        tgt_masks = tgt_masks.flatten(1)
        
        tgt_masks = tgt_masks.view(src_masks.shape)
        losses = {
            "focal": sigmoid_focal_loss(src_masks, tgt_masks, num_masks),
            "dice": dice_loss(src_masks, tgt_masks, num_masks),
        }
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_masks, indices):
        loss_map = {
            "labels": self.loss_labels, 
            "masks": self.loss_masks, 
            "ious": self.loss_ious,
        }
        assert loss in loss_map, f"Do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks, indices)

    def forward(self, outputs, targets, indices=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_masks, indices))
        return losses
    
class SetSegmentationCriterion(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx 
    
    def loss_masks(self, outputs, targets, num_masks, indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"][src_idx]
          
        # TODO use valid to mask invalid areas due to padding in loss
        masks = [t["masks"] for t in targets]
        tgt_masks, valid = nested_tensor_from_tensor_list(masks).decompose()

        tgt_masks = tgt_masks.to(src_masks)
        tgt_masks = tgt_masks[tgt_idx]
        
        src_masks = src_masks.flatten(1)
        tgt_masks = tgt_masks.flatten(1)
        
        tgt_masks = tgt_masks.view(src_masks.shape)
        losses = {
            "focal": sigmoid_focal_loss(src_masks, tgt_masks, num_masks),
            "dice": dice_loss(src_masks, tgt_masks, num_masks),
        }
        return losses

    def get_loss(self, loss, outputs, targets, num_masks, indices):
        loss_map = {
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"Do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks, indices)
    
    def forward(self, outputs, targets, indices=None, **kwargs):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["masks"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1).item()  
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_masks, indices))
        return losses


class SetRegressionCriterion(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses
        self.ssim_loss = MultiScaleSSIMLoss(channel=3, filter_size=7, reduction='mean')
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx 

    def loss_mse(self, outputs, targets, indices):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"][src_idx]
        masks = [t["masks"] for t in targets]
        
        tgt_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks = tgt_masks.to(src_masks)
        tgt_masks = tgt_masks[tgt_idx]

        src_masks = src_masks.float()
        tgt_masks = tgt_masks.float()
        
        if "weights" in targets[0].keys():
            weights = [t["weights"] for t in targets]
            tgt_weights, valid = nested_tensor_from_tensor_list(weights).decompose()
            tgt_weights = tgt_weights.to(src_masks)
            tgt_weights = tgt_weights[tgt_idx].float()
        else:
            tgt_weights = torch.ones_like(tgt_masks).to(src_masks).float()

        mse = F.mse_loss(src_masks * tgt_weights, tgt_masks * tgt_weights, reduction='sum') / tgt_weights.sum()
        return {"mse": mse}

    def loss_ssim(self, outputs, targets, indices):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"][src_idx]
        masks = [t["masks"] for t in targets]
        tgt_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks = tgt_masks.to(src_masks)
        tgt_masks = tgt_masks[tgt_idx]

        ssim = 1. - self.ssim_loss(src_masks.unsqueeze(1), tgt_masks.unsqueeze(1))
        return {"ssim": ssim}
    
    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {
            "mse": self.loss_mse,
            "ssim": self.loss_ssim,
        }
        assert loss in loss_map, f"Do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices)
    
    def forward(self, outputs, targets, indices=None, **kwargs):  
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        return losses


