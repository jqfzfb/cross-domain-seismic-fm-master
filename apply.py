import os
import random
import numpy as np

import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchmetrics import JaccardIndex

from segment_anything.utils.transforms import ResizeLongestSide
from mask_former.modeling.matcher import HungarianMatcher
from torchmetrics.classification import Accuracy, JaccardIndex, AveragePrecision
from torchmetrics.classification import MatthewsCorrCoef, CohenKappa, F1Score, Recall

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

from dataset import numpy2gray, mask_to_box
from dataset import min_max_norm, mea_std_norm
from utils import *

from PIL import Image
from sklearn.decomposition import PCA

def extract_well_match(raw_data_dict, pred_probs, valid_well_names):
    well_data = raw_data_dict['well']
    this_well_names = raw_data_dict['name']
    well_indices = []

    for well_name in valid_well_names:
        if well_name in this_well_names:
            k = this_well_names.index(well_name)
            well_indices.append((well_name, k))

    well_match = {}
    for well_name, well_index in well_indices:
        k = raw_data_dict['index'][well_index]
        well_trace_pred = pred_probs[:, k]
        well_trace_tgt = well_data[:, k]
        well_match[well_name] = [well_trace_tgt, well_trace_pred]
    
    return well_match

def pca_reduce(feats, n_components=12):
    w, h, c = feats.shape
    X = feats.reshape(w*h,c)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    feats_reduced = X_reduced.reshape(w,h,-1)
    feats_reduced = feats_reduced[:,:,:3]
    
    return feats_reduced

def data2rgb(data, mask=None):
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_scaled = (255 * data_normalized).astype(np.uint8)
    
    if mask is not None:
        data_scaled[mask == 1] = [0, 0, 0]

    image = Image.fromarray(data_scaled, 'RGB')
    image_rgb = image.convert('RGB')
    return np.array(image_rgb, np.uint8)

def is_overlapping(region1, region2):
    # Unpack the coordinates
    x1_region1, x2_region1, y1_region1, y2_region1 = region1
    x1_region2, x2_region2, y1_region2, y2_region2 = region2
    # Check for overlap
    overlap = not (x2_region1 < x1_region2 or x1_region1 > x2_region2 or
                   y2_region1 < y1_region2 or y1_region1 > y2_region2)
    return overlap

def remove_mask_topo(masks, topo, threshold=150, mode=None):
    keep = []
    boxs = mask_to_box(masks)
    for i, (box, mask) in enumerate(zip(boxs, masks)): 
        i1, i2 = (box[0] + box[2])//2, (box[1] + box[3])//2
        mask_values = topo[mask.astype(np.bool_)]
        mask_interv = max(abs(topo[i2,i1] - mask_values.max()), 
                          abs(topo[i2,i1] - mask_values.min()))
        
        if mask_interv <= threshold:
            continue
            
        topo_mean = mask_values.mean()
        if mode == '+':
            if topo[i2,i1] <= topo_mean:
                continue
        elif mode == '-':
            if topo[i2,i1] >= topo_mean:
                continue          
                
        keep.append(i)
    return keep   
    
def remove_masks_size(masks, min_size=64, max_size=None):
    keep = []
    boxes = mask_to_box(masks)
    if max_size is None:
        for k, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > min_size:
                keep.append(k)
    else:
        for k, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > min_size and area < max_size:
                keep.append(k)
    return np.array(keep)     
    
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算每个检测框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou

def non_max_suppression(boxes, scores, iou_threshold):
    assert len(boxes) == len(scores)

    # 将检测框和分数按照分数排序（从高到低）
    sorted_indices = np.argsort(scores)[::-1]

    keep = []
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)

        if len(sorted_indices) == 1:
            break

        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        remaining_boxes = [boxes[i] for i in remaining_indices]

        ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])
        remaining_indices = remaining_indices[np.where(ious <= iou_threshold)[0]]

        sorted_indices = remaining_indices

    return keep

def remove_boxes_nms(boxes, scores, iou_threshold=0.5):
    return non_max_suppression(boxes, scores, iou_threshold)  

def remove_masks_nms(masks, scores, iou_threshold=0.5):
    boxes = dataset.mask_to_box(masks)
    return non_max_suppression(boxes, scores, iou_threshold)      
    
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

def loss_masks(src_masks, tgt_masks, num_masks):
    src_masks = torch.tensor(src_masks).flatten(1)
    tgt_masks = torch.tensor(tgt_masks).flatten(1)
    tgt_masks = tgt_masks.view(src_masks.shape)
    losses = {
        "focal": sigmoid_focal_loss(src_masks, tgt_masks, num_masks),
        "dice": dice_loss(src_masks, tgt_masks, num_masks),
    }
    return losses
    
class SegmentationMetrics:
    def __init__(
        self,
    ):

        self.metrics = {
            'acc': Accuracy(task="binary"),
            'jacc': JaccardIndex(task="binary"),
            'macc': MatthewsCorrCoef(task="binary"),
            'ap': AveragePrecision(task="binary"),
            'ck': CohenKappa(task="binary"),
            'f1': F1Score(task="binary"),
            'recall': Recall(task="binary"),
        }        
        self.threshold = 10
        
    def get_metric(self, preds, targets, key):
        targets_tensor = torch.as_tensor(
            targets, 
            dtype=torch.long,
        )
        preds_tensor = torch.as_tensor(
            preds, 
            dtype=torch.float,
        )
        
        scores = []
        for k in range(preds_tensor.shape[0]):
            pred = torch.clip(
                preds_tensor[k], 
                min=-self.threshold, 
                max= self.threshold,
            )
            target = targets_tensor[k]
            scores.append(
                self.metrics[key](
                    self.min_max_norm(pred),
                    target,
            ))
        score = sum(scores) / len(scores)
        return score

    def min_max_norm(self, x):
        x = x - torch.min(x)
        x = x / torch.max(x)  
        return x
    
    def get_segmentation(self, masks, labels):
        _, w, h = masks.shape
        logits = np.zeros([w, h], dtype=np.int64)
        for msk, lab in zip(masks, labels):
            logits += msk.astype(np.int64) * (lab + 1)
        return logits    

    def __call__(self, preds, targets):    
        metric_dict = dict()
        for key in self.metrics.keys():
            metric_dict[key] = self.get_metric(preds, targets, key)
        return metric_dict

    
class RegressionMetrics:
    def __init__(self, kernel_size=5):
        self.kernel_size=kernel_size
        self.metrics = {
            'mae': MeanAbsoluteError(),
            'mse': MeanSquaredError(),
            'rmse': None,
            'psnr': PSNR(data_range=1.0),
            'ssim': SSIM(data_range=1.0, kernel_size=kernel_size),
        }
        
    def min_max_norm(self, x):
        x = x - torch.min(x)
        max_val = torch.max(x)
        return x / max_val if max_val > 0 else x

    def compute_rmse(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

    def get_metric(self, preds, targets, key):
        preds_tensor = torch.as_tensor(preds, dtype=torch.float32)
        targets_tensor = torch.as_tensor(targets, dtype=torch.float32)

        scores = []
        for k in range(preds_tensor.shape[0]):

            pred = preds_tensor[k]
            
            target = targets_tensor[k]

            if key in ['ssim', 'psnr']:
                pred = self.min_max_norm(pred[None,...])
                target = self.min_max_norm(target[None,...])

            if key == 'rmse':
                score = self.compute_rmse(pred, target)
            else:
                score = self.metrics[key](pred.unsqueeze(0), target.unsqueeze(0))  # Add batch dim

            scores.append(score)

        return sum(scores) / len(scores)

    def __call__(self, preds, targets):
        metric_dict = {}
        for key in self.metrics.keys():
            metric_dict[key] = self.get_metric(preds, targets, key)
        return metric_dict

class SequenceMetrics:
    def __init__(self):
        self.metrics = {
            'mae': self.mean_absolute_error,
            'mse': self.mean_squared_error,
            'rmse': self.root_mean_squared_error,
            'corr': self.correlation_coefficient, 
        }

    def mean_absolute_error(self, pred, target):
        return torch.mean(torch.abs(pred - target))

    def mean_squared_error(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def root_mean_squared_error(self, pred, target):
        return torch.sqrt(self.mean_squared_error(pred, target))

    def correlation_coefficient(self, pred, target):
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        num = torch.sum((pred - pred_mean) * (target - target_mean))
        den = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((target - target_mean)**2))
        return num / den if den > 0 else torch.tensor(0.0)

    def get_metric(self, preds, targets, key):
        preds_tensor = torch.as_tensor(preds, dtype=torch.float32)
        targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
        return self.metrics[key](preds_tensor, targets_tensor)

    def __call__(self, preds, targets):
        return {key: self.get_metric(preds, targets, key) for key in self.metrics}

class Valid:
    def __init__(
        self,
        param,
        model,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_GPU = param['num_GPU']
        self.net_name = param['model_name']
        self.mask_threshold = param['mask_threshold']
        self.pixel_mean, self.pixel_std = None, None
            
        self.prompt_types = param['prompt_types'] 
        self.image_encode_size = param['image_encode_size']
        self.resize_transform = ResizeLongestSide(self.image_encode_size)
        self.task = param['task_mode']
        self.device = device 
        self.model = model.module if self.num_GPU > 1 else model
        self.model = self.model.to(device)   
        
        num_multimask_outputs = param['num_multimask_outputs']
        loss_type = param['loss_type']

        self.matcher = None
        if self.task == 'match':
            self.matcher = HungarianMatcher(
                cost_class=loss_type['ce'],
                cost_mask=loss_type['focal'],
                cost_dice=loss_type['dice'],
            )

        if  self.prompt_types is not None and len(self.prompt_types):
            self.prompter = Prompter(
                self.prompt_types,
                mask_threshold = self.mask_threshold,
                image_encode_size = self.image_encode_size,
                points_per_side = param.get('points_per_side', 80),
                num_add_points = param.get('num_add_points', 1),  
                pos_point_rate = param.get('pos_point_rate', 0.5),
                task=self.task,
                mode='Valid',
            )  
        else:
            self.prompter = None
            
    def get_pred_masks(self, pred_masks_ts):
        pred_probs = pred_masks_ts.numpy().astype(np.single)   
        pred_masks = pred_probs > self.mask_threshold
        pred_masks = pred_masks.astype(np.single) 
        return pred_masks
    
    def get_pred_logits(self, pred_logits_ts, dim=0):
        pred_logits =  pred_logits_ts
        pred_labels = torch.argmax(F.softmax(pred_logits, dim=dim), dim=dim)   
        pred_labels = pred_labels.numpy().astype(np.int64)
        return pred_labels
    
    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = np.concatenate([np.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = np.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = np.concatenate([np.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = np.concatenate([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_output(self, batched_sample, batched_output, batched_index=None):  
        masks = batched_sample['masks']
        labels = batched_sample.get('labels', None)
        
        original_size = batched_sample['original_size']
        arr = np.array(original_size).astype(int).ravel()
        original_size = (int(arr[0]), int(arr[1]))  

        collect_output = dict(batched_sample)
        
        if 'mask_inputs' in collect_output.keys():
            del collect_output['mask_inputs']
            
        if 'mask_inputs' in batched_sample.keys():   

            if 'masks' in self.prompt_types:
                collect_output['prompts'] = F.interpolate(
                    torch.as_tensor(
                        batched_sample['mask_inputs'][:,None,:,:], 
                        dtype=torch.float,
                    ),
                    original_size,
                    mode='bilinear',
                    align_corners=True,
                    ).cpu().numpy().squeeze() 
                
            elif 'pred_masks' in self.prompt_types:
                collect_output['prompts'] = F.interpolate(
                    torch.as_tensor(
                        batched_sample['mask_inputs'][:,None,:,:], 
                        dtype=torch.float,
                    ),
                    original_size,
                    mode='bilinear',
                    align_corners=True,
                    ).cpu().numpy().squeeze()
                
            elif 'edges' in self.prompt_types:
                collect_output['prompts'] = F.interpolate(
                    torch.as_tensor(
                        batched_sample['mask_inputs'][:,None,:,:], 
                        dtype=torch.float,
                    ),
                    original_size,
                    mode='nearest',
                    ).cpu().numpy().squeeze()   
                
            elif 'attrs' in self.prompt_types:               
                mask_inputs = batched_sample['mask_inputs']
                collect_output['prompts'] = F.interpolate(
                    torch.as_tensor(
                        mask_inputs[None,:,:,:],
                        dtype=torch.float,
                    ),
                    original_size,
                    mode='bilinear',
                    align_corners=True,
                    ).cpu().numpy().squeeze()    
        
        if 'point_coords' in batched_sample.keys():
            collect_output['point_coords']  = batched_sample['point_coords']
            collect_output['point_labels']  = batched_sample['point_labels']
            
        if 'boxes' in batched_sample.keys():
            collect_output['boxes']  = batched_sample['boxes']

        pred_probs = batched_output['pred_masks'].clone().detach().cpu()
        pred_probs = pred_probs.numpy()

        if self.task not in ['inverse', 'regress']:
            pred_masks = self.get_pred_masks(batched_output['pred_masks'].clone().detach().cpu())
            
        if 'pred_logits' in batched_output.keys() and batched_output['pred_logits'] is not None:
            pred_labels = self.get_pred_logits(batched_output['pred_logits'].clone().detach().cpu(), dim=-1)
            
        if 'pred_ious' in batched_output.keys() and batched_output['pred_ious'] is not None:
            pred_ious = batched_output['pred_ious'].clone().detach().cpu()
            
        for j in range(len(batched_index)):
            batched_index[j] = [x.clone().detach().cpu() for x in batched_index[j]]
        
        if self.task == 'match':      
            src_idx = self.get_src_permutation_idx(batched_index)
            tgt_idx = self.get_tgt_permutation_idx(batched_index)
            
            collect_output['masks'] = masks[tgt_idx]
            collect_output['pred_ious'] = pred_ious[src_idx]
            collect_output['pred_ious'] = pred_ious[src_idx]
            collect_output['pred_probs'] = pred_probs[src_idx]
            collect_output['pred_masks'] = pred_masks[src_idx]
            collect_output['pred_labels'] = pred_labels[src_idx]
            
        elif self.task in ['segment']:
            src_idx = self.get_src_permutation_idx(batched_index)
            tgt_idx = self.get_tgt_permutation_idx(batched_index)
            
            collect_output['masks'] = masks[tgt_idx]
            collect_output['pred_probs'] = pred_probs[src_idx]
            collect_output['pred_masks'] = pred_masks[src_idx]
            
        elif self.task in ['regress', 'inverse']:
            src_idx = self.get_src_permutation_idx(batched_index)
            tgt_idx = self.get_tgt_permutation_idx(batched_index)
            
            collect_output['masks'] = masks[tgt_idx]
            collect_output['pred_probs'] = pred_probs[src_idx]
            
        return collect_output
    
    def apply(self, batched_samples, prompt_weight=0.5, norm=None):
        self.model.eval()
        collect_outputs = []
        for k, batched_sample in enumerate(batched_samples): 
            batched_index = None
            batched_outputs = None

            if self.prompter is not None:
                batched_sample = self.prompter(
                    [batched_sample], 
                     batched_outputs,
                    [batched_index],
                )[0]

            batched_inputs = batched_samples_torch(
                [batched_sample], 
                self.resize_transform, 
                self.image_encode_size, 
                self.device,
            )

            
            batched_outputs = self.model(
                batched_dict=batched_inputs, 
                prompt_weight=prompt_weight,
            )                    
    
            batched_indices = get_index(batched_outputs, batched_inputs, self.matcher)
            batched_index = batched_indices[0]
                
            collect_output = self.get_output(batched_sample, batched_outputs[0], batched_index)             
            collect_outputs.append(collect_output)
                
        return collect_outputs    
    
    def encode(self, batched_samples):
        self.model.eval()
        collect_outputs = [[]]
        for k, batched_sample in enumerate(batched_samples): 
            batched_inputs = batched_samples_torch(
                [batched_sample], 
                self.resize_transform, 
                self.pixel_mean, 
                self.pixel_std, 
                self.image_encode_size, 
                self.device,
            )

            inner_features = self.model.encode(
                batched_dict=batched_inputs, 
            )[0]              

            collect_output = dict(batched_sample) 
            collect_output['inner_features'] = inner_features.clone().detach().cpu().numpy()
            collect_outputs[0].append(collect_output)
        return collect_outputs
    
    def __call__(self, dataset, prompt_weight=0.5, path=None):
        for idx, data_sample in enumerate(dataset):
            outputs = self.apply([data_sample], prompt_weight)
            outputs_list = [outputs[j][0] for j in range(len(outputs))]
            name = '%05.d' % idx
            np.save(os.path.join(path, f"sample_{name}.npy"), outputs_list)

