import os
import glob
import time
import random
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import cv2
from skimage import measure, filters, morphology
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter
from dataset import mask_to_box, jitter_box, jitter_mask, postprocess_mask
import matplotlib.pyplot as plt
from pprint import pprint

from segment_anything.utils.transforms import ResizeLongestSide
from mask_former.modeling.matcher import HungarianMatcher
import mask_former
import dataset

class WarmupWrapper(torch.optim.lr_scheduler._LRScheduler):
    """
    Wrap any scheduler with warmup logic.
    """
    def __init__(self, optimizer, base_scheduler, warmup_steps, last_epoch=-1):
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
            
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, epoch=None, metrics=None):
        if self.last_epoch >= self.warmup_steps:
            if isinstance(self.base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step(epoch)
        super().step(epoch)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    
    if args['optimizer_type'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
        
    elif args['optimizer_type'] == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }     
        
    elif args['optimizer_type'] == 'Adamax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }        
        
    elif args['optimizer_type'] == 'SparseAdam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
        
    elif args['optimizer_type'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}
        
    elif args['optimizer_type'] == 'AdamW':
        optimizer_function = optim.AdamW
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args['lr'],
            'weight_decay': args['weight_decay']
        }
        
    kwargs['lr'] = args['lr']
    kwargs['weight_decay'] = args['weight_decay']
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args['decay_type'] == 'StepLR':
        base_scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    elif args['decay_type'] == 'MultiStepLR':
        milestones = args['decay_type'].split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        base_scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args['gamma']
        )
    elif args['decay_type'] == 'ReduceLROnPlateau':
        base_scheduler = lrs.ReduceLROnPlateau(
            my_optimizer, 
            mode='min', 
            patience=args['lr_decay'], 
            factor=args['gamma'])
    else:
        base_scheduler = None

    if args.get('warmup_steps', 0) > 0 and base_scheduler is not None:
        scheduler = WarmupWrapper(
            optimizer=my_optimizer,
            base_scheduler=base_scheduler,
            warmup_steps=args['warmup_steps']
        )
    else:
        scheduler = base_scheduler
        
    return scheduler

# 定义数据集    
class build_dataset_facies(Dataset):
    def __init__(self, 
                 param,
                 samples_list, 
                 norm=None,
                 mode='Train', 
                ):
        self.samples_list = samples_list
        self.norm = norm if norm is not None else lambda x:x
        self.mode = mode
        self.classes = param['classes']
        self.task = param['task_mode']
        self.image_type = param.get('image_type', ['data']) 
        self.selected_chans = param.get('selected_chans', [-1]) 
        self.max_num_mask = param.get('num_multimask_outputs', 10)
        
    def get(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = int(idx)
        data_dict = np.load(os.path.join(self.samples_list[idx]), 
                            allow_pickle=True).item()
        return data_dict
    
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_file_path = self.samples_list[idx]
        data_dict = np.load(data_file_path, allow_pickle=True).item()
        
        # images and masks
        images = []
        for key in self.image_type:
            data_image = data_dict[key]

            if   self.norm == "min_max":
                data_image = dataset.min_max_norm(data_image)
            elif self.norm == "mea_std":
                data_image = dataset.mea_std_norm(data_image) 
                
            images.append(data_image)
            
        if len(next(iter(images)).shape) < 3:
            image = np.stack(images, axis=0)
        else:
            image = np.concatenate(images, axis=-1)
        
        data_sample = {}
        data_sample['image'] = image
        data_sample['UID'] = np.array([idx]) 
        
        if np.array(image.shape).argmin() == 0:
            data_sample['original_size'] = np.array(image.shape[1:]) 
        else:
            data_sample['original_size'] = np.array(image.shape[:2]) 

        if 'mask' in data_dict.keys():
            masks, labels = dataset.generate_multiple_mask_label(data_dict['mask'], self.classes) 
                
            num_mask = len(masks)
            if num_mask > self.max_num_mask:
                index = random.sample(list(range(num_mask)), self.max_num_mask)
                masks, labels = masks[index], labels[index]
            data_sample['masks'] = masks[None,:,:,:]
        
        if 'labels' in data_dict.keys():
            labels = data_dict['labels']
        data_sample['labels'] = labels[None,:]

        for key in ['attributes']:
            if key in data_dict.keys():
                if data_dict[key] is None:
                    del data_dict[key]
                    continue

                attr_image = data_dict[key] 
                attr_image = dataset.min_max_norm(attr_image)
                data_sample['attrs'] = attr_image    
                
        return data_sample    
        

from scipy.ndimage import distance_transform_edt
class build_dataset_regression(Dataset):
    def __init__(self, 
                 param,
                 samples_list, 
                 norm=None,
                 mode='Train', 
                ):
        self.samples_list = samples_list
        self.norm = norm
        self.mode = mode
        self.image_type = param.get('image_type', ['data']) 

    def build_W_weak_distance(self,
                              weights,
                              tau=None,
                              sampling=None,
                              normalize=True,
                              zero_hard=False,
                              min_val=0.05,
                              max_val=5.0,
                              dtype=np.float32):
        
        Mh = np.asarray(weights).astype(bool)
        H, W_ = Mh.shape
        
        if not Mh.any():
            W = np.ones_like(Mh, dtype=dtype)
            return W
    
        D = distance_transform_edt(~Mh, sampling=sampling)
    
        if tau is None:
            tau = max(H, W_)
    
        W = np.exp(-D / float(tau)).astype(dtype)
    
        if zero_hard:
            W[Mh] = 0.0
    
        if normalize:
            W = W / (W.mean() + 1e-8)
        return np.clip(W, min_val, max_val)

    def get(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = int(idx)
        data_dict = np.load(os.path.join(self.samples_list[idx]), 
                            allow_pickle=True).item()
        return data_dict
    
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_file_path = self.samples_list[idx]
        data_dict = np.load(data_file_path, allow_pickle=True).item()
        
        # images and masks
        images = []
        for key in self.image_type:
            data_image = data_dict[key]
            if   self.norm == "min_max":
                data_image = dataset.min_max_norm(data_image)
            elif self.norm == "mea_std":
                data_image = dataset.mea_std_norm(data_image) 
            images.append(data_image)
            
        if len(next(iter(images)).shape) < 3:
            image = np.stack(images, axis=0)
        else:
            image = np.concatenate(images, axis=-1)
        
        data_sample = {}
        data_sample['image'] = image
        data_sample['UID'] = np.array([idx]) 
        
        if np.array(image.shape).argmin() == 0:
            data_sample['original_size'] = np.array(image.shape[1:]) 
        else:
            data_sample['original_size'] = np.array(image.shape[:2]) 

        if 'mask' in data_dict.keys():
            masks = data_dict['mask']
            if   self.norm == "min_max":
                masks = dataset.min_max_norm(masks)
            elif self.norm == "mea_std":
                masks = dataset.mea_std_norm(masks) 
            data_sample['masks'] = masks[None,None,:,:]

        data_sample['labels'] = np.zeros(
            1, 
            dtype=np.int64,
        )[None, :]

        if 'well' in data_dict.keys(): 
            weights = data_dict['well'] > 0.0
            weights = self.build_W_weak_distance(weights)
            data_sample['weights'] = weights[None,None,:,:]   

        if 'rgt' in data_dict.keys():
            data_sample['attrs'] = data_dict['rgt'][None,:,:]    
        
        return data_sample    

def batched_samples_torch(
    batched_samples, 
    transform, 
    image_encode_size, 
    device,
):      
    batched_inputs = []
    for batched_sample in batched_samples:
        original_size = batched_sample['original_size']
        sampled_inputs = dict()
        for key in batched_sample.keys():
            if key in ['image', 'mask_inputs']:  
                raw_image = batched_sample[key]
                c, w, h = raw_image.shape
                # resize
                image = dataset.prepare_array(raw_image, transform).float() 
                # padding
                image = dataset.preprocess_image(
                    image, 
                    image_encode_size,
                ).unsqueeze(0)
                sampled_inputs[key] = image
                
            elif key in ['labels', 'UID']:
                sampled_inputs[key] = torch.as_tensor(batched_sample[key], dtype=torch.long) 
                
            elif key in ['original_size']:
                sampled_inputs[key] = torch.as_tensor(batched_sample[key], dtype=torch.long) 
                continue 
                
            else:
                sampled_inputs[key] = torch.as_tensor(
                    batched_sample[key].copy(), 
                    dtype=torch.float,
                )
                
            sampled_inputs[key] = Variable(sampled_inputs[key]).to(device)
        batched_inputs.append(sampled_inputs)
    return batched_inputs
    
def load_checkpoint(checkpoint_file_path, model, optimizer, scheduler=None, use_checkpoint=False):
    if use_checkpoint and os.path.isfile(checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        
        if isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_key = f"module.{k}"
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler']) 
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        logs = checkpoint['logs']
        print(f"=> Loaded checkpoint '{checkpoint_file_path}' (epoch {checkpoint['epoch']})")
        
    else:
        print(f"=> No checkpoint found at '{checkpoint_file_path}'")
        start_epoch = 0
        best_loss = float('inf')
        logs = []
    
    return model, optimizer, scheduler, start_epoch, best_loss, logs

def collate_fn(batch):
    batch_samples = []
    for index, sample in enumerate(batch):
        batch_samples.append(sample)
    return batch_samples

def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2  # MB
    return 0.0

# 训练和验证
def train_valid_net(param, model, train_data, valid_data=None, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #初始参数
    max_epoch = param['epochs']
    batch_size = param['batch_size']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    test_inter = param['test_inter']
    decay_type = param['decay_type']
    loss_type = param['loss_type']  
    num_GPU = param['num_GPU']
    model_name = param['model_name']
    task = param['task_mode']
    checkpoint_path = param['checkpoint_path']
    os.makedirs(checkpoint_path, exist_ok=True)
    use_checkpoint = param['use_checkpoint']
    
    # 数据参数
    num_classes = len(param.get('classes', []))
    image_encode_size = param['image_encode_size']
    
    num_multimask_outputs = param['num_multimask_outputs']
    mask_threshold = param.get('mask_threshold', 0.0)
    points_per_side = param.get('points_per_side', 80)
    pos_point_rate = param.get('pos_point_rate', 0.5)
    prompt_weight = param.get('prompt_weight', 0.5)
    loss_weight = param.get('loss_weight', 0.1)
    num_add_points = param.get('num_add_points', 1)

    # 保存参数
    dataset.write_json(os.path.join(checkpoint_path, 'train_valid_params.json'), param)

    matcher = None
    if task == 'match':
        matcher = HungarianMatcher(
            cost_class=loss_type['ce'],
            cost_mask=loss_type['focal'],
            cost_dice=loss_type['dice'],
        )    
        
    loss_func = mask_former.make_loss_function(loss_type, num_classes, loss_weight, mode=task).to(device)
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=min(batch_size, len(train_data)),
                              collate_fn=collate_fn,
                              shuffle=True, 
                             )
    valid_loader = None
    if valid_data is not None:
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=min(batch_size, len(valid_data)),
            collate_fn=collate_fn,
            shuffle=False,
        )

    if 'warmup_steps' not in param.keys():
        total_steps = param['max_epoch'] * (len(train_loader))
        warmup_steps = int(0.02 * total_steps)
        param['warmup_steps'] = warmup_steps

    optimizer = make_optimizer(param, model)
    scheduler = make_scheduler(param, optimizer)
    
    resize_transform = ResizeLongestSide(image_encode_size)  
    
    # 提示引擎
    prompter = None
    prompt_types = param['prompt_types']
    if prompt_types is not None and len(prompt_types) > 0:
        prompter = Prompter(
            prompt_types,
            points_per_side=points_per_side,
            pos_point_rate=pos_point_rate,
            num_add_points=num_add_points,
            image_encode_size = image_encode_size,
            task=task,
        ) 

    # 主循环
    checkpoint_file_path = os.path.join(checkpoint_path, 'checkpoint-latest.pth')
    model, optimizer, scheduler, start_epoch, best_loss, logs = load_checkpoint(checkpoint_file_path, 
                                                                                model, optimizer, 
                                                                                scheduler, 
                                                                                use_checkpoint=use_checkpoint)
    
    for epoch in range(start_epoch, max_epoch, 1):

        torch.cuda.reset_peak_memory_stats() 
        epoch_start_time = time.time()      
            
        # 训练阶段
        model.train()
        
        train_per_epoch = dict( [(key, 0.0) for key in list(loss_type.keys())+["loss"]] )
        for batched_idx, batched_samples in enumerate(train_loader):          
            batched_outputs, batched_indices = None, None
            optimizer.zero_grad()  
            
            if prompter is not None:
                batched_samples = prompter(
                    batched_samples, 
                    batched_outputs,
                    batched_indices,
                )  

            batched_inputs = batched_samples_torch(
                batched_samples, 
                resize_transform,
                image_encode_size,
                device,
            )                 

            batched_outputs = model(
                batched_dict=batched_inputs,
            )
            
            batched_indices = get_index(batched_outputs, batched_inputs, matcher)     
            
            # loss
            loss_dict = get_loss(
                loss_func,
                batched_inputs, 
                batched_outputs, 
                batched_indices,
                task,
                epoch / max_epoch,
            )

            effective_losses = []
            for key in loss_type.keys():
                loss_dict_key = loss_dict[key]
                effective_losses.append(float(loss_type[key]) * loss_dict_key)
                train_per_epoch[key] += loss_dict_key.item()   
            loss = sum(effective_losses)
            
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_per_epoch['loss'] += loss.item()   

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time  
        max_memory = get_gpu_memory() 
        
        log_per_epoch = {
            "epoch":epoch,  
            "train":dict(),
            "learning_rate":optimizer.param_groups[0]['lr'], 
            "epoch_time_sec": epoch_time,
            "max_memory_MB": max_memory,
        }
        for key in train_per_epoch.keys():      
            train_per_epoch[key] /= (len(train_loader) * 1)
            log_per_epoch["train"][key] = train_per_epoch[key]
            
        # 保存中间模型
        if epoch % save_inter == 0:
            if isinstance(model, torch.nn.DataParallel):
                model_state_dict = model.module.state_dict()  
            else:
                model_state_dict = model.state_dict()
            state = {
                'epoch': epoch, 
                'state_dict':model_state_dict, 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 
                'best_loss': best_loss,                 
                'logs': logs,
            }
            filename = os.path.join(checkpoint_path, 'checkpoint-latest.pth')
            torch.save(state, filename)
            
            filename = os.path.join(checkpoint_path, f'epoch={epoch}_checkpoint.pth')
            torch.save(state, filename)                   
        
        if scheduler is not None:
            if decay_type == 'ReduceLROnPlateau':
                scheduler.step(train_per_epoch["loss"])
            else:
                scheduler.step()

        # 显示loss（
        if epoch % disp_inter == 0:
            msg = f"[TRAIN] Epoch {epoch:4d} | Loss: {train_per_epoch['loss']:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}"
            for key in loss_type.keys():
                msg += f" | {key}: {train_per_epoch[key]:.6f}"
            msg += f" | Time: {epoch_time:.2f}s"
            msg += f" | Mem: {max_memory:.1f}MB"
            print(msg)

        if valid_loader is not None and epoch % test_inter == 0:
            model.eval()
            
            valid_per_epoch = {k: 0.0 for k in list(loss_type.keys()) + ["loss"]}

            with torch.no_grad():
                for batched_idx, batched_samples in enumerate(valid_loader):
                    batched_inputs = batched_samples_torch(batched_samples, resize_transform, image_encode_size, device)
                    batched_outputs = model(batched_dict=batched_inputs)
                    batched_indices = get_index(batched_outputs, batched_inputs, matcher)     
                    loss_dict = get_loss(loss_func, batched_inputs, batched_outputs, batched_indices, task)

                    effective_losses = [loss_type[k] * loss_dict[k] for k in loss_dict]
                    total_loss = sum(effective_losses)

                    for k in loss_type.keys():
                        valid_per_epoch[k] += loss_dict[k].item()
                    valid_per_epoch["loss"] += total_loss.item()

            for k in valid_per_epoch:
                valid_per_epoch[k] /= len(valid_loader)
            log_per_epoch["valid"] = valid_per_epoch

            is_best = valid_per_epoch["loss"] < best_loss
            bold_start = "\033[1m" if is_best else ""
            bold_end   = "\033[0m" if is_best else ""
            msg = f"[VALID] Epoch {epoch:4d} | Loss: {valid_per_epoch['loss']:.6f}"
            for key in loss_type.keys():
                msg += f" | {key}: {valid_per_epoch[key]:.6f}"
            print(bold_start + msg + bold_end)
            
            if is_best:
                best_loss = valid_per_epoch["loss"]
                model_state_dict = model.module.state_dict() if num_GPU > 1 else model.state_dict()

                best_state = {
                    'epoch': epoch,
                    'state_dict': model_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'logs': logs,
                }
                torch.save(best_state, os.path.join(checkpoint_path, f'checkpoint-best.pth'))

        # 保存训练日志
        logs.append(log_per_epoch)
        if epoch % 2 == 0:
            dataset.write_json(os.path.join(checkpoint_path, 'train_valid_logs.json'), logs)

def get_loss(
    criterion,
    batched_inputs, 
    batched_outputs, 
    batched_indices,
    mode='match',
    progress=None,
):
    targets = []
    if mode == 'match':
        predictions = {
            'pred_masks': torch.cat(
                [x['pred_masks'] for x in batched_outputs], 
                dim=0),
            'pred_logits': torch.cat(
                [x['pred_logits'] for x in batched_outputs], 
                dim=0),
            'pred_ious': torch.cat(
                [x['pred_ious'] for x in batched_outputs], 
                dim=0),
            'progress': progress,
        }  

        indices = []
        for b, inputs in enumerate(batched_inputs):
            indices += batched_indices[b]
            num_mask = inputs['masks'].shape[0]
            if 'weights' in inputs.keys():
                targets += [{'masks': inputs['masks'][k], 
                             'labels': inputs['labels'][k], 
                             'weights': inputs['weights'][k],
                            } for k in range(num_mask)]
            else:
                targets += [{'masks': inputs['masks'][k], 
                             'labels': inputs['labels'][k],
                            } for k in range(num_mask)]
            
    elif mode in ['segment', 'regress', 'inverse']:
        predictions = {
            'pred_masks': torch.cat(
                [x['pred_masks'] for x in batched_outputs], 
                dim=0),
            'progress': progress,
        }  

        indices = []
        for b, inputs in enumerate(batched_inputs):
            indices += batched_indices[b]
            num_mask = inputs['masks'].shape[0]
            if 'weights' in inputs.keys():
                targets += [{'masks': inputs['masks'][k], 'weights': inputs['weights'][k]} for k in range(num_mask)]
            else:
                targets += [{'masks': inputs['masks'][k]} for k in range(num_mask)]
        
    # update indices
    loss_dict = criterion(predictions, targets, indices=indices) 
    return loss_dict     

def get_index(
    batched_outputs, 
    batched_inputs, 
    matcher=None,
):
    batched_indices = []
    for outputs, inputs in zip(batched_outputs, batched_inputs):
        predictions = {
            'pred_masks': outputs['pred_masks'],
            'pred_logits': outputs.get('pred_logits', None),
        }
        
        targets = []
        for k in range(inputs['masks'].shape[0]):
            targets.append({
                'masks': inputs['masks'][k],
                'labels': inputs['labels'][k],
            })
            
        if matcher is None:
            indices = []
            for tgt in targets:
                lab = tgt['labels']
                indices.append(
                    (lab, torch.arange(len(lab)).to(lab)), 
                )
        else:
            indices = matcher(predictions, targets)
            
        batched_indices.append(indices)
    return batched_indices

class Prompter:
    def __init__(self,
        prompt_types,         
        points_per_side = 80,
        mask_threshold = 0.0,
        image_encode_size = 1024,
        num_add_points = 1,  
        pos_point_rate = 0.5, 
        task = 'match',
        mode = 'Train',
    ):
        self.prompt_types = prompt_types
        self.mask_threshold = mask_threshold
        self.num_add_points = num_add_points
        self.point_grids_init = dataset.build_point_grid(points_per_side)  
        self.pos_point_rate = pos_point_rate
        self.low_res_size = image_encode_size // 4
        self.mode = mode
        self.task = task
        self.image_encode_size = image_encode_size
        self.transform = ResizeLongestSide(image_encode_size)  
        
    def process_masks(self, in_masks):
        masks = in_masks.clone().detach().cpu()
        masks = masks > self.mask_threshold
        masks = masks.numpy().astype(np.single)   
        return masks         
    
    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = np.concatenate([np.full_like(src.cpu(), i) for i, (src, _) in enumerate(indices)])
        src_idx = np.concatenate([src.cpu() for (src, _) in indices])
        return batch_idx, src_idx

    def get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = np.concatenate([np.full_like(tgt.cpu(), i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = np.concatenate([tgt.cpu() for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def masks2boxes(self, masks):
        return mask_to_box(masks)

    def jitter_boxes(self, boxes, jitter_factor=0.25):
        return jitter_box(boxes, jitter_factor)
    
    def jitter_masks(self, masks, jitter_list):
        return jitter_mask(masks, jitter_list)
    
    def __call__(self,
        batched_samples,
        batched_outputs=None,
        batched_indices=None,
    ):
        updated_batched_samples = []
        for b, batched_sample in enumerate(batched_samples):
            
            masks = batched_sample['masks']
            labels = batched_sample.get('labels', None)
            original_size = batched_sample['original_size']
            
            updated_sample = dict(batched_sample)

            if batched_outputs is not None and batched_indices is not None:
                indices = batched_indices[b]
                batched_output = batched_outputs[b]
                pred_masks = self.process_masks(batched_output['pred_masks'])
                
                src_idx = self.get_src_permutation_idx(indices)
                tgt_idx = self.get_tgt_permutation_idx(indices)
                matched_masks = masks[tgt_idx]
                matched_labels = labels[tgt_idx]       
                matched_pred_masks = pred_masks[src_idx]
                updated_sample.update(
                    {
                         'masks': matched_masks[:,None,:,:],
                         'labels': matched_labels[:,None], 
                    }) 
            else:
                matched_masks = masks[0]
                matched_pred_masks = None
                matched_labels = labels[0]
                updated_sample.update({
                    'masks': matched_masks[:,None,:,:],
                    'labels': matched_labels[:,None],
                })         
            
            if 'masks' in self.prompt_types:
                mask_inputs = np.zeros_like(matched_masks)
                for j in range(matched_masks.shape[0]):
                    mask_inputs[j] = gaussian_filter(matched_masks[j].astype(np.single), 40.0)
                updated_sample.update(
                    {
                        'mask_inputs': mask_inputs, 
                    }
                )         

            elif 'edges' in self.prompt_types:
                mask_inputs = np.zeros_like(matched_masks)
                jittered_masks = self.jitter_masks(matched_masks, [5])
                for j in range(matched_masks.shape[0]):
                    edge = filters.sobel(jittered_masks[j])
                    mask_inputs[j] = edge.astype(np.bool_).astype(np.single)

                updated_sample.update(
                    {
                        'mask_inputs': mask_inputs, 
                    }
                )  
                
            elif 'attrs' in self.prompt_types:   
                mask_inputs = batched_sample['attrs']
                updated_sample.update(
                    {
                         'mask_inputs': mask_inputs, 
                         'masks': matched_masks[None,:,:,:],
                         'labels': matched_labels[None,:] , 
                    })       
            
            if 'boxes' in self.prompt_types:
                matched_boxes = self.masks2boxes(matched_masks)
                updated_sample.update(
                    {
                        'boxes': self.jitter_boxes(matched_boxes), 
                    }
                )
            
            if 'points' in self.prompt_types: 
                point_scale = np.array(original_size)[None, :]
                point_grids = self.point_grids_init * point_scale  
                add_pos_points = get_prob(self.pos_point_rate)
                not_pos = point_grids.shape[0] // 2
                
                new_coords, new_labels = [], []
                for m, (mask, pred_mask) in enumerate(zip(matched_masks, matched_pred_masks)):
                    if add_pos_points:
                        mask_res = np.clip(mask - pred_mask, a_max=None, a_min=0)
                        diff_points, mask_points = dataset.get_masks_on_points(
                            point_grids, 
                            np.stack([mask_res, mask], axis=0),
                        )  
                        
                        gt0s = np.where(diff_points>0)[0]
                        if len(gt0s) < 1:
                            gt0s = np.where(mask_points>0)[0]
                            
                        if len(gt0s) < 1:
                            pos = [not_pos] * self.num_add_points
                            new_labels.append([-1]*len(pos))
                        else:   
                            pos = np.random.choice(gt0s, self.num_add_points)
                            new_labels.append([1]*len(pos))
                        
                    else:     
                        mask_non = ~mask.astype(np.bool_)
                        mask_res = np.clip(pred_mask - mask, a_max=None, a_min=0)

                        diff_points, mask_points = dataset.get_masks_on_points(
                            point_grids, 
                            np.stack([mask_res, mask_non], axis=0),
                        )  

                        lt0s = np.where(diff_points>0)[0]
        
                        if len(lt0s) < 1:   
                            lt0s = np.where((mask_points>0))[0]
                    
                        if len(lt0s) < 1:
                            pos = [not_pos] * self.num_add_points
                            new_labels.append([-1]*len(pos))
                        else: 
                            pos = np.random.choice(lt0s, self.num_add_points)
                            new_labels.append([0]*len(pos))
                            
                    new_coords.append(point_grids[pos, ::-1])
                    
                new_labels = np.stack(new_labels, axis=0).astype(np.int64)
                new_coords = np.stack(new_coords, axis=0).astype(np.single)
                
                if 'point_coords' not in batched_sample.keys():
                    updated_sample.update(
                        {
                            'point_coords':new_coords, 
                            'point_labels':new_labels,
                        }
                    )
                else:
                    updated_sample['point_coords'] = np.concatenate(
                        [
                            batched_sample['point_coords'],
                            new_coords,
                        ], 
                        axis=1)            
                    updated_sample['point_labels'] = np.concatenate(
                        [
                            batched_sample['point_labels'],
                            new_labels,
                        ], 
                        axis=1) 

            updated_batched_samples.append(updated_sample)    
        return updated_batched_samples


def get_prob(x=0.5):
    return np.random.rand() > (1.0 - x)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_aulc(loss_list):
    """
    Compute Area Under the Loss Curve (AULC) using the trapezoidal rule.
    
    Parameters:
        loss_list (list or np.ndarray): loss value at each epoch.
    
    Returns:
        float: AULC value
    """
    loss_array = np.array(loss_list, dtype=float)
    aulc = np.trapz(loss_array, dx=1)  # dx=1 since epochs are evenly spaced
    return aulc

def plot_train_valid_loss(records, figsize=(8,5)):
    """
    Plot training and validation loss from log records.

    Args:
        records (list): list of dicts, each containing epoch, train, and optional valid fields
        figsize (tuple): figure size
    """

    epochs = []
    train_loss = []
    valid_loss = []

    for r in records:
        e = r["epoch"]
        epochs.append(e)
        train_loss.append(r["train"]["loss"])

        if "valid" in r and r["valid"] is not None:
            if e > 0:
                valid_loss.append(r["valid"]["loss"])
            else:
                valid_loss.append(np.nan)
        else:
            valid_loss.append(np.nan)

    plt.figure(figsize=figsize)

    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, valid_loss, marker="s", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()