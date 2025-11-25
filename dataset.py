import os
import json 
import random
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import PIL.Image
import PIL.ImageDraw
import matplotlib as mpl
import cv2
from segment_anything.utils.amg import remove_small_regions
from torchvision.transforms.functional import resize, to_pil_image 
import pickle  
from skimage import measure, filters, morphology

from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as label_func
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from scipy.ndimage import map_coordinates

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

def gaussian_kernel2d(ks=9, sigma=3.0, device='cpu'):
    ax = torch.arange(ks, device=device) - (ks-1)/2
    g1 = torch.exp(-(ax**2)/(2*sigma**2))
    g1 /= g1.sum()
    g2 = g1[:, None] @ g1[None, :]
    g2 /= g2.sum()
    return g2

def lowpass_blur(x, ks=9, sigma=3.0):
    # x: (H, W) or (B,1,H,W)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)  # (B,1,H,W)
    B, C, H, W = x.shape
    k = gaussian_kernel2d(ks, sigma, device=x.device)
    k = k.view(1,1,ks,ks)
    pad = ks // 2
    return F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'), k).squeeze(1)  # -> (B,H,W)

def find_nonzero_column_intervals(arr):
    n_rows, n_cols = arr.shape
    result = [] 

    nonzero_cols = np.where(np.any(arr > 0, axis=0))[0]
    for col in nonzero_cols:
        column_data = arr[:, col]
        nonzero_mask = column_data > 0

        intervals = []
        in_interval = False
        start = 0

        for i, val in enumerate(nonzero_mask):
            if val and not in_interval:
                start = i
                in_interval = True
            elif not val and in_interval:
                end = i - 1
                intervals.append((start, end))
                in_interval = False

        if in_interval:
            intervals.append((start, n_rows - 1))

        result.append([int(col), intervals[0]])
    return result

def read_cube(input_data_path, size, mode=np.single):
    input_cube = np.fromfile(input_data_path, dtype=mode)
    input_cube = np.reshape(input_cube, size)
    input_cube = input_cube.transpose((2,1,0)) if len(size) == 3 else input_cube.transpose((3,2,1,0))
    return input_cube

def write_cube(data, path):                                   
    data = np.transpose(data,[2,1,0]).astype(np.single)
    data.tofile(path)


def compute_path_from_wells_index(selected_well_coords, num_inline, num_crossline):
    from scipy.interpolate import interp1d
    coords_sorted = sorted(selected_well_coords, key=lambda p: p[1])
    k2s, k3s = zip(*coords_sorted)  # crossline, inline

    k2s = np.array(k2s)
    k3s = np.array(k3s)
    inline_indices = np.arange(num_inline)

    crossline_indices = np.zeros_like(inline_indices, dtype=float)
    mask_middle = (inline_indices >= k3s[0]) & (inline_indices <= k3s[-1])
    crossline_indices[mask_middle] = np.interp(inline_indices[mask_middle], k3s, k2s)

    if len(k3s) >= 2:
        slope_left = (k2s[1] - k2s[0]) / (k3s[1] - k3s[0])
        left_inline = inline_indices[inline_indices < k3s[0]]
        crossline_indices[inline_indices < k3s[0]] = k2s[0] + slope_left * (left_inline - k3s[0])

        slope_right = (k2s[-1] - k2s[-2]) / (k3s[-1] - k3s[-2])
        right_inline = inline_indices[inline_indices > k3s[-1]]
        crossline_indices[inline_indices > k3s[-1]] = k2s[-1] + slope_right * (right_inline - k3s[-1])
    else:
        crossline_indices[:] = k2s[0]

    valid_mask = (crossline_indices >= 0) & (crossline_indices < num_crossline)
    crossline_indices = crossline_indices[valid_mask]
    inline_indices = inline_indices[valid_mask]

    if len(inline_indices) < 2:
        raise ValueError("Too few valid points after clipping. Cannot resample.")

    deltas = np.sqrt(np.diff(crossline_indices)**2 + np.diff(inline_indices)**2)
    cumulative_dist = np.concatenate([[0], np.cumsum(deltas)])
    target_dist = np.linspace(0, cumulative_dist[-1], num_inline)

    f_x = interp1d(cumulative_dist, crossline_indices)
    f_y = interp1d(cumulative_dist, inline_indices)
    resampled_crossline = f_x(target_dist)
    resampled_inline = f_y(target_dist)

    resampled_coords = np.stack([resampled_crossline, resampled_inline], axis=1)
    well_indices = []
    for k2, k3 in selected_well_coords:
        dists = np.sqrt((resampled_crossline - k2)**2 + (resampled_inline - k3)**2)
        nearest_idx = np.argmin(dists)
        well_indices.append(nearest_idx)
    
    return np.vstack([resampled_crossline, resampled_inline]), well_indices

def pca_reduce(feats, n_components=12):
    w, h, c = feats.shape
    X = feats.reshape(w*h,c)
    # 初始化PCA，设置主成分数量
    pca = PCA(n_components=n_components)
    # 对数据进行降维处理
    X_reduced = pca.fit_transform(X)
    feats_reduced = X_reduced.reshape(w,h,-1)
    feats_reduced = feats_reduced[:,:,:3]
    
    return feats_reduced

def data2rgb(data, mask=None):
    # 规范化到0-1
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

    # 缩放到0-255并转换为uint8
    data_scaled = (255 * data_normalized).astype(np.uint8)
    
    if mask is not None:
        # 应用掩码，将掩码为1的区域设置为黑色
        data_scaled[mask == 1] = [0, 0, 0]

    # 创建图像
    image = Image.fromarray(data_scaled, 'RGB')
    image_rgb = image.convert('RGB')
    return np.array(image_rgb, np.uint8)

def extract_lab_mask(msk, step=8):
    from scipy.interpolate import CubicSpline

    n1, n2 = msk.shape
    ptst = []
    ptsb = []
    for i2 in range(n2):
        index = np.where(msk[:, i2]>0)[0]
        if len(index):
            index = np.sort(index)
            ptst.append([index[0], i2])
            ptsb.append([index[-1], i2])
    ptst = np.array(ptst, dtype=np.single).T
    ptsb = np.array(ptsb, dtype=np.single).T
    cubic_spline_interp_t = CubicSpline(ptst[1][::step], ptst[0][::step])
    cubic_spline_interp_b = CubicSpline(ptsb[1][::step], ptsb[0][::step])
    f2s = np.arange(n2, dtype=np.single)
    f1st = cubic_spline_interp_t(f2s)
    f1sb = cubic_spline_interp_b(f2s)
    # return [[f1st, f2s]], [[f1sb, f2s]]
    return np.stack([f2s, f1st]).T, np.stack([f2s, f1sb]).T

def split_masks(masks_in, n=3):
    labeled_mask, num_features = label(masks_in)

    regions = []
    for i in range(1, num_features + 1):
        regions.append((i, np.sum(labeled_mask == i)))
    regions = sorted(regions, key=lambda x: x[1], reverse=True)[:n]

    region_masks = []
    for i, (label_num, _) in enumerate(regions):
        region_mask = np.where(labeled_mask == label_num, 1, 0)
        region_masks.append(region_mask)
    return np.concatenate(region_masks)

def array2rgb(in_data, mask=None):
    data = in_data.copy()
    if np.array(data.shape).argmin() == 0:
        data = data.transpose([1,2,0])
    
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_scaled = (255 * data_normalized).astype(np.uint8)
    if mask is not None:
        data_scaled[mask == 1] = [0, 0, 0]
    image = Image.fromarray(data_scaled, 'RGB')
    image_rgb = image.convert('RGB')
    return np.array(image_rgb, np.uint8)

def mask_roll(mask, shift):
    mask_arr = np.array(mask, dtype=np.single)
    shift_abs = abs(shift)
    mask_pad = np.pad(mask_arr, ((0,0),(shift_abs, shift_abs)), mode='reflect')
    rolled_mask = np.roll(mask_pad, shift, axis=1)
    return rolled_mask[:, shift_abs:-shift_abs]

def create_masks_from_connected_regions(binary_array):
    # Label connected regions of the binary array
    labeled_array, num_features = label_func(binary_array)
    
    # Initialize a list to store masks
    masks = []
    
    # Create a mask for each feature (connected region of 1's)
    for feature_id in range(1, num_features + 1):
        mask = (labeled_array == feature_id)
        masks.append(mask)
    
    return masks, num_features

def get_preprocess_shape(oldshape, long_side_length):
    """
    Compute the output size given input size and target long side length.
    """
    oldh, oldw = oldshape
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def postprocess_mask(masks, area_thresh=200):
    post_masks = []
    for mask in masks:
        mask_out = numpy2gray(mask.astype(np.single))
        mask_out = remove_small_regions(mask_out, area_thresh=area_thresh, mode='islands')[0]
        mask_out = remove_small_regions(mask_out, area_thresh=area_thresh, mode='holes')[0]
        mask_out = binary_fill_holes(mask_out).astype(mask_out.dtype)
        post_masks.append(mask_out)
    return np.stack(post_masks).astype(np.bool_).astype(masks.dtype)
    
def truncate_to_decimal(x: float, decimals: int = 1) -> float:
    factor = 10 ** decimals
    return np.floor(x * factor) / factor

def pad_to_square(arr):
    h, w = arr.shape
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')
    return padded
    
def interp2d(x0, size, mode='bilinear'):
    if isinstance(x0, list):
        x0 = np.stack(x0, axis=0)
    
    n_shape = len(x0.shape)
    if  n_shape== 2:
        x = torch.tensor(x0, dtype=torch.float32)[None,None, ...]
    elif n_shape == 3:
        x = torch.tensor(x0, dtype=torch.float32)[:, None, ...]

    if mode == 'bilinear':
        x = F.interpolate(x, size, mode=mode, align_corners=True)
    else:
        x = F.interpolate(x, size, mode=mode)
    return x.squeeze().numpy()

def read_text(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    text = []
    for line in lines:
        text.append(line)
    return text

def split_train_dataset(data_catalog, 
                        train_set_ratio=0.5, 
                        num_train_set=None,
                        shuffle=True,
                        folder=None,
                       ):
    _data_catalog = list(data_catalog)  
    if shuffle:
        random.shuffle(_data_catalog)
    
    if num_train_set is None:
        num_train_set = len(_data_catalog)
    train_data_sample = _data_catalog[:num_train_set]
    
    # 训练集/验证集划分
    valid_num = int(num_train_set * (1-train_set_ratio))
    valid_sample_list = random.sample(train_data_sample, valid_num)

    samples_train, samples_valid = [], []
    for i_sample in train_data_sample:
        j_sample = i_sample
        if folder is not None:
            j_sample = os.path.join(folder, i_sample)
            
        if j_sample not in valid_sample_list:
            samples_train.append(j_sample)
        else:
            samples_valid.append(j_sample)
    return samples_train, samples_valid

def cerate_img_from_json(json_data, inshape, line_width=2, point_size=2):
    scale = inshape[1]/json_data['imageWidth'], inshape[0]/json_data['imageHeight']
    label_dict = {}
    for idx, json_shape in enumerate(json_data['shapes']):
        label = json_shape['label']
        label_dict[str(idx)] = [json_shape]

    data_dict = dict()
    for label, json_shapes in label_dict.items():

        points = []
        for json_shape in json_shapes:
            points += json_shape['points']

        value = (np.array(points)[:, -1] * scale[-1]).mean() / inshape[0]

        msk_img = np.zeros(inshape).astype(np.bool_)

        for json_shape in json_shapes:

            mask = shape_to_mask(json_shape['points'], inshape, scale=scale, 
                                 shape_type=json_shape['shape_type'],
                                 line_width=line_width,
                                 point_size=point_size)

            msk_img = mask.astype(np.bool_) | msk_img

        data_dict[label] = msk_img.astype(np.single)
    return data_dict    

def shape_to_mask(points, img_shape, shape_type=None, scale=(1,1),
                  line_width=3, point_size=2):
    
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)

    xy = [tuple(map(lambda x,y:x*y, point, scale))for point in points]
    
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
        
    mask = np.array(mask, dtype=bool)
    return mask

def write_pickle(path, data, protocol=4):
    pickle.dump(data, open(path, 'wb'), protocol=protocol)
    
def read_pickle(path):
    return pickle.load(open(path, 'rb'))   

def numpy2rgb(x0, cmap='jet'):
    x = x0.copy()
    ni = np.isnan(x0)
    colormap = mpl.colormaps[cmap]
    y = x[~ni]
    lx = np.min(y)
    rx = np.max(y)
    x = (x - lx) / (rx - lx)
    x[ni] = x[~ni].mean()
    x = colormap(x)[:,:,:3]
    x = x * 255.0
    return x.astype(np.uint8)

def numpy2gray(x0):
    x = x0.copy()
    ni = np.isnan(x0)
    y = x[~ni]
    lx = np.min(y)
    rx = np.max(y)
    if lx == rx: 
        x = np.zeros_like(x)
        return x.astype(np.uint8)
    x = (x - lx) / (rx - lx)  
    x[ni] = x[~ni].mean()
    x = x * 255.0
    return x.astype(np.uint8)

def resize_image(image, shape):
    return np.array(resize(to_pil_image(image), shape))
    
def prepare_image(image, transform, device='cpu'):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    if len(image.shape) < 3:
        image = image.unsqueeze(-1)
    return image.permute(2, 0, 1)

def prepare_array(image, transform, device='cpu'):
    image = transform.apply_array(image)
    image = torch.as_tensor(image, device=device) 
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return image

def normalize_image(x, pixel_mean, pixel_std):
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    return x

def preprocess_image(x, image_encode_size):    
    # Pad
    h, w = x.shape[-2:]
    padh = image_encode_size - h
    padw = image_encode_size - w
    assert padh >= 0 and padw >= 0, f"encode_size {image_encode_size} < image_size {h} {w}!"  
    x = F.pad(x, (0, padw, 0, padh), mode='constant', value=-1.0)
    return x

def get_segmentation(masks, labels):
    _, w, h = masks.shape
    logits = np.zeros([w, h], dtype=np.int64)
    for msk, lab in zip(masks, labels):
        logits += msk.astype(np.int64) * (lab + 1)
    return logits    

def build_point_grid(n_per_side):
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def get_masks_on_points(points, masks):
    mask_size = masks.shape[1:]
    points_round = np.round(points).astype(np.int64)
    for i in range(2):
        points_round[:,i] = np.clip(points_round[:,i], a_min=0, a_max=mask_size[i])
    return masks[:, points_round[:,0], points_round[:,1]].astype(np.int64)

def generate_multiple_mask_label(masks_list, classes): 
    labels_dict = dict()
    num_class = len(classes)
    for i, key in enumerate(classes):
        labels_dict[key] = i

    masks, labels = [], []
    for key, msk in masks_list:
        if key == 'none':
            labels.append(num_class)
        else:
            labels.append(labels_dict[key])
        masks.append(msk)
    labels = np.array(labels).astype(np.int64)        
    masks = np.stack(masks).astype(np.single)   
    return masks, labels
    
def generate_mask_random_point(masks_dict, original_size, points_per_side=64, points_per_mask=12, create_boxes=False):
    point_grids = build_point_grid(points_per_side)
    point_scale = np.array(original_size)[None, :]
    point_grids = point_grids * point_scale
    
    masks = [masks_dict[key] for key in masks_dict.keys()]
    mask_one = np.zeros(original_size, dtype=np.bool_)
    for mask in masks:
        mask_one = mask_one | mask.astype(np.bool_)
        
    targets, boxes, ious = [], [], []
    point_coords, point_labels = [],[]
    for mask in masks:
        point_grids_labels = get_masks_on_points(point_grids, mask[None,:,:])[0]
        gt0s = np.where(point_grids_labels>0)[0] 
        if len(gt0s) < 1:
            continue

        targets.append(mask)
        ious.append(1)
        
        if create_boxes:    
            box = mask_to_box(mask[None,:,:])[0]
            boxes.append(box.astype(np.single)) 
            point_coords.append(None)
            point_labels.append(None)
        else:
            gt01 = np.random.choice(gt0s, 1)[0]
            point_coord = point_grids[gt01:gt01+1,::-1]
            point_label = np.array([1])
            point_coords.append(point_coord.astype(np.single))
            point_labels.append(point_label.astype(np.single))
            boxes.append(None)
        
    for _ in range(len(masks)*1):
        point_grids_labels = get_masks_on_points(point_grids, ~mask_one[None,:,:])[0]
        gt0s = np.where(point_grids_labels>0)[0]   
        if len(gt0s) == 0:
            break

        mask = np.zeros(original_size, dtype=np.bool_)
        targets.append(mask)
        ious.append(0)
        
        if create_boxes:    
            box = mask_to_box(mask[None,:,:])[0]
            boxes.append(box.astype(np.single)) 
            point_coords.append(None)
            point_labels.append(None)
        else:  
            gt01 = np.random.choice(gt0s, 1)[0]
            point_coord = point_grids[gt01:gt01+1,::-1]        
            point_label = np.array([0])
            point_coords.append(point_coord.astype(np.single))
            point_labels.append(point_label.astype(np.single))
            boxes.append(None)
            
    index = np.arange(len(targets))
    np.random.shuffle(index)
    index = index[:points_per_mask]
    
    targets = np.stack(targets)[index].astype(np.single)
    ious = np.stack(ious)[index].astype(np.single)
    boxes = np.stack(boxes)[index]
    point_coords = np.stack(point_coords)[index]
    point_labels = np.stack(point_labels)[index]
    return targets, ious, point_coords, point_labels, boxes

def mask_to_box(masks_numpy):
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    masks = torch.tensor(np.array(masks_numpy)).bool()
    
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
    return out.numpy()

def jitter_mask(masks_input, jitter_list):
    masks_output = np.zeros_like(masks_input)
    jitter_factor = random.choice(jitter_list)
    for j in range(len(masks_input)):
        if masks_input[j].sum() < 1:
            continue
        masks_output[j] = morphology.dilation(masks_input[j], morphology.disk(abs(jitter_factor)))
    return masks_output

def jitter_box(boxes, jitter_factor=0.5):
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        # 计算中心点
        cx, cy = x1 + width / 2, y1 + height / 2
        # 随机扰动尺寸
        jittered_width = width * random.uniform(1 - jitter_factor, 1 + jitter_factor)
        jittered_height = height * random.uniform(1 - jitter_factor, 1 + jitter_factor)
        # 计算新的边界框坐标
        new_x1 = cx - jittered_width / 2
        new_y1 = cy - jittered_height / 2
        new_x2 = cx + jittered_width / 2
        new_y2 = cy + jittered_height / 2
        new_boxes.append([new_x1, new_y1, new_x2, new_y2])
    return np.array(new_boxes)

# 归一化
def min_max_norm(x):
    if torch.is_tensor(x):
        if torch.max(x) != torch.min(x):
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    elif isinstance(x, np.ndarray):
        if np.max(x) != np.min(x):
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
    
# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x):
        if torch.std(x) != 0:
            x = (x - torch.mean(x)) / torch.std(x)
    elif isinstance(x, np.ndarray):
        if np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
    return x 

# 读取数据体
def read_data(size, path, mode="<f", order='F'):
    data = np.fromfile(path, dtype=mode)
    data = data.reshape(size, order=order)        
    return data    

def write_data(path, data):                                   
    data = np.transpose(data,list(range(len(data.shape)-1,-1,-1))).astype(np.single)
    data.tofile(path)

def write_json(path, x, indent=4):
    with open(path, 'w+', encoding='utf8') as fp:
        fp.write(json.dumps(x, indent=indent))

def read_json(path):
    with open(path, 'r+', encoding='utf8') as fp:
        json_data = json.load(fp)
    return json_data

def read_image(path, is_bgr=True):
    image = cv2.imread(path)
    if is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def write_image(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def crop_image(img):
    h, w = img.shape[:2]
    bb = np.array([255,255,255], dtype=np.uint8)
    xt, xb, xl, xr = 0, h, 0, w
    for j1 in range(0, h, 1):
        mk = img[j1, w//2] == bb
        if mk.all():
            xt = j1 + 1
        else:
            break
    for j1 in range(h-1, -1, -1):
        mk = img[j1, w//2] == bb
        if mk.all():
            xb = j1
        else:
            break        
    for j2 in range(0, w, 1):
        mk = img[h//2, j2] == bb
        if mk.all():
            xl = j2 + 1
        else:
            break
    for j2 in range(w-1, -1, -1):
        mk = img[h//2, j2] == bb
        if mk.all():
            xr = j2
        else:
            break
    return img[xt:xb-1, xl:xr-1, :]  
    
def fuse_image(img_low, img_high):
    h_low, s_low, v_low = cv2.split(
        cv2.cvtColor(img_low, cv2.COLOR_RGB2HSV))
    h_high, s_high, v_high = cv2.split(
        cv2.cvtColor(img_high, cv2.COLOR_RGB2HSV))
    HSV = cv2.merge([h_low, s_low, v_high])
    RGB = cv2.cvtColor(HSV,cv2.COLOR_HSV2RGB)
    return RGB    
    
def cerate_img_from_json(json_data, inshape=None, line_width=2, point_size=2):
    from labelme import utils as lmutils
    image = lmutils.image.img_b64_to_arr(json_data['imageData']) 
    if inshape is None:
        inshape = (json_data['imageHeight'], json_data['imageWidth'])
    scale = inshape[1]/json_data['imageWidth'], inshape[0]/json_data['imageHeight']

    label_dict = {}
    for json_shape in json_data['shapes']:
        label = json_shape['label']
        if label not in label_dict.keys():  
            label_dict[label] = [json_shape]
        else:
            label_dict[label].append(json_shape)
            
    data_dict = dict()
    for label, json_shapes in label_dict.items():
        points = []
        for json_shape in json_shapes:
            points += json_shape['points']
        value = (np.array(points)[:, -1] * scale[-1]).mean() / inshape[0]
        msk_img = np.zeros(inshape).astype(np.bool)
        for json_shape in json_shapes:
            mask = shape_to_mask(json_shape['points'], inshape, scale=scale, 
                                 shape_type=json_shape['shape_type'],
                                 line_width=line_width,
                                 point_size=point_size)
            msk_img = mask.astype(np.bool) | msk_img

        data_dict[label] = msk_img.astype(np.single)
    return data_dict, image    

def shape_to_mask(points, img_shape, shape_type=None, scale=(1,1),
                  line_width=3, point_size=2):
    
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)

    xy = [tuple(map(lambda x,y:x*y, point, scale))for point in points]
    
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
        
    mask = np.array(mask, dtype=bool)
    return mask

def compute_max_valid_length(arr_shape, start, angle_rad):
    H, W = arr_shape
    y0, x0 = start
    dy = np.sin(angle_rad)
    dx = np.cos(angle_rad)

    max_len = np.inf

    if dy != 0:
        if dy > 0:
            max_len_y = (H - 1 - y0) / dy
        else:
            max_len_y = (0 - y0) / dy
        max_len = min(max_len, max_len_y)

    if dx != 0:
        if dx > 0:
            max_len_x = (W - 1 - x0) / dx
        else:
            max_len_x = (0 - x0) / dx
        max_len = min(max_len, max_len_x)

    max_len = max(0, max_len)
    return max_len

def extract_line_fixed_sample(arr, start, angle_deg, length_sample):
    angle_rad = np.deg2rad(angle_deg)
    max_len = compute_max_valid_length(arr.shape[1:], start, angle_rad)

    if max_len == 0:
        return None

    s = np.linspace(0, max_len, length_sample)
    y_coords = start[0] + s * np.sin(angle_rad)
    x_coords = start[1] + s * np.cos(angle_rad)
    return y_coords, x_coords

def extract_line_fixed_length(arr, start, angle_deg, length_sample):
    if np.isclose(angle_deg % 180, 0, atol=1e-6):
        dx = 1.0
        dy = 0.0
    elif np.isclose(angle_deg % 180, 90, atol=1e-6):
        dx = 0.0
        dy = 1.0
    else:
        angle_rad = np.deg2rad(angle_deg)
        dy = np.sin(angle_rad)
        dx = np.cos(angle_rad)
        
    y0, x0 = start
    y1 = y0 + (length_sample-1) * dy
    x1 = x0 + (length_sample-1) * dx
    
    H, W = arr.shape[-2:]
    if not (0 <= y1 <= H-1 and 0 <= x1 <= W-1):
        return None 
        
    s = np.linspace(0, length_sample-1, length_sample)
    y_coords = y0 + s * dy
    x_coords = x0 + s * dx
    return y_coords, x_coords

def systematic_profile_extraction(arr, angles, spacing=50, length_sample=780, mode='length', mask=None):
    _, H, W = arr.shape
    all_lines = []

    top_edge = [(0, x) for x in range(0, W, spacing)]
    bottom_edge = [(H-1, x) for x in range(0, W, spacing)]
    left_edge = [(y, 0) for y in range(0, H, spacing)]
    right_edge = [(y, W-1) for y in range(0, H, spacing)]
    starting_points = top_edge + bottom_edge + left_edge + right_edge
    
    for angle in angles:
        for start in starting_points:
            if mode == 'sample':
                lines = extract_line_fixed_sample(arr, start, angle, length_sample)
            elif mode== 'length':
                lines = extract_line_fixed_length(arr, start, angle, length_sample)
                
            if lines is not None:
                if mask is not None:
                    y_coords, x_coords = lines  
                    y_coords_rounded = np.round(y_coords).astype(int)
                    x_coords_rounded = np.round(x_coords).astype(int)
            
                    valid_mask = (y_coords_rounded >= 0) & (y_coords_rounded < H) & \
                                 (x_coords_rounded >= 0) & (x_coords_rounded < W)
                    y_coords_rounded = y_coords_rounded[valid_mask]
                    x_coords_rounded = x_coords_rounded[valid_mask]
            
                    if len(y_coords_rounded) == 0 or np.any(mask[y_coords_rounded, x_coords_rounded] == 0):
                        continue  
                        
                all_lines.append(lines)     
                
    return all_lines


def extract_profiles_from_lines(arr3D, all_lines, order=1):
    Z, H, W = arr3D.shape
    all_profiles = []
    
    for (y_coords, x_coords) in all_lines:
        N = len(y_coords)
        z_grid = np.arange(Z).reshape(-1, 1).repeat(N, axis=1)  
        y_grid = np.tile(y_coords, (Z, 1))
        x_grid = np.tile(x_coords, (Z, 1))
        
        profile = map_coordinates(arr3D, [z_grid, y_grid, x_grid], order=order)
        all_profiles.append(profile)
    return all_profiles

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 