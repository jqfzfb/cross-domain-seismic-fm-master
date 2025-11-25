import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

import colorsys

model_colormap = [[0, 'rgb(255,0,0)'],
                 [0.03125, 'rgb(255,0,0)'],
                 [0.03125, 'rgb(255,128,0)'],
                 [0.0625, 'rgb(255,128,0)'],
                 [0.0625, 'rgb(255,255,0)'],
                 [0.09375, 'rgb(255,255,0)'],
                 [0.09375, 'rgb(0,255,0)'],
                 [0.125, 'rgb(0,255,0)'],
                 [0.125, 'rgb(0,128,0)'],
                 [0.15625, 'rgb(0,128,0)'],
                 [0.15625, 'rgb(0,64,0)'],
                 [0.1875, 'rgb(0,64,0)'],
                 [0.1875, 'rgb(0,255,255)'],
                 [0.21875, 'rgb(0,255,255)'],
                 [0.21875, 'rgb(0,128,255)'],
                 [0.25, 'rgb(0,128,255)'],
                 [0.25, 'rgb(0,0,255)'],
                 [0.28125, 'rgb(0,0,255)'],
                 [0.28125, 'rgb(0,0,160)'],
                 [0.3125, 'rgb(0,0,160)'],
                 [0.3125, 'rgb(0,128,192)'],
                 [0.34375, 'rgb(0,128,192)'],
                 [0.34375, 'rgb(255,128,128)'],
                 [0.375, 'rgb(255,128,128)'],
                 [0.375, 'rgb(128,128,255)'],
                 [0.40625, 'rgb(128,128,255)'],
                 [0.40625, 'rgb(128,0,255)'],
                 [0.4375, 'rgb(128,0,255)'],
                 [0.4375, 'rgb(128,0,128)'],
                 [0.46875, 'rgb(128,0,128)'],
                 [0.46875, 'rgb(255,128,255)'],
                 [0.5, 'rgb(255,128,255)'],
                 [0.5, 'rgb(255,0,255)'],
                 [0.53125, 'rgb(255,0,255)'],
                 [0.53125, 'rgb(128,64,0)'],
                 [0.5625, 'rgb(128,64,0)'],
                 [0.5625, 'rgb(128,128,128)'],
                 [0.59375, 'rgb(128,128,128)'],
                 [0.59375, 'rgb(192,192,192)'],
                 [0.625, 'rgb(192,192,192)'],
                 [0.625, 'rgb(64,0,64)'],
                 [0.65625, 'rgb(64,0,64)'],
                 [0.65625, 'rgb(231,186,50)'],
                 [0.6875, 'rgb(231,186,50)'],
                 [0.6875, 'rgb(113,149,149)'],
                 [0.71875, 'rgb(113,149,149)'],
                 [0.71875, 'rgb(134,108,124)'],
                 [0.75, 'rgb(134,108,124)'],
                 [0.75, 'rgb(183,139,113)'],
                 [0.78125, 'rgb(183,139,113)'],
                 [0.78125, 'rgb(128,128,0)'],
                 [0.8125, 'rgb(128,128,0)'],
                 [0.8125, 'rgb(192,186,224)'],
                 [0.84375, 'rgb(192,186,224)'],
                 [0.84375, 'rgb(158,219,252)'],
                 [0.875, 'rgb(158,219,252)'],
                 [0.875, 'rgb(188,66,63)'],
                 [0.90625, 'rgb(188,66,63)'],
                 [0.90625, 'rgb(226,217,160)'],
                 [0.9375, 'rgb(226,217,160)'],
                 [0.9375, 'rgb(155,240,191)'],
                 [0.96875, 'rgb(155,240,191)'],
                 [0.96875, 'rgb(159,203,27)'],
                 [1.0, 'rgb(159,203,27)']]

def pl_curl_colormap():
    raw_cmap = [
        [0.0, 'rgb(20, 29, 67)'],
        [0.05, 'rgb(25, 52, 80)'],
        [0.1, 'rgb(28, 76, 96)'],
        [0.15, 'rgb(23, 100, 110)'],
        [0.2, 'rgb(16, 125, 121)'],
        [0.25, 'rgb(44, 148, 127)'],
        [0.3, 'rgb(92, 166, 133)'],
        [0.35, 'rgb(140, 184, 150)'],
        [0.4, 'rgb(182, 202, 175)'],
        [0.45, 'rgb(220, 223, 208)'],
        [0.5, 'rgb(253, 245, 243)'],
        [0.55, 'rgb(240, 215, 203)'],
        [0.6, 'rgb(230, 183, 162)'],
        [0.65, 'rgb(221, 150, 127)'],
        [0.7, 'rgb(211, 118, 105)'],
        [0.75, 'rgb(194, 88, 96)'],
        [0.8, 'rgb(174, 63, 95)'],
        [0.85, 'rgb(147, 41, 96)'],
        [0.9, 'rgb(116, 25, 93)'],
        [0.95, 'rgb(82, 18, 77)'],
        [1.0, 'rgb(51, 13, 53)']
    ]

    colors = []
    for v, rgb_str in raw_cmap:
        rgb = tuple(int(c)/255 for c in rgb_str.strip('rgb()').split(','))
        colors.append((v, rgb))

    return LinearSegmentedColormap.from_list("pl_curl", colors)

def turbo_colormap():
    import matplotlib.colors as mcolors
    turbo_data = np.array([
        [0.18995, 0.07176, 0.23217],
        [0.20803, 0.01809, 0.36159],
        [0.23937, 0.02426, 0.52256],
        [0.30000, 0.19000, 0.70000],
        [0.40815, 0.42389, 0.88575],
        [0.63000, 0.73950, 0.73813],
        [0.84337, 0.97089, 0.44181],
        [0.98585, 0.89495, 0.14398],
        [0.98753, 0.66900, 0.00719],
        [0.83125, 0.43712, 0.00714],
        [0.60000, 0.23500, 0.00300]
    ])
    return mcolors.LinearSegmentedColormap.from_list("custom_turbo", turbo_data, N=256)

###

def plot_seismic_coverage(sx, lines, index=10, figsize=(4, 4), cmap='gray', alpha=0.6):
    plt.figure(figsize=figsize)
    H, W = sx.shape[1:]
    plt.imshow(sx[index, :, :], cmap=cmap, extent=[0, W-1, H-1, 0])
    plt.xlabel("X")
    plt.ylabel("Y")

    for (y_coords, x_coords) in lines:
        plt.plot(x_coords, y_coords, alpha=alpha)

    plt.tight_layout()
    plt.show()

def plot_gaussian_fit_comparison(
    patch: np.ndarray,
    mask: np.ndarray = None,
    cmap: str = 'deep',  
    vmin: float = None,
    vmax: float = None,
    show_contours: bool = False,
    show_colorbar: bool = False,
    lon_bounds: tuple = None,
    lat_bounds: tuple = None,
    outer_line: np.ndarray = None,
    save_path: str = None,
    fontsize: int = 12,
    contour_levels: int = 10,
    dpi: int = 200):
    import cmocean
    from matplotlib.colors import LightSource

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)

    # Apply mask
    if mask is not None:
        masked_patch = patch.copy()
        masked_patch[~mask] = np.nan
    else:
        masked_patch = patch

    # Color limits
    vmin = vmin if vmin is not None else np.nanmin(masked_patch)
    vmax = vmax if vmax is not None else np.nanmax(masked_patch)

    # Geo extent
    if lon_bounds and lat_bounds:
        extent = [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]]
    else:
        extent = None

    if cmap == 'deep':
        cmap = cmocean.cm.deep
        # 调整光照为更高角度，减弱暗影
        ls = LightSource(azdeg=315, altdeg=75)  # 更亮更自然
        rgb = ls.shade(masked_patch, cmap=cmap,
                       vert_exag=2.0,
                       blend_mode='soft',  # 更自然融合色带和阴影
                       vmin=vmin, vmax=vmax)
        # Plot shaded relief
        im0 = ax.imshow(rgb, extent=extent, origin='upper')
        
    else:
        im0 = ax.imshow(masked_patch, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        
    # Optional: Contours
    if show_contours:
        ax.contour(patch, levels=contour_levels,
                   colors='k', linewidths=0.4, extent=extent, origin='upper')

    if outer_line is not None:
        ax.plot(outer_line[:, 0], outer_line[:, 1], 'k-', linewidth=1.0)
    
    # Optional: Colorbar
    if show_colorbar:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
            ax=ax, shrink=0.7, pad=0.02
        )
        cbar.set_label("Elevation (m)", fontsize=fontsize)

    # Ticks and labels
    if extent is not None:
        lon_vals = np.linspace(lon_bounds[0], lon_bounds[1], 3)
        lat_vals = np.linspace(lat_bounds[0], lat_bounds[1], 3)

        ax.set_xticks(lon_vals)
        ax.set_yticks(lat_vals)
        ax.set_xticklabels([f"{val:.2f}°" for val in lon_vals], fontsize=fontsize)
        ax.set_yticklabels([f"{val:.2f}°" for val in lat_vals], fontsize=fontsize)
        # ax.set_xlabel("Longitude (°)", fontsize=fontsize)
        # ax.set_ylabel("Latitude (°)", fontsize=fontsize)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()

def draw_colorbar(cmap="viridis", 
                  orientation="horizontal", 
                  cmin=0, cmax=1,
                  font_size=18,
                  num_ticks=5,
                  num_round=1,
                  figsize=(6, 0.4)):
    fig, ax = plt.subplots(figsize=figsize)

    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ticks = np.linspace(cmin, cmax, num_ticks)
    if num_round is not None:
        ticks = np.round(ticks, num_round) # 只保留一位小数

    cbar = plt.colorbar(
        sm,
        cax=ax,
        orientation=orientation,
        ticks=ticks
    )

    cbar.ax.tick_params(labelsize=font_size)
    cbar.outline.set_linewidth(1.5)
    plt.show()
    
    

def get_random_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5 
        saturation = 0.9  
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors

def get_label_colors(labels):
    colors = []
    for lab in labels:
        if lab > 0:
            colors.append('g')
        else:
            colors.append('r')
    return colors

def get_label_markers(labels, num_cls):
    markers_init = ['o', '*', 'v', '^', 's', 'p', 'd']
    num_markers = len(markers_init)
    if num_markers < num_cls:
        a = num_cls // num_markers
        b = num_cls % num_markers
        markers_cls = a * markers_init + markers_init[:b]
    else:
        markers_cls = markers_init[:num_cls]
        
    markers = []    
    if num_cls > 2:
        markers = [markers_cls[i] for i in range(len(labels))]
    else:
        markers = markers_cls[0] * 2
    return markers

def colormap_reverse(cmap):
    n = len(cmap)
    cmap_out = list()
    for i in range(n):
        cmap_out.append([cmap[i][0], cmap[n-1-i][1]])
    return cmap_out   

def colormap_change(cmap):
    n = len(cmap)
    cmap_out = list()
    for i in range(n):
        rgb = cmap[i][1][4:-1].split(',')
        rgb = [int(tx) for tx in rgb]
        text = f'rgb({rgb[2]}, {rgb[1]}, {rgb[0]})'
        cmap_out.append([cmap[i][0],text])
    return cmap_out  

def colormap_mat(colormap, name='new_colormap'):
    colors = []
    for cmap in colormap:
        numbs = cmap[-1][4:-1].split(',')
        numbs = [float(i)/255.0 for i in numbs]
        numbs.append(1.0)
        colors.append(numbs)
    return LinearSegmentedColormap.from_list(name, colors)

def draw_img(img, msk=None, sct=None, ctr=None, ptsc=None, pth=None, bbx=None, text=None,
             pts=None, hrzs=None, cmap="gray", mmap='jet', interpolation="bilinear", 
             cmin=None, cmax=None, figsize=[4,4], colorbar=False, aspect=None,
             save_file=None, extent=None, maskalpha=0.6, origin='upper',
             markersize=24, markerline=1.5, marker='o', markercolor=None, 
             markerhollow=False, maskcolor=None, bbx_text=None, 
             boxcolor='g', btxcolor='c', bbx_on=True,
             xtick_num=None, ytick_num=None, tick_decimal=0,
             set_xlabel=None, set_ylabel=None, label_fontsize=14,
             dpi=300, axis_off=True, set_xlim=None, set_ylim=None):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if sct is not None:
        scatter_x, scatter_y = np.meshgrid(sct[0], sct[1]) 
        scatter_x = scatter_x.flatten()
        scatter_y = scatter_y.flatten()
        scatter_z = img.flatten()
        idxs = np.where(scatter_z > np.median(img))[0]
        im = ax.scatter(scatter_x[idxs], scatter_y[idxs], c=scatter_z[idxs], cmap=cmap, 
                        vmin=cmin, vmax=cmax)
    else:
        im = ax.imshow(img, cmap=cmap, interpolation=interpolation, origin=origin, extent=extent,
                   vmin=cmin, vmax=cmax, aspect=aspect)
    
    if msk is not None:
        nm = len(msk)
        if maskcolor is None:
            mcs = []
            colors = mpl.cm.get_cmap(mmap)
            for c in np.linspace(0,1,nm):
                mcs.append(np.array(colors(c)))
        elif isinstance(maskcolor, list):
            mcs = [np.array(maskcolor[i]) for i in range(nm)]
        else:
            mcs = [np.array(maskcolor)] * nm
        
        for mk, mc in zip(msk, mcs):
            mask_image = mk.reshape(*mk.shape[-2:], 1) * mc.reshape(1, 1, -1)
            ax.imshow(mask_image, alpha=maskalpha, origin=origin, extent=extent)
    
    if bbx is not None: 
        if isinstance(bbx_text, str):
            bbx_text = [bbx_text] * len(bbx)
            
        if isinstance(boxcolor, str):
            boxcolor = [boxcolor] * len(bbx)
            
        for b, box in enumerate(bbx):
            x, y = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            if bbx_on:
                ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor=boxcolor[b], facecolor=(0,0,0,0), linewidth=1))    
            if bbx_text is not None:
                _x, _y = -10, -4
                ax.text(x+_x, y+_y, bbx_text[b],
                              va='center', ha='center', fontsize=6, color='blue',
                              bbox=dict(facecolor=btxcolor, lw=0, pad=0))
    
    if ctr is not None:
        H, W = ctr.shape
        if extent is not None:
            x = np.linspace(extent[0], extent[1], W)
            y = np.linspace(extent[2], extent[3], H)
        else:
            x = np.arange(W)
            y = np.arange(H)
        X, Y = np.meshgrid(x, y)
        ax.contour(X, Y, ctr, np.linspace(ctr.min(), ctr.max(), 20),
                   cmap='turbo', linewidths=0.8)
        
    if pth is not None:
        cs = list(map(lambda x: color(tuple(x)), ncolors(len(pth))))
        for i, p in enumerate(pth):
            ax.plot(p[:,1], p[:,0], color=cs[i], linestyle='-', linewidth=1)    
    
    if hrzs is not None:
        cs = list(map(lambda x: color(tuple(x)), ncolors(len(hrzs))))
        for i, (x0s, x1s) in enumerate(hrzs):
            ax.plot(x1s, x0s, color=cs[i], linewidth=2)
            
    if ptsc is not None:
            nps = len(ptsc)

            if markercolor is None:
                cs = get_random_colors(len(ptsc)) 
            elif isinstance(markercolor, list):
                cs = markercolor
            elif torch.is_tensor(markercolor) or isinstance(markercolor, np.ndarray):
                cs = get_label_colors(markercolor.reshape(-1))
            else:
                cs = [markercolor]*len(ptsc)

            if markerhollow:
                edgecolors = cs
                facecolors = ['none']*len(ptsc)
            else:
                edgecolors = cs
                facecolors = cs
            for i in range(nps):
                x = ptsc[i]
                for j, p in enumerate(x):
                    ax.scatter(p[0], p[1], s=markersize, marker=marker, 
                               edgecolors=edgecolors[i], 
                               facecolors=facecolors[i],
                               linewidths=markerline)

    if pts is not None:
        for i, p in enumerate(pts):
            ax.scatter(p[0], p[1], s=markersize, marker=marker, edgecolors='w', facecolors='k', linewidths=markerline)               

    if set_xlim is not None:
        ax.set_xlim(set_xlim)
    if set_ylim is not None:
        ax.set_ylim(set_ylim)
    
    if axis_off:
        ax.axis('off')
    else:
        ytick_rotation = 90
        ax.tick_params(labelsize=label_fontsize)
        for label in ax.get_yticklabels():
            label.set_rotation(ytick_rotation)
            label.set_verticalalignment('center')

        if xtick_num is not None:
            xlim = ax.get_xlim()
            xticks = np.linspace(xlim[0], xlim[1], xtick_num)
            xticks = np.round(xticks, tick_decimal)
            ax.set_xticks(xticks)
            
        if ytick_num is not None:
            ylim = ax.get_ylim()
            yticks = np.linspace(ylim[0], ylim[1], ytick_num)
            yticks = np.round(yticks, tick_decimal)
            ax.set_yticks()
    
    if set_xlabel is not None:
        ax.set_xlabel(set_xlabel, fontsize=label_fontsize)
    if set_ylabel is not None:
        ax.set_ylabel(set_ylabel, fontsize=label_fontsize)
    
    if text is not None:
        plt.title(text, fontsize=label_fontsize)
    
    if colorbar:    
        fig.colorbar(im, fraction=0.0225, pad=0.015) 
    
    if save_file is not None:
        plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_validation_log1(x1, x2=None, x3=None, x4=None, x5=None, title=None, fontsize=16, labelsize=10, ynumticks=None,
                         ylim=None, labels=None, show_label=False, figsize=(6, 3)):
    from matplotlib.ticker import MaxNLocator
    threshold = 2.0
    markersize = 4.0
    x = np.arange(len(x1))

    if labels is None:
        labels = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    plt.figure(figsize=figsize, dpi=150)
    
    # Ground Truth
    mask1 = np.array(x1) > threshold
    plt.plot(x[mask1], np.array(x1)[mask1], color='black', linewidth=1, marker='o', markersize=markersize, label=labels[0])
    
    if x2 is not None:
        mask2 = mask1
        plt.plot(x[mask2], np.array(x2)[mask2], color='red', linewidth=1, marker='x', markersize=markersize, label=labels[1])
    
    if x3 is not None:
        mask3 = mask1
        plt.plot(x[mask3], np.array(x3)[mask3], linestyle='--', color='skyblue', 
                 linewidth=1, markersize=markersize, label=labels[2])
    
    if x4 is not None:
        mask4 = mask1
        plt.plot(x[mask4], np.array(x4)[mask4], linestyle='--', color='brown', 
                 linewidth=1, markersize=markersize, label=labels[3])
    
    if x5 is not None:
        mask5 = mask1
        plt.plot(x[mask5], np.array(x5)[mask5], linestyle='--', color='purple', 
                 linewidth=1, markersize=markersize, label=labels[4])
        
    if title is not None:
        plt.title(title, fontsize=fontsize, fontweight='bold')

    if ylim is not None:
        plt.ylim(ylim)
    else:
        print(f"Current Y-axis range: {plt.gca().get_ylim()}")

    if ynumticks is not None:
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=ynumticks))
        
    plt.xlabel("Index", fontsize=fontsize)
    plt.ylabel("Value", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.grid(True, linestyle='--', alpha=0.5)
    if show_label:
        plt.legend(fontsize=labelsize)
    plt.tight_layout()
    plt.show()


