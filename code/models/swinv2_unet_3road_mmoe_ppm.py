# -*- coding: utf-8 -*-
import os, sys, math, time, glob, random
from pathlib import Path
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support

# GDAL
from osgeo import gdal
gdal.UseExceptions()

from PIL import Image
import shutil

# ---------------------------------------------------------------------------
# 运行环境 & 随机种子
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def set_seed(seed: int = 2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(2024)

# autocast
from contextlib import nullcontext
def get_autocast():
    if device.type == "cuda":
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    else:
        return nullcontext()

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def to_2tuple(x):
    if isinstance(x, (tuple, list)): return tuple(x)
    return (x, x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in trunc_normal_.")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

# ---------------------------------------------------------------------------
# 数据集（Post/Pre/NBR/Mask 四元组）
# ---------------------------------------------------------------------------
class RemoteSensingDataset(Dataset):
    def __init__(self, post_files: List[str], pre_files: List[str], nbr_files: List[str], mask_files: List[str], transform=None):
        super().__init__()
        self.transform = transform
        self.file_quadruplets = []
        gdal.UseExceptions()

        pre_file_map  = {os.path.basename(p): p for p in pre_files}
        nbr_file_map  = {os.path.basename(n): n for n in nbr_files}
        mask_file_map = {os.path.basename(m): m for m in mask_files}

        for post_path in post_files:
            basename = os.path.basename(post_path)
            if basename in pre_file_map and basename in nbr_file_map and basename in mask_file_map:
                pre_path  = pre_file_map[basename]
                nbr_path  = nbr_file_map[basename]
                mask_path = mask_file_map[basename]
                self.file_quadruplets.append((post_path, pre_path, nbr_path, mask_path))
            else:
                print(f"Warning: unmatched file -> {post_path}")

        print(f"Successfully paired {len(self.file_quadruplets)} sets of (Post, Pre, NBR, Mask) images.")

    def __len__(self):
        return len(self.file_quadruplets)

    def __getitem__(self, idx):
        post_path, pre_path, nbr_path, mask_path = self.file_quadruplets[idx]

        def read_img(path):
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"GDAL cannot open: {path}")
            arr = ds.ReadAsArray()
            if arr.ndim == 2: arr = np.expand_dims(arr, axis=0)
            if arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 65535.0
            elif arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
            return arr

        post_img = read_img(post_path)
        pre_img  = read_img(pre_path)
        nbr_img  = read_img(nbr_path)
        nbr_img  = np.nan_to_num(nbr_img, nan=0.0, posinf=0.0, neginf=0.0)

        post_tensor = torch.from_numpy(post_img)
        pre_tensor  = torch.from_numpy(pre_img)
        nbr_tensor  = torch.from_numpy(nbr_img)

        mask_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
        if mask_ds is None:
            raise RuntimeError(f"GDAL cannot open mask: {mask_path}")
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_tensor = torch.from_numpy(mask)

        return post_tensor, pre_tensor, nbr_tensor, mask_tensor

# ---------------------------------------------------------------------------
# Swin Transformer V2 组件
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttentionV2(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8.0))
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlockV2(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttentionV2(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]; x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

class BasicLayerV2(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}x{W}) != model size ({self.img_size[0]}x{self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None: x = self.norm(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = self.expand(x)
        x = x.view(B, H, W, C * 2).permute(0, 3, 1, 2).contiguous()
        x = F.pixel_shuffle(x, self.dim_scale)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C // 2)
        x = self.norm(x)
        return x, H * 2, W * 2

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C).view(B, H, W, 4, 4, C // 16)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 4, W * 4, -1)
        x = self.norm(x)
        return x

# ---------------------------------------------------------------------------
# PPM-Lite & SCSE（全局上下文增强，位于上采样后、MMoE头之前）
# ---------------------------------------------------------------------------
class PPMLite(nn.Module):
    def __init__(self, in_ch, bins=(1, 2, 3, 6)):
        super().__init__()
        proj = max(in_ch // 4, 32)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_ch, proj, 1, bias=False),
                nn.BatchNorm2d(proj),
                nn.ReLU(inplace=True)
            ) for b in bins
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + proj * len(bins), in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):  # x: [B,C,H,W]
        outs = [x]
        for s in self.stages:
            y = s(x)
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
            outs.append(y)
        y = torch.cat(outs, dim=1)
        return self.fuse(y)

class SCSE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=True), nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(ch, 1, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):  # [B,C,H,W]
        return x * self.cSE(x) + x * self.sSE(x)

# ---------------------------------------------------------------------------
# Old-Style MMoE Head（保留；仅用于产生主 logits，训练只用 CE）
# ---------------------------------------------------------------------------
class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); x = self.act(x)
        return x

class OldStyleMMoEHead(nn.Module):
    def __init__(self, in_ch, gate_hint_ch, num_classes=4, num_experts=5,
                 expert_ch=None, tower_ch=None, topk=2, temperature=2.0):
        super().__init__()
        assert num_classes == 4
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.topk = topk
        self.temperature = temperature

        expert_ch = expert_ch or max(in_ch // 2, 32)
        tower_ch  = tower_ch  or max(in_ch // 2, 32)

        self.hint_compress = nn.Sequential(
            nn.Conv2d(gate_hint_ch, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        gate_in_ch = in_ch + 16

        self.gates = nn.ModuleList([
            nn.Conv2d(gate_in_ch, num_experts, kernel_size=1, bias=True)
            for _ in range(num_classes)
        ])

        dilations = [1, 2, 3, 1, 2]
        self.experts = nn.ModuleList([
            DWSeparableConv(in_ch, expert_ch, dilation=dilations[i % len(dilations)])
            for i in range(num_experts)
        ])

        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(expert_ch, tower_ch, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(tower_ch, 1, kernel_size=1, bias=True)
            ) for _ in range(num_classes)
        ])

    @torch.no_grad()
    def set_temperature(self, T: float):
        self.temperature = float(T)

    def _topk_sparse(self, probs, k):
        if k >= self.num_experts:
            return probs
        B, E, H, W = probs.shape
        topk_vals, topk_idx = torch.topk(probs, k, dim=1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_idx, 1.0)
        sparse = probs * mask
        denom = sparse.sum(dim=1, keepdim=True) + 1e-8
        sparse = sparse / denom
        return sparse

    def forward(self, x, gate_hint):
        B, C, H, W = x.shape
        hint = self.hint_compress(gate_hint)
        gate_in = torch.cat([x, hint], dim=1)

        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [B,E,Ce,H,W]

        per_class_logits = []
        gate_probs_all = []

        for t in range(self.num_classes):
            logits = self.gates[t](gate_in) / max(self.temperature, 1e-6)   # [B,E,H,W]
            probs  = torch.softmax(logits, dim=1)
            probs  = self._topk_sparse(probs, self.topk)
            gate_probs_all.append(probs.unsqueeze(1))                       # [B,1,E,H,W]
            fused = (expert_outs * probs.unsqueeze(2)).sum(dim=1)           # [B,Ce,H,W]
            logit_t = self.towers[t](fused)                                  # [B,1,H,W]
            per_class_logits.append(logit_t)

        logits_stack = torch.cat(per_class_logits, dim=1)                    # [B,4,H,W]
        aux = {"gate_probs": torch.cat(gate_probs_all, dim=1)}               # 仅保留路由统计（不参与损失）
        return logits_stack, aux

# ---------------------------------------------------------------------------
# 主模型（SwinUNetV2 三路输入 + PPM-Lite(+SCSE) + MMoE；★ 不含任何 DS 分支）
# ---------------------------------------------------------------------------
class SwinUNetV2_TripleInput(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans_list=[9,9,3], num_classes=4,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=8,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0,0,0,0],
                 use_mmoe=True, mmoe_topk=2, mmoe_temp=2.0):
        super().__init__()

        self.post_ch, self.pre_ch, self.nbr_ch = in_chans_list
        self.use_mmoe = use_mmoe
        self.num_classes = num_classes

        self.patch_embed_post = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=self.post_ch, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_pre  = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=self.pre_ch,  embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_nbr  = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=self.nbr_ch,  embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)

        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU()
        )

        self.num_layers  = len(depths)
        self.embed_dim   = embed_dim
        self.ape         = ape
        self.patch_norm  = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio    = mlp_ratio

        num_patches = self.patch_embed_post.num_patches
        patches_resolution = self.patch_embed_post.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.encoders = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayerV2(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)),
                depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i]
            )
            self.encoders.append(layer)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_features), nn.ReLU(inplace=True),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_features), nn.ReLU(inplace=True)
        )
        self.bottleneck_norm = norm_layer(self.num_features)
        self.bottleneck_dim = self.num_features

        self.decoders = nn.ModuleList()
        decoder_input_dim = self.bottleneck_dim
        for i in range(self.num_layers - 1):
            expand_layer = PatchExpand(dim=decoder_input_dim, dim_scale=2, norm_layer=norm_layer)
            dim_after_expand = decoder_input_dim // 2
            skip_connection_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i))
            basic_layer_input_dim = dim_after_expand + skip_connection_dim

            basic_layer = BasicLayerV2(
                dim=basic_layer_input_dim,
                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 2 - i)),
                                  patches_resolution[1] // (2 ** (self.num_layers - 2 - i))),
                depth=depths[self.num_layers - 2 - i], num_heads=num_heads[self.num_layers - 2 - i],
                window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:self.num_layers - 2 - i]):sum(depths[:self.num_layers - 1 - i])],
                norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[self.num_layers - 2 - i]
            )
            self.decoders.append(nn.ModuleList([expand_layer, basic_layer]))
            decoder_input_dim = basic_layer_input_dim

        self.up = FinalPatchExpand_X4(input_resolution=(patches_resolution[0], patches_resolution[1]), dim=decoder_input_dim)

        # === 新增：PPM-Lite(+SCSE) 全局上下文增强（默认开启；可通过 use_ppm_scse=False 关闭以做A/B） ===
        self.final_dim = decoder_input_dim
        self.use_ppm_scse = True
        self.ppm  = PPMLite(in_ch=self.final_dim)
        self.scse = SCSE(ch=self.final_dim)

        if self.use_mmoe:
            gate_hint_ch = self.nbr_ch + self.post_ch
            self.head = OldStyleMMoEHead(
                in_ch=decoder_input_dim, gate_hint_ch=gate_hint_ch, num_classes=num_classes,
                num_experts=5, expert_ch=decoder_input_dim // 2, tower_ch=decoder_input_dim // 2,
                topk=mmoe_topk, temperature=mmoe_temp
            )
        else:
            self.head = nn.Conv2d(decoder_input_dim, self.num_classes, kernel_size=1, bias=True)

        self.apply(self._init_weights)

    @torch.no_grad()
    def set_gate_temperature(self, T: float):
        if self.use_mmoe and hasattr(self.head, "set_temperature"):
            self.head.set_temperature(T)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_post, x_pre, x_nbr):
        B, _, H, W = x_post.shape

        tokens_post = self.patch_embed_post(x_post)
        tokens_pre  = self.patch_embed_pre(x_pre)
        tokens_nbr  = self.patch_embed_nbr(x_nbr)

        fused_tokens = torch.cat([tokens_post, tokens_pre, tokens_nbr], dim=2)
        x = self.fusion_layer(fused_tokens)
        if hasattr(self, "absolute_pos_embed"):
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for i, encoder in enumerate(self.encoders):
            resolution = (self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i))
            x_before_downsample = x.view(B, resolution[0], resolution[1], -1)
            features.append(x_before_downsample)
            x = encoder(x)

        final_feat_h = H // (self.patch_embed_post.patch_size[0] * (2 ** (self.num_layers - 1)))
        final_feat_w = W // (self.patch_embed_post.patch_size[1] * (2 ** (self.num_layers - 1)))

        x = x.permute(0, 2, 1).contiguous().view(B, self.num_features, final_feat_h, final_feat_w)
        x = self.bottleneck(x)

        x_norm = x.flatten(2).transpose(1, 2).contiguous()
        x_norm = self.bottleneck_norm(x_norm)
        x = x_norm.permute(0, 2, 1).contiguous().view(B, self.num_features, final_feat_h, final_feat_w)
        x = x.flatten(2).transpose(1, 2).contiguous()

        for i, decoder in enumerate(self.decoders):
            expand_layer, basic_layer = decoder
            res = (self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i)),
                   self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i)))
            x, H_up, W_up = expand_layer(x, res[0], res[1])

            skip_feature = features[-(i + 2)]
            skip_feature = skip_feature.view(B, -1, skip_feature.shape[-1])
            x = torch.cat([x, skip_feature], -1)
            x = basic_layer(x)

        x = self.up(x)                         # [B,H,W,C]
        x = x.permute(0, 3, 1, 2).contiguous() # [B,C_in,H,W]  ; C_in = final_dim

        # === PPM-Lite (+SCSE) ===
        if getattr(self, 'use_ppm_scse', True):
            x = self.ppm(x)
            x = self.scse(x)

        if not self.use_mmoe:
            return self.head(x)

        gate_hint = torch.cat([x_nbr, torch.abs(x_post - x_pre)], dim=1)
        logits, aux = self.head(x, gate_hint)
        return logits, aux

# ---------------------------------------------------------------------------
# 训练评估工具
# ---------------------------------------------------------------------------
def calculate_batch_iou(pred, target, num_classes):
    pred = pred.argmax(1).detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()
    ious, valid = [], 0
    for cls in range(num_classes):
        pm = (pred == cls)
        tm = (target == cls)
        inter = np.logical_and(pm, tm).sum()
        union = np.logical_or(pm, tm).sum()
        if union > 0:
            ious.append(inter / union)
            valid += 1
    return float(np.mean(ious)) if valid > 0 else 0.0

@torch.no_grad()
def collect_predictions(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for post, pre, nbr, targets in tqdm(data_loader, desc="收集预测", leave=False):
        post  = post.to(device).to(memory_format=torch.channels_last)
        pre   = pre.to(device).to(memory_format=torch.channels_last)
        nbr   = nbr.to(device).to(memory_format=torch.channels_last)
        targets = targets.to(device)
        with get_autocast():
            outputs = model(post, pre, nbr)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)

def calculate_metrics(y_true, y_pred, num_classes):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, labels=range(num_classes), average=None, zero_division=0
    )
    oa = np.trace(cm) / np.sum(cm)
    ious = []
    for i in range(num_classes):
        inter = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        ious.append(inter / union if union > 0 else 0.0)
    return {
        'confusion_matrix': cm, 'overall_accuracy': oa, 'mean_iou': float(np.mean(ious)),
        'ious': ious, 'precision': precision, 'recall': recall, 'f1_score': f1,
        'kappa_coefficient': cohen_kappa_score(y_true_flat, y_pred_flat), 'support': support
    }

def print_evaluation_report(metrics, class_names=None):
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics["precision"]))]
    print("\n" + "="*70)
    print("--- Swin V2 三路输入 + PPM-Lite (+SCSE) + MMoE 最终评估（CE-only） ---")  # ★ CE-only
    print("="*70)
    print(f"OA: {metrics['overall_accuracy']:.4f} | mIoU: {metrics['mean_iou']:.4f} | Kappa: {metrics['kappa_coefficient']:.4f}")
    print("-"*70)
    print(f"{'类别':<15} {'IoU':<10} {'精确率':<12} {'召回率':<10} {'F1':<10} {'样本数':<10}")
    for i, name in enumerate(class_names):
        print(f"{name:<15} {metrics['ious'][i]:<10.4f} {metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<10.4f} {metrics['f1_score'][i]:<10.4f} {metrics['support'][i]:<10.0f}")

def format_evaluation_report(metrics, class_names=None, title="Swin V2 三路输入 + PPM-Lite (+SCSE) + Old-Style MMoE"):
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics["precision"]))]
    lines = []
    lines.append(f"--- {title} 最终评估报告 ---")
    lines.append("="*70)
    lines.append(f"OA: {metrics['overall_accuracy']:.4f} | mIoU: {metrics['mean_iou']:.4f} | Kappa: {metrics['kappa_coefficient']:.4f}")
    lines.append("-"*70)
    lines.append(f"{'类别':<15} {'IoU':<10} {'精确率':<12} {'召回率':<10} {'F1':<10} {'样本数':<10}")
    for i, name in enumerate(class_names):
        lines.append(f"{name:<15} {metrics['ious'][i]:<10.4f} {metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<10.4f} {metrics['f1_score'][i]:<10.4f} {int(metrics['support'][i]):<10d}")
    return "\n".join(lines) + "\n"

def plot_confusion_matrix(cm, class_names=None, save_path=None):
    if class_names is None: class_names = [f'Class {i}' for i in range(len(cm))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - SwinV2 (3-input) + PPM-Lite(+SCSE) + MMoE (CE-only)')  # ★ CE-only
    plt.ylabel('GT'); plt.xlabel('Pred')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    plt.close()

def final_evaluate_and_report(model, data_loader, device, num_classes, class_names=None, save_dir=None):
    y_pred, y_true = collect_predictions(model, data_loader, device)
    metrics = calculate_metrics(y_true, y_pred, num_classes)
    print_evaluation_report(metrics, class_names)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cm_save_path = os.path.join(save_dir, 'confusion_matrix_swin_v2_mmoe_CEonly.png')
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_save_path)
    return metrics

# ---------------------------------------------------------------------------
# 可视化与逐图导出工具
# ---------------------------------------------------------------------------
PALETTE = np.array([
    [192, 192, 192],   # 0: 未烧 - 灰
    [0,   255, 0  ],   # 1: 低   - 绿
    [255, 255, 0  ],   # 2: 中   - 黄
    [255, 0,   0  ],   # 3: 高   - 红
], dtype=np.uint8)

def _read_img_infer(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL cannot open: {path}")
    arr = ds.ReadAsArray()
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr

def _colorize_mask(mask_np, palette=PALETTE):
    m = np.asarray(mask_np, dtype=np.int64)
    m = np.clip(m, 0, palette.shape[0]-1)
    return palette[m]

def _percentile_stretch(x, pmin=2, pmax=98):
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (y * 255.0 + 0.5).astype(np.uint8)

def _make_quicklook_rgb(post_arr):
    if post_arr.shape[0] >= 3:
        r, g, b = post_arr[0], post_arr[1], post_arr[2]
    else:
        r = g = b = post_arr[0]
    R = _percentile_stretch(r)
    G = _percentile_stretch(g)
    B = _percentile_stretch(b)
    rgb = np.stack([R, G, B], axis=-1)
    return rgb

def _save_panel(pred_rgb, gt_rgb, ql_rgb, save_path):
    H = max(pred_rgb.shape[0], gt_rgb.shape[0], ql_rgb.shape[0])
    pred_img = Image.fromarray(pred_rgb)
    gt_img   = Image.fromarray(gt_rgb)
    ql_img   = Image.fromarray(ql_rgb)

    def _resize_h(img, H_):
        if img.height == H_: return img
        new_w = int(round(img.width * (H_ / img.height)))
        return img.resize((new_w, H_), Image.NEAREST)

    pred_img = _resize_h(pred_img, H)
    gt_img   = _resize_h(gt_img, H)
    ql_img   = _resize_h(ql_img, H)

    panel = Image.new('RGB', (pred_img.width + gt_img.width + ql_img.width, H))
    x = 0
    for im in [pred_img, gt_img, ql_img]:
        panel.paste(im, (x, 0))
        x += im.width
    panel.save(save_path)

@torch.no_grad()
def export_test_predictions(model, val_dataset, out_root, device, img_size_expected=256):
    os.makedirs(out_root, exist_ok=True)
    model.eval()

    for (post_path, pre_path, nbr_path, mask_path) in tqdm(val_dataset.file_quadruplets, desc="Export test predictions"):
        base = os.path.splitext(os.path.basename(post_path))[0]
        out_dir = os.path.join(out_root, base)
        os.makedirs(out_dir, exist_ok=True)

        try:
            shutil.copy2(post_path, os.path.join(out_dir, "post.tif"))
        except Exception as e:
            print(f"[warn] copy post.tif failed for {base}: {e}")
        try:
            shutil.copy2(mask_path, os.path.join(out_dir, "gt.tif"))
        except Exception as e:
            print(f"[warn] copy gt.tif failed for {base}: {e}")

        post_np = _read_img_infer(post_path)
        pre_np  = _read_img_infer(pre_path)
        nbr_np  = _read_img_infer(nbr_path)
        gt_ds   = gdal.Open(mask_path, gdal.GA_ReadOnly)
        gt_np   = gt_ds.ReadAsArray().astype(np.int64)

        ql_rgb = _make_quicklook_rgb(post_np)

        H, W = post_np.shape[-2], post_np.shape[-1]
        assert H == img_size_expected and W == img_size_expected, \
            f"Input size {H}x{W} != expected {img_size_expected}x{img_size_expected}"

        post_t = torch.from_numpy(post_np).unsqueeze(0).to(device).to(memory_format=torch.channels_last)
        pre_t  = torch.from_numpy(pre_np ).unsqueeze(0).to(device).to(memory_format=torch.channels_last)
        nbr_t  = torch.from_numpy(nbr_np ).unsqueeze(0).to(device).to(memory_format=torch.channels_last)

        with get_autocast():
            out = model(post_t, pre_t, nbr_t)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits.argmax(1).squeeze(0).detach().cpu().numpy()

        pred_rgb = _colorize_mask(pred, PALETTE)
        gt_rgb   = _colorize_mask(gt_np, PALETTE)

        Image.fromarray(pred_rgb).save(os.path.join(out_dir, "pred.png"))
        Image.fromarray(gt_rgb).save(os.path.join(out_dir, "gt.png"))
        _save_panel(pred_rgb, gt_rgb, ql_rgb, os.path.join(out_dir, "panel.png"))

# ---------------------------------------------------------------------------
# 训练与验证（AMP + channels_last；★ Loss = 纯 CE）
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, criterion_ce, data_loader, device, epoch, num_classes, total_epochs,
                    T_start=2.0, T_end=0.5):  # ★ 去掉 alpha_bce/beta_lb/ds_weights
    model.train()
    running_loss, running_iou = 0.0, 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch:03d}", leave=False)

    # 门控温度退火（与 loss 无关）
    T = T_start + (T_end - T_start) * (epoch - 1) / max(1, total_epochs - 1)
    if hasattr(model, "set_gate_temperature"):
        model.set_gate_temperature(T)

    for post, pre, nbr, targets in progress_bar:
        post  = post.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        pre   = pre.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        nbr   = nbr.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast():
            outputs = model(post, pre, nbr)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion_ce(logits, targets)    # ★ 仅 CE

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_loss = float(loss.detach())
        batch_iou  = calculate_batch_iou(logits, targets, num_classes)
        running_loss += batch_loss
        running_iou  += batch_iou
        progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'mIoU': f'{batch_iou:.4f}', 'T': f'{T:.2f}'})

    return running_loss / len(data_loader), running_iou / len(data_loader)

@torch.no_grad()
def evaluate_micro(model, criterion_ce, data_loader, device, num_classes):
    model.eval()
    loss_sum = 0.0
    n_batches = 0

    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    progress_bar = tqdm(data_loader, desc="Evaluating(micro)", leave=False)

    for post, pre, nbr, targets in progress_bar:
        post  = post.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        pre   = pre.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        nbr   = nbr.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with get_autocast():
            outputs = model(post, pre, nbr)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion_ce(logits, targets)    # ★ 仅 CE

        loss_sum += float(loss.detach()); n_batches += 1

        pred = logits.argmax(1); t = targets
        for c in range(num_classes):
            p_c = (pred == c); t_c = (t == c)
            inter[c] += (p_c & t_c).sum().item()
            union[c] += (p_c | t_c).sum().item()

        valid = union > 0
        miou = (inter[valid] / union[valid]).mean() if valid.any() else 0.0
        progress_bar.set_postfix({'loss': f'{loss_sum/n_batches:.4f}', 'mIoU_micro': f'{miou:.4f}'})

    valid = union > 0
    miou = float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0
    return loss_sum / max(1, n_batches), miou

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    SAVE_DIR = "swin_v2_unet_triple_mmoe_results_ppm_sese"  
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据路径
    POST_TRAIN_DIR = "former_data/Training-20241111T204627Z-001/Training/Image/Post/*.tif"
    PRE_TRAIN_DIR  = "former_data/Training-20241111T204627Z-001/Training/Image/Pre/*.tif"
    NBR_TRAIN_DIR  = "former_data/Training-20241111T204627Z-001/Training/Image/NBR/*.tif"
    MASK_TRAIN_DIR = "former_data/Training-20241111T204627Z-001/Training/Mask/*.tif"

    POST_VAL_DIR = "former_data/Testing-20241111T204633Z-001/Testing/Image/Post/*.tif"
    PRE_VAL_DIR  = "former_data/Testing-20241111T204633Z-001/Testing/Image/Pre/*.tif"
    NBR_VAL_DIR  = "former_data/Testing-20241111T204633Z-001/Testing/Image/NBR/*.tif"
    MASK_VAL_DIR = "former_data/Testing-20241111T204633Z-001/Testing/Mask/*.tif"

    post_train_files = glob.glob(POST_TRAIN_DIR); pre_train_files  = glob.glob(PRE_TRAIN_DIR)
    nbr_train_files  = glob.glob(NBR_TRAIN_DIR); mask_train_files = glob.glob(MASK_TRAIN_DIR)
    post_val_files   = glob.glob(POST_VAL_DIR);  pre_val_files    = glob.glob(PRE_VAL_DIR)
    nbr_val_files    = glob.glob(NBR_VAL_DIR);   mask_val_files   = glob.glob(MASK_VAL_DIR)

    if not post_train_files or not pre_train_files or not nbr_train_files or not mask_train_files:
        print("错误：未找到训练文件。"); return False

    train_dataset = RemoteSensingDataset(post_train_files, pre_train_files, nbr_train_files, mask_train_files)
    val_dataset   = RemoteSensingDataset(post_val_files,  pre_val_files,  nbr_val_files,  mask_val_files)

    if len(train_dataset) == 0:
        print("训练集为空。"); return False

    try:
        p, pr, nb, m = train_dataset[0]
        print(f"样本Post: {p.shape} | Pre: {pr.shape} | NBR: {nb.shape} | Mask: {m.shape} | Mask范围: {m.min().item()} - {m.max().item()}")
    except Exception as e:
        print(f"样本检查失败: {e}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    NUM_CLASSES = 4
    EPOCHS = 120
    INPUT_CHANNELS_LIST = [9, 9, 3]

    # 类别权重（像素频次）
    print(">> 统计类别分布生成 class weights ...")
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for _, _, _, m in DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2):
        binc = torch.bincount(m.view(-1), minlength=NUM_CLASSES).to(torch.float64)
        class_counts += binc
    class_weights = (class_counts.sum() / (class_counts + 1e-6)).to(torch.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print("class weights:", class_weights.tolist())

    # 构建模型（★ 不使用 DS；仅 CE）
    model = SwinUNetV2_TripleInput(
        img_size=256, in_chans_list=INPUT_CHANNELS_LIST, window_size=8,
        num_classes=NUM_CLASSES, drop_path_rate=0.2, use_checkpoint=False,
        pretrained_window_sizes=[0,0,0,0], use_mmoe=True, mmoe_topk=2, mmoe_temp=2.0
    ).to(device)
    model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 总可训练参数量: {total_params/1e6:.2f} M")

    # 优化器 & 调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, (EPOCHS - warmup))
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)  # ★ 仅 CE

    BEST_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_swin_v2_triple_mmoe_ppm_scse.pth")
    best_val_iou, best_epoch = 0.0, 0

    print("\n=== 🚀 开始训练（CE-only | AMP bfloat16 + channels_last + PPM-Lite(+SCSE) + MMoE） ===")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(
            model, optimizer, criterion_ce, train_loader, device, epoch, NUM_CLASSES, EPOCHS,
            T_start=2.0, T_end=0.5
        )
        val_loss, val_iou_micro = evaluate_micro(model, criterion_ce, val_loader, device, NUM_CLASSES)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        cur_T = float(model.head.temperature) if hasattr(model, "head") and hasattr(model.head, "temperature") else None

        print(f"Epoch {epoch:03d}/{EPOCHS} -> 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
              f"训练mIoU(batch): {train_iou:.4f} | 验证mIoU(micro): {val_iou_micro:.4f} | "
              f"LR: {current_lr:.2e}" + (f" | T: {cur_T:.2f}" if cur_T is not None else ""))

        if val_iou_micro > best_val_iou:
            best_val_iou = val_iou_micro; best_epoch = epoch
            ckpt = {"state_dict": model.state_dict(), "gate_temperature": cur_T if cur_T is not None else 0.5, "epoch": epoch}
            torch.save(ckpt, BEST_MODEL_SAVE_PATH)
            print(f"✅ 保存新最好模型（micro mIoU: {best_val_iou:.4f}, T={ckpt['gate_temperature']:.2f}）")

    minutes = (time.time() - start_time) / 60
    print(f"\n⏱️  总训练时间: {minutes:.1f} 分钟")

    # 最终评估
    print("\n--- 🔍 加载最好模型做最终评估（CE-only） ---")
    final_model = SwinUNetV2_TripleInput(
        img_size=256, in_chans_list=INPUT_CHANNELS_LIST, window_size=8,
        num_classes=NUM_CLASSES, drop_path_rate=0.2, use_checkpoint=False,
        pretrained_window_sizes=[0,0,0,0], use_mmoe=True, mmoe_topk=2, mmoe_temp=0.5
    ).to(device)
    final_model.to(memory_format=torch.channels_last)

    ckpt = torch.load(BEST_MODEL_SAVE_PATH, map_location=device)
    final_model.load_state_dict(ckpt["state_dict"])
    if "gate_temperature" in ckpt:
        final_model.set_gate_temperature(float(ckpt["gate_temperature"]))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class_names = ['Unburn', 'Low', 'Moderate', 'Severe']
    final_metrics = final_evaluate_and_report(final_model, val_loader, device, NUM_CLASSES, class_names, SAVE_DIR)

    # === 覆盖写入完整评估报告（指定格式）
    with open(os.path.join(SAVE_DIR, "final_results_swin_v2_mmoe_ppm_scse.txt"), 'w') as f:
        report = format_evaluation_report(final_metrics, class_names, title="Swin V2 三路输入 + PPM-Lite (+SCSE) + Old-Style MMoE")
        f.write(report)
        f.write("\n")
        f.write(f"Best Val mIoU(micro): {best_val_iou:.4f} (Epoch {best_epoch})\n")

    TEST_EXPORT_DIR = os.path.join(SAVE_DIR, "test_outputs")
    export_test_predictions(final_model, val_dataset, TEST_EXPORT_DIR, device, img_size_expected=256)
    print(f"[OK] 测试导出完成：{TEST_EXPORT_DIR}")

    print("\n=== ✅ 训练完成（CE-only） ===")
    return True

if __name__ == "__main__":
    main()
