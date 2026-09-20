"""Unified trainer for every TPSM-SwinV2-UNet revision experiment.

Differences from the published scripts, all deliberate:

* Checkpoints are selected on a held-out *August* validation split, never on the
  Mosquito test event. The original scripts passed the Mosquito patches in as
  `val_loader` and kept the best epoch by that score, so every published number
  carries an optimistic bias. `split.json` supplies a spatially blocked 610/80
  split of the August grid; Mosquito is touched exactly once, at the end.
* Backbone blocks are imported from the original model file rather than
  retyped, so the architecture is bit-for-bit what produced the submitted
  results.
* GeoTIFF reading uses rasterio instead of osgeo.gdal (no GDAL on the servers).
* Every variant in the paper is one `--variant` flag, and every run writes a
  JSON with per-class metrics, the confusion matrix, ordinal metrics and timing.

Usage:
    python tpsm_train.py --variant A11_TPSM --seed 1 --out runs/A11_TPSM_s1
"""
import argparse
import glob
import importlib.machinery
import importlib.util
import json
import os
import random
import sys
import time
import types

os.environ.setdefault("MPLBACKEND", "Agg")
# Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.  It
# must be set before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import rasterio

# Several jobs share one GPU; expandable segments keep fragmentation from
# turning a fitting workload into an OOM (see OPERATION_LOG, 2026-07-25).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Import the original backbone definitions
# --------------------------------------------------------------------------


class _Stub(types.ModuleType):
    """Stands in for a module the original script imports but this trainer never calls."""

    def __init__(self, name):
        super().__init__(name)
        # torch._dynamo walks sys.modules and calls find_spec on every entry; a
        # module with __spec__ set to None makes that walk raise.
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _Stub(f"{self.__name__}.{name}")
        sys.modules[sub.__name__] = sub
        setattr(self, name, sub)
        return sub

    def __call__(self, *a, **k):
        return None


class _TqdmStub:
    """No-op progress bar that is still a usable base class.

    segmentation_models_pytorch subclasses tqdm at import time, so the stand-in
    cannot be a plain function.
    """

    def __init__(self, iterable=None, *a, **k):
        self._it = iterable

    def __iter__(self):
        return iter([] if self._it is None else self._it)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, *a, **k):
        pass

    def close(self):
        pass

    def set_postfix(self, *a, **k):
        pass

    def set_description(self, *a, **k):
        pass


def load_original(code_root):
    """Load the manuscript's model module.

    Only the network definitions are wanted. The file also pulls in GDAL and a
    plotting stack for its own training/export helpers, none of which are used
    here, so those imports are satisfied with stubs rather than installed.
    """
    # Stub only what is genuinely missing. Shadowing an installed package breaks
    # anything else that imports it for real: the lambda tqdm stub killed every
    # D-group baseline (smp subclasses tqdm) and the PIL stub then broke
    # torchvision, which wants `from PIL import __version__` (2026-07-26).
    for name in ("osgeo", "seaborn", "matplotlib", "tqdm", "PIL", "sklearn"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = _Stub(name)

    for parent, child in (("osgeo", "gdal"), ("matplotlib", "pyplot"),
                          ("sklearn", "metrics")):
        mod = sys.modules.get(parent)
        if isinstance(mod, _Stub):
            sys.modules[f"{parent}.{child}"] = getattr(mod, child)
    if isinstance(sys.modules.get("tqdm"), _Stub):
        sys.modules["tqdm"].tqdm = _TqdmStub

    path = os.path.join(code_root, "models", "swinv2_unet_3road_mmoe_ppm.py")
    spec = importlib.util.spec_from_file_location("tpsm_orig", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tpsm_orig"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Variant table
# --------------------------------------------------------------------------
# streams: 1 = post only, 2 = pre+post, 3 = pre+post+NBR composite
VARIANTS = {
    # -- Table 3 / Table 4 four-class ablation --------------------------------
    "A01_SwinV2":      dict(streams=1, unet=False, ppm=False, scse=False, mmoe=False),
    "A02_SwinV2UNet":  dict(streams=1, unet=True,  ppm=False, scse=False, mmoe=False),
    "A03_Bi":          dict(streams=2, unet=True,  ppm=False, scse=False, mmoe=False),
    "A04_T":           dict(streams=3, unet=True,  ppm=False, scse=False, mmoe=False),
    "A05_TP":          dict(streams=3, unet=True,  ppm=True,  scse=False, mmoe=False),
    "A06_TS":          dict(streams=3, unet=True,  ppm=False, scse=True,  mmoe=False),
    "A07_TPS":         dict(streams=3, unet=True,  ppm=True,  scse=True,  mmoe=False),
    "A08_TM":          dict(streams=3, unet=True,  ppm=False, scse=False, mmoe=True),
    "A09_TPM":         dict(streams=3, unet=True,  ppm=True,  scse=False, mmoe=True),
    "A10_TSM":         dict(streams=3, unet=True,  ppm=False, scse=True,  mmoe=True),
    "A11_TPSM":        dict(streams=3, unet=True,  ppm=True,  scse=True,  mmoe=True),
    # -- Referee 1.1: does the model still work without the label-source index? -
    "C3_TPSM_noNBR":   dict(streams=2, unet=True,  ppm=True,  scse=True,  mmoe=True),
    # Full TPSM with the index stream rebuilt from NDVI instead of the NBR
    # family the labels were thresholded on (Referee 1.1).
    "C4_TPSM_altindex": dict(streams=3, unet=True, ppm=True,  scse=True,  mmoe=True,
                             index_source="ndvi"),
    # -- Referee 1.4: the paper claims four experts, the code used five --------
    "E1_TPSM_4exp":    dict(streams=3, unet=True,  ppm=True,  scse=True,  mmoe=True, experts=4),
    # -- Referee 1.4: is the change-aware gate hint doing anything? ------------
    "E5_TPSM_nohint":  dict(streams=3, unet=True,  ppm=True,  scse=True,  mmoe=True, gate_hint=False),
    # E5 removes the hint entirely, which also removes the channels that carry
    # it -- so a drop there could be lost capacity rather than lost information.
    # These two keep the hint tensor's exact shape and destroy only its content.
    # "noise" matches each channel's per-sample mean and std; "shuffle" feeds a
    # real hint from a different patch in the batch, preserving spatial
    # structure and marginals exactly and destroying only the correspondence
    # with the image the head is segmenting.
    "E6_TPSM_hintnoise": dict(streams=3, unet=True, ppm=True, scse=True, mmoe=True,
                              hint_mode="noise"),
    "E7_TPSM_hintshuf":  dict(streams=3, unet=True, ppm=True, scse=True, mmoe=True,
                              hint_mode="shuffle"),
    "E6_TPM_hintnoise":  dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                              hint_mode="noise"),
    "E7_TPM_hintshuf":   dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                              hint_mode="shuffle"),
    # -- MoE design sweep -----------------------------------------------------
    # The head ships with five experts and top-k=2; neither number has been
    # probed. Expert count and k are already constructor arguments, so these
    # arms need no model code.
    "E2_TPM_2exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, experts=2),
    "E3_TPM_3exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, experts=3),
    "E4_TPM_4exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, experts=4),
    "E6_TPM_6exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, experts=6),
    "E8_TPM_8exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, experts=8),
    "K1_TPM_top1": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=1),
    "K3_TPM_top3": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=3),
    # 2026-09-17: complete the top-k sweep (k = 1..5) so k can be chosen on the
    # August validation split. k = 5 keeps all five experts (dense mixture).
    "K4_TPM_top4": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=4),
    "K5_TPM_top5": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=5),
    # k = 3 counterparts of the component and routing controls (used only if
    # k = 3 is selected on the validation split).
    "K3_TM_top3": dict(streams=3, unet=True, ppm=False, scse=False, mmoe=True, topk=3),
    "K3_SG_TPM_sharedgate": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                                 topk=3, shared_gate=True),
    "K3_E6_TPM_hintnoise": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                                topk=3, hint_mode="noise"),
    "K3_E7_TPM_hintshuf": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                               topk=3, hint_mode="shuffle"),
    # expert-count sensitivity at k = 3 (experts must be >= k)
    "K3_X3_TPM_3exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=3, experts=3),
    "K3_X4_TPM_4exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=3, experts=4),
    "K3_X6_TPM_6exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=3, experts=6),
    "K3_X8_TPM_8exp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True, topk=3, experts=8),
    # -- hint without MMoE ----------------------------------------------------
    # E5/E6/E7 show MMoE gains nothing once the gate hint is destroyed. These are
    # the missing other half: the same full-resolution hint reaches a plain head
    # with NO mixture-of-experts. If they match TPM, the gain is hint access, not
    # routing; if they stay at backbone level, the two are only useful together.
    # topk/experts are inert here because there is no MMoE head.
    "HH_TP_hinthead_mlp": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=False,
                               hint_head="mlp"),
    "HH_TP_hinthead_lin": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=False,
                               hint_head="linear"),
}
# Same counterpart set for k = 4 and k = 5, used only if validation selects one of them.
for _k in (4, 5):
    VARIANTS["K%d_TM_top%d" % (_k, _k)] = dict(streams=3, unet=True, ppm=False, scse=False, mmoe=True, topk=_k)
    VARIANTS["K%d_SG_TPM_sharedgate" % _k] = dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                                                  topk=_k, shared_gate=True)
    VARIANTS["K%d_E6_TPM_hintnoise" % _k] = dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                                                 topk=_k, hint_mode="noise")
    VARIANTS["K%d_E7_TPM_hintshuf" % _k] = dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                                                topk=_k, hint_mode="shuffle")
    for _e in (4, 5, 6, 8):
        if _e >= _k:
            VARIANTS["K%d_X%d_TPM_%dexp" % (_k, _e, _e)] = dict(streams=3, unet=True, ppm=True, scse=False,
                                                              mmoe=True, topk=_k, experts=_e)
VARIANTS.update({
    # Referee 1.4 asks for evidence that the routing's class-awareness earns its
    # keep. The head builds one gate per class; this arm makes all four classes
    # share a single gate, so the experts still mix per pixel but no longer per
    # class. Everything else -- expert count, top-k, temperature, towers -- is
    # unchanged, so the difference isolates class-awareness alone.
    "SG_TPM_sharedgate": dict(streams=3, unet=True, ppm=True, scse=False, mmoe=True,
                              shared_gate=True),
})


def variant_cfg(name):
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name}; choose from {sorted(VARIANTS)}")
    cfg = dict(experts=5, topk=2, gate_hint=True, index_source="nbr",
               hint_mode="real", shared_gate=False, hint_head=None)
    cfg.update(VARIANTS[name])
    return cfg


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Aligned (post, pre, nbr, mask) GeoTIFF quadruplets addressed by stem."""

    def __init__(self, root, stems, streams, binary=False, cache=True,
                 index_source="nbr", label_dir=None, layout="manuscript"):
        if layout == "ravg":
            # The prebuilt RAVG dataset keeps its streams side by side and its
            # labels in Label/, with 255 marking pixels outside the perimeter.
            self.dirs = {
                "post": os.path.join(root, "Image", "Post"),
                "pre": os.path.join(root, "Image", "Pre"),
                "nbr": os.path.join(root, "Image", "NBR"),
                "mask": label_dir or os.path.join(root, "Label"),
            }
            self.dirs["ndvi_pre"] = self.dirs["ndvi_post"] = self.dirs["nbr"]
            self.index_source = "nbr"
            self.stems = list(stems)
            self.streams = streams
            self.binary = binary
            self.cache = [self._load(s) for s in self.stems] if cache else None
            return

        # label_dir swaps the target for an externally derived product (RAVG
        # CBI4) while every input stream stays exactly as published. Pixels it
        # marks 255 are outside that product's perimeter, where it is undefined
        # rather than unburned, and are masked from loss and metrics.
        self.dirs = {
            "post": os.path.join(root, "Image", "Post"),
            "pre": os.path.join(root, "Image", "Pre"),
            "mask": label_dir or os.path.join(root, "Mask"),
        }
        nbr_train = os.path.join(root, "NBR_ALL", "NBR_train")
        nbr_test = os.path.join(root, "NBR_test")
        self.dirs["nbr"] = nbr_train if os.path.isdir(nbr_train) else nbr_test
        # Referee 1.1 asks whether the accuracy merely reflects fitting the dNBR
        # thresholds the labels came from. "ndvi" swaps the index stream for one
        # built from a different index family (NIR/Red rather than
        # NIR-narrow/SWIR-2), which took no part in generating the masks.
        self.dirs["ndvi_pre"] = os.path.join(root, "NDVI", "Pre_NDVI")
        self.dirs["ndvi_post"] = os.path.join(root, "NDVI", "Post_NDVI")
        self.index_source = index_source
        self.stems = list(stems)
        self.streams = streams
        self.binary = binary
        # The whole event is only a few GB decoded; holding it in RAM removes
        # GeoTIFF decoding from the training loop, which otherwise dominates.
        self.cache = [self._load(s) for s in self.stems] if cache else None

    def __len__(self):
        return len(self.stems)

    def channels(self):
        """Band counts of (post, pre, index), read from the data itself."""
        post, pre, nbr, _ = self[0]
        return [int(post.shape[0]),
                int(pre.shape[0]) if pre.numel() else 0,
                int(nbr.shape[0]) if nbr.numel() else 0]

    def _load(self, stem):
        post = torch.from_numpy(self._read(os.path.join(self.dirs["post"], stem)))
        pre = (torch.from_numpy(self._read(os.path.join(self.dirs["pre"], stem)))
               if self.streams >= 2 else torch.zeros(0))
        nbr = (torch.from_numpy(self._index_stream(stem))
               if self.streams >= 3 else torch.zeros(0))
        with rasterio.open(os.path.join(self.dirs["mask"], stem)) as ds:
            m = ds.read(1).astype(np.int64)
        if self.binary:
            m = (m > 0).astype(np.int64)
        return post, pre, nbr, torch.from_numpy(m)

    def _index_stream(self, stem):
        """Three-channel change composite for the index stream.

        "nbr"  -> the shipped [dNBR, RdNBR, dNBR2] raster, i.e. the same family
                  the severity thresholds were cut on.
        "ndvi" -> [dNDVI, NDVI_pre, NDVI_post], assembled here so the stream
                  carries an equally strong change signal from an index that had
                  no role in labelling. Same shape, so the model is unchanged.
        """
        if self.index_source == "nbr":
            return self._read(os.path.join(self.dirs["nbr"], stem))

        # The NDVI rasters carry a "_ndvi" suffix the other streams do not.
        ndvi_name = stem.replace(".tif", "_ndvi.tif")
        pre = self._read(os.path.join(self.dirs["ndvi_pre"], ndvi_name))
        post = self._read(os.path.join(self.dirs["ndvi_post"], ndvi_name))
        return np.concatenate([pre - post, pre, post], axis=0).astype(np.float32)

    @staticmethod
    def _read(path):
        with rasterio.open(path) as ds:
            arr = ds.read()
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, i):
        if self.cache is not None:
            return self.cache[i]
        return self._load(self.stems[i])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class SMPBaseline(nn.Module):
    """Referee 1.3 asks for modern segmentation baselines under comparable input.

    The three streams are concatenated channel-wise (9+9+3 = 21) and fed to an
    off-the-shelf architecture, so the comparison isolates the backbone rather
    than the input protocol. Encoders start from scratch, matching the Swin
    models, which use no pretrained weights either.
    """

    def __init__(self, arch, streams=3, num_classes=4, encoder="resnet34",
                 chans=None):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.streams = streams
        # Channel counts differ between the manuscript's Sentinel-2 patches
        # (9/9/3) and the RAVG Landsat dataset (6/6/2), so they are detected
        # from the data rather than assumed.
        in_ch = sum((chans or [9, 9, 3])[:streams])
        builders = {
            "unet": smp.Unet,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "segformer": smp.Segformer,
        }
        if arch not in builders:
            raise SystemExit(f"unknown arch {arch}; choose from {sorted(builders)}")
        self.net = builders[arch](
            encoder_name=encoder, encoder_weights=None,
            in_channels=in_ch, classes=num_classes)

    def set_gate_temperature(self, T):
        return None

    def forward(self, post, pre, nbr, return_aux=False):
        parts = [post, pre, nbr][:self.streams]
        x = torch.cat(parts, dim=1)
        logits = self.net(x)
        return (logits, None) if return_aux else logits


def placebo_hint(hint, mode, generator=None):
    """Destroy the hint's information while keeping its shape and statistics.

    Referee 1.4 asks whether the change-aware gate hint contributes anything.
    Deleting it (E5) also deletes the channels that carry it, so a drop there
    is ambiguous between lost information and lost capacity. These two keep the
    tensor identical in shape -- and therefore the head identical in size --
    and remove only what the hint knows about this particular patch.

    noise    per-channel Gaussian matched to each sample's own mean and std.
             Marginals are right; all spatial structure is gone.
    shuffle  a real hint from another sample in the batch. Marginals AND
             spatial structure are exactly those of a genuine hint; only the
             correspondence with the image being segmented is broken. This is
             the stronger control -- it cannot be passed by a model that is
             merely exploiting the hint's texture statistics.
    """
    if mode == "noise":
        m = hint.mean(dim=(2, 3), keepdim=True)
        s = hint.std(dim=(2, 3), keepdim=True)
        z = torch.randn(hint.shape, device=hint.device, dtype=hint.dtype,
                        generator=generator)
        # Standardise the draw before rescaling. Without this the realised mean
        # and std of each channel wander by ~1/sqrt(H*W) -- harmless in
        # expectation, but it leaves a nuisance difference between arms that
        # has nothing to do with the information being removed.
        z = (z - z.mean(dim=(2, 3), keepdim=True)) / \
            z.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return z * s + m
    if mode == "shuffle":
        if hint.shape[0] == 1:
            # Nothing to swap with; fall back to noise rather than silently
            # handing the model its own hint back and calling it a placebo.
            return placebo_hint(hint, "noise", generator)
        return torch.roll(hint, shifts=1, dims=0)
    raise ValueError("unknown hint_mode %r" % mode)


class SwinUNetV2Flex(nn.Module):
    """One backbone covering every ablation cell in the paper."""

    def __init__(self, O, streams=3, unet=True, ppm=True, scse=True, mmoe=True,
                 experts=5, topk=2, gate_hint=True, num_classes=4, img_size=256,
                 patch_size=4, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=8, drop_path_rate=0.2,
                 chans=None, hint_mode="real", shared_gate=False, hint_head=None):
        super().__init__()
        self.streams, self.use_unet = streams, unet
        self.use_ppm, self.use_scse, self.use_mmoe = ppm, scse, mmoe
        self.use_gate_hint = gate_hint
        self.hint_head = hint_head
        self.hint_mode = hint_mode
        self.num_classes = num_classes
        # 9/9/3 for the manuscript's Sentinel-2 patches, 6/6/2 for the RAVG
        # Landsat dataset; passed in from whatever the loader actually found.
        chans = list((chans or [9, 9, 3])[:streams])
        self.chans = chans

        norm_layer = nn.LayerNorm
        self.embeds = nn.ModuleList([
            O.PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=c,
                         embed_dim=embed_dim, norm_layer=norm_layer) for c in chans
        ])
        self.fusion = (nn.Sequential(nn.LayerNorm(embed_dim * streams),
                                     nn.Linear(embed_dim * streams, embed_dim),
                                     nn.GELU()) if streams > 1 else None)

        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        pr = self.embeds[0].patches_resolution
        self.patches_resolution = pr
        self.pos_drop = nn.Dropout(0.0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.encoders = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoders.append(O.BasicLayerV2(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(pr[0] // (2 ** i), pr[1] // (2 ** i)),
                depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=O.PatchMerging if (i < self.num_layers - 1) else None,
                use_checkpoint=False, pretrained_window_size=0))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_features), nn.ReLU(inplace=True),
            nn.Conv2d(self.num_features, self.num_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_features), nn.ReLU(inplace=True))
        self.bottleneck_norm = norm_layer(self.num_features)

        if unet:
            self.decoders = nn.ModuleList()
            dim = self.num_features
            for i in range(self.num_layers - 1):
                expand = O.PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
                skip = int(embed_dim * 2 ** (self.num_layers - 2 - i))
                merged = dim // 2 + skip
                layer = O.BasicLayerV2(
                    dim=merged,
                    input_resolution=(pr[0] // (2 ** (self.num_layers - 2 - i)),
                                      pr[1] // (2 ** (self.num_layers - 2 - i))),
                    depth=depths[self.num_layers - 2 - i],
                    num_heads=num_heads[self.num_layers - 2 - i],
                    window_size=window_size, mlp_ratio=4., qkv_bias=True,
                    drop=0., attn_drop=0.,
                    drop_path=dpr[sum(depths[:self.num_layers - 2 - i]):
                                  sum(depths[:self.num_layers - 1 - i])],
                    norm_layer=norm_layer, downsample=None, use_checkpoint=False,
                    pretrained_window_size=0)
                self.decoders.append(nn.ModuleList([expand, layer]))
                dim = merged
            self.up = O.FinalPatchExpand_X4(input_resolution=(pr[0], pr[1]), dim=dim)
            self.final_dim = dim
        else:
            # Post-only SwinV2 baseline: classify at bottleneck stride and
            # bilinearly upsample, exactly as swinorg.py does.
            self.final_dim = self.num_features

        if self.use_ppm:
            self.ppm = O.PPMLite(in_ch=self.final_dim)
        if self.use_scse:
            self.scse = O.SCSE(ch=self.final_dim)

        if self.use_mmoe:
            hint_ch = (sum(chans) if gate_hint and streams >= 2 else 0)
            if gate_hint and streams >= 2:
                hint_ch = (chans[2] if streams >= 3 else 0) + chans[0]
            # OldStyleMMoEHead carries `assert num_classes == 4` from the
            # published code, which blocks the binary (Table 6) runs. Its gates
            # and towers are built independently per class and forward() loops
            # over self.num_classes, so building at 4 and truncating yields
            # exactly the module a 2-class construction would have produced --
            # without editing the archived model file.
            self.head = O.OldStyleMMoEHead(
                in_ch=self.final_dim, gate_hint_ch=max(hint_ch, 1),
                num_classes=4, num_experts=experts,
                expert_ch=self.final_dim // 2, tower_ch=self.final_dim // 2,
                topk=topk, temperature=2.0)
            if num_classes != 4:
                self.head.num_classes = num_classes
                self.head.gates = nn.ModuleList(list(self.head.gates)[:num_classes])
                self.head.towers = nn.ModuleList(list(self.head.towers)[:num_classes])
            if shared_gate:
                # One gate for every class instead of one per class. Listing the
                # same module repeatedly shares its weights -- parameters() sees
                # it once -- so the experts still mix per pixel but the mixture
                # no longer depends on which class is being predicted. Towers,
                # experts, top-k and temperature are untouched.
                g = self.head.gates[0]
                self.head.gates = nn.ModuleList([g] * self.head.num_classes)
            self.hint_ch = hint_ch
        elif hint_head:
            # Control for "is the MMoE gain the mixture, or just the gate hint?".
            # The MMoE head is replaced by a plain head that receives the SAME
            # full-resolution hint the gate would have seen, concatenated to the
            # decoder features. "mlp" matches the MMoE tower width so the control
            # is not simply starved of capacity; "linear" is the minimal version.
            self.hint_ch = (chans[2] if streams >= 3 else 0) + chans[0]
            if hint_head == "mlp":
                self.head = nn.Sequential(
                    nn.Conv2d(self.final_dim + self.hint_ch, self.final_dim // 2, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(self.final_dim // 2, num_classes, 1, bias=True))
            elif hint_head == "linear":
                self.head = nn.Conv2d(self.final_dim + self.hint_ch, num_classes, 1, bias=True)
            else:
                raise ValueError("unknown hint_head %r" % hint_head)
        else:
            self.head = nn.Conv2d(self.final_dim, num_classes, 1, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            O = sys.modules["tpsm_orig"]
            O.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_gate_temperature(self, T):
        if self.use_mmoe:
            self.head.set_temperature(T)

    def forward(self, post, pre, nbr, return_aux=False):
        B, _, H, W = post.shape
        ins = [post, pre, nbr][:self.streams]
        toks = [e(x) for e, x in zip(self.embeds, ins)]
        x = torch.cat(toks, dim=2) if self.streams > 1 else toks[0]
        if self.fusion is not None:
            x = self.fusion(x)
        x = self.pos_drop(x)

        feats = []
        for i, enc in enumerate(self.encoders):
            res = (self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i))
            feats.append(x.view(B, res[0], res[1], -1))
            x = enc(x)

        fh = H // (self.embeds[0].patch_size[0] * (2 ** (self.num_layers - 1)))
        fw = W // (self.embeds[0].patch_size[1] * (2 ** (self.num_layers - 1)))
        x = x.permute(0, 2, 1).contiguous().view(B, self.num_features, fh, fw)
        x = self.bottleneck(x)
        xs = self.bottleneck_norm(x.flatten(2).transpose(1, 2).contiguous())

        if self.use_unet:
            x = xs
            for i, (expand, layer) in enumerate(self.decoders):
                res = (self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i)),
                       self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i)))
                x, _, _ = expand(x, res[0], res[1])
                skip = feats[-(i + 2)]
                x = torch.cat([x, skip.view(B, -1, skip.shape[-1])], -1)
                x = layer(x)
            x = self.up(x).permute(0, 3, 1, 2).contiguous()
        else:
            x = xs.permute(0, 2, 1).contiguous().view(B, self.num_features, fh, fw)

        if self.use_ppm:
            x = self.ppm(x)
        if self.use_scse:
            x = self.scse(x)
        # (placebo substitution happens just below, where the hint is built)

        if self.use_mmoe:
            if self.use_gate_hint and self.streams >= 3:
                hint = torch.cat([nbr, torch.abs(post - pre)], dim=1)
            elif self.use_gate_hint and self.streams == 2:
                hint = torch.abs(post - pre)
            else:
                hint = torch.zeros(B, 1, x.shape[-2], x.shape[-1],
                                   device=x.device, dtype=x.dtype)
            if self.hint_mode != "real" and self.use_gate_hint:
                hint = placebo_hint(hint, self.hint_mode)
            logits, aux = self.head(x, hint)
        elif self.hint_head:
            hint = torch.cat([nbr, torch.abs(post - pre)], dim=1) if self.streams >= 3 \
                else torch.abs(post - pre)
            if self.hint_mode != "real":
                hint = placebo_hint(hint, self.hint_mode)
            if hint.shape[-2:] != x.shape[-2:]:
                hint = F.interpolate(hint, size=x.shape[-2:], mode="bilinear", align_corners=False)
            logits, aux = self.head(torch.cat([x, hint], dim=1)), None
        else:
            logits, aux = self.head(x), None

        if not self.use_unet:
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return (logits, aux) if return_aux else logits


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.weight, self.gamma = weight, gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        channels = logits.shape[1]
        flat_logits = logits.movedim(1, -1).reshape(-1, channels)
        ce = F.cross_entropy(flat_logits, target.reshape(-1),
                             weight=self.weight, reduction="none",
                             ignore_index=self.ignore_index).reshape_as(target)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SoftIoULoss(nn.Module):
    def __init__(self, num_classes, weight=None, eps=1e-6):
        super().__init__()
        self.n, self.weight, self.eps = num_classes, weight, eps

    def forward(self, logits, target):
        p = torch.softmax(logits.float(), dim=1)
        t = F.one_hot(target, self.n).permute(0, 3, 1, 2).float()
        inter = (p * t).sum(dim=(0, 2, 3))
        union = (p + t - p * t).sum(dim=(0, 2, 3))
        iou = (inter + self.eps) / (union + self.eps)
        if self.weight is not None:
            return 1.0 - (iou * self.weight).sum() / self.weight.sum()
        return 1.0 - iou.mean()


class FlatCrossEntropyLoss(nn.Module):
    """Pixelwise CE through CUDA's deterministic 2-D NLL implementation.

    Passing [B,C,H,W] directly selects nll_loss2d on Windows CUDA, whose
    reduction is nondeterministic.  Flattening pixels to [B*H*W,C] is
    mathematically identical and permits strict deterministic algorithms.
    """

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        channels = logits.shape[1]
        flat_logits = logits.movedim(1, -1).reshape(-1, channels)
        return F.cross_entropy(flat_logits, target.reshape(-1),
                               weight=self.weight,
                               ignore_index=self.ignore_index)


def build_loss(name, weights, n, ignore_index=-100):
    if name == "ce":
        return FlatCrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(weight=weights, ignore_index=ignore_index)
    if name == "iou":
        return SoftIoULoss(n, weight=weights)
    raise SystemExit(f"unknown loss {name}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def confusion(pred, true, n):
    k = (true * n + pred).astype(np.int64)
    return np.bincount(k, minlength=n * n).reshape(n, n).astype(np.float64)


def metrics_from_cm(cm):
    n = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    total = cm.sum()
    prec = np.divide(tp, tp + fp, out=np.zeros(n), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(n), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros(n), where=(tp + fp + fn) > 0)
    oa = tp.sum() / total if total else 0.0
    pe = ((cm.sum(0) / total) * (cm.sum(1) / total)).sum() if total else 0.0
    kappa = (oa - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    # Ordinal metrics (Referee 1.8): severity levels are ordered, so distance matters.
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    mae = (cm * dist).sum() / total if total else 0.0
    w = dist.astype(float) ** 2
    if n > 1:
        w /= w.max()
    exp = np.outer(cm.sum(1), cm.sum(0)) / total if total else np.zeros_like(cm)
    num, den = (w * cm).sum(), (w * exp).sum()
    qwk = 1 - num / den if den > 0 else 0.0

    return dict(
        precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(), iou=iou.tolist(),
        mean_precision=float(prec.mean()), mean_recall=float(rec.mean()),
        mean_f1=float(f1.mean()), mIoU=float(iou.mean()),
        OA=float(oa), kappa=float(kappa),
        ordinal_mae=float(mae), qwk=float(qwk),
        confusion_matrix=cm.astype(np.int64).tolist(),
    )


@torch.no_grad()
def evaluate(model, loader, device, n, criterion=None):
    model.eval()
    cm = np.zeros((n, n), dtype=np.float64)
    loss_sum, nb = 0.0, 0
    for post, pre, nbr, y in loader:
        post = post.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        pre = pre.to(device, non_blocking=True) if pre.numel() else pre
        nbr = nbr.to(device, non_blocking=True) if nbr.numel() else nbr
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(post, pre, nbr)
            if criterion is not None:
                loss_sum += float(criterion(logits.float(), y))
                nb += 1
        p = logits.argmax(1).cpu().numpy().ravel()
        t = y.cpu().numpy().ravel()
        keep = t < n                    # drops the ignore label, if any
        cm += confusion(p[keep], t[keep], n)
    m = metrics_from_cm(cm)
    m["loss"] = loss_sum / nb if nb else None
    return m


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_loader_worker(worker_id):
    """Seed NumPy/Python workers from the DataLoader's private RNG stream."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--arch", default="tpsm",
                    choices=["tpsm", "unet", "deeplabv3plus", "segformer"],
                    help="tpsm uses --variant; the others are referee 1.3 baselines "
                         "on the same tri-stream input")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--loss", default="ce", choices=["ce", "focal", "iou"])
    ap.add_argument("--classes", type=int, default=4, choices=[2, 4])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true",
                    help="stream patches from disk instead of preloading into RAM")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-test-root", default=None,
                    help="optional RAVG-layout dataset whose Testing directory "
                         "is evaluated with the same validation-selected checkpoint")
    ap.add_argument("--extra-test-name", default="cross_fire",
                    help="JSON key used for --extra-test-root metrics")
    ap.add_argument("--label-dir-train", default=None,
                    help="override the target for the train/val split, e.g. RAVG CBI4")
    ap.add_argument("--label-dir-test", default=None,
                    help="override the target for the held-out event")
    ap.add_argument("--ignore-index", type=int, default=-100,
                    help="label value excluded from loss and metrics (255 for RAVG)")
    ap.add_argument("--layout", choices=("manuscript", "ravg"), default="manuscript",
                    help="directory layout: the paper's two-event tree, or the "
                         "prebuilt RAVG dataset with its own Train/Val/Test split")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="opt out of deterministic CUDA/DataLoader controls")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    set_seed(args.seed)
    deterministic = not args.nondeterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    # PPM-Lite uses adaptive_avg_pool2d; its CUDA backward has no deterministic
    # implementation in the current PyTorch build.  Warn-only keeps every arm
    # on the same cuDNN/RNG protocol while making that unavoidable exception
    # explicit instead of crashing only the PPM arms.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    O = load_original(args.code_root)
    cfg = variant_cfg(args.variant)
    binary = args.classes == 2

    cache = not args.no_cache
    # Workers would each fork their own copy of the RAM cache, so caching and
    # multiprocess loading are mutually exclusive.
    workers = 0 if cache else args.workers
    t_load = time.time()
    idx_src = cfg["index_source"]

    if args.layout == "ravg":
        # This dataset ships its own fire-level split as three directories, so
        # the stems come from disk rather than from a split file.
        roots = {k: os.path.join(args.data_root, k)
                 for k in ("Training", "Validation", "Testing")}
        stems = {k: sorted(os.path.basename(p) for p in
                           glob.glob(os.path.join(v, "Label", "*.tif")))
                 for k, v in roots.items()}
        train_root, val_root, test_root = (roots["Training"], roots["Validation"],
                                           roots["Testing"])
        ds_tr = PatchDataset(train_root, stems["Training"], cfg["streams"], binary,
                             cache, idx_src, None, "ravg")
        ds_va = PatchDataset(val_root, stems["Validation"], cfg["streams"], binary,
                             cache, idx_src, None, "ravg")
        ds_te = PatchDataset(test_root, stems["Testing"], cfg["streams"], binary,
                             cache, idx_src, None, "ravg")
        split = {"train": stems["Training"], "val": stems["Validation"],
                 "test": stems["Testing"]}
        label_root = os.path.join(train_root, "Label")
    else:
        with open(args.split, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_root = os.path.join(args.data_root, "Training", "Training")
        test_root = os.path.join(args.data_root, "Testing", "Testing")
        ds_tr = PatchDataset(train_root, split["train"], cfg["streams"], binary, cache,
                             idx_src, args.label_dir_train)
        ds_va = PatchDataset(train_root, split["val"], cfg["streams"], binary, cache,
                             idx_src, args.label_dir_train)
        ds_te = PatchDataset(test_root, split["test"], cfg["streams"], binary, cache,
                             idx_src, args.label_dir_test)
        label_root = args.label_dir_train or os.path.join(train_root, "Mask")

    ds_extra = None
    if args.extra_test_root:
        if args.layout != "ravg":
            raise SystemExit("--extra-test-root currently requires --layout ravg")
        extra_root = os.path.join(args.extra_test_root, "Testing")
        extra_stems = sorted(os.path.basename(p) for p in
                             glob.glob(os.path.join(extra_root, "Label", "*.tif")))
        if not extra_stems:
            raise SystemExit(f"no extra-test labels below {extra_root}")
        ds_extra = PatchDataset(extra_root, extra_stems, cfg["streams"], binary,
                                cache, idx_src, None, "ravg")

    in_chans = ds_tr.channels()
    print(f"[data] detected channels post/pre/index = {in_chans}", flush=True)
    print(f"[data] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} "
          f"cache={cache} load={time.time() - t_load:.1f}s", flush=True)
    if ds_extra is not None:
        if ds_extra.channels() != in_chans:
            raise SystemExit(
                f"extra-test channels {ds_extra.channels()} != training {in_chans}"
            )
        print(f"[data] extra_test={args.extra_test_name} patches={len(ds_extra)}",
              flush=True)

    # A trailing batch of one breaks PPM-Lite: its 1x1 pooled branch reaches
    # BatchNorm with a single value per channel. Shuffling means a different
    # handful is dropped each epoch, so nothing is systematically excluded.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=True,
                       persistent_workers=workers > 0,
                       worker_init_fn=seed_loader_worker,
                       generator=loader_generator)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)
    dl_extra = (DataLoader(ds_extra, batch_size=args.batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)
                if ds_extra is not None else None)

    # Class weights from the TRAIN split only.
    counts = np.zeros(args.classes, dtype=np.float64)
    for stem in split["train"]:
        with rasterio.open(os.path.join(label_root, stem)) as d:
            m = d.read(1)
        if binary:
            m = (m > 0).astype(np.int64)
        # Ignored pixels must not enter the frequencies the weights come from.
        m = m[m < args.classes]
        counts += np.bincount(m.ravel(), minlength=args.classes)[:args.classes]
    w = counts.sum() / (counts + 1e-6)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print(f"[data] class weights {w.tolist()}", flush=True)

    def build_model():
        if args.arch == "tpsm":
            m = SwinUNetV2Flex(
                O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"], scse=cfg["scse"],
                mmoe=cfg["mmoe"], experts=cfg["experts"], topk=cfg["topk"],
                gate_hint=cfg["gate_hint"], num_classes=args.classes,
                chans=in_chans, hint_mode=cfg["hint_mode"],
                shared_gate=cfg["shared_gate"], hint_head=cfg["hint_head"]).to(device)
        else:
            m = SMPBaseline(args.arch, streams=cfg["streams"],
                            num_classes=args.classes, chans=in_chans).to(device)
        m.to(memory_format=torch.channels_last)
        return m

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.variant} params={n_params / 1e6:.2f}M", flush=True)

    criterion = build_loss(args.loss, w, args.classes, args.ignore_index)
    warmup = 5

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))

    # The optimiser and scheduler are built per attempt, inside the loop below,
    # so that a restart after divergence starts from a clean state.

    ckpt_path = os.path.join(args.out, "best.pth")
    t0 = time.time()
    recoveries, skipped = [], {"nan_batches": 0, "oom_batches": 0}

    # A single NaN batch used to be fatal: NaN grads survive clip_grad_norm_ and
    # AdamW writes them into every weight, so the run produced nan loss for all
    # 120 epochs (A_A03_Bi_s1, 2026-07-25). Bad batches are now dropped before
    # backward, and a run that still diverges restarts on a fresh seed.
    MAX_ATTEMPTS = 3
    for attempt in range(MAX_ATTEMPTS):
        cur_lr = args.lr * (0.5 ** attempt)
        if attempt:
            cur_seed = args.seed + 1000 * attempt
            print(f"[recover] diverged; attempt {attempt + 1}/{MAX_ATTEMPTS} "
                  f"reinit seed={cur_seed} lr={cur_lr:.2e}", flush=True)
            recoveries.append({"attempt": attempt, "seed": cur_seed, "lr": cur_lr})
            set_seed(cur_seed)
            loader_generator.manual_seed(cur_seed)
            del model
            torch.cuda.empty_cache()
            model = build_model()

        best = {"mIoU": -1.0, "epoch": -1}
        history = []
        opt = torch.optim.AdamW(model.parameters(), lr=cur_lr,
                                weight_decay=0.05, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        nan_streak, diverged = 0, False

        for ep in range(1, args.epochs + 1):
            model.train()
            T = 2.0 + (0.5 - 2.0) * (ep - 1) / max(1, args.epochs - 1)
            model.set_gate_temperature(T)
            run_loss, nb = 0.0, 0
            for post, pre, nbr, y in dl_tr:
                try:
                    post = post.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    pre = pre.to(device, non_blocking=True) if pre.numel() else pre
                    nbr = nbr.to(device, non_blocking=True) if nbr.numel() else nbr
                    y = y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(post, pre, nbr)
                        loss = criterion(logits.float(), y)
                    if not torch.isfinite(loss):
                        opt.zero_grad(set_to_none=True)
                        skipped["nan_batches"] += 1
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    run_loss += float(loss.detach())
                    nb += 1
                except RuntimeError as e:
                    # Exhausting VRAM surfaces as torch.OutOfMemoryError from the
                    # allocator but as torch.AcceleratorError from a .to() call;
                    # both subclass RuntimeError, so match on the message instead
                    # of the class. A neighbouring job on the same GPU spiked --
                    # drop this batch rather than losing the whole run.
                    if "out of memory" not in str(e).lower():
                        raise
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    skipped["oom_batches"] += 1
                    if skipped["oom_batches"] > 200:
                        raise
                    time.sleep(2.0)
                    continue
            sched.step()

            ep_loss = run_loss / max(nb, 1)
            if nb == 0 or not np.isfinite(ep_loss):
                nan_streak += 1
                print(f"[{args.variant} s{args.seed}] ep {ep:03d}/{args.epochs} "
                      f"loss=nan (streak {nan_streak})", flush=True)
                if nan_streak >= 2:
                    diverged = True
                    break
                continue
            nan_streak = 0

            # Validation runs outside the per-batch guard above, and a
            # neighbouring job spiking here used to kill the whole run. Retry
            # once with a cleared cache, then skip this epoch's validation.
            va = None
            for attempt_va in range(2):
                try:
                    va = evaluate(model, dl_va, device, args.classes, criterion)
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    skipped["oom_batches"] += 1
                    time.sleep(5.0)
            if va is None:
                print(f"[{args.variant} s{args.seed}] ep {ep:03d} validation skipped (OOM)",
                      flush=True)
                continue

            history.append({"epoch": ep, "train_loss": ep_loss,
                            "val_mIoU": va["mIoU"], "val_OA": va["OA"], "gate_T": T})
            if va["mIoU"] > best["mIoU"]:
                best = {"mIoU": va["mIoU"], "epoch": ep}
                torch.save({"state_dict": model.state_dict(), "epoch": ep,
                            "gate_temperature": T, "val_mIoU": va["mIoU"]}, ckpt_path)
            print(f"[{args.variant} s{args.seed}] ep {ep:03d}/{args.epochs} "
                  f"loss={ep_loss:.4f} val_mIoU={va['mIoU']:.4f} "
                  f"best={best['mIoU']:.4f}@{best['epoch']}", flush=True)

        if not diverged:
            break

    train_minutes = (time.time() - t0) / 60

    # Final test pass on the held-out event, using the val-selected checkpoint.
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.set_gate_temperature(ck.get("gate_temperature", 0.5))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    te = evaluate(model, dl_te, device, args.classes)
    va = evaluate(model, dl_va, device, args.classes)
    extra_te = (evaluate(model, dl_extra, device, args.classes)
                if dl_extra is not None else None)

    t1 = time.time()
    _ = evaluate(model, dl_te, device, args.classes)
    infer_s = time.time() - t1

    result = {
        "variant": args.variant, "arch": args.arch, "seed": args.seed, "loss": args.loss,
        "classes": args.classes, "config": cfg,
        "params_M": n_params / 1e6,
        "best_val_epoch": best["epoch"], "best_val_mIoU": best["mIoU"],
        "train_minutes": train_minutes,
        "recoveries": recoveries, "skipped_batches": skipped,
        "test_inference_seconds": infer_s,
        "test_patches": len(ds_te),
        "fps": len(ds_te) / infer_s if infer_s > 0 else None,
        "val_final": va, "test": te,
        "extra_test": ({"name": args.extra_test_name,
                        "root": os.path.abspath(args.extra_test_root),
                        "patches": len(ds_extra), "metrics": extra_te}
                       if ds_extra is not None else None),
        "n_train": len(ds_tr), "n_val": len(ds_va),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "deterministic": deterministic,
        "determinism_controls": {
            "private_dataloader_generator": True,
            "cudnn_deterministic": deterministic,
            "torch_deterministic_algorithms": deterministic,
            "torch_deterministic_warn_only": deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pixelwise_ce_flattened": args.loss in ("ce", "focal"),
            "bit_exact_checkpoint_guarantee": False,
        },
        "data_root": os.path.abspath(args.data_root),
        "train_order_seed": args.seed,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] {args.variant} s{args.seed} "
          f"TEST mIoU={te['mIoU']:.4f} OA={te['OA']:.4f} kappa={te['kappa']:.4f} "
          f"QWK={te['qwk']:.4f} | {train_minutes:.1f} min", flush=True)
    if extra_te is not None:
        print(f"[EXTRA] {args.extra_test_name} patches={len(ds_extra)} "
              f"mIoU={extra_te['mIoU']:.4f} OA={extra_te['OA']:.4f} "
              f"QWK={extra_te['qwk']:.4f}", flush=True)


if __name__ == "__main__":
    main()
