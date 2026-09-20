"""How much of the paper's mIoU comes from the easy unburned margin?

The paper's Mosquito grid is 77% unburned on the August side and tiles 4.3x the
fire's area; MTBS tiles tightly around the perimeter and is 83% burned. So the
paper's 0.86 and the MTBS 0.75 are not measured on the same field, and a
reviewer comparing them directly would draw the wrong conclusion.

Split the paper's own test metrics by whether a pixel falls inside the BAER
mapped perimeter. Inside is the part that is actually comparable to an
operational product; outside is background the model gets almost for free.

Inference only, batch 1, eval mode -- it runs alongside training without
crowding the card, as the bootstrap run already demonstrated.
"""
import glob
import importlib.util
import json
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = r"D:\former_files\former_data (2)\former_data"
SBS_MOSQ = os.path.join(ROOT, "results", "sbs_labels", "mosquito")
CODE = os.path.join(ROOT, "code_base", "wildfire-TPSM-SwinV2-UNet-main")
SPLIT = os.path.join(ROOT, "results", "split.json")
NAMES = ["Unburned", "Low", "Moderate", "Severe"]
RUNS = ["runs/A_A04_T_s1", "runs/A_A09_TPM_s6", "runs/A_A11_TPSM_s6"]


def load_trainer():
    spec = importlib.util.spec_from_file_location("tpsm_train", os.path.join(HERE, "tpsm_train.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(cm):
    n = cm.shape[0]
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tot = cm.sum()
    iou = np.divide(tp, tp + fp + fn, out=np.zeros(n), where=(tp + fp + fn) > 0)
    oa = tp.sum() / tot if tot else 0.0
    return iou, float(iou.mean()), float(oa), int(tot)


T = load_trainer()
O = T.load_original(CODE)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = json.load(open(SPLIT, encoding="utf-8"))
test_root = os.path.join(DATA, "Testing", "Testing")

# Perimeter mask per patch, reprojected straight from the BAER raster.
#
# An earlier version read pre-built label files from results/sbs_labels, but
# that builder was never run and the directory was empty, so every patch came
# back "0% inside" -- a silently wrong answer rather than an error. Deriving
# the mask from the source raster here removes the dependency, and the count
# printed below fails loudly if the source is missing.
from rasterio.warp import Resampling, reproject

sbs_src = glob.glob(os.path.join(ROOT, "results", "baer_sbs", "mosquito", "*.tif"))
if not sbs_src:
    raise SystemExit("no BAER SBS raster for Mosquito; run scripts/sbs_extract.py first")

inside = {}
with rasterio.open(sbs_src[0]) as src:
    for stem in split["test"]:
        with rasterio.open(os.path.join(test_root, "Mask", stem)) as ref:
            h, w = ref.height, ref.width
            tr, crs = ref.transform, ref.crs
        buf = np.zeros((h, w), dtype=np.uint8)
        reproject(source=rasterio.band(src, 1), destination=buf,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        m = np.isin(buf, [1, 2, 3, 4])      # BAER classes; 0 and 15 are outside
        if m.any():
            inside[stem] = m

cov = sum(int(m.sum()) for m in inside.values())
print("patches intersecting the BAER perimeter: %d of %d  (%d pixels)"
      % (len(inside), len(split["test"]), cov))
if not inside:
    raise SystemExit("no overlap found -- check the reprojection before trusting anything")

out = {}
for run in RUNS:
    rd = os.path.join(ROOT, run)
    rj, ck = os.path.join(rd, "result.json"), os.path.join(rd, "best.pth")
    if not (os.path.exists(rj) and os.path.exists(ck)):
        print("%-18s no checkpoint" % os.path.basename(run))
        continue
    res = json.load(open(rj, encoding="utf-8"))
    cfg = res["config"]
    ds = T.PatchDataset(test_root, split["test"], cfg["streams"], False, False,
                        cfg.get("index_source", "nbr"), None)
    model = T.SwinUNetV2Flex(
        O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"], scse=cfg["scse"],
        mmoe=cfg["mmoe"], experts=cfg["experts"], topk=cfg["topk"],
        gate_hint=cfg["gate_hint"], num_classes=4, chans=ds.channels()).to(dev)
    model.to(memory_format=torch.channels_last)
    c = torch.load(ck, map_location=dev)
    model.load_state_dict(c["state_dict"])
    if hasattr(model, "set_gate_temperature"):
        model.set_gate_temperature(c.get("gate_temperature", 0.5))
    model.eval()

    cm_in = np.zeros((4, 4))
    cm_out = np.zeros((4, 4))
    with torch.no_grad():
        for i, (post, pre, nbr, y) in enumerate(
                DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)):
            post = post.to(dev).to(memory_format=torch.channels_last)
            pre = pre.to(dev) if pre.numel() else pre
            nbr = nbr.to(dev) if nbr.numel() else nbr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(post, pre, nbr)
            p = lo.argmax(1)[0].cpu().numpy()
            t = y[0].numpy()
            stem = split["test"][i]
            mask = inside.get(stem)
            valid = t < 4
            if mask is not None:
                a, b = valid & mask, valid & ~mask
            else:
                a, b = np.zeros_like(valid), valid
            if a.any():
                cm_in += T.confusion(p[a], t[a], 4)
            if b.any():
                cm_out += T.confusion(p[b], t[b], 4)
    del model
    torch.cuda.empty_cache()

    tag = os.path.basename(run)
    ii, mi, oi, ni = metrics(cm_in)
    io, mo, oo, no = metrics(cm_out)
    all_cm = cm_in + cm_out
    ia, ma, oa, na = metrics(all_cm)
    print()
    print("=== %s ===" % tag)
    print("  whole grid       mIoU=%.4f  OA=%.4f  (%d px)  [recorded %.4f]"
          % (ma, oa, na, res["test"]["mIoU"]))
    print("  inside perimeter mIoU=%.4f  OA=%.4f  (%d px, %.1f%%)"
          % (mi, oi, ni, 100.0 * ni / max(na, 1)))
    print("  outside          mIoU=%.4f  OA=%.4f  (%d px, %.1f%%)"
          % (mo, oo, no, 100.0 * no / max(na, 1)))
    print("  per-class IoU inside : %s" % {NAMES[k]: round(float(ii[k]), 4) for k in range(4)})
    print("  per-class IoU outside: %s" % {NAMES[k]: round(float(io[k]), 4) for k in range(4)})
    out[tag] = {"whole": ma, "inside": mi, "outside": mo,
                "px_inside": ni, "px_outside": no,
                "iou_inside": ii.tolist(), "iou_outside": io.tolist()}

p = os.path.join(ROOT, "results", "inside_perimeter.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print()
print("written %s" % p)
