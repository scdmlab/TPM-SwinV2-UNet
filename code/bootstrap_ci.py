# -*- coding: utf-8 -*-
"""Patch-level bootstrap confidence intervals (Referee 1.7).

The manuscript reports single numbers on a 36-patch test set. Referee 1.7 asks
for spread. Seed-to-seed spread is one source and is already reported; the
other, larger one is which patches happen to be in the test set, and that is
what a patch-level bootstrap measures.

Predictions come from each run's saved best.pth, so nothing is retrained. The
paired difference is resampled on the same patch draw for both arms, which is
the comparison that matters: it asks whether TPM beats T on this test set, not
whether two independently drawn test sets would rank them the same way.
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpsm_train as TT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--treatment", required=True)
ap.add_argument("--data-root", default=r"D:\former_files\former_data (2)\former_data")
ap.add_argument("--split", default=os.path.join(os.path.dirname(HERE), "results", "split.json"))
ap.add_argument("--code-root", default=os.path.join(os.path.dirname(HERE), "code_base",
                                                    "wildfire-TPSM-SwinV2-UNet-main"))
ap.add_argument("--boot", type=int, default=5000)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
O = TT.load_original(args.code_root)
split = json.load(open(args.split, encoding="utf-8"))
test_root = os.path.join(args.data_root, "Testing", "Testing")


def predict(rd):
    rj = json.load(open(os.path.join(rd, "result.json")))
    cfg = TT.variant_cfg(rj["variant"])
    ds = TT.PatchDataset(test_root, split["test"], cfg["streams"], False, True,
                         cfg.get("index_source", "nbr"), None)
    m = TT.SwinUNetV2Flex(O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"],
                          scse=cfg["scse"], mmoe=cfg["mmoe"], experts=cfg["experts"],
                          topk=cfg["topk"], gate_hint=cfg["gate_hint"], num_classes=4,
                          chans=ds.channels(), hint_mode=cfg["hint_mode"],
                          shared_gate=cfg["shared_gate"]).to(dev)
    ck = torch.load(os.path.join(rd, "best.pth"), map_location=dev)
    m.load_state_dict(ck["state_dict"])
    if hasattr(m, "set_gate_temperature"):
        m.set_gate_temperature(ck.get("gate_temperature", 0.5))
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 2):
            xs = [ds[j] for j in range(i, min(i + 2, len(ds)))]
            b = [torch.stack([torch.as_tensor(x[k]) for x in xs]).to(dev) for k in range(3)]
            lo = m(*b)
            if isinstance(lo, (tuple, list)):
                lo = lo[0]
            out.append(lo.argmax(1).cpu().numpy())
    return np.concatenate(out)


truth = []
for stem in split["test"]:
    with rasterio.open(os.path.join(test_root, "Mask", stem)) as d:
        truth.append(d.read(1))
truth = np.stack(truth)
valid = truth < 4

pa = predict(args.baseline)
pb = predict(args.treatment)
n = len(truth)

# per-patch confusion matrices, so a bootstrap draw is a sum of matrices
def per_patch_cm(pred):
    cms = np.zeros((n, 4, 4), dtype=np.int64)
    for i in range(n):
        t, p, v = truth[i], pred[i], valid[i]
        np.add.at(cms[i], (t[v].astype(int), p[v].astype(int)), 1)
    return cms


def miou(cm):
    tp = np.diag(cm).astype(float)
    d = tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.divide(tp, d, out=np.zeros(4), where=d > 0).mean())


ca, cb = per_patch_cm(pa), per_patch_cm(pb)
obs_a, obs_b = miou(ca.sum(0)), miou(cb.sum(0))
rng = np.random.default_rng(2024)
da = np.empty(args.boot); db = np.empty(args.boot)
for k in range(args.boot):
    idx = rng.integers(0, n, n)
    da[k] = miou(ca[idx].sum(0))
    db[k] = miou(cb[idx].sum(0))
diff = db - da


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


print("test patches: %d, bootstrap draws: %d" % (n, args.boot))
print("%-28s %8s   %s" % ("run", "mIoU", "95% CI (patch bootstrap)"))
for tag, obs, v in ((os.path.basename(args.baseline), obs_a, da),
                    (os.path.basename(args.treatment), obs_b, db)):
    lo, hi = ci(v)
    print("%-28s %8.4f   [%.4f, %.4f]" % (tag, obs, lo, hi))
lo, hi = ci(diff)
print("%-28s %+8.4f   [%+.4f, %+.4f]   P(diff>0) = %.3f"
      % ("paired difference", obs_b - obs_a, lo, hi, float((diff > 0).mean())))
