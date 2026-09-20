# -*- coding: utf-8 -*-
r"""Score each index channel as a rule in its own right.

The table lists three untrained rules -- dNBR, RdNBR, dNBR2 -- and the existing
threshold baseline reports only one number, the best channel, without recording
which channel that was. So the three rows cannot be filled from what is on
disk.

Each is the same rule applied to a different channel of the composite the
network receives: digitise at the three cut points, no fitting of any kind.
Scoring them separately says which part of the composite carries the labels,
which is the point of listing three rules rather than one.

Pixels marked 255 are excluded, as everywhere else in this project; on the
released dataset there are none, but the per-fire datasets are half of them.

Each channel is written as its own run so it joins the results tree on the same
footing as everything else.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import rasterio

CUTS = [0.164, 0.400, 0.800]
IGNORE = 255
NAMES = ["dNBR rule", "RdNBR rule", "dNBR2 rule",
         "index channel 3 rule", "index channel 4 rule"]


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("tpsm_train", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dirs_for(root, split):
    base = os.path.join(root, split)
    idx = os.path.join(base, "Image", "NBR")
    lab = os.path.join(base, "Label")
    if not os.path.isdir(idx):
        inner = os.path.join(base, split)
        for cand in ("NBR_ALL/NBR_train", "NBR_test"):
            c = os.path.join(inner, *cand.split("/"))
            if os.path.isdir(c):
                idx = c
                break
        lab = os.path.join(inner, "Mask")
    return idx, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cuts", nargs=3, type=float, default=None)
    a = ap.parse_args()

    T = load_trainer(a.trainer)
    cuts = list(a.cuts) if a.cuts else list(CUTS)
    for cand in ("label_meta.json", "replay_meta.json"):
        p = os.path.join(a.data_root, cand)
        if a.cuts is None and os.path.exists(p):
            try:
                j = json.load(open(p, "r", encoding="utf-8"))
                if isinstance(j.get("cuts"), list) and len(j["cuts"]) == 3:
                    cuts = [float(x) for x in j["cuts"]]
                    print("cuts %s from %s" % ([round(c, 4) for c in cuts], cand))
                    break
            except (ValueError, OSError):
                pass
    else:
        if a.cuts is None:
            print("cuts %s (released dataset default)" % cuts)

    idx_d, lab_d = dirs_for(a.data_root, "Testing")
    stems = sorted(os.path.basename(p) for p in glob.glob(os.path.join(lab_d, "*.tif")))
    if not stems:
        raise SystemExit("no test labels under %s" % lab_d)

    chans = None
    preds, ys = None, []
    for s in stems:
        ip = os.path.join(idx_d, s)
        if not os.path.exists(ip):
            cand = [f for f in os.listdir(idx_d) if f.startswith(s[:-4])]
            if not cand:
                continue
            ip = os.path.join(idx_d, cand[0])
        with rasterio.open(ip) as f:
            d = f.read().astype(np.float32)
        with rasterio.open(os.path.join(lab_d, s)) as f:
            y = f.read(1)
        m = y != IGNORE
        if chans is None:
            chans = d.shape[0]
            preds = [[] for _ in range(chans)]
        for c in range(chans):
            preds[c].append(np.digitize(d[c][m], cuts))
        ys.append(y[m])

    Y = np.concatenate(ys)
    print("test pixels %d, index channels %d" % (Y.size, chans))
    print()
    for c in range(chans):
        P = np.concatenate(preds[c])
        met = T.metrics_from_cm(T.confusion(P, Y, 4))
        name = NAMES[c] if c < len(NAMES) else "index channel %d rule" % c
        print("  %-22s mIoU %.4f  OA %.4f  kappa %.4f  QWK %.4f  per-class %s"
              % (name, met["mIoU"], met["OA"], met.get("kappa") or float("nan"),
                 met.get("qwk") or float("nan"), [round(v, 4) for v in met["iou"]]))
        rid = "%s_rule_ch%d" % (a.tag, c)
        d = os.path.join(a.out, rid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": "baseline_%s" % name.replace(" ", "_"),
                       "arch": "threshold", "seed": 1, "loss": "none",
                       "classes": 4,
                       "config": {"cuts": cuts, "index_channel": c,
                                  "rule": name,
                                  "note": "no fitting; digitize one index channel"},
                       "params_M": 0.0, "test": met, "epochs": 0,
                       "batch_size": 0, "gpu": "CPU",
                       "data_root": a.data_root}, f, indent=2)


if __name__ == "__main__":
    main()
