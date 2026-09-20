# -*- coding: utf-8 -*-
r"""The non-deep baselines, on any dataset built by the new-style builder.

The existing ml_baselines.py is bound to the original release, whose layout puts
the labels in Mask/ and the index stream in NBR_ALL/NBR_train. Every dataset this
project builds itself uses Image/NBR and Label instead, so the referee's method
comparison has never been run on any of them.

This runs the same three non-deep baselines on any root of either shape, with the
hyperparameters copied verbatim from ml_baselines.py so the numbers stay
comparable with the ones already reported on the original release:

    zero-parameter threshold   digitize the delivered index at the three cut
                               points, no fitting of any kind
    random forest              300 trees, min_samples_leaf 2, balanced subsample,
                               150,000 pixels per class
    support vector machine     RBF, C = 10, gamma scale, standardised,
                               12,500 pixels per class

Features are the same 21 per-pixel channels the tri-stream network sees: nine
post-fire bands, nine pre-fire bands, three index channels, each scaled by its
storage dtype. Metrics come from the trainer module, so the confusion matrix is
aggregated over the whole test split before the per-class IoU is averaged --
the same convention as every trained result in this project.
"""
import argparse
import glob
import importlib.util
import json
import os
import time

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CUTS = [0.164, 0.400, 0.800]


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("tpsm_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rd(path):
    with rasterio.open(path) as ds:
        a = ds.read()
    if a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    elif a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def dirs_for(root, split):
    """Return (post, pre, index, label) directories for either layout."""
    base = os.path.join(root, split)
    post = os.path.join(base, "Image", "Post")
    pre = os.path.join(base, "Image", "Pre")
    idx = os.path.join(base, "Image", "NBR")
    lab = os.path.join(base, "Label")
    if not os.path.isdir(idx):                      # original release shape
        inner = os.path.join(base, split)
        post = os.path.join(inner, "Image", "Post")
        pre = os.path.join(inner, "Image", "Pre")
        for cand in ("NBR_ALL/NBR_train", "NBR_test"):
            c = os.path.join(inner, *cand.split("/"))
            if os.path.isdir(c):
                idx = c
                break
        lab = os.path.join(inner, "Mask")
    return post, pre, idx, lab


def stems_of(root, split):
    post, _, _, _ = dirs_for(root, split)
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(post, "*.tif")))


STREAM_SETS = {
    "all": ("post", "pre", "index"),
    "post": ("post",),
    "pre": ("pre",),
    "index": ("index",),
    "prepost": ("post", "pre"),
    "postindex": ("post", "index"),
}


def read_patch(root, split, stem, streams=("post", "pre", "index")):
    post, pre, idx, lab = dirs_for(root, split)
    src = {"post": post, "pre": pre, "index": idx}
    x = np.concatenate([rd(os.path.join(src[s], stem)) for s in streams], axis=0)
    with rasterio.open(os.path.join(lab, stem)) as ds:
        y = ds.read(1).astype(np.int64)
    return x.reshape(x.shape[0], -1).T, y.ravel()


def gather(root, split, stems, per_class=None, seed=0,
           streams=("post", "pre", "index")):
    xs, ys = [], []
    for s in stems:
        x, y = read_patch(root, split, s, streams)
        xs.append(x)
        ys.append(y)
    X, Y = np.concatenate(xs), np.concatenate(ys)
    if per_class is None:
        return X, Y
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(4):
        i = np.flatnonzero(Y == c)
        if i.size > per_class:
            i = rng.choice(i, per_class, replace=False)
        keep.append(i)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], Y[keep]


def write_result(out_dir, rid, name, m, extra, streams="all"):
    d = os.path.join(out_dir, rid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"variant": "baseline_%s" % name, "arch": name,
                   "seed": extra.get("seed", 1), "loss": "none", "classes": 4,
                   "config": dict(extra, streams=streams), "params_M": 0.0, "best_val_epoch": 0,
                   "best_val_mIoU": None,
                   "train_minutes": extra.get("train_minutes", 0.0),
                   "test": m, "epochs": 0, "batch_size": 0, "gpu": "CPU"},
                  f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--trainer", required=True, help="path to tpsm_train.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="prefix for the run ids")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rf-per-class", type=int, default=150_000)
    ap.add_argument("--svm-per-class", type=int, default=12_500)
    ap.add_argument("--skip", default="", help="comma list: threshold,rf,svm")
    ap.add_argument("--cuts", nargs=3, type=float, default=None,
                    help="the three cut points this dataset's labels use. "
                         "Read from its label_meta.json when present, else the "
                         "released dataset's 0.164 0.400 0.800. Scoring a "
                         "dataset with cut points it was not built with "
                         "understates the untrained rule and flatters "
                         "everything measured against it.")
    ap.add_argument("--streams", default="all", choices=sorted(STREAM_SETS),
                    help="which input streams the tree and kernel baselines see; "
                         "'pre' asks whether the labels are predictable from "
                         "imagery taken before the fire")
    args = ap.parse_args()

    # cut points: explicit, else the dataset's own record, else the published
    # values. Which of the three applied is printed, so a table can be audited.
    cuts, cuts_src = list(CUTS), "released dataset default"
    # a dataset records its cuts in label_meta.json when it was relabelled and
    # in replay_meta.json when it came out of the residual replay; both are
    # checked, because a degraded dataset has only the second
    meta_p = None
    for _cand in ("label_meta.json", "replay_meta.json"):
        _p = os.path.join(args.data_root, _cand)
        if os.path.exists(_p):
            meta_p = _p
            break
    meta_p = meta_p or os.path.join(args.data_root, "label_meta.json")
    if args.cuts:
        cuts, cuts_src = list(args.cuts), "given on the command line"
    elif os.path.exists(meta_p):
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                mj = json.load(f)
            if isinstance(mj.get("cuts"), list) and len(mj["cuts"]) == 3:
                cuts, cuts_src = [float(x) for x in mj["cuts"]], os.path.basename(meta_p)
        except (ValueError, OSError):
            pass
    if sorted(cuts) != cuts:
        raise SystemExit("cut points must increase: %r" % cuts)
    print("cut points %s  (%s)" % ([round(c, 4) for c in cuts], cuts_src))

    T = load_trainer(args.trainer)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    tr_stems = stems_of(args.data_root, "Training")
    te_stems = stems_of(args.data_root, "Testing")
    print("dataset %s" % args.data_root)
    IGNORE = 255
    print("  train patches %d, test patches %d" % (len(tr_stems), len(te_stems)))

    Xte, Yte = gather(args.data_root, "Testing", te_stems, streams=STREAM_SETS[args.streams])
    # Pixels marked 255 are outside the reference perimeter, where severity is
    # undefined. Every trained run excludes them via --ignore-index; the
    # baselines must exclude them too, or they are scored on a class that does
    # not exist and fitted on one they should never see.
    keep_te = Yte != IGNORE
    if keep_te.sum() != Yte.size:
        print("  dropping %d of %d test pixels marked %d (%.1f%%)"
              % (Yte.size - keep_te.sum(), Yte.size, IGNORE,
                 100.0 * (1 - keep_te.mean())))
        Xte, Yte = Xte[keep_te], Yte[keep_te]
    print("  test pixels %d, features %d" % (Xte.shape[0], Xte.shape[1]))
    print("  test class counts %s" % np.bincount(Yte, minlength=4).tolist())

    # ---- zero-parameter threshold ---------------------------------------
    if "threshold" not in skip:
        print("\nzero-parameter threshold on the delivered index ...")
        post_dir, pre_dir, idx_dir, _ = dirs_for(args.data_root, "Testing")

        def _count(d):
            with rasterio.open(os.path.join(d, te_stems[0])) as ds:
                return ds.count

        n_post, n_pre, n_idx = _count(post_dir), _count(pre_dir), _count(idx_dir)
        base = n_post + n_pre
        # The composite is not the same everywhere: three channels on the
        # released data with the differenced index first, five on the per-fire
        # data with it elsewhere. Score each and keep the best, so the rule is
        # never handicapped by being pointed at the wrong channel.
        with rasterio.open(os.path.join(idx_dir, te_stems[0])) as ds:
            dt = ds.dtypes[0]
        scale = 65535.0 if dt == "uint16" else (255.0 if dt == "uint8" else 1.0)

        best = None
        for c in range(n_idx):
            col = base + c
            if col >= Xte.shape[1]:
                break
            d = Xte[:, col] * scale
            mm = T.metrics_from_cm(T.confusion(np.digitize(d, cuts), Yte, 4))
            print("    index channel %d: mIoU=%.4f" % (c, mm["mIoU"]))
            if best is None or mm["mIoU"] > best[0]["mIoU"]:
                best = (mm, c)
        m, chan = best
        print("  best index channel %d of %d" % (chan, n_idx))
        print("  mIoU=%.4f OA=%.4f QWK=%.4f" % (m["mIoU"], m["OA"], m["qwk"]))
        print("  per-class IoU %s" % [round(v, 4) for v in m["iou"]])
        write_result(args.out, "%s_threshold" % args.tag, "threshold", m,
                     {"seed": args.seed, "cuts": cuts, "cuts_source": cuts_src,
                      "index_channel": chan, "index_channels": n_idx,
                      "bands_post_pre_index": [n_post, n_pre, n_idx],
                      "note": "no fitting; digitize the best index channel"},
                     args.streams)

    # ---- random forest ---------------------------------------------------
    if "rf" not in skip:
        print("\nrandom forest, %d px/class ..." % args.rf_per_class)
        Xtr, Ytr = gather(args.data_root, "Training", tr_stems,
                          args.rf_per_class, args.seed,
                          streams=STREAM_SETS[args.streams])
        k = Ytr != IGNORE
        if k.sum() != Ytr.size:
            print("  dropping %d of %d training pixels marked %d"
                  % (Ytr.size - k.sum(), Ytr.size, IGNORE))
            Xtr, Ytr = Xtr[k], Ytr[k]
        print("  %d samples, class counts %s"
              % (Xtr.shape[0], np.bincount(Ytr, minlength=4).tolist()))
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    n_jobs=-1, random_state=args.seed,
                                    class_weight="balanced_subsample")
        rf.fit(Xtr, Ytr)
        mins = (time.time() - t0) / 60
        pred = np.empty(Yte.shape, np.int64)
        for i in range(0, Xte.shape[0], 500_000):
            pred[i:i + 500_000] = rf.predict(Xte[i:i + 500_000])
        m = T.metrics_from_cm(T.confusion(pred, Yte, 4))
        print("  fitted in %.1f min  mIoU=%.4f OA=%.4f" % (mins, m["mIoU"], m["OA"]))
        print("  per-class IoU %s" % [round(v, 4) for v in m["iou"]])
        write_result(args.out, "%s_randomforest" % args.tag, "randomforest", m,
                     {"seed": args.seed, "train_minutes": mins,
                      "features": int(Xtr.shape[1]),
                      "note": "300 trees, balanced_subsample"}, args.streams)

    # ---- support vector machine -----------------------------------------
    if "svm" not in skip:
        print("\nsupport vector machine, %d px/class ..." % args.svm_per_class)
        Xs, Ys = gather(args.data_root, "Training", tr_stems,
                        args.svm_per_class, args.seed,
                        streams=STREAM_SETS[args.streams])
        k = Ys != IGNORE
        if k.sum() != Ys.size:
            print("  dropping %d of %d training pixels marked %d"
                  % (Ys.size - k.sum(), Ys.size, IGNORE))
            Xs, Ys = Xs[k], Ys[k]
        t0 = time.time()
        svm = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", C=10.0, gamma="scale",
                                class_weight="balanced", cache_size=1000,
                                random_state=args.seed))
        svm.fit(Xs, Ys)
        mins = (time.time() - t0) / 60
        pred = np.empty(Yte.shape, np.int64)
        for i in range(0, Xte.shape[0], 500_000):
            pred[i:i + 500_000] = svm.predict(Xte[i:i + 500_000])
        m = T.metrics_from_cm(T.confusion(pred, Yte, 4))
        print("  fitted in %.1f min  mIoU=%.4f OA=%.4f" % (mins, m["mIoU"], m["OA"]))
        print("  per-class IoU %s" % [round(v, 4) for v in m["iou"]])
        write_result(args.out, "%s_svm" % args.tag, "svm", m,
                     {"seed": args.seed, "train_minutes": mins,
                      "features": int(Xs.shape[1]),
                      "note": "RBF C=10 gamma=scale, standardised"}, args.streams)


if __name__ == "__main__":
    main()
