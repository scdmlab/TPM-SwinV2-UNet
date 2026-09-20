"""Where the errors sit on the ground, not just how many there are (R1-6).

Section 15 established WHAT the confusions are (Low-Moderate before the
modules, Unburned-Low after). Referee 1.6 also wants to see WHERE. Two claims
in particular need a spatial check rather than a tabular one:

  - that the residual Unburned->Low errors sit on the fire perimeter, where
    section 8.2 found the first dNBR cut point to be least stable
  - that the modules remove interior Low/Moderate confusion rather than simply
    trading one error for another

Produces, per model, a PNG per test patch plus a summary: the fraction of
errors falling within N pixels of a class boundary in the reference mask.
Boundary distance is computed on the LABEL, so it does not depend on the
model being evaluated and the same denominator applies to every model.

    python scripts/error_maps.py --runs runs/A_A09_TPM_s6 runs/A_A04_T_s1
"""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DATA = r"D:\former_files\former_data (2)\former_data"
DEFAULT_CODE = os.path.join(ROOT, "code_base", "wildfire-TPSM-SwinV2-UNet-main")
DEFAULT_SPLIT = os.path.join(ROOT, "results", "split.json")

# Unburned, Low, Moderate, Severe -- sequential, so severity reads as intensity.
PALETTE = np.array([[237, 237, 237], [253, 208, 122], [232, 119, 34], [150, 33, 27]],
                   dtype=np.uint8)


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "tpsm_train", os.path.join(HERE, "tpsm_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boundary_distance(lbl, max_r=8):
    """For each pixel, how far (in pixels, capped) to the nearest different class.

    Plain dilation-style search: cheap on 256x256 and avoids a scipy dependency
    the servers do not have.
    """
    h, w = lbl.shape
    dist = np.full((h, w), max_r + 1, dtype=np.int16)
    diff = np.zeros((h, w), dtype=bool)
    diff[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    diff[:, 1:] |= lbl[:, :-1] != lbl[:, 1:]
    diff[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    diff[1:, :] |= lbl[:-1, :] != lbl[1:, :]
    dist[diff] = 0
    cur = diff.copy()
    for r in range(1, max_r + 1):
        grown = np.zeros_like(cur)
        grown[:, :-1] |= cur[:, 1:]
        grown[:, 1:] |= cur[:, :-1]
        grown[:-1, :] |= cur[1:, :]
        grown[1:, :] |= cur[:-1, :]
        new = grown & (dist > r)
        dist[new] = r
        cur = grown
    return dist


def build(T, O, run_dir, res, in_chans, classes, device):
    cfg = res["config"]
    if res.get("arch", "tpsm") == "tpsm":
        m = T.SwinUNetV2Flex(
            O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"],
            scse=cfg["scse"], mmoe=cfg["mmoe"], experts=cfg["experts"],
            topk=cfg["topk"], gate_hint=cfg["gate_hint"],
            num_classes=classes, chans=in_chans).to(device)
    else:
        m = T.SMPBaseline(res["arch"], streams=cfg["streams"],
                          num_classes=classes, chans=in_chans).to(device)
    m.to(memory_format=torch.channels_last)
    ck = torch.load(os.path.join(run_dir, "best.pth"), map_location=device)
    m.load_state_dict(ck["state_dict"])
    if hasattr(m, "set_gate_temperature"):
        m.set_gate_temperature(ck.get("gate_temperature", 0.5))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--code-root", default=DEFAULT_CODE)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "error_maps"))
    ap.add_argument("--max-png", type=int, default=8,
                    help="write this many patch figures per model; stats use all")
    args = ap.parse_args()

    dirs = []
    for p in args.runs:
        dirs.extend(sorted(glob.glob(p)))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "best.pth"))]
    if not dirs:
        raise SystemExit("no runs with best.pth")

    # Import matplotlib BEFORE the trainer. tpsm_train stubs any module it
    # cannot import, and a stub imports cleanly -- so importing it afterwards
    # yields a fake whose plt.subplots() returns None. Same failure shape as
    # the suppressed python3 error in operation log 36: absence disguised as
    # availability.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        can_plot = hasattr(plt, "subplots") and plt.subplots.__module__.startswith("matplotlib")
        if not can_plot:
            print("matplotlib resolved to a stub; writing statistics only")
    except Exception as e:
        print("matplotlib unavailable (%s); writing statistics only" % e)
        can_plot = False

    T = load_trainer()
    O = T.load_original(args.code_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)

    summary = {}
    for run_dir in dirs:
        tag = os.path.basename(run_dir)
        with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
            res = json.load(f)
        classes = res.get("classes", 4)
        if classes != 4:
            print("%s is not a 4-class run, skipping" % tag)
            continue

        ds = T.PatchDataset(os.path.join(args.data_root, "Testing", "Testing"),
                            split["test"], res["config"]["streams"], False, False,
                            res["config"].get("index_source", "nbr"), None)
        model = build(T, O, run_dir, res, ds.channels(), classes, device)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        near = far = 0
        per_class_near = np.zeros(4)
        per_class_err = np.zeros(4)
        made = 0
        with torch.no_grad():
            for i, (post, pre, nbr, y) in enumerate(dl):
                post = post.to(device).to(memory_format=torch.channels_last)
                pre = pre.to(device) if pre.numel() else pre
                nbr = nbr.to(device) if nbr.numel() else nbr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(post, pre, nbr)
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.int16)
                lbl = y[0].numpy().astype(np.int16)
                valid = lbl < 4
                wrong = (pred != lbl) & valid

                d = boundary_distance(lbl)
                near += int((wrong & (d <= 2)).sum())
                far += int((wrong & (d > 2)).sum())
                for c in range(4):
                    m = wrong & (lbl == c)
                    per_class_err[c] += int(m.sum())
                    per_class_near[c] += int((m & (d <= 2)).sum())

                if can_plot and made < args.max_png:
                    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))
                    ax[0].imshow(PALETTE[np.clip(lbl, 0, 3)])
                    ax[0].set_title("reference")
                    ax[1].imshow(PALETTE[np.clip(pred, 0, 3)])
                    ax[1].set_title("prediction")
                    over = PALETTE[np.clip(lbl, 0, 3)].copy()
                    over[wrong] = [0, 90, 200]
                    ax[2].imshow(over)
                    ax[2].set_title("errors (blue)")
                    for a in ax:
                        a.set_xticks([]); a.set_yticks([])
                    fig.suptitle("%s  patch %d" % (tag, i))
                    fig.tight_layout()
                    fig.savefig(os.path.join(args.outdir, "%s_p%02d.png" % (tag, i)), dpi=110)
                    plt.close(fig)
                    made += 1

        del model
        torch.cuda.empty_cache()

        tot = near + far
        summary[tag] = {
            "errors": tot,
            "near_boundary_frac": near / tot if tot else 0.0,
            "per_class_near_frac": [
                (per_class_near[c] / per_class_err[c]) if per_class_err[c] else 0.0
                for c in range(4)],
            "per_class_errors": per_class_err.tolist(),
        }
        print("%-22s errors=%9d   within 2 px of a label boundary: %.1f%%"
              % (tag, tot, 100 * summary[tag]["near_boundary_frac"]))
        print("     by true class  %s"
              % ["%.0f%%" % (100 * v) for v in summary[tag]["per_class_near_frac"]])

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print("written %s" % args.outdir)


if __name__ == "__main__":
    main()
