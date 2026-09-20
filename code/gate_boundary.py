"""Does the MMoE gate route differently at class boundaries? (R1-4 + mechanism)

Two results currently sit side by side without a link between them:

  §15/§16  the module gain is entirely a boundary phenomenon -- boundary errors
           halve (169.7k -> 80.9k) while interior errors do not move
  §gate_viz  the four class gates do spread over the five experts rather than
           collapsing, which is what the referee asked to see

If the routing is what produces the boundary repair, the gate should behave
measurably differently on boundary pixels than on interior ones. If it does
not, the two findings are independent and the boundary repair has to be
attributed to PPM's multi-scale context rather than to expert routing -- which
would itself be worth knowing, since §17.2 already found the MTBS gain comes
entirely from PPM with MMoE contributing zero.

Either answer is publishable; this script is written to be able to return
"no difference" and say so.

Inference only -- no retraining. Boundary distance is computed on the label
with the same routine as `error_maps.py`, so bands here mean what they mean
in §16.
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

NAMES = ["Unburned", "Low", "Moderate", "Severe"]
# Bands in pixels from the nearest label boundary. The 0-2 / >8 contrast is the
# comparison of interest; the middle bands show whether any effect decays with
# distance rather than being a step at the edge.
BANDS = [(0, 0, "on edge"), (1, 2, "1-2 px"), (3, 8, "3-8 px"), (9, 99, ">8 px")]


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("tpsm_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boundary_distance(lbl, max_r=8):
    """Distance to the nearest different-class pixel, capped at max_r + 1.

    Copied from error_maps.py rather than imported so the two analyses cannot
    drift apart silently; also avoids a scipy dependency the servers lack.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default="scripts/tpsm_train.py")
    ap.add_argument("--variant", default="A09_TPM")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--code-root",
                    default="code_base/wildfire-TPSM-SwinV2-UNet-main")
    ap.add_argument("--split", default=None)
    ap.add_argument("--layout", choices=("manuscript", "ravg"),
                    default="manuscript")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--null", type=int, default=32,
                    help="rolled-null draws; costs no extra forward passes")
    ap.add_argument("--max-patches", type=int, default=0,
                    help="evenly subsample the test set (0 = all). The full "
                         "MTBS test set is 932 patches, which is hours of CPU "
                         "inference for a statistic that converges long before "
                         "that; subsample evenly rather than taking a prefix, "
                         "since stems are sorted by fire.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    T = load_trainer(args.trainer)
    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device == "cpu" else "cpu")
    O = T.load_original(args.code_root)
    cfg = T.variant_cfg(args.variant)
    if not cfg["mmoe"]:
        raise SystemExit("%s has no MMoE head" % args.variant)

    # Mirror the trainer's two layouts exactly: the MTBS/RAVG trees carry their
    # own fire-level split as directories, the manuscript tree needs split.json.
    if args.layout == "ravg":
        import glob as _glob
        test_root = os.path.join(args.data_root, "Testing")
        stems = sorted(os.path.basename(p) for p in
                       _glob.glob(os.path.join(test_root, "Label", "*.tif")))
    else:
        with open(args.split, encoding="utf-8") as f:
            stems = json.load(f)["test"]
        test_root = os.path.join(args.data_root, "Testing", "Testing")
    if args.max_patches and len(stems) > args.max_patches:
        step = len(stems) / args.max_patches
        stems = [stems[int(i * step)] for i in range(args.max_patches)]
        print("subsampled to %d patches (every ~%.1f)" % (len(stems), step))
    ds = T.PatchDataset(test_root, stems, cfg["streams"], False, True,
                        cfg.get("index_source", "nbr"), None, args.layout)

    chans = ds.channels() if hasattr(ds, "channels") else None
    kw = {"chans": chans} if chans else {}
    model = T.SwinUNetV2Flex(
        O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"],
        scse=cfg["scse"], mmoe=cfg["mmoe"], experts=cfg["experts"],
        topk=cfg["topk"], gate_hint=cfg["gate_hint"], num_classes=4,
        **kw).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    if "gate_temperature" in ck and hasattr(model, "set_gate_temperature"):
        model.set_gate_temperature(float(ck["gate_temperature"]))
    model.eval()

    E, K = cfg["experts"], cfg["topk"]
    nb = len(BANDS)
    # Null model. Gate maps are spatially smooth, so ANY spatially structured
    # pixel set would show some contrast against its complement -- a raw spread
    # above 1 proves nothing on its own. Rolling the distance map on a torus
    # keeps the exact geometry and size of each band but decouples it from the
    # real label boundaries, which is the specific thing under test.
    rng = np.random.default_rng(0)
    n_null = args.null
    mass_null = np.zeros((n_null, nb, 4, E), dtype=np.float64)
    npx_null = np.zeros((n_null, nb), dtype=np.float64)
    # Summed gate mass and pixel counts per band, so the per-band mean is a
    # true pixel-weighted mean rather than an average of per-patch averages
    # (patches differ a lot in how much boundary they contain).
    mass = np.zeros((nb, 4, E), dtype=np.float64)      # band, class gate, expert
    sel = np.zeros((nb, 4, E), dtype=np.float64)       # top-k selection counts
    ent = np.zeros((nb, 4), dtype=np.float64)          # summed routing entropy
    npx = np.zeros(nb, dtype=np.float64)
    n = 0

    with torch.no_grad():
        for i in range(len(ds)):
            post, pre, nbr, y = ds[i]
            post = post.unsqueeze(0).to(dev).to(memory_format=torch.channels_last)
            pre = pre.unsqueeze(0).to(dev) if pre.numel() else pre
            nbr = nbr.unsqueeze(0).to(dev) if nbr.numel() else nbr
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(post, pre, nbr, return_aux=True)
            else:
                out = model(post, pre, nbr, return_aux=True)
            if not (isinstance(out, tuple) and out[1] and "gate_probs" in out[1]):
                raise SystemExit("model did not return gate_probs")
            g = out[1]["gate_probs"].float()[0]                 # [C, E, h, w]

            lbl = y.numpy()
            if g.shape[-2:] != lbl.shape:
                g = torch.nn.functional.interpolate(
                    g.unsqueeze(0), size=lbl.shape, mode="nearest")[0]
            g = g.cpu().numpy()

            dist = boundary_distance(lbl)
            valid = lbl != 255
            # Entropy of the routing distribution, normalised so 1.0 is uniform
            # over experts and 0.0 is a single expert.
            p = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-12)
            h_pix = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(E)  # [C, H, W]
            # Which experts top-k actually picks, which is what runs at
            # inference -- soft mass can be spread while selection is not.
            order = np.argsort(-g, axis=1)[:, :K]                     # [C, K, H, W]
            picked = np.zeros_like(g, dtype=bool)
            for k in range(K):
                np.put_along_axis(picked, order[:, k:k + 1], True, axis=1)

            for b, (lo, hi, _) in enumerate(BANDS):
                m = valid & (dist >= lo) & (dist <= hi)
                c = int(m.sum())
                if not c:
                    continue
                npx[b] += c
                mass[b] += g[:, :, m].sum(axis=2)
                sel[b] += picked[:, :, m].sum(axis=2)
                ent[b] += h_pix[:, m].sum(axis=1)

            for r in range(n_null):
                dy, dx = rng.integers(0, lbl.shape[0]), rng.integers(0, lbl.shape[1])
                rolled = np.roll(dist, (int(dy), int(dx)), axis=(0, 1))
                for b, (lo, hi, _) in enumerate(BANDS):
                    m = valid & (rolled >= lo) & (rolled <= hi)
                    c = int(m.sum())
                    if not c:
                        continue
                    npx_null[r, b] += c
                    mass_null[r, b] += g[:, :, m].sum(axis=2)
            n += 1

    mean_mass = mass / np.maximum(npx[:, None, None], 1)
    mean_sel = sel / np.maximum(npx[:, None, None], 1)
    mean_ent = ent / np.maximum(npx[:, None], 1)

    print("patches=%d  experts=%d  top-k=%d  ckpt=%s\n"
          % (n, E, K, os.path.basename(os.path.dirname(args.ckpt))))
    print("pixels per band")
    for b, (_, _, name) in enumerate(BANDS):
        print("  %-8s %10d  (%.1f%%)" % (name, npx[b], 100 * npx[b] / npx.sum()))

    print("\nmean gate mass, summed over the four class gates")
    print("  %-8s %s" % ("band", " ".join("   e%d" % j for j in range(E))))
    for b, (_, _, name) in enumerate(BANDS):
        tot = mean_mass[b].sum(axis=0) / 4
        print("  %-8s %s" % (name, " ".join("%5.3f" % v for v in tot)))

    print("\nboundary specialisation: mass(on edge + 1-2 px) / mass(>8 px)")
    near = (mass[0] + mass[1]).sum(axis=0) / max(npx[0] + npx[1], 1)
    far = mass[3].sum(axis=0) / max(npx[3], 1)
    ratios = near / np.maximum(far, 1e-12)
    for j in range(E):
        bar = "+" * int(round(abs(ratios[j] - 1) * 40))
        print("  e%d  %.3f  %s" % (j, ratios[j], bar))
    spread = float(ratios.max() - ratios.min())
    print("  spread across experts: %.3f" % spread)

    # Expert index is arbitrary -- seeds permute it freely -- so the quantity
    # that can be compared across runs is the sorted profile, not e0..e4.
    print("\n  sorted ratio profile (seed-invariant): %s"
          % " ".join("%.3f" % v for v in np.sort(ratios)[::-1]))

    null_spread = np.empty(n_null)
    for r in range(n_null):
        nn = (mass_null[r, 0] + mass_null[r, 1]).sum(axis=0) / \
            max(npx_null[r, 0] + npx_null[r, 1], 1)
        nf = mass_null[r, 3].sum(axis=0) / max(npx_null[r, 3], 1)
        rr = nn / np.maximum(nf, 1e-12)
        null_spread[r] = rr.max() - rr.min()
    p = (1 + int((null_spread >= spread).sum())) / (n_null + 1)
    print("\n  null (band geometry rolled off the real boundaries, %d draws)"
          % n_null)
    print("    null spread %.3f +- %.3f   range %.3f-%.3f"
          % (null_spread.mean(), null_spread.std(),
             null_spread.min(), null_spread.max()))
    # Raw spread is not comparable across datasets -- the null level itself
    # differs several-fold with test-set size and how smooth the gate maps are.
    # Excess over the null in null-sigma units is.
    z = (spread - null_spread.mean()) / max(null_spread.std(), 1e-12)
    print("    observed %.3f   p = %.3f   excess = %.1f null-sigma" % (spread, p, z))

    print("\ntop-k selection frequency (share of pixels where the expert is picked)")
    print("  %-8s %s" % ("band", " ".join("   e%d" % j for j in range(E))))
    for b, (_, _, name) in enumerate(BANDS):
        tot = mean_sel[b].sum(axis=0) / 4
        print("  %-8s %s" % (name, " ".join("%5.3f" % v for v in tot)))

    print("\nrouting entropy (1.0 = uniform over experts, 0.0 = one expert)")
    print("  %-8s %s   mean" % ("band", " ".join("%9s" % c for c in NAMES)))
    for b, (_, _, name) in enumerate(BANDS):
        print("  %-8s %s   %.3f"
              % (name, " ".join("%9.3f" % v for v in mean_ent[b]),
                 mean_ent[b].mean()))

    d_ent = mean_ent[3].mean() - (ent[0] + ent[1]).mean() / max(npx[0] + npx[1], 1)
    print("\n  entropy(>8 px) - entropy(near boundary) = %+.4f" % d_ent)

    verdict = ("routing IS boundary-dependent (beyond the rolled null, p=%.3f)" % p
               if p <= 0.05 else
               "routing contrast is within what band geometry alone produces "
               "(p=%.3f) -- no boundary specialisation demonstrated" % p)
    print("\n  => %s" % verdict)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "ckpt": args.ckpt, "patches": n,
                   "experts": E, "topk": K,
                   "bands": [b[2] for b in BANDS],
                   "band_pixels": npx.tolist(),
                   "mean_gate_mass": mean_mass.tolist(),
                   "mean_selection": mean_sel.tolist(),
                   "mean_entropy": mean_ent.tolist(),
                   "boundary_ratio_per_expert": ratios.tolist(),
                   "sorted_ratio_profile": np.sort(ratios)[::-1].tolist(),
                   "ratio_spread": spread,
                   "null_spread_mean": float(null_spread.mean()),
                   "null_spread_std": float(null_spread.std()),
                   "null_spread_max": float(null_spread.max()),
                   "excess_null_sigma": float(z),
                   "p_value": p,
                   "entropy_far_minus_near": float(d_ent),
                   "verdict": verdict}, f, indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
