"""Expert-routing evidence for the MMoE head (Referee 1.4).

The referee asks for gate-weight or expert-usage visualisations showing that
class-aware routing actually does something, rather than collapsing to one
expert or spreading uniformly.

The head already returns per-class gate probabilities of shape [B, C, E, H, W],
so nothing needs retraining: run the held-out event through a trained
checkpoint and summarise how the four class gates distribute their mass over
the five experts.
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

NAMES = ["Unburned", "Low", "Moderate", "Severe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default="code/tpsm_train.py")
    ap.add_argument("--variant", default="K3_TPM_top3")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--code-root", default="code")
    ap.add_argument("--split", default="data/segmentation_split.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("tpsm_train", args.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    O = T.load_original(args.code_root)
    cfg = T.variant_cfg(args.variant)
    if not cfg["mmoe"]:
        raise SystemExit(f"{args.variant} has no MMoE head")

    with open(args.split, encoding="utf-8") as f:
        stems = json.load(f)["test"]
    test_root = os.path.join(args.data_root, "Testing", "Testing")
    try:
        ds = T.PatchDataset(test_root, stems, cfg["streams"], False, True,
                            cfg.get("index_source", "nbr"))
    except TypeError:
        ds = T.PatchDataset(test_root, stems, cfg["streams"], False, True)

    chans = ds.channels() if hasattr(ds, "channels") else None
    kw = {"chans": chans} if chans else {}
    model = T.SwinUNetV2Flex(
        O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"],
        scse=cfg["scse"], mmoe=cfg["mmoe"], experts=cfg["experts"],
        topk=cfg["topk"], gate_hint=cfg["gate_hint"], num_classes=4,
        **kw).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    if "gate_temperature" in ck:
        model.set_gate_temperature(float(ck["gate_temperature"]))
    model.eval()

    E = cfg["experts"]
    # Mass each class gate puts on each expert, and the same conditioned on the
    # true class, which is what shows routing is class-aware rather than global.
    total = np.zeros((4, E), dtype=np.float64)
    by_true = np.zeros((4, 4, E), dtype=np.float64)
    true_px = np.zeros(4, dtype=np.float64)
    n = 0

    with torch.no_grad():
        for i in range(len(ds)):
            post, pre, nbr, y = ds[i]
            post = post.unsqueeze(0).to(dev).to(memory_format=torch.channels_last)
            pre = pre.unsqueeze(0).to(dev) if pre.numel() else pre
            nbr = nbr.unsqueeze(0).to(dev) if nbr.numel() else nbr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(post, pre, nbr, return_aux=True)
            if not (isinstance(out, tuple) and out[1] and "gate_probs" in out[1]):
                raise SystemExit("model did not return gate_probs")
            g = out[1]["gate_probs"].float()[0].cpu().numpy()   # [C, E, H, W]
            total += g.reshape(4, E, -1).mean(axis=2)
            yy = y.numpy()
            for c in range(4):
                m = yy == c
                if m.any():
                    by_true[c] += g[:, :, m].mean(axis=2)
                    true_px[c] += int(m.sum())
            n += 1

    total /= max(n, 1)
    for c in range(4):
        if true_px[c] > 0:
            by_true[c] /= max(n, 1)

    print(f"patches: {n}   experts: {E}\n")
    print("mean gate mass per class gate (rows) over experts (cols)")
    hdr = "  " + " ".join(f"  e{j}" for j in range(E))
    print(f"  {'gate':<10}{hdr}")
    for c in range(4):
        row = " ".join(f"{v:5.3f}" for v in total[c])
        top = int(np.argmax(total[c]))
        print(f"  {NAMES[c]:<10}  {row}   -> peak e{top}")

    # Concentration: 1.0 means a single expert, 1/E means uniform.
    print("\nrouting concentration (max share, and entropy ratio)")
    for c in range(4):
        p = total[c] / total[c].sum()
        ent = -(p * np.log(p + 1e-12)).sum() / np.log(E)
        print(f"  {NAMES[c]:<10} max={p.max():.3f}  entropy/max={ent:.3f}")

    print("\nper-class gate mass conditioned on the true class")
    for c in range(4):
        if true_px[c] == 0:
            continue
        row = " ".join(f"{v:5.3f}" for v in by_true[c][c])
        print(f"  true={NAMES[c]:<10} own gate: {row}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "ckpt": args.ckpt, "experts": E,
                   "patches": n, "gate_mass": total.tolist(),
                   "gate_mass_by_true_class": by_true.tolist(),
                   "true_pixels": true_px.tolist()}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
