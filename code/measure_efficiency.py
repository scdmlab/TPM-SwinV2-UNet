"""Complexity metrics on one machine (Referee 1.9).

The referee asks for Params, FLOPs, FPS, latency, memory and training time
"on the same hardware". The numbers currently in summary_tables.md fail that:
the baselines were timed on the 3090 and most TPSM variants on the 5090, so the
21x FPS gap they appear to show is partly just the two cards.

Everything here is measured in one process on one device, with warm-up and
repeated timing, so the comparison is about the models.
"""
import argparse
import importlib.util
import json
import os
import time

import torch

ROOT = r"D:\wildfire project\revision_2026"

VARIANTS = [
    ("A01_SwinV2", "tpsm"), ("A02_SwinV2UNet", "tpsm"), ("A03_Bi", "tpsm"),
    ("A04_T", "tpsm"), ("A05_TP", "tpsm"), ("A06_TS", "tpsm"),
    ("A07_TPS", "tpsm"), ("A08_TM", "tpsm"), ("A09_TPM", "tpsm"),
    ("A10_TSM", "tpsm"), ("A11_TPSM", "tpsm"),
    ("A04_T", "unet"), ("A04_T", "deeplabv3plus"), ("A04_T", "segformer"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default=os.path.join(ROOT, "scripts", "tpsm_train.py"))
    ap.add_argument("--code-root",
                    default=os.path.join(ROOT, "code_base",
                                         "wildfire-TPSM-SwinV2-UNet-main"))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "efficiency.json"))
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("tpsm_train", args.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    print(f"device: {gpu}\n")

    O = T.load_original(args.code_root)
    rows = []

    for variant, arch in VARIANTS:
        cfg = T.variant_cfg(variant)
        name = variant if arch == "tpsm" else f"baseline_{arch}"
        try:
            if arch == "tpsm":
                model = T.SwinUNetV2Flex(
                    O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"],
                    scse=cfg["scse"], mmoe=cfg["mmoe"], experts=cfg["experts"],
                    topk=cfg["topk"], gate_hint=cfg["gate_hint"],
                    num_classes=4).to(dev)
                streams = cfg["streams"]
            else:
                model = T.SMPBaseline(arch, streams=3, num_classes=4).to(dev)
                streams = 3
            model.eval().to(memory_format=torch.channels_last)
        except Exception as e:
            print(f"{name}: build failed ({type(e).__name__})")
            continue

        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        post = torch.randn(1, 9, 256, 256, device=dev).to(memory_format=torch.channels_last)
        pre = torch.randn(1, 9, 256, 256, device=dev) if streams >= 2 else torch.zeros(0, device=dev)
        nbr = torch.randn(1, 3, 256, 256, device=dev) if streams >= 3 else torch.zeros(0, device=dev)

        flops = None
        try:
            from thop import profile
            m2 = model
            flops, _ = profile(m2, inputs=(post, pre, nbr), verbose=False)
            flops = flops / 1e9
        except Exception:
            pass

        with torch.no_grad():
            for _ in range(5):
                model(post, pre, nbr)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                model(post, pre, nbr)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.reps

        mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
        row = {"model": name, "params_M": round(params, 2),
               "gflops": round(flops, 1) if flops else None,
               "latency_ms": round(dt * 1000, 2),
               "fps": round(1.0 / dt, 1),
               "peak_mem_MiB": round(mem, 0) if mem else None,
               "gpu": gpu}
        rows.append(row)
        print(f"  {name:<24} params={row['params_M']:>7.2f}M  "
              f"GFLOPs={str(row['gflops']):>8}  {row['latency_ms']:>7.2f} ms  "
              f"{row['fps']:>6.1f} FPS  {str(row['peak_mem_MiB']):>7} MiB")

        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"device": gpu, "input": "1x(9+9+3)x256x256",
                   "reps": args.reps, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
