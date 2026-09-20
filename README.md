# TPM-SwinV2-UNet

Code, fixed split metadata, reported metrics, and publication figures for **“Enhancing multi-class wildfire burn severity mapping via transformer-based class-aware mixture-of-experts learning on bitemporal imagery.”**

TPM-SwinV2-UNet combines three aligned Sentinel-2 streams (pre-fire reflectance, post-fire reflectance, and an NBR-family change composite), a SwinV2-UNet backbone, PPM-Lite multi-scale context, and a class-gated mixture-of-experts prediction head. The reported main configuration uses five shared experts and top-3 class-specific routing.

## Reported main result

On the event-disjoint Mosquito Fire test set, the three-run TPM estimate is:

| Metric | Value |
|---|---:|
| mIoU | 0.9049 ± 0.0042 |
| OA | 0.9564 |
| Cohen's kappa | 0.9124 |
| QWK | 0.9733 |
| OMAE | 0.0442 |

Class-wise IoUs are 0.9460, 0.7755, 0.9278, and 0.9704 for Unburned, Low, Moderate, and Severe, respectively. The complete comparison is in [`results/cross_event_mosquito.csv`](results/cross_event_mosquito.csv).

The independent CBI field-reference results are in [`results/cbi_regression.csv`](results/cbi_regression.csv). Under training from scratch, TPM obtains fire-averaged Macro RMSE 0.5970 ± 0.0296 and Macro MAE 0.5387 ± 0.0058 on the same 28 held-out locations.

## Repository contents

- `code/`: the unified trainer, model implementation, conventional baselines, evaluation utilities, and archived CBI experiment scripts.
- `data/`: the fixed segmentation filename split and a portable CBI location/split manifest.
- `results/`: the paper's quantitative tables in machine-readable form, plus the CBI classical-baseline predictions.
- `figures/`: publication figures corresponding to the released results.

## Environment

The archived neural runs used Python 3.10–3.12 and PyTorch 2.8.0+cu128 or 2.11.0+cu128. Install a CUDA-compatible PyTorch build first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Segmentation training

The trainer takes an explicit data root, code root, split manifest, output directory, model variant, and seed. For the final top-3 TPM configuration:

```bash
python code/tpsm_train.py \
  --variant K3_TPM_top3 \
  --seed 1 \
  --data-root /path/to/data \
  --code-root code \
  --split data/segmentation_split.json \
  --out runs/K3_TPM_top3_seed1
```

Use `A04_T`, `A05_TP`, and `K3_TM_top3` for the tri-stream backbone, PPM-Lite-only, and top-3 MMoE-only counterparts. The trainer selects checkpoints on the held-out validation split and evaluates the test event after training.

## Data availability

The repository contains the exact split metadata and reported results, rather than duplicating large third-party Sentinel-2 rasters. Sentinel-2 imagery is available through the Copernicus Data Space Ecosystem. Field CBI observations are available from the USGS data release identified in [`data/README.md`](data/README.md). Reconstruct the directory layout described there before running the archived scripts.

## Result provenance

All CSV values are transcribed from the final manuscript tables. `results/cbi_classical_baselines.json` preserves the test predictions and per-fire errors for the CBI classical baselines. The figure assets match the final manuscript versions released with this repository.

