# CBI experiment scripts

These files are the archived scripts used for the supplementary CBI regression evaluation:

- `train.py`: full-network fine-tuning and training-from-scratch workflows.
- `frozen_bootstrap_ridge.py`: frozen feature extraction followed by PCA and ridge regression.
- `classical.py`: mean/median, spectral-index ridge, SVR, random forest, and XGBoost controls.
- `train_topk.py`: top-k routing variant used by the final model configuration.
- `prepare.py` and `prepare_spectral.py`: field-window and spectral-feature preparation.
- `post_gpu.py`, `report.py`, and `audit_table7_provenance.py`: aggregation and provenance checks.

The scripts retain the execution-time relative directory layout for provenance. Before rerunning them, place the CBI imagery arrays, auxiliary source windows, and source-trained checkpoints in the locations expected by the scripts, and run the documented sanity checks before full training.

