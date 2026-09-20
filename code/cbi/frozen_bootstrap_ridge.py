"""Training-only bootstrap replicas for deterministic frozen-feature ridge adapters."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

threadpool_limits(2)
R = Path(__file__).resolve().parent
M = json.loads((R.parent / "cbi_adaptation_v3_20260914" / "manifest.json").read_text())
VARIANTS = ("A04_T", "A05_TP", "A08_TM", "A09_TPM")
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def metric(rows):
    by_fire = {}
    for event in sorted({row["event"] for row in rows}):
        error = np.asarray([row["prediction"] - row["cbi"] for row in rows if row["event"] == event])
        by_fire[event] = {
            "n": len(error),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    return {
        "by_fire": by_fire,
        "macro": {key: float(np.mean([value[key] for value in by_fire.values()])) for key in ("rmse", "mae", "bias")},
    }


def feature_path(variant):
    local = R / "frozen" / f"{variant}_features.npz"
    remote = R / "remote_snapshot" / "frozen" / f"{variant}_features.npz"
    path = local if local.exists() else remote
    assert path.exists(), variant
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["status"] == "complete" and metadata["sha256"] == sha(path)
    return path, metadata


def layout(variant):
    path, metadata = feature_path(variant)
    raw = np.load(path)
    X, y, ids = raw["X"], raw["y"], raw["ids"]
    lookup = {str(identifier): i for i, identifier in enumerate(ids)}
    index = {
        split: np.asarray([lookup[point["event"] + "::" + point["id"]] for point in M[split]], dtype=int)
        for split in ("train", "val", "test")
    }
    assert len(X) == len(y) == len(ids) == sum(len(M[split]) for split in ("train", "val", "test"))
    assert len(set(index["train"]).intersection(index["val"])) == 0
    assert len(set(index["train"]).intersection(index["test"])) == 0
    return X, y, index, metadata


def stratified_bootstrap(index, seed):
    rng = np.random.default_rng(10000 + seed)
    groups = {}
    for point, row in zip(M["train"], index["train"]):
        groups.setdefault(point["event"], []).append(int(row))
    drawn = []
    for event in sorted(groups):
        group = np.asarray(groups[event], dtype=int)
        drawn.extend(rng.choice(group, size=len(group), replace=True).tolist())
    drawn = np.asarray(drawn, dtype=int)
    assert len(drawn) == len(index["train"]) and set(drawn).issubset(set(index["train"]))
    return drawn


def rows(prediction, split):
    return [
        {"id": point["id"], "event": point["event"], "cbi": point["cbi"], "prediction": float(prediction[i])}
        for i, point in enumerate(M[split])
    ]


def run_one(variant, seed, persist=True):
    X, y, index, metadata = layout(variant)
    train_index = stratified_bootstrap(index, seed)
    components = min(32, X.shape[1], len(train_index))

    def fit(alpha, labels=y):
        model = make_pipeline(StandardScaler(), PCA(n_components=components, svd_solver="full"), Ridge(alpha=alpha))
        model.fit(X[train_index], labels[train_index])
        return model

    changed = y.copy()
    changed[index["val"]] = 3.0
    changed[index["test"]] = 0.0
    assert np.allclose(fit(1.0).predict(X), fit(1.0, changed).predict(X), rtol=0, atol=1e-12)
    selection = []
    for alpha in ALPHAS:
        validation_rows = rows(np.clip(fit(alpha).predict(X[index["val"]]), 0, 3), "val")
        selection.append({"alpha": alpha, "validation": metric(validation_rows)})
    chosen = min(selection, key=lambda item: item["validation"]["macro"]["rmse"])
    test_rows = rows(np.clip(fit(chosen["alpha"]).predict(X[index["test"]]), 0, 3), "test")
    result = {
        "status": "complete",
        "variant": variant,
        "replicate_type": "stratified_training_location_bootstrap",
        "seed": seed,
        "train_samples": len(train_index),
        "unique_train_locations": int(len(set(train_index))),
        "selected": chosen,
        "pca_components": components,
        "heldout_label_invariance": True,
        "feature_meta": metadata,
        "test": metric(test_rows),
        "predictions": test_rows,
    }
    assert np.isfinite(result["test"]["macro"]["rmse"])
    assert np.isfinite(result["test"]["macro"]["mae"])
    if persist:
        base = R / "frozen_bootstrap" / variant
        save(base / f"seed{seed}_selection.json", {"candidates": selection, "selected": chosen, "pca_components": components})
        save(base / f"seed{seed}_result.json", result)
    return result


def summarize():
    output = {}
    for variant in VARIANTS:
        results = [json.loads((R / "frozen_bootstrap" / variant / f"seed{seed}_result.json").read_text()) for seed in (1, 2, 3)]
        output[variant] = {
            "replicate_type": "stratified_training_location_bootstrap",
            "seeds": [1, 2, 3],
            "rmse_mean": float(np.mean([result["test"]["macro"]["rmse"] for result in results])),
            "rmse_sd": float(np.std([result["test"]["macro"]["rmse"] for result in results], ddof=1)),
            "mae_mean": float(np.mean([result["test"]["macro"]["mae"] for result in results])),
            "mae_sd": float(np.std([result["test"]["macro"]["mae"] for result in results], ddof=1)),
        }
    save(R / "frozen_bootstrap" / "summary.json", output)
    return output


parser = ArgumentParser()
parser.add_argument("--sanity", action="store_true")
args = parser.parse_args()
if args.sanity:
    result = run_one("A04_T", 1, persist=False)
    save(R / "frozen_bootstrap" / "sanity.json", {
        "status": "passed", "variant": "A04_T", "seed": 1,
        "replicate_type": result["replicate_type"], "train_samples": result["train_samples"],
        "unique_train_locations": result["unique_train_locations"],
        "heldout_label_invariance": result["heldout_label_invariance"],
        "macro": result["test"]["macro"],
    })
    print("FROZEN_BOOTSTRAP_SANITY_PASSED")
else:
    for variant in VARIANTS:
        for seed in (1, 2, 3):
            run_one(variant, seed)
    print(json.dumps(summarize(), indent=2))
