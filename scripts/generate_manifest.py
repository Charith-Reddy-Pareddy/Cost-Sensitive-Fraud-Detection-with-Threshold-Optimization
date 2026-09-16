"""Generate `results/manifest.json`: a single record of exactly what produced the numbers in this
report, so a claim like "PR-AUC 0.758" can be traced back to a specific dataset checksum, split
size, package version set, seed, and commit — not just trusted on faith.

This does not re-run any experiment. It fingerprints the current state of the repo, the raw data,
the processed splits, the trained model artifacts, and the installed package versions, and writes
that fingerprint to disk. Re-run it any time after touching data, code, or dependencies to keep
the manifest current; a stale manifest (wrong commit SHA, wrong checksum) is a signal something
changed without being re-verified, not just an inconvenience.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRACKED_PACKAGES = [
    "pandas", "numpy", "scikit-learn", "xgboost", "imbalanced-learn", "torch", "shap", "mlflow", "fastapi",
]

FIXED_SEEDS = {
    "note": "This project does not use one global seed; every place randomness matters pins its own.",
    "bootstrap_ci_default_seed": 0,
    "sklearn_random_state (SMOTE, BalancedRandomForest)": 0,
    "numpy_default_rng_seed (synthetic sweeps, cluster-effect tests)": 0,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return len(pd.read_parquet(path))


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def main() -> None:
    manifest = {
        "generated_at_commit": _git_commit_sha(),
        "working_tree_dirty_when_generated": _git_is_dirty(),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "seeds": FIXED_SEEDS,
        "raw_data": {
            "primary": _file_record(ROOT / "data" / "raw" / "creditcard.csv"),
            "sparkov_train": _file_record(ROOT / "data" / "raw" / "sparkov" / "fraudTrain.csv"),
            "sparkov_test": _file_record(ROOT / "data" / "raw" / "sparkov" / "fraudTest.csv"),
        },
        "processed_splits": {
            "primary": {
                split: _parquet_row_count(ROOT / "data" / "processed" / f"{split}.parquet")
                for split in ("train", "val", "test")
            },
            "sparkov": {
                split: _parquet_row_count(ROOT / "data" / "processed" / "sparkov" / f"{split}.parquet")
                for split in ("train", "val", "test")
            },
        },
        "model_artifacts": {
            name: _file_record(ROOT / "models" / "artifacts" / f"{name}.joblib")
            for name in ("xgboost", "logistic_regression", "production_pipeline", "sparkov_production_pipeline")
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    print(f"commit: {manifest['generated_at_commit']} (dirty: {manifest['working_tree_dirty_when_generated']})")


if __name__ == "__main__":
    main()
