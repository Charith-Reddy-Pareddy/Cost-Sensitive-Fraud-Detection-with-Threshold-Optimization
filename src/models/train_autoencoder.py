"""Train the autoencoder on legitimate-only transactions and score the test set by
reconstruction error, logged to MLflow for a direct comparison against the Day 2 baselines."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor
from src.mlflow_utils import use_local_tracking_store
from src.models.autoencoder import fit_autoencoder, reconstruction_error

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-autoencoder")

    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    legit_mask = (y_train == 0).to_numpy()
    X_train_legit = X_train_scaled[legit_mask]

    with mlflow.start_run(run_name="autoencoder"):
        epochs = 20
        mlflow.log_param("model", "autoencoder")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_train_legit", len(X_train_legit))
        mlflow.log_param("bottleneck_dim", 8)

        model = fit_autoencoder(np.asarray(X_train_legit, dtype=np.float32), epochs=epochs)

        fraud_scores = reconstruction_error(model, np.asarray(X_test_scaled, dtype=np.float32))
        pr_auc = average_precision_score(y_test, fraud_scores)
        roc_auc = roc_auc_score(y_test, fraud_scores)

        # For reference, the error a fraud row gets relative to a legit row — a value near
        # 1.0 means reconstruction error does not separate the classes at all.
        fraud_mean_error = fraud_scores[y_test == 1].mean()
        legit_mean_error = fraud_scores[y_test == 0].mean()

        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "fraud_mean_reconstruction_error": float(fraud_mean_error),
                "legit_mean_reconstruction_error": float(legit_mean_error),
            }
        )

        print(f"autoencoder: pr_auc={pr_auc:.4f} roc_auc={roc_auc:.4f}")
        print(f"mean reconstruction error — fraud: {fraud_mean_error:.4f}, legit: {legit_mean_error:.4f}")


if __name__ == "__main__":
    main()
