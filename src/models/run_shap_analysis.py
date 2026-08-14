"""Compute SHAP values for the class-weighted XGBoost model on a sample of the test set,
identify the top drivers among the anonymized PCA components, and save a summary plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_utils import use_local_tracking_store
from src.models.imbalance_comparison import build_pipeline
from src.models.interpretability import compute_shap_values, mean_abs_shap_by_feature

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
SHAP_SAMPLE_SIZE = 2000


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-interpretability")

    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")

    pipeline = build_pipeline("class_weight")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)

    # oversample fraud rows into the SHAP sample so the explanation isn't dominated entirely
    # by legitimate-transaction rows given the severe class imbalance
    fraud_rows = X_test[y_test == 1]
    legit_sample = X_test[y_test == 0].sample(
        n=min(SHAP_SAMPLE_SIZE - len(fraud_rows), (y_test == 0).sum()), random_state=0
    )
    X_sample = pd.concat([fraud_rows, legit_sample]).sort_index()

    with mlflow.start_run(run_name="shap_interpretability"):
        mlflow.log_param("shap_sample_size", len(X_sample))

        shap_values, X_transformed = compute_shap_values(pipeline, X_sample)
        importance = mean_abs_shap_by_feature(shap_values, list(X_transformed.columns))

        for feature, value in importance.head(10).items():
            mlflow.log_metric(f"mean_abs_shap_{feature}", value)

        plt.figure(figsize=(7, 5))
        importance.head(10).sort_values().plot(kind="barh")
        plt.xlabel("mean |SHAP value|")
        plt.title("Top 10 features driving fraud predictions")
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        shap_plot_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(shap_plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(shap_plot_path))

        print("Top 10 features by mean |SHAP value|:")
        print(importance.head(10).to_string())


if __name__ == "__main__":
    main()
