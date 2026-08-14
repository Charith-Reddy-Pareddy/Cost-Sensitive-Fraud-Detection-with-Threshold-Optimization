"""Train and evaluate baseline models (Logistic Regression, XGBoost) with MLflow tracking.

These are plain baselines with no special imbalance handling — Day 3 compares SMOTE vs.
class-weighting on top of this. The point here is an honest look at what these two model
families do out of the box under severe class imbalance, before any cost-sensitivity is
layered in on Day 4.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def build_xgboost_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    eval_metric="aucpr",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "f1_at_0.5": f1_score(y_test, preds),
    }


def train_and_log(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> Pipeline:
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_train_fraud", int(y_train.sum()))

        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ARTIFACTS_DIR / f"{name}.joblib")

        print(f"{name}: {metrics}")
    return pipeline


def main() -> None:
    mlflow.set_experiment("fraud-detection-baselines")

    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")

    train_and_log(
        "logistic_regression", build_logistic_regression_pipeline(), X_train, y_train, X_test, y_test
    )
    train_and_log("xgboost", build_xgboost_pipeline(), X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
