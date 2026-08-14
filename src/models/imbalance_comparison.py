"""Compare imbalance-handling strategies on the XGBoost baseline: none, class-weighting, and
SMOTE. SMOTE is deliberately treated as one experimental comparison point, not the default —
synthetic minority oversampling can produce artifacts that don't generalize to a real
distribution shift, whereas cost-sensitive learning (Day 4) is the more ML-systems-appropriate
approach for this problem. Resampling always happens inside the pipeline, fit per training
fold only, so it can never leak into the held-out test set.
"""

from pathlib import Path

import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor
from src.mlflow_utils import use_local_tracking_store

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def build_pipeline(strategy: str) -> ImbPipeline:
    """strategy: 'none' | 'class_weight' | 'smote'"""
    if strategy not in {"none", "class_weight", "smote"}:
        raise ValueError(f"unknown strategy: {strategy}")

    steps = [("preprocess", build_preprocessor())]

    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=0)))
        classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1)
    elif strategy == "class_weight":
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="aucpr",
            n_jobs=-1,
            scale_pos_weight=None,  # set at fit time based on training data, see train()
        )
    else:
        classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1)

    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


def evaluate(pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "f1_at_0.5": f1_score(y_test, preds),
    }


def train_strategy(strategy: str, X_train: pd.DataFrame, y_train: pd.Series, X_test, y_test) -> dict[str, float]:
    pipeline = build_pipeline(strategy)

    if strategy == "class_weight":
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)

    pipeline.fit(X_train, y_train)
    return evaluate(pipeline, X_test, y_test)


def main() -> None:
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-imbalance-comparison")

    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")

    for strategy in ("none", "class_weight", "smote"):
        with mlflow.start_run(run_name=f"xgboost_{strategy}"):
            mlflow.log_param("imbalance_strategy", strategy)
            metrics = train_strategy(strategy, X_train, y_train, X_test, y_test)
            mlflow.log_metrics(metrics)
            print(f"{strategy}: {metrics}")


if __name__ == "__main__":
    main()
