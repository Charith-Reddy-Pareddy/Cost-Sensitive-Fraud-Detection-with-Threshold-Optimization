"""Visual companions to the metrics already reported as numbers elsewhere: a confusion matrix
at the production threshold, an ROC curve, and a precision-recall curve — all computed on the
real production pipeline's predictions over the untouched test split.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.serving.model_loader import load_production_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    pipeline, metadata = load_production_pipeline()
    threshold = metadata["optimal_threshold"]

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # confusion matrix at the production threshold
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["legitimate", "fraud"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix @ threshold={threshold:.2f} (test set)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # precision-recall curve — the more informative one under this severe an imbalance
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = y_test.mean()
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    plt.axhline(baseline, linestyle="--", color="gray", label=f"chance (fraud rate = {baseline:.4f})")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=150)
    plt.close()

    print(f"threshold={threshold:.2f}")
    print(f"confusion matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print(f"ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  fraud_rate={baseline:.4f}")


if __name__ == "__main__":
    main()
