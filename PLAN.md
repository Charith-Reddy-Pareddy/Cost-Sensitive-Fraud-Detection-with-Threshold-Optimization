# 7-Day Build Plan

Research question: which modeling approach holds up best under severe class imbalance, and how does
optimizing the decision threshold against a cost function (rather than accuracy) change the fraud-detection /
false-positive tradeoff as the cost ratio changes?

- [x] **Day 1 — Scaffold, data, EDA.** Repo/GitHub setup, data ingestion, chronological train/test split
      (no leakage), compact EDA notebook.
- [x] **Day 2 — Baselines.** Feature pipeline, Logistic Regression baseline, RandomForest/XGBoost baseline,
      MLflow tracking wired in.
- [x] **Day 3 — Autoencoder + imbalance handling.** Unsupervised autoencoder (reconstruction error), honest
      writeup of why it over/underperforms, SMOTE vs. class-weighting comparison.
- [x] **Day 4 — Cost-sensitive decision engine.** Illustrative FN/FP costs, threshold optimizer, cost-vs-
      threshold curve, cost-ratio sensitivity sweep.
- [x] **Day 5 — Evaluation + interpretability.** Precision/recall/F1, PR-AUC/ROC-AUC, confusion matrix,
      precision@K, SHAP over the PCA components, main README assembled.
- [x] **Day 6 — Inference service + streaming stretch.** FastAPI service with p50/p95 latency logging,
      Docker, Kafka/Redpanda → Redis sliding-window feature stretch goal (only kept if it fully works).
- [x] **Day 7 — Controlled debugging exercise.** Three defects intentionally introduced and documented,
      reproduced via failing tests, diagnosed, fixed, regression-tested, full suite green.

## Known limitations (stated upfront)

- Single-day sample limits generalizability to model concept drift over time.
- Heavy PCA anonymization removes real-world feature semantics.
- Risk of temporal leakage if train/test splits aren't done chronologically — mitigated by splitting on
  `Time` rather than randomly.
- Illustrative fraud costs used in the cost engine are assumptions for demonstrating the method, not sourced
  figures.
