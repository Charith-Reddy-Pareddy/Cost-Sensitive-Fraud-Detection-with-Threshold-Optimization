# Cost-Sensitive Fraud Detection with Threshold Optimization

[![Tests](https://github.com/Charith-Reddy-Pareddy/Cost-Sensitive-Fraud-Detection-with-Threshold-Optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/Charith-Reddy-Pareddy/Cost-Sensitive-Fraud-Detection-with-Threshold-Optimization/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live dashboard →](https://charith-reddy-pareddy.github.io/Cost-Sensitive-Fraud-Detection-with-Threshold-Optimization/)**
— headline metrics, confusion matrix, ROC/PR curves, and every robustness finding in one page.

Fraud detection isn't an accuracy-maximization problem — missing a fraudulent transaction and
blocking a legitimate customer cost different amounts. This project optimizes the decision
threshold against a cost function instead of 0.5, then asks the harder question: **how robust is
that improvement**, once checked against a held-out validation split, across time windows, under
cost-ratio uncertainty, and on a second, structurally different dataset?

**Core research question:** which modeling approach performs best under severe class imbalance,
and how does optimizing the decision threshold against a cost function change the tradeoff
between fraud detection and false positives — including how robust that optimum is under
temporal shift, cost-ratio uncertainty, and across datasets, not just whether it beats 0.5 on
one test split?

**Full writeup with every experiment: [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md).** This README
is the short version.

| | |
|---|---|
| Best model | XGBoost (class-weighted) |
| Cost-optimal threshold (val-selected, $500/$5 illustrative costs) | 0.09 |
| Cost reduction vs. default (untouched test) | 2.4% (95% CI: −9.3% to 18.7%) |
| Beat default across 4 walk-forward windows | 2 of 4 |
| Replicates on a second dataset (Sparkov) | **No** |
| Inference latency | 1.42ms p50 / 2.14ms p95 |

That table is honest rather than flattering on purpose. Every technique here — XGBoost, class
weighting, SMOTE, autoencoders, threshold optimization, calibration, SHAP, bootstrap CIs — is
established, not novel; the contribution is the robustness checks, and most of them come back
negative or null. See [`PLAN.md`](PLAN.md) for the original 7-day build plan.

## Dataset

**Primary:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(Kaggle) — anonymized PCA-transformed transactions, 284,807 rows, 492 fraud (0.17%). Split
chronologically: train 68% / val 17% / test 15%.

**Secondary (external validation):** [Sparkov](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
— 881,739 rows, 4,989 fraud, with real merchant/category/geolocation fields.

**Limitations, stated upfront:** single day (no genuine multi-day drift — walk-forward only
tests intra-day windows); heavy PCA anonymization on the primary dataset; only 492 fraud rows
total (52 in test), so every interval below is wide; the $500/$5 costs are illustrative, not
sourced (see the cost-uncertainty sweep). Detailed EDA lives in
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Methodology

Earlier versions of this project selected the threshold using the test set's own labels, which
is how an inflated 13.3% cost reduction became the corrected 2.4%. Every script now follows one
rule: fit on **train**, select every threshold/calibrator on **validation**, touch **test**
exactly once, for final reporting. Full derivation: [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md).

## Architecture

```mermaid
flowchart TD
    A["Data"] --> B["Train / val / test split"]
    B --> C["LR / XGBoost / Autoencoder"]
    C --> D["Class weighting + cost function"]
    D --> E["Threshold optimization (val) + calibration (val)"]
    E --> F["Production pipeline: XGBoost @ 0.09"]
    E --> G["Walk-forward + cost-uncertainty checks"]
    F --> H["Sparkov external validation"]
    F -->|"POST /predict"| I["FastAPI service"]
```

Every training run logs to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Results

| Model | PR-AUC | ROC-AUC | F1 @ 0.5 |
|---|---|---|---|
| Logistic Regression | 0.647 | 0.966 | 0.581 |
| XGBoost (no weighting) | 0.757 | 0.984 | 0.813 |
| Autoencoder (unsupervised) | 0.273 | 0.947 | — |

Autoencoder trained on legitimate transactions only; fraud gets ~21x higher reconstruction error
but PR-AUC still collapses, because reconstruction-anomalous and fraudulent only partly overlap.

| Imbalance strategy | PR-AUC | F1 @ 0.5 | Cost-weighted training | Calibration |
|---|---|---|---|---|
| None | 0.757 | **0.813** | standard @ 0.5: **$7,510** | raw: $6,415 |
| Class weighting | 0.758 | 0.736 | cost-weighted @ 0.5: **$6,530** | Platt: $6,605 |
| SMOTE | **0.760** | 0.611 | cost-weighted + tuned: $7,070 | **isotonic: $6,235** |

PR-AUC barely moves across imbalance strategies (within noise) while F1@0.5 gets steadily worse
— ranking and default-threshold precision aren't the same thing. Cost-weighted training beats
both threshold-tuning alone *and* the two combined (training-time and decision-time
cost-sensitivity partially substitute for each other, not add). Isotonic calibration wins on
both Brier score and cost; Platt scaling improves Brier but still costs more. Full discussion:
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md#experiments).

### How the production model detects fraud

The production pipeline (class-weighted XGBoost @ threshold 0.09) on the real, untouched test
set (42,721 rows, 52 fraud) — accuracy alone is meaningless at this imbalance (99.8% accuracy is
achievable by predicting "not fraud" every time), so this is confusion matrix, ROC, and
precision-recall, not a single accuracy number
([`src/models/run_evaluation_plots.py`](src/models/run_evaluation_plots.py)):

<img src="reports/figures/confusion_matrix.png" width="360" alt="Confusion matrix">
<img src="reports/figures/roc_curve.png" width="420" alt="ROC curve">
<img src="reports/figures/pr_curve.png" width="420" alt="Precision-recall curve">

**40 of 52 fraud cases caught (recall 0.77), 83 false alarms out of 42,669 legitimate
transactions (precision 0.33), ROC-AUC 0.983, PR-AUC 0.758.** The PR curve's chance line
(fraud rate 0.0012) is the point of showing it at all: ROC-AUC looks uniformly excellent even
for a mediocre model here, because true negatives are so abundant they flood the false-positive
rate axis — PR-AUC is far more sensitive to what actually happens at realistic operating
thresholds, which is why it's the metric used for model selection throughout this project.

### Cost-sensitive threshold optimization

![Expected cost vs. decision threshold](reports/figures/cost_vs_threshold.png)
![Optimal threshold vs. cost ratio](reports/figures/cost_ratio_sensitivity.png)

Default (0.50) costs $6,575 on test; the val-selected threshold (0.09) costs $6,415 — a 2.4%
reduction. Sweeping the cost ratio instead of fixing it shows *why* one static threshold isn't
the point: optimal threshold falls from 0.87 (ratio=1) to 0.01 (ratio=500) as missing fraud gets
relatively more expensive, recall climbing from 0.71 to 0.81 as precision falls from 0.82 to
0.08. (Threshold 0.05 actually beats the val-selected 0.09 on test cost — the val→test
generalization gap, made visible on purpose.)

### Is any of this robust?

![Cost uncertainty threshold distribution](reports/figures/cost_uncertainty_threshold_distribution.png)

- **Walk-forward (4 time windows):** optimized threshold beat default in only 2/4 folds; the
  threshold itself swings from 0.04 to 0.62 across windows.
- **Cost-ratio uncertainty (500 draws, cost_fn~U(100,1000), cost_fp~U(1,20)):** median threshold
  0.09 matches the point estimate, but the range is [0.01, 0.75].
- **Bootstrap (1,000 resamples):** cost reduction 2.4%, 95% CI **[−9.3%, 18.7%]** — crosses zero.

Combined, the honest read: cost-sensitive threshold optimization has a positive expected effect
here but isn't a reliable win on a 492-fraud-row dataset. Full numbers:
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md#statistical-analysis).

### Interpretability

![Top 10 SHAP features](reports/figures/shap_summary.png)

`V4`, `V14`, `V12` are the strongest drivers (SHAP, `TreeExplainer`) — consistent with published
analyses of this dataset, despite the components having no semantic meaning on their own.

## External validation: does any of this generalize?

Same protocol and costs, applied to Sparkov
([`src/models/run_sparkov_validation.py`](src/models/run_sparkov_validation.py) and siblings):

| | Primary | Sparkov (no velocity feature) | Sparkov (+ per-card velocity feature) |
|---|---|---|---|
| Baseline PR-AUC | 0.757 | 0.909 | **0.969** |
| Class weighting | helps marginally | **hurts** (0.909→0.882) | — |
| Threshold optimization | +2.4% cost reduction | no effect (0%) | **−1.8%** (worse) |
| Cost-reduction 95% CI | [−9.3%, 18.7%] | [0.0%, 0.0%] | [−9.1%, 1.5%] |
| Walk-forward: beats default | 2/4 folds | 2/4 folds | **1/4 folds** |
| Cost-weighted training beats tuning | yes | — | **yes** |
| Cost-ratio-uncertainty median threshold | 0.09 | — | 0.60 (anchored, not swinging) |

**Neither primary-dataset conclusion replicates**, and the disagreement gets *stronger* with a
real per-card feature (`card_txn_count_24h`, causally windowed via Redis — added because
Sparkov's `cc_num` makes a genuine per-entity feature possible, unlike the primary dataset).
Likely reason: Sparkov's engineered features already separate classes almost perfectly, leaving
little room for cost-sensitive machinery to help. Every Sparkov robustness check agrees in
direction with itself even as it disagrees with the primary dataset. Full breakdown of all 4
repeated experiments: [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md#5-does-any-of-this-generalize-to-a-structurally-different-dataset).

The per-card feature isn't just computed offline — `src/serving/sparkov_app.py` is a standing
FastAPI service that looks it up live from Redis on every request (`docker compose up -d
sparkov-api redis`), verified to score identically inside and outside Docker.

## Inference service

`src/serving/app.py` serves the production pipeline at its cost-optimized threshold (0.09).

```bash
uvicorn src.serving.app:app --reload   # then see http://localhost:8000/docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"Time": 120000, "V1": -2.31, ...}'
# -> {"fraud_probability": 0.913, "is_fraud": true, "threshold": 0.09, "latency_ms": 1.6}
```

| Endpoint | |
|---|---|
| `GET /health` | liveness + threshold |
| `POST /predict` | score one transaction |
| `POST /replay?n=&delay_ms=` | replay real test rows |
| `GET /latency` | p50/p95 so far |

p50 **1.42ms**, p95 **2.14ms** from 5,000 real served predictions
(![histogram](reports/figures/latency_histogram.png)).

## Streaming prototype (primary dataset) — not model-connected

```mermaid
flowchart LR
    A["Producer"] -->|"Kafka"| B["Consumer"]
    B --> C["Redis sliding window"]
    B -->|"POST /predict"| D["FastAPI"]
    C -. "not wired into the model" .-> D
```

`docker compose up --build` runs a producer/consumer replaying transactions through a global
(not per-card — no entity ID on this dataset) Redis sliding window, scored against the live
model. The window is computed and logged, and **deliberately not fed into the model** — doing so
would mean retraining on a feature the static dataset doesn't have. Sparkov's version above
closes that gap for real, because `cc_num` makes a genuine per-entity feature possible there.

## Repository structure

```
├── RESEARCH_REPORT.md   full methodology + 5 experiments (start here for depth)
├── PLAN.md              original 7-day build plan
├── src/
│   ├── data/            ingestion + chronological split (primary and Sparkov)
│   ├── features/        shared preprocessing
│   ├── models/          training, cost engine, calibration, bootstrap, SHAP, robustness checks
│   ├── serving/         FastAPI services (primary + standing Sparkov)
│   └── streaming/       Kafka/Redpanda → Redis (per-card for Sparkov, global for primary)
├── tests/               pytest suite
├── scripts/             CI-only synthetic fixture generator (no real data needed)
├── .github/workflows/   CI: tests + Docker build/smoke-test on every push
├── docker-compose.yml
└── requirements.txt
```

## Reproducing this locally

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw
unzip data/raw/creditcardfraud.zip -d data/raw && rm data/raw/creditcardfraud.zip
python -m src.data.ingest

# primary-dataset experiments (each is self-contained, see src/models/)
python -m src.models.train_production_model   # then: uvicorn src.serving.app:app

# optional Sparkov external validation (~200MB download)
kaggle datasets download -d kartik2112/fraud-detection -p data/raw/sparkov
unzip data/raw/sparkov/fraud-detection.zip -d data/raw/sparkov
python -m src.data.ingest_sparkov
python -m src.models.train_sparkov_production_model   # then: docker compose up -d sparkov-api redis

pytest tests/ -q
```

Every individual experiment's script is linked inline where it's discussed in
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) — this is just the core reproducible flow.

**Environment note:** XGBoost and PyTorch each bundle their own OpenMP runtime — set
`OMP_NUM_THREADS=1` if running both in one process on macOS (already handled in
[`src/__init__.py`](src/__init__.py)). `xgboost` is pinned below 3.0 for SHAP compatibility.

## What I'd do with more data

Multi-day data to study real concept drift, and more fraud examples in the primary dataset — the
bootstrap CIs are wide because of 492 positive rows, not a weak method. Full list:
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md#future-work).

## License

[MIT](LICENSE)
