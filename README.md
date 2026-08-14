# Cost-Sensitive Fraud Detection with Threshold Optimization

A production-oriented fraud detection system that handles severe class imbalance and optimizes
fraud decisions against real business cost — not accuracy maximization. Three modeling
approaches are compared honestly (including failure modes), decision thresholds are optimized
against a financial loss function, and the result ships as a working low-latency inference
service rather than a notebook.

**Core research question:** which modeling approach performs best under severe class imbalance,
and how does optimizing the decision threshold against a cost function change the tradeoff
between fraud detection and false positives — including how that optimum shifts as the cost
ratio changes?

See [`PLAN.md`](PLAN.md) for the day-by-day build plan this repo followed.

## Dataset

**Primary:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(Kaggle) — anonymized (PCA-transformed) European card transactions, single day, 284,807 rows,
492 fraud (0.17%).

**Known limitations (stated upfront):**
- Single-day sample limits the ability to model concept drift over time.
- Heavy PCA anonymization (`V1`..`V28`) removes real-world feature semantics.
- Risk of temporal leakage if train/test splits aren't done chronologically — mitigated here by
  splitting on `Time` rather than randomly (see [`src/data/ingest.py`](src/data/ingest.py)).
- The fraud/blocked-transaction dollar costs used in the cost engine below are **illustrative
  assumptions** for demonstrating the method, not sourced figures.

Detailed EDA (class balance, amount/time distributions, correlation structure) lives in
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) rather than here, to keep this README focused
on architecture, cost curves, and latency.

## Architecture

```
data/raw/creditcard.csv
        │
        ▼
 src/data/ingest.py           chronological train/test split (no leakage)
        │
        ▼
 src/features/pipeline.py     shared preprocessing (scale Amount/Time, passthrough V1..V28)
        │
        ├──▶ src/models/train_baseline.py        Logistic Regression, XGBoost
        ├──▶ src/models/autoencoder.py            unsupervised reconstruction-error model
        ├──▶ src/models/imbalance_comparison.py   none vs. class-weight vs. SMOTE
        ├──▶ src/models/cost_engine.py            cost-sensitive threshold optimization
        └──▶ src/models/interpretability.py       SHAP over the PCA components
                    │
                    ▼
         src/serving/          FastAPI inference service (replay + latency logging)
```

Every training run logs params/metrics/artifacts to MLflow (project-local SQLite store,
`mlflow.db`, gitignored — run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse).

## Modeling: three approaches, compared honestly

| Model | PR-AUC | ROC-AUC | F1 @ 0.5 |
|---|---|---|---|
| Logistic Regression | 0.744 | 0.979 | 0.678 |
| XGBoost (no imbalance handling) | 0.671 | 0.976 | 0.756 |
| **Autoencoder** (reconstruction error, unsupervised) | **0.272** | 0.922 | — |

The autoencoder is trained only on legitimate transactions and never sees a labeled fraud
example. Fraud rows get ~18x higher mean reconstruction error than legitimate ones (4.70 vs.
0.25), which is why ROC-AUC still looks reasonable (0.922) — but PR-AUC collapses to 0.272. That
gap is the finding: reconstruction error assumes fraud is statistically anomalous, which is only
partly true here — some fraud reconstructs about as well as an ordinary transaction, and some
unusual-but-legitimate transactions reconstruct as badly as fraud does. Full writeup:
[`docs/day3_autoencoder_and_imbalance_analysis.md`](docs/day3_autoencoder_and_imbalance_analysis.md).

## Handling class imbalance

Resampling is applied **inside** the training pipeline only (never before the train/test split)
to avoid leakage. SMOTE is treated as one experimental comparison point, not the default.

| Strategy (XGBoost) | PR-AUC | ROC-AUC | F1 @ 0.5 |
|---|---|---|---|
| None | 0.671 | 0.976 | 0.756 |
| **Class weighting** | **0.786** | **0.986** | **0.765** |
| SMOTE | 0.777 | 0.977 | 0.667 |

Class-weighting wins outright, including F1 at the default threshold. SMOTE improves PR-AUC over
the unweighted baseline but costs real precision at the default threshold (F1 drops from 0.756 to
0.667) — synthetic minority samples shift the decision boundary in a way that doesn't fully
generalize. Class-weighting is the strategy carried forward into the cost-sensitive threshold
work below.

## Cost-sensitive decision engine

Illustrative costs: **$500** per missed fraud (false negative), **$5** per blocked legitimate
transaction (false positive) — assumptions for demonstrating the method, not sourced figures.

![Expected cost vs. decision threshold](reports/figures/cost_vs_threshold.png)

| | Threshold | Expected cost (test set) |
|---|---|---|
| Default | 0.50 | $9,085 |
| **Cost-optimized** | **0.08** | **$7,880** |

**13.3% reduction in expected financial loss** versus the default 0.5 threshold, on the
class-weighted XGBoost model.

### Cost-ratio sensitivity sweep

A single cost ratio only shows one operating point. Sweeping the ratio of
(cost of missed fraud) / (cost of a blocked transaction) shows how the optimal threshold moves
as that ratio changes — this is the strongest differentiator of the cost-sensitive approach over
a single static threshold:

![Optimal threshold vs. cost ratio](reports/figures/cost_ratio_sensitivity.png)

| Cost ratio | Optimal threshold |
|---|---|
| 5 | 0.84 |
| 10–25 | 0.42 |
| 50 | 0.12 |
| 75–100 | 0.08 |
| 150–200 | 0.02 |
| 300–500 | 0.01 |

As missing fraud becomes relatively more expensive, the optimizer accepts more false positives
to catch more fraud — the threshold falls accordingly.

## Model interpretability

SHAP (`TreeExplainer`) over the class-weighted XGBoost model, sampled 2,000 test rows
(all fraud rows plus a matched legitimate sample, since fraud is too rare to show up reliably in
a plain random sample):

![Top 10 SHAP features](reports/figures/shap_summary.png)

`V14`, `V4`, and `V12` are the strongest drivers, consistent with published analyses of this
dataset that repeatedly identify `V14`/`V12`/`V10`/`V17` as the components carrying the most
fraud signal — despite `V1`..`V28` having no semantic meaning on their own (PCA components, not
named business features).

## Inference service

A FastAPI service (`src/serving/app.py`) loads the persisted production pipeline — the
class-weighted XGBoost model at its cost-optimized threshold (0.08) — and exposes:

| Endpoint | Description |
|---|---|
| `GET /health` | liveness + current decision threshold |
| `POST /predict` | score a single transaction |
| `POST /replay?n=&delay_ms=` | replay `n` real test-set rows with a small delay, scoring each |
| `GET /latency` | p50/p95 over every prediction served so far |

Latency, measured from 5,000 real served predictions (replay + individual calls):

![Inference latency histogram](reports/figures/latency_histogram.png)

**p50: 1.42ms, p95: 2.14ms** — comfortably low-latency for a synchronous scoring call.

```bash
uvicorn src.serving.app:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"Time": 0, "V1": ..., "Amount": 149.62}'
curl -X POST "localhost:8000/replay?n=500&delay_ms=1"
curl localhost:8000/latency
```

### Streaming stretch goal: Kafka → FastAPI → Redis sliding window

`docker-compose.yml` also brings up a full streaming demo: a producer replays test-set
transactions onto a Redpanda (Kafka-API-compatible) topic with a small delay; a consumer reads
that stream, maintains a Redis-backed sliding-window feature (rolling transaction count and
amount sum over the last 30 seconds — global rather than per-card, since this dataset has no
card/merchant ID to key on), scores each transaction against the live FastAPI service, and logs
the result. Verified end-to-end via `docker compose up`:

```
amount=$892.16 fraud_probability=0.1286 is_fraud=True window[30s]: count=148 sum=$24602.63
```

**Honest scope note:** the Redis sliding-window aggregate is genuinely computed and logged
end-to-end, but it isn't yet wired back into the model as a trained input — that would mean
retraining on a feature that doesn't exist in the static offline dataset used here. It
demonstrates the full mechanism the architecture calls for; feeding it into the model is exactly
the kind of thing the "what I'd do with more data" section below gestures at.

```bash
docker compose up --build
```


## Repository structure

```
├── PLAN.md                          7-day build plan
├── DEBUGGING_EXERCISE.md            controlled debugging exercise writeup (Day 7)
├── docs/                            per-day analysis writeups
├── data/
│   ├── raw/                         gitignored — populated via kaggle CLI, see below
│   └── processed/                   gitignored — populated via src/data/ingest.py
├── notebooks/                       appendix EDA notebook
├── reports/figures/                 generated plots referenced above
├── src/
│   ├── data/                        ingestion, chronological split
│   ├── features/                    shared preprocessing pipeline
│   ├── models/                      training pipelines, cost engine, SHAP
│   ├── serving/                     FastAPI inference service
│   └── streaming/                   Kafka/Redpanda → Redis stretch goal
├── tests/                           pytest suite
├── docker-compose.yml
└── requirements.txt
```

## Reproducing this locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw
unzip data/raw/creditcardfraud.zip -d data/raw && rm data/raw/creditcardfraud.zip

python -m src.data.ingest
python -m src.models.train_baseline
python -m src.models.train_autoencoder
python -m src.models.imbalance_comparison
python -m src.models.run_cost_analysis
python -m src.models.run_shap_analysis
python -m src.models.train_production_model

pytest tests/ -q
```

**Environment note:** XGBoost and PyTorch each bundle their own OpenMP runtime; running both in
one process crashes on macOS unless `OMP_NUM_THREADS=1` is set (already handled in
[`src/__init__.py`](src/__init__.py)). `xgboost` is pinned below 3.0 because SHAP 0.49.1's
`TreeExplainer` can't yet parse XGBoost 3.x's `base_score` serialization format.

## What I'd do with more data

- Multi-day/multi-month data to actually study concept drift, instead of a single-day snapshot.
- Real merchant/geo/entity fields (e.g. the Sparkov-simulated dataset) for genuine feature
  engineering — spatial distance between sequential transactions, per-card velocity — instead of
  working only with anonymized PCA components.
- A live transaction stream instead of a static replay for the serving layer.

This section is meant to signal awareness of this project's ceiling, not to pretend it's bigger
than it is.
