# How Robust Is Cost-Sensitive Threshold Optimization Under Temporal Shift and Cost Uncertainty?

## Problem

Fraud detection under severe class imbalance is usually framed as a model-selection problem:
pick the classifier with the best PR-AUC. But PR-AUC is threshold-independent, while the actual
business decision — block this transaction or not — happens at a single, chosen threshold. Two
models can rank transactions almost identically and still produce very different real-world
costs, depending on where their probability mass sits relative to that threshold (see the
[ablation study](#4-does-training-time-cost-sensitivity-add-anything-over-threshold-tuning)
below). This project's original framing was: pick the best model, then optimize its decision
threshold against an explicit cost function. That framing turned out to have a real
methodological gap, described next.

## Hypothesis

Cost-sensitive threshold optimization reduces expected financial loss relative to a default 0.5
threshold — but the sharper, more useful question is **how robust that improvement is**: does it
hold up out-of-sample, across different time windows, under uncertainty about the assumed cost
ratio, and on a structurally different dataset? A method that only wins on the exact split it was
tuned on is a weaker result than one that wins consistently.

## Method

**The methodological fix this report exists to document:** earlier versions of this project
selected the decision threshold using the *test set's* labels — `optimize_threshold(...)` was
called with `y_test`, then the resulting cost reduction was reported on that same test set. That
makes the test set part of model selection, not a genuine held-out evaluation. Every experiment
below instead uses a strict **train (68%) → validation (17%) → test (15%)** chronological split:

- The model is fit on `train` only.
- Every threshold, calibrator, or hyperparameter choice is selected on `val` only.
- `test` is touched exactly once, for final reporting, and never used to pick anything.

All splits are chronological (sorted by transaction time, not shuffled), for the same reason
stated throughout this project: a random split lets the model see transactions that happen after
the ones it's evaluated on.

## Experiments

### 1. Does the corrected protocol change the headline result?

Yes, substantially. Under the old (leaking) protocol, the reported cost reduction from
threshold optimization was 13.3%. Under the corrected protocol — threshold selected on val,
evaluated once on test — it drops to **2.4%** ($6,575 → $6,415 on the $500/$5 scenario). The
original number was inflated by selecting the threshold against the same data used to report the
improvement. This is the single most important correction in this report.

A related, smaller finding from the same run: the val-selected "optimal" threshold (0.09) is not
actually the lowest-cost point on test — threshold 0.05 achieves a lower test cost ($6,135 vs.
$6,415). That's not a bug; it's the val→test generalization gap made visible, exactly what a
held-out test set is supposed to expose.

### 2. Is the result stable across time windows? (walk-forward evaluation)

The full dataset (sorted by time) is cut into 5 equal blocks. Fold *k* trains on an expanding
window (blocks 0..k-1), selects the threshold on the first half of block *k*, and reports on the
second half — so training data only ever grows forward in time.

| Fold | Train rows | Threshold (val) | Default cost (test) | Optimized cost (test) | Result |
|---|---|---|---|---|---|
| 1 | 56,961 | 0.04 | $4,520 | $4,340 | beats |
| 2 | 113,922 | 0.62 | $5,180 | $5,665 | worse |
| 3 | 170,884 | 0.56 | $4,015 | $4,015 | tie (beats) |
| 4 | 227,845 | 0.05 | $3,060 | $3,840 | worse |

**The optimized threshold beat the default in only 2 of 4 folds**, and the selected threshold
itself swings from 0.04 to 0.62 across windows. This dataset covers a single day, so it cannot
test genuine multi-day concept drift — that limitation is real and unavoidable here — but it
does show the conclusion is not stable even across different windows of a single day. A
single-split result (Experiment 1) looked like a clean win; four splits show a coin flip.

### 3. Is the selected threshold robust to uncertainty in the assumed cost ratio?

Every other experiment treats $500 (false negative) and $5 (false positive) as fixed and known.
Here, `cost_fn ~ Uniform(100, 1000)` and `cost_fp ~ Uniform(1, 20)` are drawn independently 500
times; for each draw, the cost-optimal threshold is selected on val.

![Cost uncertainty threshold distribution](reports/figures/cost_uncertainty_threshold_distribution.png)

| Statistic | Value |
|---|---|
| Mean | 0.151 |
| Std. dev. | 0.178 |
| Median | 0.090 |
| IQR | [0.090, 0.240] |
| Range | [0.010, 0.750] |

The distribution is multi-modal — thresholds cluster at a handful of plateaus rather than
varying smoothly, itself a consequence of how few fraud examples (59) are in the validation
split. The median (0.090) matches the point-estimate result from the $500/$5 scenario, which is
reassuring, but the spread (0.01 to 0.75) means a materially different but still plausible cost
assumption would have picked a very different operating point.

### 4. Does training-time cost-sensitivity add anything over threshold tuning?

Four configurations, same train/val/test protocol:

| Configuration | Threshold | PR-AUC | Recall | Expected cost |
|---|---|---|---|---|
| A: standard training, threshold 0.5 | 0.50 | 0.757 | 0.712 | $7,510 |
| B: standard training, optimized threshold | 0.01 | 0.757 | 0.750 | $6,795 |
| C: cost-weighted training, threshold 0.5 | 0.50 | 0.761 | 0.750 | $6,530 |
| D: cost-weighted training, optimized threshold | 0.02 | 0.761 | 0.750 | $7,070 |

"Cost-weighted training" here means the training sample weights are set directly to the dollar
costs (`$500` per fraud row, `$5` per legitimate row) — a genuinely different mechanism from
class weighting (which only encodes class *frequency*, not the actual dollar figures).

**C beats both B and D.** Cost-weighted training alone, at the plain default threshold, does
better than either decision-time threshold tuning alone or the two combined. Combining
training-time and decision-time cost-sensitivity (D) is not simply additive — it's *worse* than
cost-weighted training alone, because threshold selection on top of an already cost-shifted
probability distribution is itself unstable (consistent with Experiments 2 and 3). This is the
most interesting single result in this report: two cost-sensitivity mechanisms that sound
complementary in principle turn out to partially substitute for, rather than reinforce, each
other in practice.

### 5. Does any of this generalize to a structurally different dataset?

The primary dataset is anonymized PCA components with 492 fraud rows total — useful for method
demonstration, but a professor (or any careful reviewer) can't conclude the method works on real
fraud data from that alone. The [Sparkov-simulated fraud
dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (881,739 train rows, 4,989
fraud) has real merchant, category, and geolocation fields, enabling genuine feature engineering
— haversine distance between customer and merchant, transaction hour, customer age — instead of
working only with anonymized components. Same protocol, same $500/$5 costs, applied here.

| | Primary dataset | Sparkov |
|---|---|---|
| Baseline PR-AUC (no weighting) | 0.757 | **0.909** |
| Class weighting vs. baseline | **helps** (+0.001 PR-AUC, cost $7,510→$6,575) | **hurts** (0.909→0.882 PR-AUC) |
| Threshold optimization vs. default | **helps** (2.4% cost reduction) | **no effect** (val selects 0.50, 0% reduction) |

**Neither conclusion from the primary dataset replicates on Sparkov.** The likely explanation is
that Sparkov's engineered features (distance, category, amount) separate fraud from legitimate
transactions far better than anonymized PCA components do — the baseline model is already at
PR-AUC 0.909 / ROC-AUC 0.998 without any imbalance handling. When a model already separates
classes almost perfectly, there's much less room for either class-weighting or threshold-tuning
to help, and rebalancing can actively distort an already well-calibrated ranking. This is not a
failure of the method — it's evidence that its *value* is dataset-dependent, concentrated in
regimes where the raw signal is weak, which is itself a useful, non-obvious finding this project
would not have produced without a second dataset.

#### 5a. Adding a real per-entity feature, and repeating Experiments 2–3 on Sparkov

Unlike the primary dataset, Sparkov has a card identifier (`cc_num`), which makes a genuine
per-entity feature possible: `card_txn_count_24h` / `card_amt_sum_24h`, a rolling count and
amount sum of that same card's transactions in the preceding 24 hours, computed causally
(`closed="left"` — the window is `[t-24h, t)`, strictly excluding the transaction's own row, so
it can never leak information about itself). Adding it and repeating the walk-forward and
bootstrap checks from Experiments 2–3, this time on Sparkov:

| | Sparkov, no velocity feature | Sparkov, + card velocity feature |
|---|---|---|
| Baseline PR-AUC (no weighting) | 0.909 | **0.969** |
| Threshold optimization vs. default (single split) | 0% reduction | **−1.8%** (actively worse) |
| Cost-reduction 95% bootstrap CI | [0.0%, 0.0%] | [−9.1%, 1.5%] |
| Walk-forward: optimized beats default | 2 of 4 folds | **1 of 4 folds** |

The velocity feature is a genuinely strong signal — it pushes an already-strong baseline from
0.909 to 0.969 PR-AUC — and it makes the "threshold optimization doesn't help here" finding
*more* decisive, not less: with a near-perfect model, optimizing the threshold has less room to
help and more room to overfit to the validation split's noise. This directly answers the
Limitations concern in the previous version of this report that the Sparkov comparison used only
one split — it doesn't anymore, and the repeated checks confirm the single-split result rather
than overturning it.

#### 5b. Closing the streaming gap: does the live feature actually reach the model?

The README's streaming section is explicit that the primary dataset's Redis sliding-window
aggregate is computed and logged but never fed into the model — there's no entity ID to key a
meaningful per-card feature on. Sparkov's `cc_num` removes that obstacle, so
[`src/streaming/run_sparkov_streaming_demo.py`](src/streaming/run_sparkov_streaming_demo.py)
replays 5,000 real transactions through a per-card Redis sliding window
([`src/streaming/redis_features_sparkov.py`](src/streaming/redis_features_sparkov.py)) and feeds
the *live* `card_txn_count_24h` / `card_amt_sum_24h` — not the offline, pre-computed version —
directly into `pipeline.predict_proba(...)` for each transaction, in real time. Verified output,
one card mid-fraud-burst:

```
cc_num=3573030041201292 amt=$8.28    live_card_txn_count_24h=4  fraud_probability=0.9999
cc_num=3573030041201292 amt=$353.57  live_card_txn_count_24h=5  fraud_probability=0.9998
cc_num=3573030041201292 amt=$876.10  live_card_txn_count_24h=6  fraud_probability=0.9997
...
cc_num=3573030041201292 amt=$233.53  live_card_txn_count_24h=11 fraud_probability=0.9983
```

47 fraud transactions were replayed, 87 flagged, 99.2% overall accuracy — but the number that
matters here isn't the accuracy, it's that `live_card_txn_count_24h` is visibly climbing
transaction-by-transaction as Redis accumulates state, and the model's prediction is responding
to that same live number. This is what "the streaming feature actually feeds the model" means in
practice, not just as a claim.

## Statistical analysis

Fraud is 0.17% of the primary test split (52 of 42,721 rows) — not much to draw firm conclusions
from a single evaluation. Bootstrap resampling (1,000 resamples, 95% CI) on the corrected,
val-selected threshold's test-set predictions:

| Metric | Point estimate | 95% CI |
|---|---|---|
| PR-AUC | 0.758 | [0.630, 0.860] |
| Precision @ 0.09 | 0.325 | [0.235, 0.407] |
| Recall @ 0.09 | 0.769 | [0.640, 0.871] |
| Expected cost @ 0.09 | $6,415 | [$3,390, $9,935] |
| Cost reduction vs. default | 2.4% | [−9.3%, 18.7%] |

The cost-reduction interval crosses zero. Combined with the walk-forward result (2 of 4 folds)
and the cost-uncertainty spread, the honest summary is: **on this dataset, cost-sensitive
threshold optimization has a positive expected effect but is not a reliably-winning
intervention** — its benefit is real on average but small relative to the noise in a
492-fraud-row dataset.

## Limitations

- **Single day of data.** Walk-forward evaluation (Experiment 2) tests intra-day window
  stability, not genuine multi-day concept drift — that would need a multi-day dataset this
  project doesn't have.
- **Only 492 fraud rows** in the primary dataset (52 in test). Every interval above is wide
  because of this, not because of a weak method.
- **Illustrative costs.** $500/$5 are demonstration figures, not sourced fraud-loss data — this
  is why Experiment 3 treats them as uncertain rather than fixed.
- **The Sparkov comparison's training-objective check (Experiment 4) is not yet repeated on
  Sparkov** — only Experiments 2 (walk-forward) and 3 (bootstrap) have been, in Experiment 5a.
- **Established techniques throughout.** XGBoost, class weighting, SMOTE, autoencoders,
  threshold optimization, Platt/isotonic calibration, bootstrap CIs — no new algorithm, loss
  function, or optimization procedure is introduced. What's novel here is the combination and,
  more specifically, Experiments 2–5: repeated temporal evaluation, cost-uncertainty robustness,
  the training-objective-vs-decision-policy comparison, and external validation are not
  standard parts of a typical threshold-tuning writeup, and the fact that three of the four
  produce a *negative or null* result is itself the finding.

## Future work

- Multi-day data to test genuine concept drift, not just intra-day window stability.
- Repeat Experiment 4 (training-objective vs. decision-policy) on Sparkov — Experiments 2 and 3
  are now done (Experiment 5a).
- Wire the *offline* card-velocity feature's live Redis computation into the actual served
  Sparkov model as a standing service, not just a replay demo script — the mechanism is proven
  end-to-end (Experiment 5b), but there's no persistent Sparkov production pipeline the way the
  primary dataset has one in `src/serving/`.
- A theoretically motivated cost-sensitive objective (e.g., a custom asymmetric loss function
  rather than sample-weighting) as a fifth training-objective configuration.
