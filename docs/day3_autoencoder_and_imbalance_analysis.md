# Day 3 Analysis: Autoencoder Diagnosis + Imbalance Handling Comparison

## Autoencoder: why it underperforms the supervised baselines

Trained only on legitimate transactions (never sees a labeled fraud example), scored by
reconstruction error on the held-out test set.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic Regression (Day 2) | 0.744 | 0.979 |
| XGBoost, no imbalance handling (Day 2) | 0.671 | 0.976 |
| **Autoencoder** | **0.272** | **0.922** |

Mean reconstruction error: fraud rows average **4.70**, legitimate rows average **0.25** — an
~18x gap, and on its own that looks like a strong signal (it's why ROC-AUC is still a
respectable 0.922). But PR-AUC — the metric that actually matters here, since it's sensitive to
precision under severe imbalance rather than reflecting performance across an enormous default-
class majority — collapses to 0.272. That gap between a fine ROC-AUC and a poor PR-AUC is
itself the diagnosis: the autoencoder separates classes *on average*, but at any operating
threshold where recall is worth having, it's flagging far more unusual-but-legitimate
transactions than actual fraud.

The underlying reason is the framing assumption itself: reconstruction error assumes fraud is
statistically anomalous relative to normal spending. That's only partly true here. Some fraud
is genuinely unusual (large amount, atypical PCA-component combination) and gets a high
reconstruction error along with legitimate outliers — the same tail territory a large but
completely legitimate purchase would land in. Meanwhile some fraud is amount-typical and
component-typical enough that it reconstructs almost as well as normal traffic, since the
model never saw *any* fraud example to learn what specifically distinguishes it from a normal
outlier. The supervised baselines don't have this problem because they're trained directly on
the distinction that matters (fraud vs. not), rather than on a proxy (anomalous vs. not) that
only partially overlaps with it.

This isn't a case for throwing the autoencoder out — it's a useful complementary signal (fraud
does skew toward higher reconstruction error, so it could feed a downstream ensemble or a
secondary review queue), but on its own it is not the modeling approach to ship as the primary
decision-maker for this dataset.

## Imbalance handling: none vs. class-weighting vs. SMOTE

All three use the same XGBoost baseline; only the imbalance-handling strategy changes. SMOTE
and class-weighting are both fit inside the pipeline on the training fold only, so neither can
leak into the test set.

| Strategy | PR-AUC | ROC-AUC | F1 @ 0.5 |
|---|---|---|---|
| None | 0.671 | 0.976 | 0.756 |
| **Class weighting** | **0.786** | **0.986** | **0.765** |
| SMOTE | 0.777 | 0.977 | 0.667 |

Class-weighting wins on every metric here, including F1 at the default threshold. SMOTE does
meaningfully improve PR-AUC over the unweighted baseline (0.671 → 0.777), so it isn't useless —
but it comes at a real cost: F1@0.5 drops to 0.667, well below both alternatives, because the
synthetic minority samples shift the default decision boundary in a way that trades away
precision. This is consistent with the concern stated up front in the project plan: synthetic
oversampling can produce artifacts that don't generalize as cleanly as a distributional
reweighting of the real data. Class-weighting achieves a comparable or better recall-side
signal without inventing synthetic transactions, and is the strategy carried forward as the
default for the cost-sensitive threshold work in Day 4 — consistent with the project's stated
emphasis on cost-sensitive learning as the more ML-systems-appropriate approach, with SMOTE
kept as a documented comparison rather than the default.
