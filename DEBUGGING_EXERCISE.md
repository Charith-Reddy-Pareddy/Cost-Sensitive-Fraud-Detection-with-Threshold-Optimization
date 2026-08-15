# Controlled Debugging Exercise

**This is a deliberate exercise, not a record of organically discovered bugs.** Three small,
realistic defects are about to be introduced into otherwise-working, tested code, on purpose,
to practice and demonstrate a full debugging workflow: introduce a controlled failure,
reproduce it via the test suite, diagnose the root cause, fix it properly, add a regression
test, and verify the full suite is green again. Each defect gets its own commit clearly labeled
as an intentional-defect commit, so the git history itself documents that these were planted,
not stumbled into during ordinary development.

## Planned defects

1. **Wrong reduction in the cost threshold optimizer** — `src/models/cost_engine.py`,
   `optimize_threshold()`. Picks the threshold that *maximizes* expected cost instead of
   minimizing it.
2. **Leakage reintroduced in the train/test split** — `src/data/ingest.py`,
   `chronological_split()`. Reverts the chronological split back to a random shuffle split,
   which is exactly the temporal-leakage failure mode this project's README and PLAN.md call
   out as a known risk if splitting isn't done chronologically.
3. **Amount/Time field swap in the serving path** — `src/serving/app.py`, `_row_to_frame()`.
   Swaps the `Amount` and `Time` values before scoring, so the API silently scores requests
   against the wrong feature values.

## What happens next in this document

Once all three defects are committed, the full test suite gets run to capture the actual
failures, followed by a root-cause diagnosis, the fix, and a regression test for each — appended
below this line as it happens.

---

## Reproducing the failures

With all three defects committed (`1d4ee61`, `99ea60f`, `8e73694`), running the full suite
produced four failures:

```
FAILED tests/test_cost_engine.py::test_optimize_threshold_beats_or_matches_default
FAILED tests/test_cost_engine.py::test_higher_cost_ratio_lowers_optimal_threshold
FAILED tests/test_ingest.py::test_split_is_chronological_not_random
FAILED tests/test_serving.py::test_predict_matches_offline_batch_scoring
4 failed, 21 passed, 2 skipped
```

Defect 1 alone broke two tests; defects 2 and 3 each broke one.

## Diagnosis and fixes

### Defect 1 — threshold optimizer picked the worst threshold, not the best

**Root cause:** `optimize_threshold()` in `src/models/cost_engine.py` used
`np.argmax(costs)` instead of `np.argmin(costs)` when selecting the threshold from the swept
cost curve — a one-word reduction-function mixup that silently inverts the entire optimizer's
objective. Nothing about the return type or shape changes, so it fails silently rather than
raising — exactly the kind of bug that's dangerous in production ML code, since it would have
shipped a threshold that *maximizes* expected financial loss.

**Fix:** reverted to `np.argmin(costs)`.

**Regression test added:** `test_optimize_threshold_picks_the_minimum_not_the_maximum` in
`tests/test_cost_engine.py` — a hand-crafted, deterministic 3-point cost curve (`[70, 0, 30]`)
with an unambiguous, non-endpoint minimum, so an argmin/argmax mixup fails immediately and
obviously rather than relying on random test data happening to expose it (the property-based
tests that already existed did catch this regression, but only indirectly).

### Defect 2 — chronological split reverted to a random shuffle

**Root cause:** `chronological_split()` in `src/data/ingest.py` was rewritten to call
`sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=42)` — a
plausible-looking "simplification" that reintroduces exactly the temporal-leakage risk this
project's README and PLAN.md call out explicitly: a random split lets the model see
transactions that happen chronologically *after* the ones it's evaluated on, which doesn't
match how the model is actually used in production.

**Fix:** reverted to sorting by `Time` and slicing positionally, so the test set is strictly
later in time than the training set.

**Regression test:** the existing `test_split_is_chronological_not_random` already pins this
precisely (asserts `train["Time"].max() < test["Time"].min()`), which is exactly the property a
random-shuffle regression violates — no new test needed, the existing one is a direct,
unambiguous pin for this defect class.

### Defect 3 — Amount/Time swapped before scoring in the serving path

**Root cause:** `_row_to_frame()` in `src/serving/app.py` swapped the `Amount` and `Time`
values (`row["Amount"], row["Time"] = row["Time"], row["Amount"]`) after building the row dict
by name — a classic copy-paste-during-a-refactor mistake. Because the swap happens *after* the
name-based lookup, it's easy to miss on a diff: the column names are still correct, only the
values under them are wrong. The model silently scored every request against the wrong `Amount`
and `Time`, without erroring.

**Fix:** removed the swap.

**Regression test:** the existing `test_predict_matches_offline_batch_scoring` already pins this
precisely — it asserts the API's prediction for a real row matches the pipeline scoring that
same row directly, offline. Any mismatch between what the serving path sends the model and what
it was trained on fails this test immediately.

## Final verification

Full suite green after all three fixes, including the two Redis-backed integration tests
(skipped above only because Redis wasn't running locally at reproduction time):

```
28 passed
```

