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
