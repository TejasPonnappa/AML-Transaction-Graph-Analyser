# Tree-model baselines vs the GNNs

Measured with `python src/baselines.py <amlsim|ibm|elliptic> --seeds 3` (IBM also `--neg-frac 0.1`). Results are also written to `outputs/baselines_<dataset>.json` (gitignored). Nothing here comes from the literature; there are no unverified numbers in this file.

## Method

- **Same splits as the GNN:** the saved train/val/test masks (AMLSim random, IBM chronological, Elliptic steps 1-34 / 35-38 / 39-49). AMLSim is also run on a temporal split (sorted by `TIMESTAMP`, 70/15/15).
- **Leak-free features:** account aggregates (counts, sums, means) and pair counts come from training rows only and are frozen; training rows have their own contribution subtracted (leave-one-out). Held-out rows cannot influence any feature (tested in `tests/test_baselines.py`).
- **Threshold:** maximum F1 on the validation precision-recall curve, applied once to test with `>=`.
- **Models:** scikit-learn `HistGradientBoostingClassifier` (hgb) and `RandomForestClassifier` (rf), class-balanced. LightGBM is not installed. hgb is near-deterministic, so its seed std mainly reflects tie-breaking; rf std is real.
- **GNN row:** the numbers in `outputs/metrics*.json` as they stand, which are leak-prone (see `model_review.md`). The IBM predictions CSV is stale, so the IBM GNN row uses the metrics file only.
- **Runtime (8 cores, 3 seeds):** Elliptic 67 s, AMLSim 233 s, IBM 328 s. IBM fits on all positives plus 10% of training negatives (weighted back); validation and test use every row.

## Results (test set, mean +/- std over 3 seeds)

### AMLSim (258 positives in the random split)

| model | split | PR-AUC | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|
| hgb | random (as GNN) | 0.977 +/- 0.007 | 0.992 | 0.984 | 0.969 | 0.977 |
| rf | random (as GNN) | 0.984 +/- 0.001 | 0.997 | 0.992 | 0.968 | 0.980 |
| hgb | temporal | 0.991 +/- 0.001 | 0.996 | 0.996 | 0.985 | 0.991 |
| rf | temporal | 0.989 +/- 0.000 | 0.998 | 1.000 | 0.985 | 0.993 |
| GNN (reported) | random | 0.847 | 0.990 | 0.888 | 0.895 | 0.892 |

### IBM HI-Small (chronological, 1,561 test positives)

| model | PR-AUC | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|
| hgb | 0.307 +/- 0.015 | 0.987 | 0.344 | 0.340 | 0.342 |
| rf | 0.323 +/- 0.007 | 0.985 | 0.404 | 0.297 | 0.341 |
| GNN (reported) | 0.257 | 0.990 | 0.223 | 0.693 | 0.337 |

### Elliptic (717 test positives, steps 39-49)

| model | features | PR-AUC | precision | recall | F1 |
|---|---|---|---|---|---|
| hgb | A: local only | 0.689 +/- 0.004 | 0.921 | 0.553 | 0.691 |
| rf | A: local only | 0.690 +/- 0.001 | 0.941 | 0.574 | 0.713 |
| hgb | B: all 165, no time_step | 0.714 +/- 0.002 | 0.964 | 0.614 | 0.750 |
| rf | B: all 165, no time_step | 0.699 +/- 0.001 | 0.955 | 0.615 | 0.748 |
| hgb | C: local + neighbour mean | 0.707 +/- 0.004 | 0.923 | 0.608 | 0.733 |
| rf | C: local + neighbour mean | 0.706 +/- 0.002 | 0.943 | 0.603 | 0.735 |
| GNN (reported) | 166 incl. time_step | 0.470 | 0.727 | 0.400 | 0.516 |

Per-time-step test F1 at each model's validation-tuned threshold (the GNN uses its saved threshold on its existing predictions):

| model | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb, local only | 0.88 | 0.68 | 0.91 | 0.82 | 0.00 | 0.06 | 0.00 | 0.29 | 0.00 | 0.00 | 0.02 |
| rf, local only | 0.94 | 0.72 | 0.91 | 0.84 | 0.00 | 0.05 | 0.00 | 0.47 | 0.00 | 0.00 | 0.01 |
| hgb, all 165 | 0.93 | 0.77 | 0.96 | 0.88 | 0.00 | 0.07 | 0.00 | 0.40 | 0.00 | 0.00 | 0.03 |
| rf, all 165 | 0.95 | 0.77 | 0.95 | 0.88 | 0.00 | 0.06 | 0.00 | 0.44 | 0.00 | 0.00 | 0.03 |
| hgb, local + neighbours | 0.92 | 0.75 | 0.93 | 0.87 | 0.00 | 0.06 | 0.00 | 0.36 | 0.00 | 0.00 | 0.03 |
| rf, local + neighbours | 0.94 | 0.74 | 0.93 | 0.88 | 0.00 | 0.05 | 0.00 | 0.50 | 0.00 | 0.00 | 0.02 |
| GNN (existing predictions) | 0.60 | 0.60 | 0.58 | 0.71 | 0.00 | 0.05 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| illicit nodes in step | 81 | 112 | 116 | 239 | 24 | 24 | 5 | 2 | 22 | 36 | 56 |

## What the numbers say

1. **AMLSim is trivially separable, and the GNN is worse than a tree.** Gradient boosting reaches F1 about 0.98 on the same random split (GNN 0.89), and about 0.99 on a temporal split, so the split is not what made the GNN look good. All 1,719 fraud amounts are below 20 and `TX_TYPE` is constant (checked on the CSV). One caveat: the `amount < 20` flag was added after seeing this, so it is dataset knowledge, though a tree finds the same split from `log1p(amount)` alone.
2. **IBM: the GNN does not beat trees.** Trees on edge features plus train-only aggregates reach PR-AUC 0.31-0.32 against the GNN's 0.257, with equal F1 (0.34). They have higher precision and lower recall: the GNN's 0.95 threshold sits at its tuning-grid boundary and trades precision for recall. These baselines use no temporal or structural features beyond aggregates and a pair count, so a graph or rolling-window model could still gain; none has been shown to.
3. **Elliptic: trees clearly beat the GNN** (F1 0.69-0.75 vs 0.52). The graph neighbourhood adds nothing over the provided features (C vs B: no gain). The GNN is below trees on every step before 43.
4. **Every model fails after step 43** (F1 near 0 on steps 43-49, with step 46 having only 2 illicit nodes). That is a property of the data split, not of the GNN. Pooled test numbers therefore mix a usable period (steps 39-42) with a period where no model works. Why the break happens (a dark-market shutdown has been suggested) is unverified here.

## Not done yet

- The leaky-aggregate ablation (same trees with whole-dataset aggregates), which would quantify how much leakage inflates the GNN-style features.
- IBM rolling-window (past-only) features, per-time-step IBM metrics, and a full-negative IBM run.
- A GNN control with `time_step` removed (this needs retraining).
- LightGBM or XGBoost (not installed).
