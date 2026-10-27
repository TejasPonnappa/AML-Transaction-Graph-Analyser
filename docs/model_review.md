# Model review and Phase 3 plan

> **Note:** AMLSim was later removed from the project (it is trivially separable, as the results below show), and the Elliptic GNN was retrained without the time step; see `docs/elliptic_results.md`. This document is kept as a record of the measurements at the time.

Three read-only reviews (AMLSim, IBM AML, Elliptic) of `src/`, `outputs/metrics*.json` and `tests/`, combined into one prioritised plan. Nothing was trained or changed. "Verified" means read in the code (or, for AMLSim data facts, checked on the raw CSV); "estimate" means reasoning, not a measurement in this repo. **Any number attributed to the literature is marked "(unverified)": it was recalled from memory and has not been checked against the papers.**

## 1. Can the reported metrics be trusted?

| | Reported (test) | Verdict |
|---|---|---|
| AMLSim | ROC-AUC 0.990, PR-AUC 0.847, F1 0.892 (thr 0.90), 258 positives | **Optimistic.** Random edge split, full-data features. Single seed. |
| IBM | ROC-AUC 0.990, PR-AUC 0.257, F1 0.337 (thr 0.95), 1,561 positives | **Not a clean forward-in-time estimate.** Leaky features, test-informed tuning, censored threshold. |
| Elliptic | F1 0.516, PR-AUC 0.470, recall 0.40 (thr 0.85), 717 positives | **Valid measurement, but understates the model.** Split is correct; `time_step` fed as a feature, mismatched selection and threshold. |

### Verified issues

**AMLSim**
- Random stratified 70/15/15 edge split (`data_preprocessing.py:98-104`) although the CSV has a `TIMESTAMP` column (200 distinct values) that is ignored. Fraud is account-structured (1,719 fraud edges, 1,639 accounts), so train and test share accounts.
- Node aggregates (degree, sums, means) and z-score statistics are computed over all edges (`:55-90`), including test edges and the fraud edges themselves.
- Message passing runs over the full `edge_index` in training and evaluation (`model.py:22-30`, `train.py:61`).
- Data facts (checked on the CSV): all 1,719 fraud amounts are below 19.92, while non-fraud has a median of 157. `TX_TYPE` is a single value (`TRANSFER`), so that feature is constant. The benchmark is close to separable on amount alone. `ALERT_ID` is a perfect label leak and is correctly not used.
- Threshold and checkpoint chosen on validation only; no test leakage there. Single seed, 258 test positives, so F1 carries roughly +/-0.03 to 0.04 of noise (estimate).

**IBM**
- Chronological split is correct (`data_preprocessing_ibm.py:117-125`). But node aggregates (`:96-111`) and pair frequency (`:81`) are computed over the full timeline, so a training edge's features include future transactions. Pair frequency is a feature of the edge itself, which makes this the most direct leak.
- Normalisation statistics are global (minor). Message passing sees val/test structure (transductive).
- `POS_WEIGHT_MULTIPLIER` was tuned against test results: the code comment (`train_ibm.py:23-28`) cites test-set precision and FP counts. The origin of those numbers is inferred, because only test confusion matrices are printed.
- Threshold grid is `0.05..0.95` (`:108`) and the chosen value is 0.95, the last grid point, so the optimum is censored. Default-threshold F1 is 0.086.
- Checkpoint selection on validation ROC-AUC with about 0.2% positives; PR-AUC and F1 are nearly unconstrained by it. One full-batch step per epoch.
- Metrics file is internally consistent (TP+FN=1561; precision 0.223).

**Elliptic**
- Temporal split is correct (`data_preprocessing_bitcoin.py:83-88`), unlabeled nodes are excluded from loss and evaluation, and BatchNorm statistics include test-period features (unsupervised, mild).
- `time_step` is included as an unscaled feature (`:27`, `:34`): train sees 1-34, test sees 39-49, so the model extrapolates on a raw integer.
- Checkpoints chosen by validation F1 at a hardcoded 0.5 (`train_bitcoin.py:175`, `:79`), while the loss is class-weighted and the deployed threshold is 0.85. The scheduler steps on the same noisy F1. Validation is only 4 time steps.
- Precision 0.73 vs recall 0.40 is consistent with a threshold too high for shifted test data. Test pools steps 39-49, so a post-step-43 collapse (attributed to a dark-market shutdown; the cause is unverified, the per-step results are measured, see `docs/baseline_results.md`) is averaged in. No per-step metrics exist, so how much of the loss comes from it is unconfirmed.
- No seeds set anywhere; one run reported. Tests don't cover Elliptic preprocessing or any leakage property.

## 2. Why the models underperform, by expected impact

**AMLSim**: (1) random split ignores time; (2) task is trivially separable on amount, so the score mostly measures separation from the 22% of normal rows under 20; (3) dead `tx_type`; (4) 5-feature bottleneck to 2 dims and `edge_attr` unused in message passing; (5) `pos_weight` near 1,000 inflates scores (best threshold 0.9; F1 at 0.5 is 0.40).

**IBM**: (1) no temporal or structural features (velocity, fan-in/out, cycles, scatter-gather), and mean-aggregating SAGEConv cannot count them; (2) 5-feature node input squeezed to 4 dims; (3) `edge_attr` not used in message passing; (4) selection on ROC-AUC, single seed, test-tuned weight, censored threshold; (5) mixed-currency amounts with no conversion and `Amount Received` unused; (6) ordinal payment-format encoding. These rankings are estimates.

**Elliptic**: (1) distribution shift after step 43 plus a threshold tuned on pre-shutdown validation; (2) raw `time_step` feature; (3) graph smoothing over features that already contain aggregated neighbours (published GCNs reportedly lose to Random Forest here; unverified); (4) selection/threshold mismatch; (5) BatchNorm/dropout instability, no seed.

## 3. Honest baselines (estimates, not measured here)

- **AMLSim:** with a temporal split and training-only aggregates, PR-AUC roughly 0.3-0.7 (down from 0.85), ROC-AUC 0.95-0.99, tuned F1 roughly 0.4-0.75. A simple `amount < 20` rule already gets 100% recall at about 0.6% precision, so tree models will likely match the GNN.
- **IBM:** gradient boosting on edge features only, no leakage: PR-AUC about 0.05-0.15; with past-only sender/receiver aggregates and time features about 0.2-0.4. The current 0.26 contains leaked features, so the leak-free value is likely lower. Reported edge-aware GNN results on HI-Small are said to be higher (F1 around 0.5-0.6; **unverified**, recalled from the literature, not checked).
- **Elliptic:** the original paper (train steps 1-34, test 35-49) is recalled as reporting about 0.78-0.80 illicit F1 for Random Forest and about 0.63 for a GCN (**both unverified**). Measured here (see `docs/baseline_results.md`), Random Forest and gradient boosting reach about 0.70-0.75 on steps 39-49. The GNN's 0.52 is likely below a trivial baseline; fixing `time_step` and the selection criterion might reach 0.60-0.70.

## 4. Cheap comparison experiments (to run in Phase 3, none run yet)

Common protocol: same masks, threshold tuned on validation (PR curve), applied once on test, five seeds, report PR-AUC, ROC-AUC, precision, recall, F1, precision@k (k = number of test positives).

- **AMLSim:** run both the existing random split and a temporal split (sort by `TIMESTAMP`, 70/15/15). Features: log amount, `amount < 20`, sender/receiver aggregates computed from training rows only (counts, sum/mean/std/min/max, distinct counterparties, share under 20, `is_new_account`). Models: HistGradientBoosting/LightGBM and RandomForest. If trees match the GNN on the random split, the GNN adds nothing; if both fall on the temporal split, the old numbers were split-driven.
- **IBM:** same chronological cut. Edge features: log paid/received amounts, one-hot format and currencies, mismatch and self-loop flags, hour/day-of-week. Past-only aggregates via sorted cumulative/rolling groupbys excluding the current row (sender/receiver and pair counts, distinct counterparties, time since last transaction, 1h/1d/7d windows, fan-in/out). Models: LightGBM (early stop on validation average precision), logistic regression as floor. Ablation: boosting with the current leaky aggregates vs past-only versions; the gap is the leakage. Then rerun the GNN with past-only features.
- **Elliptic:** (A) local features only (columns 2-94, no `time_step`); (B) all 165 non-time features; (C) local plus mean/max of neighbours' local features from an undirected adjacency, never neighbour labels. RandomForest(500, balanced_subsample) and HistGradientBoosting. Report per-step F1 for 39-49 and before/after step 43. One GNN control: current model without `time_step`, checkpoint chosen by validation PR-AUC.

## 5. Findings shared across the three models

1. Checkpoint selection on ROC-AUC (AMLSim, IBM) or F1 at 0.5 (Elliptic) instead of PR-AUC.
2. Fixed `0.05..0.95` threshold grid; IBM hits the edge, AMLSim (0.90) is near it. Train uses `>` and predict uses `>=`.
3. Large `pos_weight` inflates scores, so thresholds sit high and scores are not probabilities (matters for the dashboard wording).
4. No seeds or repeats anywhere; single-run metrics on small positive counts.
5. No tabular baseline anywhere.
6. Predictions CSVs mix train, validation and test rows without a split flag.
7. Full-graph transductive training; features and normalisation fitted on all rows (AMLSim and IBM confirmed; Elliptic only BatchNorm).
8. `AMLGraphSAGE` (AMLSim and IBM) has the feature bottleneck and ignores `edge_attr` in message passing.
9. Tests check shapes and plumbing, not leakage or split semantics.

## 6. Prioritised Phase 3 plan

Each step is a separate commit, and metrics will drop after the leakage fixes; that is the honest baseline. Nothing here is implemented; it needs approval first.

1. **Baselines first (no model change).** Add `src/baselines.py` running the Section 4 experiments on the current splits, to quantify how much of each score is leakage or trivial separability and whether the GNN beats trees. Highest information per unit of effort.
2. **Stop test-informed tuning.** Remove the `POS_WEIGHT_MULTIPLIER` justification based on test results; choose it (or a grid of it) on validation only, and record the selection in the metrics JSON.
3. **Fix selection and thresholds in all three train scripts.** Select checkpoints on validation PR-AUC; choose the threshold from `precision_recall_curve` on validation (no grid edge); use one comparison operator everywhere; set seeds and report mean +/- std over several seeds.
4. **Remove leakage in IBM, then AMLSim preprocessing.** Compute node aggregates, pair frequency and normalisation from the training window only; for evaluation windows use past-only values; restrict message passing at train time to training edges. Make the AMLSim split temporal on `TIMESTAMP`.
5. **Elliptic fixes.** Drop or scale `time_step`, report per-step metrics and the step-43 break, select on PR-AUC, compare against the Section 4 baselines.
6. **Add a split column to all prediction CSVs** (`train`/`val`/`test`) and make the dashboard default to the test split. (The dashboard work reads a split column from a UI-only file in the meantime; this moves it to the source.)
7. **Model improvements, only if baselines justify them.** Remove the `initial_lin` bottleneck, use an edge-aware conv (e.g. GINE/GATv2 with `edge_attr`), one-hot payment format, per-currency amount handling, temporal and structural features for IBM. Drop the constant `tx_type` in AMLSim.
8. **Tests for the above.** Assert no future information in features (perturb test-period rows and check training features are unchanged), split chronology for AMLSim, threshold selection, and seed reproducibility.
9. **Calibrate scores** (Platt/isotonic on validation) before presenting them as probabilities.

## 7. Caveats on this review

- The Elliptic claim that edges never cross time steps, the attribution of the step-43 drop to a dark-market shutdown, and every published baseline number (Elliptic RF/GCN, IBM edge-aware GNNs) come from recalled knowledge of the datasets and papers, not from this repo, and are **unverified**.
- The AMLSim and IBM reviewers read the other pipelines only partly; cross-model findings marked "confirmed" were checked in the code, the rest are inferred.
- No experiment in this document has been run.
