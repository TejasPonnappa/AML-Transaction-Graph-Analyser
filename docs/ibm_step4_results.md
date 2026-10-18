# IBM step 4: removing future information, before and after

All numbers are measured in this repo. Nothing here comes from the literature. Test set: 761,752 transactions, 1,561 laundering, evaluated on every row (no negative sampling) at a threshold tuned on the full validation set.

## What changed

- **Features:** one past-only feature set (30 columns, `src/ibm_features.py`) used by BOTH the GNN (as `edge_attr`) and the tree baselines. Every history feature uses transactions with a strictly earlier timestamp and never labels: prior counts and sums per role, seconds since last transaction, 1-hour and 1-day counts, and pair history. Static columns: log paid and received amount, currency mismatch, self-loop, bank mismatch, hour, day of week, payment format one-hot.
- **GNN graph:** node features from training-period edges only; edge features standardised on training statistics; message passing restricted to training edges (`mp_edge_index`), so validation and test edges never reach an embedding.
- **Selection:** checkpoint and threshold chosen on validation only (PR-AUC and the PR curve); 3 seeds; `POS_WEIGHT_MULTIPLIER` back at 1.0 (the earlier 0.5 had been chosen with test results in view).
- **Predictions:** `outputs/ibm_predictions.csv` regenerated from the new checkpoint with a `SPLIT` column; its test-row confusion matrix matches the metrics file exactly (TP 690, FP 2038).

## Before and after (test set)

| model | features / setup | PR-AUC | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|
| GNN, before | whole-timeline aggregates, full-graph message passing, multiplier 0.5, threshold 0.95 (grid edge) | 0.257 | 0.990 | 0.223 | 0.693 | 0.337 |
| GNN, after (selected seed 1) | past-only, training-edge message passing | 0.246 | 0.986 | 0.253 | 0.442 | 0.322 |
| GNN, after, mean +/- std of 3 seeds | same | 0.226 +/- 0.017 | 0.985 | 0.240 | 0.430 | 0.308 +/- 0.015 |
| HistGradientBoosting, before | frozen training-period aggregates | 0.307 +/- 0.015 | 0.987 | 0.344 | 0.340 | 0.342 |
| RandomForest, before | same | 0.323 +/- 0.007 | 0.985 | 0.404 | 0.297 | 0.341 |
| HistGradientBoosting, after | the same 30 past-only features as the GNN | 0.375 +/- 0.085 | 0.989 | 0.343 | 0.622 | 0.442 +/- 0.017 |
| RandomForest, after | same | **0.537 +/- 0.007** | 0.986 | 0.482 | 0.587 | **0.528 +/- 0.014** |

Per-seed GNN results: seed 0: val PR-AUC 0.130, test PR-AUC 0.228; seed 1: 0.138 / 0.246 (selected on validation); seed 2: 0.129 / 0.203. Training stopped early at epochs 104, 113 and 65.

## Reading it

1. **Removing future information did not hurt the trees; it helped them a lot.** History features (recency, burst counts, pair history) carry real signal: RandomForest PR-AUC went from 0.32 to 0.54 and F1 from 0.34 to 0.53.
2. **The GNN fell slightly (PR-AUC 0.257 to 0.246 for the selected seed, 0.226 on average), and on identical inputs it is far behind the RandomForest** (0.23 vs 0.54). The GNN does not exploit these features better than a forest, and message passing over training-period edges adds nothing visible here.
3. **The before/after GNN change mixes several effects.** Leakage removal, the pos-weight multiplier (0.5 to 1.0), PR-AUC-based selection and the new threshold rule all changed together, so I cannot say how much of the GNN's change is due to leakage alone. The leaky-aggregate ablation for trees, which would isolate that, has not been run.
4. **Validation looks harder than test for the GNN** (val PR-AUC 0.13-0.14 versus test 0.20-0.25), so thresholds picked on validation are at 0.993 and trade recall (0.44) for precision. HistGradientBoosting is noisy across seeds (PR-AUC 0.26 to 0.47), because it is fit on a random 10% negative sample without early stopping.
5. **The GNN no longer beats trees on any IBM metric that matters here**, even with identical inputs.

## Caveats

- Trees fit on all positives plus 10% of training negatives (weighted back); validation and test use every row.
- The GNN is a plain GraphSAGE with edge features only in the prediction head; I did not tune architecture, learning rate or epochs. A stronger graph model could still do better; this only shows the current one does not.
- Same-minute transactions are treated as simultaneous (not visible to each other), which is conservative.
- AMLSim and Elliptic were not retrained and still use their previous leakage-prone preprocessing.
