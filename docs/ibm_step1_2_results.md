# IBM: positive rate per split, the GNN's validation/test gap, and graph-structure features

All numbers are measured in this repo; nothing is from the literature.

## 1. Positive rate per split

| split | rows | laundering | rate | period |
|---|---|---|---|---|
| train | 3,554,841 | 2,856 | 0.080% | 2022-09-01 00:00 to 09-07 14:55 |
| validation | 761,752 | 760 | 0.100% | 09-07 14:55 to 09-09 03:16 |
| test | 761,752 | 1,561 | 0.205% | 09-09 03:16 to 09-18 16:18 |

The test rate is twice the validation rate, and the reason is concentrated in the last eight days. Transaction volume collapses from about 480k per day to a few hundred per day from 2022-09-11, while laundering continues:

| test segment | rows | laundering | rate |
|---|---|---|---|
| Sep 9-10 (normal volume) | 760,644 | 906 | 0.119% |
| Sep 11-18 (low-volume tail) | 1,108 | 655 | 59.1% |

So **42% of all test positives sit in 1,108 rows** (0.15% of the test set) where the base rate is about 59%. Validation contains none of this tail.

## 1b. Why the GNN's validation and test PR-AUC differ (0.138 vs 0.246)

It is mostly composition, not a better model on test:

- PR-AUC scales with prevalence. Lift over the base rate went **down**, from about 139x (validation) to about 120x (test). ROC-AUC is flat (0.983 validation, 0.986 test).
- On the GNN's existing predictions: the low-volume tail has PR-AUC 0.940 (precision 0.946, recall 0.530 at its threshold); the **normal-volume test days have PR-AUC 0.120**, which is below the validation 0.138. On the part of test that looks like validation, the model is not better.
- The validation-tuned threshold (0.993) never saw the tail regime, so the tail is both easier to score and dominated by the base rate.

Consequence: **every pooled IBM test metric reported so far is inflated by the tail.** From here the normal-volume days are reported separately, with the same validation-tuned threshold.

## 2. Graph-structure features for the RandomForest

New past-only features (13, appended to the 30 shared features, 43 in total). Each uses strictly earlier timestamps only and never labels:
- distinct receivers the sender has used (all-time, new in the last hour, new in the last day);
- distinct senders into the receiver (same three), distinct senders into the sender, distinct receivers of the receiver;
- 2-hop chains (x to y to s) ending at the sender and at the receiver;
- reverse-pair history (earlier receiver-to-sender transactions: count, seconds since last);
- the number of 3-cycles the transaction closes.

Leak tests (`tests/test_ibm_features.py`): altering every later transaction leaves all earlier rows' features unchanged; legs of a cycle that arrive in the same minute are not counted as earlier; hand-checked cycle, distinct-counterparty and reverse-pair cases; features are invariant to input row order.

### Results (test, 3 seeds, mean +/- std; thresholds tuned on full validation, all test rows evaluated)

| model | features | pooled PR-AUC | pooled F1 | normal-volume PR-AUC | normal-volume F1 (precision / recall) | tail PR-AUC |
|---|---|---|---|---|---|---|
| RandomForest | 30 past-only | 0.537 +/- 0.007 | 0.528 | 0.286 +/- 0.008 | 0.348 (0.291 / 0.436) | 0.971 |
| **RandomForest** | **43 (+ graph)** | **0.663 +/- 0.005** | **0.639** | **0.433 +/- 0.008** | **0.457 (0.499 / 0.425)** | 0.970 |
| HistGradientBoosting | 30 past-only | 0.375 +/- 0.085 | 0.442 | 0.187 +/- 0.051 | 0.275 | 0.967 |
| HistGradientBoosting | 43 (+ graph) | 0.523 +/- 0.028 | 0.513 | 0.287 +/- 0.025 | 0.333 | 0.959 |
| GNN, existing (30 features) | one run | 0.246 | 0.322 | 0.120 | about 0.21 (0.145 / 0.379) | 0.940 |

(The GNN's normal-volume F1 is derived from its printed precision and recall; the other GNN figures are computed from `ibm_predictions.csv`.)

### Do the graph features help?

Yes, clearly, where it counts. On normal-volume days the RandomForest's PR-AUC goes from 0.286 to 0.433 and F1 from 0.348 to 0.457, with a seed spread of under 0.01. HistGradientBoosting gains by about the same margin. The tail is already near its ceiling (about 0.97) and does not change, so the gain is entirely in the regime that resembles real operation.

## Caveats

- No per-feature ablation was run, so I cannot say which of the 13 features helps.
- Trees are fit on all positives plus 10% of training negatives (weighted back); validation and test use every row. The forest is untuned.
- The tail regime has only 1,108 rows (655 positives) from eight days of a simulation; treat its numbers as descriptive, not as a performance estimate.
- The GNN row is a single run on the earlier 30-feature graph, not retrained with the new features.
