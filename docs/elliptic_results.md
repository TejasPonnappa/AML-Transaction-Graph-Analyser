# Elliptic: GNN retrained without the time step, results before and after step 43

Measured in this repo; nothing here comes from the literature. Test set: 12,367 labelled transactions (717 illicit) in time steps 39-49, evaluated at a threshold tuned on validation only.

## What changed from the earlier GNN run

- The raw time step is no longer a model input (it was column 1 of the feature file). It is kept only for the split and reporting.
- Checkpoints and early stopping now use validation PR-AUC (before: F1 at 0.5), the threshold comes from the validation precision-recall curve (before: a 0.05 grid), and 3 seeds are trained and the best on validation is kept.

## Overall test results

| model | PR-AUC | precision | recall | F1 |
|---|---|---|---|---|
| GNN before (time step as a feature, selection on F1 at 0.5) | 0.470 | 0.727 | 0.400 | 0.516 |
| **GNN after (no time step; selected seed, 3 seeds)** | **0.382** (0.370 +/- 0.009 over seeds) | 0.277 | 0.559 | 0.370 |
| RandomForest, 165 features | 0.699 +/- 0.001 | 0.955 | 0.615 | 0.748 |
| HistGradientBoosting, 165 features | 0.714 +/- 0.002 | 0.964 | 0.614 | 0.750 |

**My earlier expectation was wrong.** I had suggested the raw time step was hurting the GNN and that removing it, together with better selection, might lift it to about 0.60-0.70 F1. It got worse, not better (F1 0.52 to 0.37). Three things changed at once (no time step, PR-AUC selection, a new threshold rule), so I cannot say which one caused the drop. What is clear is that the GNN trails the tree models by a wide margin either way.

## Before and from step 43 (test, same thresholds)

Steps 39-42 contain 548 illicit transactions and steps 43-49 contain 169.

| model | steps 39-42: PR-AUC / precision / recall / F1 | steps 43-49: PR-AUC / precision / recall / F1 |
|---|---|---|
| GNN (no time step) | 0.604 / 0.478 / 0.717 / 0.573 | 0.040 / 0.013 / 0.047 / 0.020 |
| RandomForest, 165 features | 0.888 / 0.989 / 0.799 / 0.884 | 0.042 / 0.162 / 0.018 / 0.032 |
| HistGradientBoosting, 165 features | 0.891 / 0.989 / 0.797 / 0.883 | 0.055 / 0.224 / 0.018 / 0.033 |
| RandomForest, local features only | 0.885 / 0.982 / 0.747 / 0.849 | 0.031 / 0.111 / 0.014 / 0.025 |
| RandomForest, local + neighbour mean | 0.890 / 0.983 / 0.784 / 0.872 | 0.045 / 0.127 / 0.016 / 0.028 |

Per-step F1 of the GNN: 0.63, 0.50, 0.51, 0.62 for steps 39-42, then 0.00 to 0.08 for steps 43-49.

## Reading it

1. **Every model fails from step 43.** The tree models score 0.85-0.88 F1 before step 43 and about 0.03 after; the GNN goes from 0.57 to 0.02. The collapse is a property of the test period, not of one model. Why it happens (a dark-market shutdown has been suggested) is unverified here.
2. **Before step 43 the trees are far ahead** (F1 about 0.88 against 0.57), and neighbour features add nothing over the provided features (0.872 against 0.884).
3. Pooled numbers mix a usable period (548 illicit) with an unusable one (169 illicit), which is why they look mediocre. The dashboard's Evaluation view shows both segments.

## Caveats

- The GNN's three seeds vary little (PR-AUC 0.363-0.382), but the comparison with the earlier run is confounded as described above.
- A GNN control with only the time step removed (selection and threshold as before) would isolate that change; it has not been run.
- Thresholds were tuned on time steps 35-38, which come before the break, so they are too high for the later period for every model.
