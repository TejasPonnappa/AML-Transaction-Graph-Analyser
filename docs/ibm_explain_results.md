# IBM: laundering schemes, pattern detectors and analyst workload

Measured with `python src/explain_ibm.py` on the RandomForest model of record (43 past-only features, alert threshold 0.6255 tuned on validation). Test period, **normal-volume days** unless stated (2022-09-09 to 09-10; the last eight days of the data are a low-volume tail with about 59% laundering and are excluded, see `ibm_step1_2_results.md`). Nothing here comes from the literature.

## Benchmark scheme labels

`HI-Small_Patterns.txt` lists 370 laundering attempts in eight schemes. 3,209 of its transaction lines match rows of the transactions file; the other **1,968 of 5,177 laundering rows (38%) are listed in no attempt**. They are called UNATTRIBUTED below. Scheme labels are benchmark ground truth, used for evaluation and shown in the Investigate view as a benchmark label; the monitoring system itself never sees them.

## Does the model catch each scheme? (recall at the alert threshold)

| scheme | test rows (normal days) | recall | recall (pooled test) |
|---|---|---|---|
| scatter-gather | 106 | 0.755 | 0.800 |
| fan-out | 63 | 0.714 | 0.829 |
| fan-in | 56 | 0.661 | 0.770 |
| cycle | 52 | 0.654 | 0.667 |
| gather-scatter | 125 | 0.624 | 0.784 |
| bipartite | 40 | 0.575 | 0.468 |
| stack | 81 | 0.420 | 0.456 |
| random | 41 | 0.415 | 0.500 |
| **unattributed** | 342 | **0.029** | 0.029 |

The model finds most laundering that belongs to a labelled scheme but almost none of the unattributed laundering. Stack, random and bipartite are the weakest schemes. I have not investigated what the unattributed rows are; they are 38% of all laundering rows, so this is the largest gap.

## Rule-based pattern detectors

Four detectors (fan-out, fan-in, cycle, pass-through) turn one past-only signal into a flag. Thresholds are tuned on the training period only (best F0.5 against the benchmark scheme labels the detector targets).

| detector | threshold | flagged (test, normal days) | precision against any laundering | recall on its own schemes | validation precision |
|---|---|---|---|---|---|
| fan-out | 3 new receivers in a day | 150 | 0.453 | 0.413 | 0.435 |
| cycle | 2 closed 3-cycles | 39 | 0.231 | 0.000 | 0.619 |
| fan-in | 4 new senders in a day | 10,580 | 0.007 | 0.125 | 0.007 |
| pass-through | closeness 0.944 | 5,925 | 0.002 | 0.013 | 0.003 |

Only **fan-out and cycle** are precise enough to be shown as badges (the dashboard shows a detector only if its validation precision is at least 20%). Fan-in and pass-through as defined here are noise: they flag thousands of ordinary transactions. The cycle detector is precise but rare, so it catches almost nothing. The forest, which sees all of these signals together, recalls far more than any single rule. The fan-out threshold gives 0.1% precision on the training period because every counterparty looks new on the first day of the data; it works on validation and test.

## Surrogate decision tree

A depth-8 tree (113 nodes) fitted on validation rows to imitate the forest's alert decisions. On the test set it agrees with the forest on 99.9% of rows, but that is dominated by non-alerts: of the forest's alerts it also flags **77%** (normal days: 70%), and 61% of its own alerts are also forest alerts (normal days: 47%). It is an approximation for illustration and can disagree with the forest on individual transactions; the dashboard shows the forest's own vote counts next to it.

## Analyst workload (account level, test period, normal-volume days)

An account's score is the highest score among its transactions; a laundering account takes part in at least one laundering transaction. There are 332,186 active accounts, 1,327 of them laundering.

| accounts reviewed (highest score first) | accounts | laundering accounts caught | share of all laundering accounts | precision of the list |
|---|---|---|---|---|
| top 0.1% | 332 | 264 | 19.9% | 79.5% |
| top 0.5% | 1,661 | 573 | 43.2% | 34.5% |
| **top 1%** | 3,322 | 728 | **54.9%** | 21.9% |
| top 2% | 6,644 | 946 | 71.3% | 14.2% |
| top 5% | 16,609 | 1,115 | 84.0% | 6.7% |

At the alert threshold the model raises about 330 alerts per day on normal-volume days. Transaction-level, reaching 50% recall needs about 1,223 alerts (37% precision) and 80% recall about 9,683 alerts (7.5% precision). The "per day" figures use two partial days and should be read as rough.

## Caveats

- The benchmark is synthetic; the schemes and the 38% unattributed share are properties of its generator.
- Scheme and workload results cover a short test window (two normal-volume days, 1,327 laundering accounts).
- Detector thresholds were tuned against the same benchmark labels used to score them, on the training period.
