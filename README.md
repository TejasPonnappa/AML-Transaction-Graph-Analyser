# AML Transaction Graph Analyser

Scores bank and Bitcoin transactions for money-laundering and illicit-activity alerts, and lets an analyst investigate any account as a network: its counterparties, the patterns around it, and why the model flagged it.

| Dataset | Data | Model of record |
|---|---|---|
| IBM AML (HI-Small) | 5.08M public synthetic bank transactions, 5,177 laundering | RandomForest on 43 past-only features |
| Elliptic | 203k real Bitcoin transactions, 4,545 illicit | GraphSAGE (tree models are stronger, see below) |

## What was found

- **Features that only see the past.** Every history feature uses strictly earlier transactions (tests perturb later rows to prove it). Graph-structure features (new counterparties per window, 2-hop chains, reverse pairs, time-aware 3-cycles) lift the IBM RandomForest's PR-AUC on normal-volume days from 0.29 to 0.43.
- **A benchmark artifact.** In the IBM data 42% of test laundering sits in 1,108 low-volume rows (about 59% laundering). Pooled metrics (PR-AUC 0.66) are inflated by it; on normal-volume days the RandomForest gets PR-AUC 0.43, precision 0.54, recall 0.40. The dashboard defaults to normal-volume days.
- **A GNN did not beat the baselines.** On identical past-only inputs, GraphSAGE reaches PR-AUC 0.23 against the forest's 0.54. On Elliptic the GNN's F1 is 0.37 against about 0.75 for the tree models, and every model fails from time step 43.
- **Analyst workload (IBM, test, normal-volume days).** Reviewing the top 1% of accounts by score catches 54.9% of laundering accounts; the top 0.1% catches 19.9% at 79.5% precision.
- **Detectors and schemes.** The model recalls 42-76% of each labelled laundering scheme but only 3% of the 38% of laundering rows the benchmark does not attribute to a scheme. Simple rule-based detectors are weak; only fan-out and cycle are precise enough to show as badges.

Details and caveats: `docs/ibm_step1_2_results.md`, `docs/ibm_step4_results.md`, `docs/ibm_explain_results.md`, `docs/elliptic_results.md`, `docs/baseline_results.md` and `docs/model_review.md`.

## Dashboard

`streamlit run streamlit_app.py`. For each dataset: a KPI row and views for Overview (largest alert clusters as a network), Alerts (sortable, CSV export), Investigate (account search, 2-hop network, why a transaction was flagged) and Evaluation. IBM also has "How it decides": a depth-8 tree that imitates the forest, with a token travelling the path of the transaction you pick. Scores are model scores, not calibrated probabilities.

## Run it

Raw data goes in `data/raw/` (not in git): `ibm_aml/HI-Small_Trans.csv`, `ibm_aml/HI-Small_Patterns.txt`, and the three `elliptic_txs_*.csv` files. Run from the repo root.

```
pip install -r requirements.txt
# IBM
python src/data_preprocessing_ibm.py     # chronological split, past-only graph for the GNN comparison
python src/predict_ibm_rf.py             # fit the forest, tune the threshold on validation, score all rows
python src/explain_ibm.py                # surrogate tree, forest votes, scheme and workload metrics
# Elliptic
python src/data_preprocessing_bitcoin.py && python src/train_bitcoin.py --seeds 3 && python src/predict_bitcoin.py
# comparisons
python src/baselines.py ibm --neg-frac 0.1 --ibm-features past_graph     # also: elliptic
python src/data_preprocessing_ibm.py && python src/train_ibm.py --seeds 3   # IBM GraphSAGE
python -m pytest -q
```

## Layout

- `src/ibm_features.py`, `ibm_patterns.py`, `ibm_detectors.py`: past-only features, benchmark scheme labels, pattern detectors.
- `src/predict_ibm_rf.py`, `explain_ibm.py`: the IBM model of record and its explanations. `baselines.py`: tree baselines on the GNN splits.
- `src/train_ibm.py`, `train_bitcoin.py`, `training_utils.py`, `model*.py`: the GNNs and the shared training rules (validation PR-AUC selection, threshold from the validation precision-recall curve, test used only for the report).
- `ui/`: Streamlit pages (`ibm_dashboard.py`, `elliptic_dashboard.py`), `components/` (Cytoscape network and D3 tree, vendored JS) and the data layer (`data_store.py`, a Parquet cache in `outputs/.graph_cache`).
- `tests/`: pytest suite on synthetic data, including no-future-leakage tests. CI runs it on every push.
