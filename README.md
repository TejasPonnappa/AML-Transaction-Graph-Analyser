# AML Transaction Graph Analyser ("Inferno")

Graph neural network (GraphSAGE) models for financial-crime detection on three datasets, with a Streamlit dashboard to explore the results.

| Project | Task | Dataset | Unit scored |
|---|---|---|---|
| AMLSim | edge classification | `data/raw/amlsim_transactions.csv` (simulated bank transactions) | transaction |
| Elliptic | node classification | `data/raw/elliptic_txs_{features,edgelist,classes}.csv` (Bitcoin transaction graph) | transaction node |
| IBM AML | edge classification | `data/raw/ibm_aml/HI-Small_{Trans,accounts}.csv` | transaction |

Raw data is not stored in git (`data/raw/` is gitignored). Place the files at the paths above. The IBM and Elliptic datasets are available on Kaggle.

## Setup

```
pip install -r requirements.txt
```

`torch` and `torch_geometric` may need a platform-specific install; see their docs if `pip` fails.

## Usage

Run everything from the repository root, because paths in the scripts are relative to it. Each project is a three-step pipeline: preprocess, train, predict.

```
# IBM AML (swap the suffix: none for AMLSim, _bitcoin for Elliptic)
python src/data_preprocessing_ibm.py   # -> data/processed/graph_data_ibm.pt
python src/train_ibm.py                # -> outputs/model_checkpoint_ibm.pth, metrics_ibm.json, training_curves_ibm.png
python src/predict_ibm.py              # -> outputs/ibm_predictions.csv

streamlit run streamlit_app.py
```

AMLSim writes `outputs/suspicion_scores.csv`, `metrics.json` and `model_checkpoint.pth` (no suffix). Elliptic writes `outputs/bitcoin_predictions.csv` and `*_bitcoin.*`.

## How it works

- **Preprocessing** builds a PyTorch Geometric graph. Node features are per-account aggregates (degree, amounts sent and received). Edge features are log-amount, payment type and, for IBM, a currency-mismatch flag and pair frequency. IBM uses a chronological split and Elliptic a time-step split; AMLSim uses a stratified random split.
- **Model** (`src/model.py`) applies two `SAGEConv` layers, then an MLP on `[sender embedding, receiver embedding, edge features]` to score each transaction. Elliptic uses `src/model_bitcoin.py`, a node classifier with BatchNorm and a residual connection.
- **Training** (`src/training_utils.py`) uses a class-weighted loss, selects the checkpoint and stops early on validation PR-AUC, and takes the decision threshold from the validation precision-recall curve (max F1). Test data is used only for the final report. `--seeds N` trains N seeds, keeps the best on validation and records the spread in the metrics JSON. `python src/baselines.py <dataset>` runs tree-model baselines on the same splits (see `docs/baseline_results.md`).
- **Prediction** scores every row, including the training rows, and applies the saved threshold. The dashboard reads the CSV and metrics JSON from `outputs/` and takes its thresholds and headline metrics from the JSON.

## Dashboard

`streamlit_app.py` lets you pick a project in the sidebar. Each project has a global view (metrics, score distribution, leaderboards, interactive suspicious-network map) and a single-account deep dive with an ego-network graph.

## Caveats

- Predictions cover all rows, including the ones the model trained on, so the dashboard's lists and counts mix training and held-out data. The metrics JSON reports test-set results only.
- Node aggregates and normalisation are computed over the full dataset, so the evaluation is not leak-free. The AMLSim random split in particular makes its metrics look better than a time-based split would.
- There are no automated tests yet.
