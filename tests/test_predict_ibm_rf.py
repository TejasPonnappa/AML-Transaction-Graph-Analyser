import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import predict_ibm_rf as rf

N = 400


def _raw(tmp_path, labels):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2022-09-01", periods=N, freq="45min").strftime("%Y/%m/%d %H:%M")
    frame = pd.DataFrame(
        [ts, rng.choice(["010", "020"], N), [f"A{i}" for i in rng.integers(0, 15, N)], rng.choice(["010", "020"], N),
         [f"A{i}" for i in rng.integers(0, 15, N)], rng.uniform(1, 500, N).round(2),
         rng.choice(["US Dollar", "Euro"], N), rng.uniform(1, 500, N).round(2), rng.choice(["US Dollar", "Euro"], N),
         rng.choice(["ACH", "Cheque", "Wire"], N), labels]).T
    frame.columns = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account', 'Amount Received', 'Receiving Currency',
                     'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']     # duplicate header, as in the real file
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    return path


def _graph(tmp_path, labels):
    train = torch.arange(N) < 280
    val = (torch.arange(N) >= 280) & (torch.arange(N) < 340)
    test = torch.arange(N) >= 340
    path = tmp_path / "g.pt"
    torch.save(Data(y=torch.tensor(labels, dtype=torch.float), train_mask=train, val_mask=val, test_mask=test), path)
    return path


def _run(monkeypatch, tmp_path, labels):
    monkeypatch.setattr(rf, "RAW_TRANS_PATH", str(_raw(tmp_path, labels)))
    monkeypatch.setattr(rf, "GRAPH_PATH", str(_graph(tmp_path, labels)))
    monkeypatch.setattr(rf, "OUTPUT_SCORES_PATH", str(tmp_path / "o" / "p.csv"))
    monkeypatch.setattr(rf, "METRICS_PATH", str(tmp_path / "o" / "m.json"))
    monkeypatch.setattr(rf, "MODEL_PATH", str(tmp_path / "o" / "m.joblib"))
    metrics = rf.generate_rf_predictions(neg_frac=1.0, seed=0, n_estimators=10, log=lambda *_: None)
    return metrics, pd.read_csv(tmp_path / "o" / "p.csv")


@pytest.fixture
def labels():
    y = np.zeros(N, dtype=int)
    y[np.random.default_rng(1).choice(N, 60, replace=False)] = 1
    return y


def test_predictions_cover_every_row_with_a_split_and_the_saved_threshold(monkeypatch, tmp_path, labels):
    metrics, df = _run(monkeypatch, tmp_path, labels)
    assert len(df) == N
    assert df["SPLIT"].value_counts().to_dict() == {"train": 280, "val": 60, "test": 60}
    assert ((df["SUSPICION_SCORE"] >= 0) & (df["SUSPICION_SCORE"] <= 1)).all()
    expected = (df["SUSPICION_SCORE"] >= metrics["best_threshold"]).astype(int)
    assert df["PREDICTED_FRAUD"].tolist() == expected.tolist()
    assert {"SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TX_TYPE", "IS_FRAUD"} <= set(df.columns)
    assert metrics["n_features"] == len(metrics["feature_names"]) == 43
    assert {"pooled", "normal-volume days", "low-volume tail days"} <= set(metrics["regimes"])
    assert {"test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1_tuned_thresh",
            "confusion_matrix", "best_threshold"} <= set(metrics)       # keys the dashboard reads


def test_threshold_and_model_never_use_test_labels(monkeypatch, tmp_path, labels):
    base, _ = _run(monkeypatch, tmp_path, labels)
    flipped = labels.copy()
    flipped[340:] = 1 - flipped[340:]                                    # corrupt ONLY the test labels
    other, _ = _run(monkeypatch, tmp_path, flipped)
    assert base["best_threshold"] == other["best_threshold"]
    assert base["test_pr_auc"] != other["test_pr_auc"]                   # only the final report moves
    assert json.loads((tmp_path / "o" / "m.json").read_text())["best_threshold"] == other["best_threshold"]
