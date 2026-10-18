import json

import pandas as pd
import pytest
import torch

from conftest import make_graph
from model import AMLGraphSAGE

N_EDGES = 12


@pytest.fixture
def env(load_script, tmp_path, monkeypatch):
    graph = make_graph(n_nodes=8, n_edges=N_EDGES, x_dim=5, edge_dim=4)
    mod = load_script("predict_ibm", "data/processed/graph_data_ibm.pt", graph)
    model = AMLGraphSAGE(mod.IN_CHANNELS_X, mod.IN_CHANNELS_EDGE,
                         mod.HIDDEN_CHANNELS, mod.OUT_CHANNELS_NODE)
    ckpt = tmp_path / "ckpt.pth"
    torch.save(model.state_dict(), ckpt)
    out = tmp_path / "outputs" / "preds.csv"
    metrics = tmp_path / "metrics.json"
    monkeypatch.setattr(mod, "MODEL_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setattr(mod, "OUTPUT_SCORES_PATH", str(out))
    monkeypatch.setattr(mod, "METRICS_PATH", str(metrics))

    def write_raw(n):
        raw = tmp_path / f"raw_{n}.csv"
        pd.DataFrame({
            "Account": [f"A{i}" for i in range(n)],
            "Account.1": [f"B{i}" for i in range(n)],
            "Amount Paid": [10.0 + i for i in range(n)],
            "Payment Format": ["ACH"] * n,
            "Is Laundering": [i % 2 for i in range(n)],
        }).to_csv(raw, index=False)
        monkeypatch.setattr(mod, "RAW_DATA_PATH", str(raw))

    return mod, out, metrics, write_raw


@pytest.mark.parametrize("n", [N_EDGES - 1, N_EDGES + 1])
def test_row_count_mismatch_raises(env, n):
    mod, out, _, write_raw = env
    write_raw(n)
    with pytest.raises(ValueError, match="Row count mismatch"):
        mod.generate_predictions()
    assert not out.exists()


@pytest.mark.parametrize("threshold", [0.0, 0.5, 2.0])
def test_predicted_fraud_uses_saved_threshold(env, threshold):
    pytest.importorskip("tabulate")
    mod, out, metrics, write_raw = env
    write_raw(N_EDGES)
    metrics.write_text(json.dumps({"best_threshold": threshold}))
    mod.generate_predictions()
    df = pd.read_csv(out)
    assert len(df) == N_EDGES
    assert {"SUSPICION_SCORE", "PREDICTED_FRAUD", "IS_FRAUD"} <= set(df.columns)
    assert ((df["SUSPICION_SCORE"] >= 0) & (df["SUSPICION_SCORE"] <= 1)).all()
    expected = (df["SUSPICION_SCORE"] >= threshold).astype(int)
    assert df["PREDICTED_FRAUD"].tolist() == expected.tolist()
    if threshold == 0.0:
        assert df["PREDICTED_FRAUD"].all()
    if threshold == 2.0:
        assert not df["PREDICTED_FRAUD"].any()
