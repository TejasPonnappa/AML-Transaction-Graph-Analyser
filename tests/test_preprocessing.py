import pandas as pd
import pytest
import torch

import data_preprocessing_ibm as ibm
from ibm_features import FEATURE_NAMES


def _node_map(sender, receiver):
    accounts = pd.unique(pd.concat([pd.Series(sender), pd.Series(receiver)]))
    return {a: i for i, a in enumerate(accounts)}


def _assert_masks(data, n):
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    assert len(tr) == len(va) == len(te) == n
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    assert tr.sum() + va.sum() + te.sum() == n
    assert tr.any() and va.any() and te.any()


def _run_ibm(monkeypatch, tmp_path, csv):
    out = tmp_path / "out" / "g_ibm.pt"
    monkeypatch.setattr(ibm, "RAW_TRANS_PATH", str(csv))
    monkeypatch.setattr(ibm, "PROCESSED_DATA_PATH", str(out))
    ibm.preprocess_ibm_data()
    return torch.load(out, weights_only=False)


def test_ibm_edges_labels_and_attr(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    data = _run_ibm(monkeypatch, tmp_path, csv)
    nm = _node_map(df["Account"], df["Account.1"])
    assert data.edge_index.tolist() == [
        [nm[a] for a in df["Account"]],
        [nm[a] for a in df["Account.1"]],
    ]
    assert data.edge_attr.shape == (len(df), len(FEATURE_NAMES))
    assert data.y.tolist() == df["Is Laundering"].astype(float).tolist()
    assert torch.isfinite(data.x).all() and torch.isfinite(data.edge_attr).all()


def test_ibm_masks_cover_and_disjoint(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    _assert_masks(_run_ibm(monkeypatch, tmp_path, csv), len(df))


def test_ibm_split_is_chronological(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    data = _run_ibm(monkeypatch, tmp_path, csv)
    t = pd.to_datetime(df["Timestamp"]).to_numpy()
    tr, va, te = (t[m.numpy()] for m in (data.train_mask, data.val_mask, data.test_mask))
    assert tr.max() <= va.min()
    assert va.max() <= te.min()
    assert tr.max() <= te.min()
    assert len(tr) > len(va) and len(tr) > len(te)


def test_ibm_without_currency_columns_keeps_the_same_width(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    bad = tmp_path / "nocur.csv"
    df.drop(columns=["Payment Currency", "Receiving Currency"]).to_csv(bad, index=False)
    data = _run_ibm(monkeypatch, tmp_path, bad)
    assert data.edge_attr.shape == (len(df), len(FEATURE_NAMES))
    assert data.edge_feature_names.index('currency_mismatch') == 2
    assert torch.allclose(data.edge_attr[:, 2], torch.zeros(len(df)))


def test_ibm_requires_a_timestamp(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    bad = tmp_path / "nots.csv"
    df.drop(columns=["Timestamp"]).to_csv(bad, index=False)
    monkeypatch.setattr(ibm, "RAW_TRANS_PATH", str(bad))
    monkeypatch.setattr(ibm, "PROCESSED_DATA_PATH", str(tmp_path / "o" / "g.pt"))
    with pytest.raises(ValueError, match="Timestamp"):
        ibm.preprocess_ibm_data()


def test_ibm_message_passing_uses_training_edges_only(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    data = _run_ibm(monkeypatch, tmp_path, csv)
    assert data.mp_edge_index.shape[1] == int(data.train_mask.sum())
    assert data.mp_edge_index.tolist() == data.edge_index[:, data.train_mask].tolist()


def test_ibm_node_features_ignore_validation_and_test_rows(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    base = _run_ibm(monkeypatch, tmp_path, csv)
    changed = df.copy()
    held_out = (base.val_mask | base.test_mask).numpy()
    changed.loc[held_out, "Amount Paid"] = changed.loc[held_out, "Amount Paid"] * 1000 + 7
    alt = tmp_path / "changed.csv"
    changed.to_csv(alt, index=False)
    other = _run_ibm(monkeypatch, tmp_path, alt)
    assert torch.allclose(base.x, other.x)                       # node features come from training rows only
    # training rows only ever look at earlier rows, and the held-out rows are all later in time
    assert torch.allclose(base.edge_attr[base.train_mask], other.edge_attr[base.train_mask])


def test_ibm_missing_column(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    bad = tmp_path / "bad.csv"
    df.drop(columns=["Amount Paid"]).to_csv(bad, index=False)
    monkeypatch.setattr(ibm, "RAW_TRANS_PATH", str(bad))
    monkeypatch.setattr(ibm, "PROCESSED_DATA_PATH", str(tmp_path / "o" / "g.pt"))
    with pytest.raises(ValueError, match="Amount Paid"):
        ibm.preprocess_ibm_data()
