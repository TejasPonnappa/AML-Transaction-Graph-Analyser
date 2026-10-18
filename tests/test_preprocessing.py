import pandas as pd
import pytest
import torch

import data_preprocessing as amlsim
import data_preprocessing_ibm as ibm


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


def _run_amlsim(monkeypatch, tmp_path, csv):
    out = tmp_path / "out" / "g.pt"
    monkeypatch.setattr(amlsim, "RAW_DATA_PATH", str(csv))
    monkeypatch.setattr(amlsim, "PROCESSED_DATA_PATH", str(out))
    amlsim.preprocess_amlsim_data()
    return torch.load(out, weights_only=False)


def _run_ibm(monkeypatch, tmp_path, csv):
    out = tmp_path / "out" / "g_ibm.pt"
    monkeypatch.setattr(ibm, "RAW_TRANS_PATH", str(csv))
    monkeypatch.setattr(ibm, "PROCESSED_DATA_PATH", str(out))
    ibm.preprocess_ibm_data()
    return torch.load(out, weights_only=False)


def test_amlsim_edges_labels_and_attr(monkeypatch, tmp_path, amlsim_csv):
    csv, df = amlsim_csv
    data = _run_amlsim(monkeypatch, tmp_path, csv)
    nm = _node_map(df["sender-account-id"], df["receiver-account-id"])
    assert data.edge_index.tolist() == [
        [nm[a] for a in df["sender-account-id"]],
        [nm[a] for a in df["receiver-account-id"]],
    ]
    assert data.edge_attr.shape == (len(df), 2)
    assert data.y.tolist() == df["is-fraud"].astype(float).tolist()
    assert data.x.shape[0] == len(nm)
    assert torch.isfinite(data.x).all() and torch.isfinite(data.edge_attr).all()


def test_amlsim_masks(monkeypatch, tmp_path, amlsim_csv):
    csv, df = amlsim_csv
    _assert_masks(_run_amlsim(monkeypatch, tmp_path, csv), len(df))


def test_amlsim_missing_column(monkeypatch, tmp_path, amlsim_csv):
    csv, df = amlsim_csv
    bad = tmp_path / "bad.csv"
    df.drop(columns=["tx-type"]).to_csv(bad, index=False)
    monkeypatch.setattr(amlsim, "RAW_DATA_PATH", str(bad))
    monkeypatch.setattr(amlsim, "PROCESSED_DATA_PATH", str(tmp_path / "o" / "g.pt"))
    with pytest.raises(ValueError, match="TX_TYPE"):
        amlsim.preprocess_amlsim_data()


def test_ibm_edges_labels_and_attr(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    data = _run_ibm(monkeypatch, tmp_path, csv)
    nm = _node_map(df["Account"], df["Account.1"])
    assert data.edge_index.tolist() == [
        [nm[a] for a in df["Account"]],
        [nm[a] for a in df["Account.1"]],
    ]
    assert data.edge_attr.shape == (len(df), 4)
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


def test_ibm_without_currency_columns_keeps_width_4(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    bad = tmp_path / "nocur.csv"
    df.drop(columns=["Payment Currency", "Receiving Currency"]).to_csv(bad, index=False)
    data = _run_ibm(monkeypatch, tmp_path, bad)
    assert data.edge_attr.shape == (len(df), 4)
    assert torch.allclose(data.edge_attr[:, 2], torch.zeros(len(df)))


def test_ibm_without_timestamp_falls_back_to_random_split(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    bad = tmp_path / "nots.csv"
    df.drop(columns=["Timestamp"]).to_csv(bad, index=False)
    _assert_masks(_run_ibm(monkeypatch, tmp_path, bad), len(df))


def test_ibm_missing_column(monkeypatch, tmp_path, ibm_csv):
    csv, df = ibm_csv
    bad = tmp_path / "bad.csv"
    df.drop(columns=["Amount Paid"]).to_csv(bad, index=False)
    monkeypatch.setattr(ibm, "RAW_TRANS_PATH", str(bad))
    monkeypatch.setattr(ibm, "PROCESSED_DATA_PATH", str(tmp_path / "o" / "g.pt"))
    with pytest.raises(ValueError, match="Amount Paid"):
        ibm.preprocess_ibm_data()
