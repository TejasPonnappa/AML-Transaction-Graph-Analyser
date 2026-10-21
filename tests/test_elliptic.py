import numpy as np
import pandas as pd
import pytest
import torch

import data_preprocessing_bitcoin as btc
from train_bitcoin import SHUTDOWN_STEP, time_step_report

N_FEATURES = 6


@pytest.fixture
def elliptic_files(tmp_path):
    rng = np.random.default_rng(0)
    n = 98
    steps = np.repeat(np.arange(1, 50), 2)                     # two nodes per time step 1..49
    ids = 1000 + np.arange(n)
    feats = pd.DataFrame(np.column_stack([ids, steps, rng.normal(size=(n, N_FEATURES))]))
    feats.to_csv(tmp_path / "features.csv", header=False, index=False)
    pd.DataFrame({"txId1": ids[:-1], "txId2": ids[1:]}).to_csv(tmp_path / "edges.csv", index=False)
    labels = rng.choice(["1", "2", "unknown"], n)
    pd.DataFrame({"txId": ids, "class": labels}).to_csv(tmp_path / "classes.csv", index=False)
    return tmp_path, steps, labels


def _run(monkeypatch, files):
    path = files[0]
    monkeypatch.setattr(btc, "FEATURES_PATH", str(path / "features.csv"))
    monkeypatch.setattr(btc, "EDGES_PATH", str(path / "edges.csv"))
    monkeypatch.setattr(btc, "CLASSES_PATH", str(path / "classes.csv"))
    monkeypatch.setattr(btc, "PROCESSED_DATA_PATH", str(path / "out" / "g.pt"))
    btc.preprocess_bitcoin_data()
    return torch.load(path / "out" / "g.pt", weights_only=False)


def test_time_step_is_not_a_model_input_but_is_kept_for_reporting(monkeypatch, elliptic_files):
    data = _run(monkeypatch, elliptic_files)
    assert data.x.size(1) == N_FEATURES                                  # txId and time step both dropped
    assert data.time_step.tolist() == elliptic_files[1].tolist()


def test_split_follows_time_steps_and_ignores_unlabelled_nodes(monkeypatch, elliptic_files):
    data = _run(monkeypatch, elliptic_files)
    steps = data.time_step
    assert (steps[data.train_mask] <= 34).all()
    assert ((steps[data.val_mask] > 34) & (steps[data.val_mask] <= 38)).all()
    assert (steps[data.test_mask] > 38).all()
    assert not (data.train_mask & data.val_mask).any() and not (data.val_mask & data.test_mask).any()
    assert (data.y[data.train_mask | data.val_mask | data.test_mask] != -1).all()


def test_time_step_report_splits_before_and_from_step_43():
    steps = np.array([40, 41, 42, 43, 44, 49] * 10)
    y = np.array([1, 0, 1, 1, 0, 1] * 10)
    prob = np.where(y == 1, 0.9, 0.1)
    prob[steps >= SHUTDOWN_STEP] = 0.0                                   # the model "fails" from step 43 on
    per_step, regimes = time_step_report(np.arange(60), prob, y, steps, threshold=0.5)
    before, after = list(regimes.values())
    assert before['f1'] == 1.0 and after['recall'] == 0.0
    assert before['rows'] == 30 and after['rows'] == 30
    assert set(per_step) == {40, 41, 42, 43, 44, 49}
