import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def make_graph(n_nodes=12, n_edges=40, x_dim=5, edge_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randn(n_nodes, x_dim, generator=g),
        edge_index=torch.randint(0, n_nodes, (2, n_edges), generator=g),
        edge_attr=torch.randn(n_edges, edge_dim, generator=g),
        y=torch.randint(0, 2, (n_edges,), generator=g).float(),
    )


@pytest.fixture
def tiny_graph():
    return make_graph()


def _accounts(rng, n, pool):
    return [f"ACC{i}" for i in rng.integers(0, pool, n)]


@pytest.fixture
def ibm_csv(tmp_path):
    """40 rows, unique shuffled timestamps, 10 laundering rows."""
    rng = np.random.default_rng(2)
    n = 40
    ts = pd.date_range("2022-09-01", periods=n, freq="h").strftime("%Y/%m/%d %H:%M").to_numpy()
    ts = ts[rng.permutation(n)]
    laund = np.zeros(n, dtype=int)
    laund[rng.choice(n, 10, replace=False)] = 1
    df = pd.DataFrame({
        "Timestamp": ts,
        "Account": _accounts(rng, n, 9),
        "Account.1": _accounts(rng, n, 9),
        "Amount Paid": rng.uniform(5, 5000, n).round(2),
        "Payment Currency": rng.choice(["US Dollar", "Euro"], n),
        "Receiving Currency": rng.choice(["US Dollar", "Euro"], n),
        "Payment Format": rng.choice(["Cheque", "ACH", "Wire", "Other"], n),
        "Is Laundering": laund,
    })
    path = tmp_path / "ibm.csv"
    df.to_csv(path, index=False)
    return path, df


@pytest.fixture
def load_script(tmp_path, monkeypatch):
    """Import a predict* module from a temp cwd containing its processed graph file."""
    loaded = []

    def _load(name, graph_rel_path, graph):
        monkeypatch.chdir(tmp_path)
        full = tmp_path / graph_rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, full)
        sys.modules.pop(name, None)
        mod = importlib.import_module(name)
        loaded.append(name)
        return mod

    yield _load
    for name in loaded:
        sys.modules.pop(name, None)
