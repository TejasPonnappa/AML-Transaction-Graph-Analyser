import pytest
import torch

from conftest import make_graph
from model import AMLGraphSAGE
from model_bitcoin import BitcoinGraphSAGE


@pytest.mark.parametrize("edge_dim", [2, 4])
def test_aml_graphsage_shapes_and_range(edge_dim):
    g = make_graph(n_nodes=10, n_edges=25, x_dim=5, edge_dim=edge_dim)
    model = AMLGraphSAGE(5, edge_dim, 8, 4).eval()
    with torch.no_grad():
        raw, score = model(g)
    assert raw.shape == (25,) and score.shape == (25,)
    assert ((score >= 0) & (score <= 1)).all()
    assert torch.allclose(score, torch.sigmoid(raw))


def test_aml_graphsage_rejects_wrong_edge_width():
    g = make_graph(edge_dim=4)
    model = AMLGraphSAGE(5, 2, 8, 4).eval()
    with pytest.raises(RuntimeError):
        model(g)


def test_bitcoin_graphsage_shapes_and_range():
    g = make_graph(n_nodes=16, n_edges=40, x_dim=6)
    model = BitcoinGraphSAGE(6, 8, 4).eval()
    with torch.no_grad():
        raw, score = model(g)
    assert raw.shape == (16, 2)
    assert score.shape == (16,)
    assert ((score >= 0) & (score <= 1)).all()
    assert torch.allclose(score, torch.softmax(raw, dim=1)[:, 1])
