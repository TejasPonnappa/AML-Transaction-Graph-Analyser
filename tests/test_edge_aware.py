import numpy as np
import pandas as pd
import torch

from conftest import make_graph
from eval_ibm_regimes import regime_report
from model import EdgeAwareGNN
from training_utils import run_training


def _graph(train_edges=30):
    data = make_graph(n_nodes=12, n_edges=40, x_dim=5, edge_dim=6)
    mask = torch.zeros(40, dtype=torch.bool)
    mask[:train_edges] = True
    data.mp_edge_index = data.edge_index[:, mask]
    data.mp_edge_mask = mask
    return data


def _model():
    torch.manual_seed(0)
    return EdgeAwareGNN(5, 6, 8, 4).eval()


def test_scores_for_all_edges_or_a_subset_agree():
    data, model = _graph(), _model()
    with torch.no_grad():
        raw_all, p_all = model(data)
        idx = torch.tensor([3, 17, 39])
        raw_sub, p_sub = model(data, idx)
    assert raw_all.shape == (40,) and p_all.min() >= 0 and p_all.max() <= 1
    assert torch.allclose(raw_all[idx], raw_sub, atol=1e-6)


def test_message_passing_ignores_held_out_edges():
    data, model = _graph(train_edges=30), _model()
    with torch.no_grad():
        before = model(data)[0].clone()
        data.edge_attr[30:] += 100.0                  # perturb ONLY edges outside the message-passing set
        after = model(data)[0]
    assert torch.allclose(before[:30], after[:30], atol=1e-5)      # training edges' scores cannot move
    assert not torch.allclose(before[30:], after[30:])             # the perturbed edges' own scores do


def test_edge_attributes_reach_the_node_embeddings():
    data, model = _graph(), _model()
    with torch.no_grad():
        before = model(data)[0].clone()
        data.edge_attr[:30] += 5.0                    # change message-passing edges but not edge 35's own attributes
        after = model(data)[0]
    assert not torch.allclose(before[35], after[35])               # edge-aware: neighbours' attributes matter


def test_time_limit_stops_training_and_keeps_the_best_checkpoint():
    data = _graph()
    y = (data.edge_attr[:, 0] > 0.5).float()
    masks = {k: torch.zeros(40, dtype=torch.bool) for k in ('train', 'val', 'test')}
    masks['train'][:20] = True
    masks['val'][20:30] = True
    masks['test'][30:] = True
    result = run_training(
        build_model=lambda: EdgeAwareGNN(5, 6, 8, 4), loss_fn=torch.nn.BCEWithLogitsLoss(),
        outputs_fn=lambda m, d, idx: tuple(t.reshape(-1) for t in m(d, idx)),
        data=data, y=y, masks=masks, device=torch.device('cpu'), seeds=[0], num_epochs=500, lr=0.01,
        patience=500, log=lambda *_: None, max_seconds=0.05)
    run = result['runs'][0]
    assert run['stopped_by_time'] and run['epochs_run'] < 500 and run['state'] is not None


def test_regime_report_separates_the_low_volume_tail():
    days = ["2022/09/01 10:00"] * 200 + ["2022/09/02 10:00"] * 200 + ["2022/09/03 10:00"]
    ts = pd.to_datetime(pd.Series(days), format='%Y/%m/%d %H:%M')
    labels = np.array([0] * 396 + [1] * 4 + [1])
    prob = np.where(labels == 1, 0.9, 0.1)
    rep = regime_report(np.arange(401), prob, 0.5, ts, labels)
    assert rep['pooled']['rows'] == 401 and rep['low-volume tail days']['rows'] == 1
    assert rep['low-volume tail days']['positives'] == 1 and rep['normal-volume days']['positives'] == 4
