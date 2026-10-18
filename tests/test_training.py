import numpy as np
import torch
from torch_geometric.data import Data

from model import AMLGraphSAGE
from training_utils import binary_report, run_training, summarise_runs, tune_threshold


def _tiny_graph(n_nodes=60, n_edges=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    attr = torch.randn(n_edges, 2, generator=g)
    y = (attr[:, 0] > 0.8).float()                      # learnable from edge_attr
    order = torch.randperm(n_edges, generator=g)
    masks = {k: torch.zeros(n_edges, dtype=torch.bool) for k in ('train', 'val', 'test')}
    masks['train'][order[:250]] = True
    masks['val'][order[250:325]] = True
    masks['test'][order[325:]] = True
    data = Data(x=torch.randn(n_nodes, 5, generator=g), edge_index=torch.stack([src, dst]), edge_attr=attr, y=y)
    return data, y, masks


def _train(data, y, masks, seeds=(0,), epochs=30):
    return run_training(
        build_model=lambda: AMLGraphSAGE(5, 2, 8, 4),
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        outputs_fn=lambda m, d, idx: tuple(t.reshape(-1) for t in m(d, idx)),
        data=data, y=y, masks=masks, device=torch.device('cpu'),
        seeds=list(seeds), num_epochs=epochs, lr=0.01, patience=10, log=lambda *_: None)


def test_same_seed_is_reproducible():
    data, y, masks = _tiny_graph()
    a = _train(data, y, masks)['runs'][0]
    b = _train(data, y, masks)['runs'][0]
    assert a['val_pr_auc'] == b['val_pr_auc'] and a['test']['threshold'] == b['test']['threshold']


def test_selection_and_threshold_never_look_at_test_labels():
    data, y, masks = _tiny_graph()
    base = _train(data, y, masks)['runs'][0]
    y_flipped = y.clone()
    y_flipped[masks['test']] = 1 - y_flipped[masks['test']]     # corrupt ONLY the test labels
    changed = _train(data, y_flipped, masks)['runs'][0]
    assert base['val_pr_auc'] == changed['val_pr_auc']          # checkpoint choice unchanged
    assert base['test']['threshold'] == changed['test']['threshold']
    assert base['test']['pr_auc'] != changed['test']['pr_auc']  # only the final report differs


def test_best_seed_is_chosen_on_validation_and_summary_has_spread():
    data, y, masks = _tiny_graph()
    result = _train(data, y, masks, seeds=(0, 1, 2), epochs=15)
    vals = [r['val_pr_auc'] for r in result['runs']]
    assert result['best'] == int(np.argmax(vals))
    payload = summarise_runs(result)
    assert payload['selected_seed'] == result['runs'][result['best']]['seed']
    assert len(payload['seeds']) == 3 and 'pr_auc' in payload['seed_spread_test']
    assert {'test_roc_auc', 'test_pr_auc', 'test_precision', 'test_recall', 'test_f1_tuned_thresh',
            'confusion_matrix', 'best_threshold'} <= set(payload)      # keys the dashboard reads


def test_threshold_is_not_clipped_to_a_coarse_grid():
    y = np.array([0] * 50 + [1] * 5)
    p = np.concatenate([np.linspace(0.0, 0.97, 50), np.array([0.981, 0.985, 0.99, 0.992, 0.995])])
    thr = tune_threshold(y, p)
    assert 0.97 < thr <= 0.981                                     # between the classes, not on a 0.05 step


def test_binary_report_handles_a_single_class_and_uses_greater_equal():
    rep = binary_report(np.zeros(5, dtype=int), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.3)
    assert rep['confusion_matrix'] == {'tp': 0, 'fp': 3, 'fn': 0, 'tn': 2}
    assert rep['recall'] == 0.0 and rep['precision'] == 0.0
