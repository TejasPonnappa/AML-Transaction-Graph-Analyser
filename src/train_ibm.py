"""
GraphSAGE edge classifier for the IBM AML data (kept as a research comparison; the model of record is the RandomForest in
predict_ibm_rf.py). See training_utils.py for the selection rules: checkpoints and early stopping use validation PR-AUC,
the threshold comes from the validation precision-recall curve, and test data is used only for the final report.

    python src/data_preprocessing_ibm.py
    python src/train_ibm.py [--seeds 3] [--pos-weight-multiplier 1.0]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch_geometric.data.data

from model import AMLGraphSAGE
from training_utils import run_training, summarise_runs

torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

PROCESSED_DATA_PATH = 'data/processed/graph_data_ibm.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_ibm.pth'
METRICS_PATH = 'outputs/metrics_ibm.json'
CURVES_PATH = 'outputs/training_curves_ibm.png'
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
EARLY_STOP_PATIENCE = 25
HIDDEN_CHANNELS = 64      # must match predict_ibm.py
OUT_CHANNELS_NODE = 32


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['epoch'], history['train_loss'], label='Train loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss vs epoch')
    axes[1].plot(history['epoch'], history['val_pr_auc'], label='Val PR-AUC')
    axes[1].set(xlabel='Epoch', ylabel='PR-AUC', title='Validation PR-AUC vs epoch')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(seeds, pos_weight_multiplier=1.0, log=print):
    log("1. Loading data...")
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
    in_x, in_edge = data.x.size(1), data.edge_attr.size(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    y = data.y.float()

    num_pos = y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    base_pos_weight = num_neg / num_pos
    pos_weight = torch.tensor([base_pos_weight * pos_weight_multiplier], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log(f"   device {device} | edge_attr dim {in_edge} | train positives {int(num_pos)}/{int(data.train_mask.sum())} "
        f"| pos_weight {pos_weight.item():.2f} (base {base_pos_weight:.2f} x {pos_weight_multiplier})")

    log(f"\n2. Training {len(seeds)} seed(s); checkpoints chosen on validation PR-AUC...")
    result = run_training(
        build_model=lambda: AMLGraphSAGE(in_x, in_edge, HIDDEN_CHANNELS, OUT_CHANNELS_NODE),
        loss_fn=criterion,
        outputs_fn=lambda model, d, idx: tuple(t.reshape(-1) for t in model(d, idx)),
        data=data, y=y, masks={'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask},
        device=device, seeds=seeds, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE, log=log)

    best = result['runs'][result['best']]
    os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
    torch.save(best['state'], MODEL_CHECKPOINT_PATH)
    payload = summarise_runs(result, extra={
        'pos_weight_multiplier': pos_weight_multiplier, 'edge_attr_dim': in_edge, 'num_epochs_max': NUM_EPOCHS,
        'train_seconds': [round(r['train_seconds']) for r in result['runs']],
        'epochs_run': [r['epochs_run'] for r in result['runs']]})
    with open(METRICS_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    plot_training_curves(best['history'], CURVES_PATH)

    t = best['test']
    log(f"\n3. Test (selected seed {best['seed']}, threshold {t['threshold']:.4f} from validation): "
        f"PR-AUC {t['pr_auc']:.4f} | ROC-AUC {t['roc_auc']:.4f} | precision {t['precision']:.4f} | "
        f"recall {t['recall']:.4f} | F1 {t['f1']:.4f}")
    if len(seeds) > 1:
        s = payload['seed_spread_test']
        log(f"   over {len(seeds)} seeds: PR-AUC {s['pr_auc']['mean']:.4f}+/-{s['pr_auc']['std']:.4f}, "
            f"F1 {s['f1']['mean']:.4f}+/-{s['f1']['std']:.4f}")
    log(f"   metrics -> {METRICS_PATH}; checkpoint -> {MODEL_CHECKPOINT_PATH}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IBM AML GraphSAGE edge classifier.")
    parser.add_argument('--seeds', type=int, default=1, help='number of seeds; the best on validation is kept')
    parser.add_argument('--pos-weight-multiplier', type=float, default=1.0,
                        help='scale on neg/pos in the loss; choose it on VALIDATION numbers only, never test')
    args = parser.parse_args()
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
        print("Please run 'python src/data_preprocessing_ibm.py' first.")
    else:
        train(list(range(args.seeds)), args.pos_weight_multiplier)
