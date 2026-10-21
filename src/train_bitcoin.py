import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model_bitcoin import BitcoinGraphSAGE
import numpy as np
from training_utils import binary_report, run_training, safe_average_precision, safe_roc_auc, summarise_runs

# --- Configuration (Elliptic) ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_bitcoin.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_bitcoin.pth'
METRICS_PATH = 'outputs/metrics_bitcoin.json'
CURVES_PATH = 'outputs/training_curves_bitcoin.png'
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
EARLY_STOP_PATIENCE = 25
SHUTDOWN_STEP = 43       # test results are also reported before / from this time step
HIDDEN_CHANNELS = 128     # must match predict_bitcoin.py
OUT_CHANNELS_NODE = 64


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


def time_step_report(test_idx, test_prob, y, time_step, threshold):
    """Per-time-step and before/after-step-43 test metrics at the validation-tuned threshold."""
    y_test = np.asarray(y)[test_idx]
    steps = np.asarray(time_step)[test_idx]

    def metrics(mask):
        yy, pp = y_test[mask], test_prob[mask]
        rep = binary_report(yy, pp, threshold)
        return dict(rows=int(mask.sum()), positives=int((yy == 1).sum()), pr_auc=safe_average_precision(yy, pp),
                    roc_auc=safe_roc_auc(yy, pp), precision=rep['precision'], recall=rep['recall'], f1=rep['f1'])

    per_step = {int(s): metrics(steps == s) for s in sorted(set(steps.tolist()))}
    first, last = int(steps.min()), int(steps.max())
    regimes = {f'steps {first}-{SHUTDOWN_STEP - 1} (before step {SHUTDOWN_STEP})': metrics(steps < SHUTDOWN_STEP),
               f'steps {SHUTDOWN_STEP}-{last} (from step {SHUTDOWN_STEP})': metrics(steps >= SHUTDOWN_STEP)}
    return per_step, regimes


def train(seeds, log=print):
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    labelled = data.y != -1
    masks = {k: getattr(data, f'{k}_mask') & labelled for k in ('train', 'val', 'test')}

    num_pos = data.y[masks['train']].sum().item()
    num_neg = masks['train'].sum().item() - num_pos
    class_weights = torch.tensor([1.0, num_neg / num_pos], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    log(f"device {device} | train nodes {int(masks['train'].sum())} | illicit weight {class_weights[1].item():.2f} "
        f"| {len(seeds)} seed(s); checkpoints chosen on validation PR-AUC")

    def outputs(model, d, idx):
        raw, _ = model(d)
        raw = raw[idx]
        return raw, F.softmax(raw, dim=1)[:, 1]

    result = run_training(
        build_model=lambda: BitcoinGraphSAGE(data.x.size(1), HIDDEN_CHANNELS, OUT_CHANNELS_NODE),
        loss_fn=criterion, outputs_fn=outputs, data=data, y=data.y, masks=masks, device=device,
        seeds=seeds, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE, log=log)

    best = result['runs'][result['best']]
    os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
    torch.save(best['state'], MODEL_CHECKPOINT_PATH)
    per_step, regimes = time_step_report(best['test_idx'], best['test_prob'], data.y.cpu().numpy(),
                                         data.time_step.cpu().numpy(), best['test']['threshold'])
    payload = summarise_runs(result, extra={'illicit_class_weight': class_weights[1].item(),
                                            'num_epochs_max': NUM_EPOCHS, 'time_step_is_a_feature': False,
                                            'regimes': regimes, 'per_step': per_step})
    with open(METRICS_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    plot_training_curves(best['history'], CURVES_PATH)
    t = best['test']
    log(f"Test (selected seed {best['seed']}, threshold {t['threshold']:.4f} from validation): PR-AUC "
        f"{t['pr_auc']:.4f} | precision {t['precision']:.4f} | recall {t['recall']:.4f} | F1 {t['f1']:.4f}")
    for name, r in regimes.items():
        log(f"  {name}: {r['positives']} illicit | PR-AUC {r['pr_auc']:.3f} | precision {r['precision']:.3f} | "
            f"recall {r['recall']:.3f} | F1 {r['f1']:.3f}")
    log(f"metrics -> {METRICS_PATH}; checkpoint -> {MODEL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Elliptic node classifier.")
    parser.add_argument('--seeds', type=int, default=1, help='number of seeds; the best on validation is kept')
    args = parser.parse_args()
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
        print("Please run 'python src/data_preprocessing_bitcoin.py' first.")
    else:
        train(list(range(args.seeds)))
