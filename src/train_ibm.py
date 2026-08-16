import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from model import AMLGraphSAGE  # Import the model you defined

import torch_geometric.data.data
torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_ibm.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_ibm.pth'
METRICS_PATH = 'outputs/metrics_ibm.json'
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
EARLY_STOP_PATIENCE = 25

# --- Hyperparameters for the Model ---
data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
IN_CHANNELS_X = data.x.size(1)
IN_CHANNELS_EDGE = data.edge_attr.size(1)
HIDDEN_CHANNELS = 64
OUT_CHANNELS_NODE = 32


def train():
    """Implements the training and validation loop."""
    print("1. Initializing Model and Data...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AMLGraphSAGE(IN_CHANNELS_X, IN_CHANNELS_EDGE, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"   -> Device: {device}")
    print(f"   -> Training edges (Fraud/Total): {num_pos}/{data.train_mask.sum().item()}")
    print(f"   -> Positive Weight for BCE Loss: {pos_weight.item():.2f}")

    best_val_auc = 0.0
    patience_counter = 0

    # UPDATED: track per-epoch history for the loss/metric curve plot
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}

    print("\n2. Starting Training Loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        raw_scores, _ = model(data)
        loss = criterion(raw_scores[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        val_loss, val_metrics = evaluate(model, data, data.val_mask, criterion)
        scheduler.step(val_metrics['roc_auc'])

        history['epoch'].append(epoch)
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['roc_auc'])
        history['val_f1'].append(val_metrics['f1'])

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            save_status = "-> Model Saved!"
        else:
            patience_counter += 1
            save_status = ""

        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f} {save_status}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val AUC improvement in {EARLY_STOP_PATIENCE} epochs).")
            break

    # --- 3. Threshold sweep on validation set ---
    print("\n3. Sweeping thresholds on validation set...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    _, _, val_probs, val_labels = evaluate(model, data, data.val_mask, criterion, return_probs=True)

    best_thresh, best_f1_at_thresh = 0.5, 0.0
    for t in [i / 100 for i in range(5, 96, 5)]:
        f1_t = f1_score(val_labels, (val_probs > t).astype(int), zero_division=0)
        if f1_t > best_f1_at_thresh:
            best_f1_at_thresh, best_thresh = f1_t, t
    print(f"   -> Best threshold: {best_thresh} (Val F1={best_f1_at_thresh:.4f})")

    # --- 4. Final Test Evaluation ---
    print("\n4. Final Test Evaluation...")
    test_loss, test_metrics, test_probs, test_labels = evaluate(
        model, data, data.test_mask, criterion, return_probs=True
    )
    test_preds_at_best_thresh = (test_probs > best_thresh).astype(int)
    test_f1_at_best_thresh = f1_score(test_labels, test_preds_at_best_thresh, zero_division=0)
    test_pr_auc = average_precision_score(test_labels, test_probs)

    test_precision = precision_score(test_labels, test_preds_at_best_thresh, zero_division=0)
    test_recall = recall_score(test_labels, test_preds_at_best_thresh, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds_at_best_thresh).ravel()

    print(f"TEST RESULTS: Loss: {test_loss:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | "
          f"F1 (thresh=0.5): {test_metrics['f1']:.4f} | F1 (tuned thresh={best_thresh}): "
          f"{test_f1_at_best_thresh:.4f} | PR-AUC: {test_pr_auc:.4f}")
    print(f"   -> Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")
    print(f"   -> Confusion Matrix -> TP: {tp} FP: {fp} FN: {fn} TN: {tn}")

    # --- 5. Save metrics for predict.py / dashboard to consume ---
    os.makedirs('outputs', exist_ok=True)
    metrics_out = {
        'test_roc_auc': test_metrics['roc_auc'],
        'test_f1_default_thresh': test_metrics['f1'],
        'test_f1_tuned_thresh': test_f1_at_best_thresh,
        'test_pr_auc': test_pr_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)},
        'best_threshold': best_thresh
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n   -> Metrics saved to {METRICS_PATH}")

    # --- 6. Plot and save training curves ---
    plot_training_curves(history)

    print("\nPipeline stage 2 COMPLETE. Trained model checkpoint saved.")


def plot_training_curves(history):
    """UPDATED: Saves a loss-vs-epoch and metric-vs-epoch plot to outputs/."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['epoch'], history['val_auc'], label='Val ROC-AUC')
    axes[1].plot(history['epoch'], history['val_f1'], label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics vs Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = 'outputs/training_curves_ibm.png'
    plt.savefig(out_path, dpi=150)
    print(f"   -> Training curves saved to {out_path}")
    plt.close()


def evaluate(model, data, mask, criterion, return_probs=False):
    """Evaluates the model on a given mask (val or test)."""
    model.eval()
    with torch.no_grad():
        raw_scores, suspicion_scores = model(data)

        loss = criterion(raw_scores[mask], data.y[mask])

        preds = suspicion_scores[mask].cpu().numpy()
        labels = data.y[mask].cpu().numpy()

        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.0

        binary_preds = (preds > 0.5).astype(int)
        f1 = f1_score(labels, binary_preds, zero_division=0)

        metrics = {'roc_auc': auc, 'f1': f1}

        if return_probs:
            return loss.item(), metrics, preds, labels
        return loss.item(), metrics


if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
        print("Please run 'python src/data_preprocessing.py' first.")
    else:
        train()