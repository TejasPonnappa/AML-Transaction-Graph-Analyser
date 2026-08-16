import torch
import torch.nn.functional as F
from model_bitcoin import BitcoinGraphSAGE
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, confusion_matrix
import os
import json
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_bitcoin.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_bitcoin.pth'
METRICS_PATH = 'outputs/metrics_bitcoin.json'
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
EARLY_STOP_PATIENCE = 25

# --- Hyperparameters ---
if not os.path.exists(PROCESSED_DATA_PATH):
    raise FileNotFoundError(f"{PROCESSED_DATA_PATH} not found. Run data_preprocessing_bitcoin.py first.")

data = torch.load(PROCESSED_DATA_PATH, weights_only=False)

IN_CHANNELS_X = data.x.size(1)
HIDDEN_CHANNELS = 128
OUT_CHANNELS_NODE = 64
NUM_CLASSES = 2


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BitcoinGraphSAGE(IN_CHANNELS_X, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    labeled_nodes = (data.y != -1)
    num_pos = data.y[data.train_mask & labeled_nodes].sum().item()
    num_neg = (data.train_mask & labeled_nodes).sum().item() - num_pos

    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    class_weights = torch.tensor([1.0, pos_weight.item()], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    patience_counter = 0

    # UPDATED: track per-epoch history for the loss/metric curve plot
    history = {'epoch': [], 'train_loss': [], 'val_f1': [], 'val_pr_auc': []}

    train_mask = data.train_mask & labeled_nodes
    val_mask = data.val_mask & labeled_nodes
    test_mask = data.test_mask & labeled_nodes

    print(f"1. Initializing Model and Data...")
    print(f"   -> Device: {device}")
    print(f"   -> Training nodes (Labeled): {train_mask.sum().item():,}")
    print(f"   -> Positive Weight for Illicit Class: {pos_weight.item():.2f}")
    print(f"\n2. Starting Training Loop (Node Classification)...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        raw_scores, _ = model(data)
        loss = criterion(raw_scores[train_mask], data.y[train_mask])

        loss.backward()
        optimizer.step()

        val_f1, val_pr_auc = evaluate(model, data, val_mask)
        scheduler.step(val_f1)

        history['epoch'].append(epoch)
        history['train_loss'].append(loss.item())
        history['val_f1'].append(val_f1)
        history['val_pr_auc'].append(val_pr_auc)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            save_status = "-> Model Saved!"
        else:
            patience_counter += 1
            save_status = ""

        if epoch % 10 == 0 or save_status:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | "
                  f"Val PR-AUC: {val_pr_auc:.4f} {save_status}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val F1 improvement in {EARLY_STOP_PATIENCE} epochs).")
            break

    # --- 3. Threshold sweep on validation set ---
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    _, _, val_probs, val_labels = evaluate(model, data, val_mask, return_probs=True)

    best_thresh, best_f1_at_thresh = 0.5, 0.0
    for t in [i / 100 for i in range(5, 96, 5)]:
        f1_t = f1_score(val_labels, (val_probs > t).astype(int), zero_division=0)
        if f1_t > best_f1_at_thresh:
            best_f1_at_thresh, best_thresh = f1_t, t
    print(f"\n3. Best threshold from val sweep: {best_thresh} (F1={best_f1_at_thresh:.4f})")

    # --- 4. Final Test Evaluation ---
    _, _, test_probs, test_labels = evaluate(model, data, test_mask, return_probs=True)
    test_preds = (test_probs > best_thresh).astype(int)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_pr_auc = average_precision_score(test_labels, test_probs)

    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

    print(f"\nTEST RESULTS (threshold={best_thresh}): F1: {test_f1:.4f} | PR-AUC: {test_pr_auc:.4f}")
    print(f"   -> Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")
    print(f"   -> Confusion Matrix -> TP: {tp} FP: {fp} FN: {fn} TN: {tn}")

    # --- 5. Save metrics for predict_bitcoin.py / dashboard to consume ---
    os.makedirs('outputs', exist_ok=True)
    metrics_out = {
        'test_f1': test_f1,
        'test_pr_auc': test_pr_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)},
        'best_threshold': best_thresh
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"   -> Metrics saved to {METRICS_PATH}")

    # --- 6. Plot and save training curves ---
    plot_training_curves(history)

    print("Pipeline stage 2 COMPLETE. Trained Bitcoin model checkpoint saved.")


def plot_training_curves(history):
    """UPDATED: Saves a loss-vs-epoch and metric-vs-epoch plot to outputs/."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss vs Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['epoch'], history['val_f1'], label='Val F1')
    axes[1].plot(history['epoch'], history['val_pr_auc'], label='Val PR-AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics vs Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = 'outputs/training_curves_bitcoin.png'
    plt.savefig(out_path, dpi=150)
    print(f"   -> Training curves saved to {out_path}")
    plt.close()


def evaluate(model, data, mask, return_probs=False):
    model.eval()
    with torch.no_grad():
        raw_scores, suspicion_scores = model(data)
        probs = suspicion_scores[mask].cpu().numpy()
        labels = data.y[mask].cpu().numpy()

        preds = (probs > 0.5).astype(int)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        pr_auc = average_precision_score(labels, probs)

        if return_probs:
            return f1, pr_auc, probs, labels
        return f1, pr_auc


if __name__ == "__main__":
    train()