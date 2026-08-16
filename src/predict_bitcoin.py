import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
from torch_geometric.data import Data
from model_bitcoin import BitcoinGraphSAGE

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_bitcoin.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_bitcoin.pth'
OUTPUT_SCORES_PATH = 'outputs/bitcoin_predictions.csv'
FEATURES_PATH = 'data/raw/elliptic_txs_features.csv'
CLASSES_PATH = 'data/raw/elliptic_txs_classes.csv'
METRICS_PATH = 'outputs/metrics_bitcoin.json'  # UPDATED: written by train_bitcoin.py
DEFAULT_THRESHOLD = 0.5

# --- Hyperparameters (Must match train_bitcoin.py) ---
# UPDATED: removed bare `except: exit()` — fail loudly with the real error instead
if not os.path.exists(PROCESSED_DATA_PATH):
    raise FileNotFoundError(
        f"{PROCESSED_DATA_PATH} not found. Run data_preprocessing_bitcoin.py first."
    )
data = torch.load(PROCESSED_DATA_PATH, weights_only=False)

IN_CHANNELS_X = data.x.size(1)
HIDDEN_CHANNELS = 128
OUT_CHANNELS_NODE = 64


def load_decision_threshold():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics.get('best_threshold', DEFAULT_THRESHOLD)
    print(f"Warning: {METRICS_PATH} not found. Using default threshold {DEFAULT_THRESHOLD}.")
    return DEFAULT_THRESHOLD


def generate_bitcoin_predictions():
    """Loads the model and generates illicit probability scores for all addresses."""
    print("1. Loading Model and Data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BitcoinGraphSAGE(IN_CHANNELS_X, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()
    data.to(device)

    # --- 2. Generating Scores ---
    print("2. Running inference on all Bitcoin addresses...")
    with torch.no_grad():
        raw_scores, _ = model(data)
        probabilities = F.softmax(raw_scores, dim=1)
        suspicion_scores = probabilities[:, 1].cpu().numpy()

    # --- 3. Merging with Original Data ---
    print("3. Merging scores with original Bitcoin transaction IDs...")

    df_features = pd.read_csv(FEATURES_PATH, header=None)
    df_features.rename(columns={0: 'txId', 1: 'TimeStep'}, inplace=True)
    df_classes = pd.read_csv(CLASSES_PATH)

    # UPDATED: safety check — node order in df_features must match node_map order
    # used when building graph_data_bitcoin.pt, or scores get misassigned to wrong txIds.
    if len(df_features) != len(suspicion_scores):
        raise ValueError(
            f"Row count mismatch: elliptic_txs_features.csv has {len(df_features)} rows "
            f"but model produced {len(suspicion_scores)} scores. Check node ordering in "
            f"data_preprocessing_bitcoin.py."
        )

    df_pred = pd.DataFrame({
        'txId': df_features['txId'].values,
        'SUSPICION_SCORE': suspicion_scores
    })

    df_output = pd.merge(df_pred, df_classes, on='txId', how='left')
    df_output = pd.merge(df_output, df_features[['txId', 'TimeStep']], on='txId', how='left')

    df_output['class'] = df_output['class'].replace({'1': 'Illicit', '2': 'Licit', 'unknown': 'Unlabeled'})

    # UPDATED: explicit binary prediction using the tuned threshold from train_bitcoin.py
    threshold = load_decision_threshold()
    df_output['PREDICTED_ILLICIT'] = (df_output['SUSPICION_SCORE'] >= threshold).astype(int)
    print(f"   -> Using decision threshold: {threshold:.5f}")

    # --- 4. Saving Results ---
    os.makedirs('outputs', exist_ok=True)
    df_output.to_csv(OUTPUT_SCORES_PATH, index=False)

    print(f"4. \u2705 Bitcoin Prediction scores saved to: {OUTPUT_SCORES_PATH}")

    top_5 = df_output.sort_values(by='SUSPICION_SCORE', ascending=False).head(5)
    print("\nTop 5 Most Suspicious Bitcoin Addresses:")
    print(top_5[['txId', 'TimeStep', 'class', 'SUSPICION_SCORE']].to_markdown(index=False))

    print("\nBitcoin Prediction pipeline complete.")


if __name__ == "__main__":
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        generate_bitcoin_predictions()
    else:
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. Run train_bitcoin.py first.")