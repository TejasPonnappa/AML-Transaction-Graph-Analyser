import torch
import pandas as pd
import numpy as np
import os
import json
from torch_geometric.data import Data  # Import to ensure correct loading of graph data

# Import the model definition from your model.py file
from model import AMLGraphSAGE

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint.pth'
OUTPUT_SCORES_PATH = 'outputs/suspicion_scores.csv'
RAW_DATA_PATH = 'data/raw/amlsim_transactions.csv'  # Need raw data for account IDs
METRICS_PATH = 'outputs/metrics.json'  # UPDATED: written by train.py, read here for a real threshold

# --- Hyperparameters (Must match train.py) ---
data = torch.load(PROCESSED_DATA_PATH, weights_only=False)

IN_CHANNELS_X = data.x.size(1)
IN_CHANNELS_EDGE = data.edge_attr.size(1)
HIDDEN_CHANNELS = 64
OUT_CHANNELS_NODE = 32
DEFAULT_THRESHOLD = 0.5  # fallback only if metrics.json isn't present yet
# --- End Hyperparameters ---


def load_decision_threshold():
    """
    UPDATED: Load the threshold tuned on the validation set during training,
    instead of hardcoding 0.5 or a stale value inside the UI files.
    """
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics.get('best_threshold', DEFAULT_THRESHOLD)
    print(f"Warning: {METRICS_PATH} not found. Using default threshold {DEFAULT_THRESHOLD}.")
    return DEFAULT_THRESHOLD


def generate_predictions():
    """Loads the model and generates suspicion scores for all transactions."""
    print("1. Loading Model and Data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model structure
    model = AMLGraphSAGE(IN_CHANNELS_X, IN_CHANNELS_EDGE, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()
    data.to(device)

    # --- 2. Generating Scores ---
    print("2. Running inference on all transactions...")
    with torch.no_grad():
        raw_scores, suspicion_scores = model(data)
        scores = suspicion_scores.cpu().numpy()

    # --- 3. Merging with Original Data ---
    print("3. Merging scores with original transactions...")

    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw.columns = [col.upper().replace('-', '_') for col in df_raw.columns]

    # UPDATED: Safety check — this assumes row order in the CSV exactly matches
    # edge order in graph_data.pt. If data_preprocessing.py ever filters, sorts,
    # or dedupes rows before building edge_index, this silently misaligns
    # scores with the wrong transactions. Fail loudly instead of silently.
    if len(df_raw) != len(scores):
        raise ValueError(
            f"Row count mismatch: raw CSV has {len(df_raw)} rows but model produced "
            f"{len(scores)} scores. Check that data_preprocessing.py builds edge_index "
            f"in the same order as amlsim_transactions.csv, with no filtering/dedup in between."
        )

    df_raw['SUSPICION_SCORE'] = scores

    # UPDATED: Apply the tuned decision threshold to produce an explicit binary flag,
    # instead of leaving thresholding entirely to the UI layer.
    threshold = load_decision_threshold()
    df_raw['PREDICTED_FRAUD'] = (df_raw['SUSPICION_SCORE'] >= threshold).astype(int)
    print(f"   -> Using decision threshold: {threshold:.5f}")

    df_output = df_raw[[
        'SENDER_ACCOUNT_ID',
        'RECEIVER_ACCOUNT_ID',
        'TX_AMOUNT',
        'TX_TYPE',
        'IS_FRAUD',
        'SUSPICION_SCORE',
        'PREDICTED_FRAUD'
    ]].copy()

    # --- 4. Saving Results ---
    os.makedirs(os.path.dirname(OUTPUT_SCORES_PATH), exist_ok=True)
    df_output.to_csv(OUTPUT_SCORES_PATH, index=False)

    print(f"4. \u2705 Suspicion Scores saved to: {OUTPUT_SCORES_PATH}")

    top_5 = df_output.sort_values(by='SUSPICION_SCORE', ascending=False).head(5)
    print("\nTop 5 Highest Suspicion Scores:")
    print(top_5[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'IS_FRAUD', 'SUSPICION_SCORE']].to_markdown(index=False))

    print("\nPipeline stage 3 COMPLETE. Ready for visualization!")


if __name__ == "__main__":
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}.")
        print("Please run 'python src/train.py' first.")
    elif not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
        print("Please run 'python src/data_preprocessing.py' first.")
    else:
        generate_predictions()