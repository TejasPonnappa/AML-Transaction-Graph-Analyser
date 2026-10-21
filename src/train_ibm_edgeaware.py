"""
Time-boxed run of the edge-aware GNN (GINE layers) on the IBM graph that carries the 43 past-only features.

    python src/data_preprocessing_ibm.py --graph-features
    python src/train_ibm_edgeaware.py --minutes 45

Outputs go to separate files (..._ibm_gine...) so the existing GraphSAGE artifacts are untouched. Training stops at the
wall-clock limit and keeps the best checkpoint by validation PR-AUC.
"""
import argparse
import os

from data_preprocessing_ibm import GRAPH_FEATURES_DATA_PATH
from train_edge_common import train_edge_model

MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_ibm_gine.pth'
METRICS_PATH = 'outputs/metrics_ibm_gine.json'
CURVES_PATH = 'outputs/training_curves_ibm_gine.png'
SCORES_PATH = 'outputs/ibm_gine_test_scores.npz'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-boxed edge-aware GNN on the IBM graph with structure features.")
    parser.add_argument('--minutes', type=float, default=45.0, help='wall-clock limit for training, per seed')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--pos-weight-multiplier', type=float, default=1.0)
    args = parser.parse_args()
    if not os.path.exists(GRAPH_FEATURES_DATA_PATH):
        print(f"Error: {GRAPH_FEATURES_DATA_PATH} not found. Run 'python src/data_preprocessing_ibm.py --graph-features'.")
    else:
        train_edge_model(processed_path=GRAPH_FEATURES_DATA_PATH, checkpoint_path=MODEL_CHECKPOINT_PATH,
                         metrics_path=METRICS_PATH, curves_path=CURVES_PATH, seeds=list(range(args.seeds)),
                         num_epochs=args.epochs, patience=25, pos_weight_multiplier=args.pos_weight_multiplier,
                         model_kind='gine', max_seconds=args.minutes * 60, scores_path=SCORES_PATH)
