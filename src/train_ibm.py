import argparse
import os

from train_edge_common import train_edge_model

# --- Configuration (IBM AML) ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_ibm.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_ibm.pth'
METRICS_PATH = 'outputs/metrics_ibm.json'
CURVES_PATH = 'outputs/training_curves_ibm.png'
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
EARLY_STOP_PATIENCE = 25

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IBM AML edge classifier.")
    parser.add_argument('--seeds', type=int, default=1, help='number of seeds; the best on validation is kept')
    parser.add_argument('--pos-weight-multiplier', type=float, default=1.0,
                        help='scale on neg/pos in the loss; choose it on VALIDATION numbers only, never test')
    args = parser.parse_args()
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
        print("Please run 'python src/data_preprocessing_ibm.py' first.")
    else:
        train_edge_model(processed_path=PROCESSED_DATA_PATH, checkpoint_path=MODEL_CHECKPOINT_PATH,
                         metrics_path=METRICS_PATH, curves_path=CURVES_PATH, seeds=list(range(args.seeds)),
                         num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE,
                         pos_weight_multiplier=args.pos_weight_multiplier)
