import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
RAW_DATA_PATH = 'data/raw/amlsim_transactions.csv'
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'

# TX_TYPE categories AMLSim typically uses. Adjust this list if your CSV has
# different labels — check with: pd.read_csv(RAW_DATA_PATH)['TX_TYPE'].unique()
TX_TYPE_CATEGORIES = ['TRANSFER', 'DEPOSIT', 'WITHDRAWAL', 'PAYMENT', 'CASH_OUT']


def preprocess_amlsim_data():
    """
    Loads amlsim_transactions.csv, builds node features from account-level
    transaction behavior (instead of a dummy constant), builds edge_attr from
    amount + transaction type, and saves everything as a PyG Data object for
    EDGE classification (predicting IS_FRAUD per transaction).
    """
    print("1. Loading raw AMLSim data...")
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = [col.upper().replace('-', '_') for col in df.columns]

    required_cols = {'SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {RAW_DATA_PATH}: {missing}")

    print(f"   -> Total transactions: {len(df):,}")
    print(f"   -> Fraud ratio: {df['IS_FRAUD'].mean() * 100:.4f}%")

    # 2. Build a node index for every unique account (sender or receiver)
    all_accounts = pd.unique(pd.concat([df['SENDER_ACCOUNT_ID'], df['RECEIVER_ACCOUNT_ID']]))
    node_map = {acc_id: idx for idx, acc_id in enumerate(all_accounts)}
    num_nodes = len(all_accounts)
    print(f"   -> Total unique accounts (nodes): {num_nodes:,}")

    df['SENDER_IDX'] = df['SENDER_ACCOUNT_ID'].map(node_map)
    df['RECEIVER_IDX'] = df['RECEIVER_ACCOUNT_ID'].map(node_map)

    # 3. Build EDGE_INDEX — order here fixes the order predict.py must match later,
    #    since predict.py assumes row order == edge order.
    edge_index = torch.tensor(
        [df['SENDER_IDX'].values, df['RECEIVER_IDX'].values], dtype=torch.long
    )

    # 4. Build EDGE_ATTR: [normalized amount, tx_type one-hot-ish encoded as an index]
    #    Kept to 2 columns to match model.py's IN_CHANNELS_EDGE=2, but using a real
    #    signal (log-amount) instead of raw amount, which has a long-tail distribution
    #    that hurts GNN training if left unnormalized.
    log_amount = np.log1p(df['TX_AMOUNT'].values)
    log_amount_norm = (log_amount - log_amount.mean()) / (log_amount.std() + 1e-8)

    tx_type_map = {t: i for i, t in enumerate(TX_TYPE_CATEGORIES)}
    tx_type_idx = df['TX_TYPE'].map(lambda t: tx_type_map.get(t, len(TX_TYPE_CATEGORIES))).values
    tx_type_norm = (tx_type_idx - tx_type_idx.mean()) / (tx_type_idx.std() + 1e-8)

    edge_attr = torch.tensor(
        np.stack([log_amount_norm, tx_type_norm], axis=1), dtype=torch.float
    )

    # 5. Build EDGE LABELS (y) — one label per transaction/edge
    y = torch.tensor(df['IS_FRAUD'].values, dtype=torch.float)

    # 6. Build NODE FEATURES (x) — replaces the old dummy constant-1 feature.
    #    Per-account behavioral stats computed from the whole transaction history:
    #    out-degree, in-degree, total sent, total received, avg tx amount sent.
    print("2. Engineering node features from transaction behavior...")

    sent_stats = df.groupby('SENDER_IDX').agg(
        out_degree=('TX_AMOUNT', 'count'),
        total_sent=('TX_AMOUNT', 'sum'),
        avg_sent=('TX_AMOUNT', 'mean'),
    )
    recv_stats = df.groupby('RECEIVER_IDX').agg(
        in_degree=('TX_AMOUNT', 'count'),
        total_received=('TX_AMOUNT', 'sum'),
    )

    node_features = pd.DataFrame(index=range(num_nodes))
    node_features = node_features.join(sent_stats).join(recv_stats).fillna(0.0)

    # log-transform + z-score normalize skewed monetary/count features
    for col in ['out_degree', 'total_sent', 'avg_sent', 'in_degree', 'total_received']:
        vals = np.log1p(node_features[col].values)
        node_features[col] = (vals - vals.mean()) / (vals.std() + 1e-8)

    X = torch.tensor(node_features.values, dtype=torch.float)
    print(f"   -> Node feature dimension: {X.size(1)} "
          f"(out_degree, total_sent, avg_sent, in_degree, total_received)")

    # 7. Train/Val/Test split — stratified on IS_FRAUD, done at the EDGE level
    #    since this is edge classification, not node classification.
    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.3, stratify=y.numpy(), random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y.numpy()[temp_idx], random_state=42
    )

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"   -> Train/Val/Test edges: {train_mask.sum().item():,}/"
          f"{val_mask.sum().item():,}/{test_mask.sum().item():,}")

    # 8. Build and save the PyG Data object
    data = Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    torch.save(data, PROCESSED_DATA_PATH, _use_new_zipfile_serialization=False)
    print(f"3. \u2705 AMLSim Graph Data saved successfully to: {PROCESSED_DATA_PATH}")


if __name__ == '__main__':
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data not found at {RAW_DATA_PATH}.")
    else:
        preprocess_amlsim_data()