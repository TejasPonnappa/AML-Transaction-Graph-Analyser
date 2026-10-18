import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
RAW_TRANS_PATH = 'data/raw/ibm_aml/HI-Small_Trans.csv'
RAW_ACCOUNTS_PATH = 'data/raw/ibm_aml/HI-Small_accounts.csv'
PROCESSED_DATA_PATH = 'data/processed/graph_data_ibm.pt'

PAYMENT_FORMAT_CATEGORIES = ['Cheque', 'Credit Card', 'ACH', 'Wire', 'Cash', 'Bitcoin', 'Reinvestment']


def preprocess_ibm_data():
    """
    Loads the IBM AML HI-Small transaction data and builds a PyG Data object
    for EDGE classification (predicting 'Is Laundering' per transaction).

    UPDATED edge_attr (was 2 features, now 4):
      [log-amount, payment-format-idx, currency-mismatch flag, pairwise tx frequency]
    Rationale: cross-currency transactions and repeated transfers between the
    same account pair are known laundering signals not previously used at all.
    """
    print("1. Loading raw IBM AML data...")
    df = pd.read_csv(RAW_TRANS_PATH)
    df.columns = [c.strip() for c in df.columns]

    required_cols = {'Account', 'Account.1', 'Amount Paid', 'Payment Format', 'Is Laundering'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. "
            f"Actual columns found: {list(df.columns)}. "
            f"Update the column names in this script to match."
        )

    print(f"   -> Total transactions: {len(df):,}")
    print(f"   -> Laundering ratio: {df['Is Laundering'].mean() * 100:.4f}%")

    # 2. Build node index
    sender_col, receiver_col = 'Account', 'Account.1'
    all_accounts = pd.unique(pd.concat([df[sender_col], df[receiver_col]]))
    node_map = {acc_id: idx for idx, acc_id in enumerate(all_accounts)}
    num_nodes = len(all_accounts)
    print(f"   -> Total unique accounts (nodes): {num_nodes:,}")

    df['SENDER_IDX'] = df[sender_col].map(node_map)
    df['RECEIVER_IDX'] = df[receiver_col].map(node_map)

    # 3. EDGE_INDEX
    edge_index = torch.tensor(np.array([df['SENDER_IDX'].values, df['RECEIVER_IDX'].values]), dtype=torch.long)

    # 4. EDGE_ATTR — now 4 features instead of 2
    amount_col = 'Amount Paid'
    log_amount = np.log1p(df[amount_col].values.astype(float))
    log_amount_norm = (log_amount - log_amount.mean()) / (log_amount.std() + 1e-8)

    fmt_map = {f: i for i, f in enumerate(PAYMENT_FORMAT_CATEGORIES)}
    fmt_idx = df['Payment Format'].map(lambda f: fmt_map.get(f, len(PAYMENT_FORMAT_CATEGORIES))).values
    fmt_norm = (fmt_idx - fmt_idx.mean()) / (fmt_idx.std() + 1e-8)

    print("2. Engineering new edge features (currency mismatch, pairwise frequency)...")

    # --- NEW: currency mismatch flag ---
    # 1 if Payment Currency != Receiving Currency, else 0. IBM's cross-currency
    # transfers are a known laundering pattern (layering across currencies).
    if 'Payment Currency' in df.columns and 'Receiving Currency' in df.columns:
        currency_mismatch = (df['Payment Currency'] != df['Receiving Currency']).astype(float).values
    else:
        print("   -> WARNING: currency columns not found, defaulting mismatch flag to 0.")
        currency_mismatch = np.zeros(len(df))
    # already binary 0/1, no normalization needed, but center it slightly for scale consistency
    currency_mismatch_norm = currency_mismatch - currency_mismatch.mean()

    # --- NEW: pairwise transaction frequency ---
    # How many total transactions exist between this exact (sender, receiver)
    # pair across the whole dataset. Repeated transfers between the same two
    # accounts is a structuring/smurfing signal.
    pair_counts = df.groupby(['SENDER_IDX', 'RECEIVER_IDX'])[amount_col].transform('count').values
    log_pair_freq = np.log1p(pair_counts.astype(float))
    log_pair_freq_norm = (log_pair_freq - log_pair_freq.mean()) / (log_pair_freq.std() + 1e-8)

    edge_attr = torch.tensor(
        np.stack([log_amount_norm, fmt_norm, currency_mismatch_norm, log_pair_freq_norm], axis=1),
        dtype=torch.float
    )
    print(f"   -> edge_attr dimension: {edge_attr.size(1)} (was 2, now 4)")

    # 5. EDGE LABELS
    y = torch.tensor(df['Is Laundering'].values, dtype=torch.float)

    # 6. NODE FEATURES — unchanged
    print("3. Engineering node features from transaction behavior...")
    sent_stats = df.groupby('SENDER_IDX').agg(
        out_degree=(amount_col, 'count'),
        total_sent=(amount_col, 'sum'),
        avg_sent=(amount_col, 'mean'),
    )
    recv_stats = df.groupby('RECEIVER_IDX').agg(
        in_degree=(amount_col, 'count'),
        total_received=(amount_col, 'sum'),
    )

    node_features = pd.DataFrame(index=range(num_nodes))
    node_features = node_features.join(sent_stats).join(recv_stats).fillna(0.0)

    for col in ['out_degree', 'total_sent', 'avg_sent', 'in_degree', 'total_received']:
        vals = np.log1p(node_features[col].values)
        node_features[col] = (vals - vals.mean()) / (vals.std() + 1e-8)

    X = torch.tensor(node_features.values, dtype=torch.float)
    print(f"   -> Node feature dimension: {X.size(1)}")

    # 7. Train/Val/Test split — unchanged (temporal)
    if 'Timestamp' in df.columns:
        df_sorted_idx = df['Timestamp'].argsort().values
        n = len(df_sorted_idx)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_idx = df_sorted_idx[:train_end]
        val_idx = df_sorted_idx[train_end:val_end]
        test_idx = df_sorted_idx[val_end:]
        print("   -> Using TEMPORAL split (chronological by Timestamp)")
    else:
        all_idx = np.arange(len(df))
        train_idx, temp_idx = train_test_split(all_idx, test_size=0.3, stratify=y.numpy(), random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y.numpy()[temp_idx], random_state=42)
        print("   -> Timestamp column not found, falling back to stratified random split")

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"   -> Train/Val/Test edges: {train_mask.sum().item():,}/"
          f"{val_mask.sum().item():,}/{test_mask.sum().item():,}")

    # 8. Build and save
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
    print(f"4. \u2705 IBM AML Graph Data saved successfully to: {PROCESSED_DATA_PATH}")


if __name__ == '__main__':
    if not os.path.exists(RAW_TRANS_PATH):
        print(f"Error: Raw data not found at {RAW_TRANS_PATH}. Download HI-Small_Trans.csv from Kaggle first.")
    else:
        preprocess_ibm_data()