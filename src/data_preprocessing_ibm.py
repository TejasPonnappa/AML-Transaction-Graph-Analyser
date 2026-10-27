import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ibm_features import build_past_only_features

# --- Configuration ---
RAW_TRANS_PATH = 'data/raw/ibm_aml/HI-Small_Trans.csv'
RAW_ACCOUNTS_PATH = 'data/raw/ibm_aml/HI-Small_accounts.csv'
PROCESSED_DATA_PATH = 'data/processed/graph_data_ibm.pt'
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.85   # chronological cut points


def _zscore(values, reference):
    """Standardise columns with the mean/std of the reference rows only (the training period)."""
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    return (values - mean) / (std + 1e-8)


def preprocess_ibm_data():
    """
    Builds a PyG Data object for EDGE classification (predicting 'Is Laundering' per transaction) with no
    information from the future:

      * the split is chronological (70/15/15 by Timestamp);
      * edge_attr holds past-only features (see ibm_features.py), standardised with training-period statistics;
      * node features are aggregates of TRAINING-period edges only, standardised on those statistics;
      * `mp_edge_index` is the set of training edges: message passing never sees validation or test edges.

    Row order of the CSV is preserved (predict_ibm.py relies on it).
    """
    print("1. Loading raw IBM AML data...")
    df = pd.read_csv(RAW_TRANS_PATH)
    df.columns = [c.strip() for c in df.columns]

    required_cols = {'Account', 'Account.1', 'Amount Paid', 'Payment Format', 'Is Laundering', 'Timestamp'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. "
            f"Actual columns found: {list(df.columns)}. "
            f"Update the column names in this script to match."
        )
    n = len(df)
    print(f"   -> Total transactions: {n:,}")
    print(f"   -> Laundering ratio: {df['Is Laundering'].mean() * 100:.4f}%")

    # 2. Node index
    sender_col, receiver_col = 'Account', 'Account.1'
    all_accounts = pd.unique(pd.concat([df[sender_col], df[receiver_col]]))
    node_map = {acc_id: idx for idx, acc_id in enumerate(all_accounts)}
    num_nodes = len(all_accounts)
    print(f"   -> Total unique accounts (nodes): {num_nodes:,}")
    src = df[sender_col].map(node_map).to_numpy(dtype=np.int64)
    dst = df[receiver_col].map(node_map).to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # 3. Chronological split (stable sort: ties keep file order)
    timestamps = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    order = np.argsort(timestamps.to_numpy(), kind='stable')
    train_end, val_end = int(n * TRAIN_FRACTION), int(n * VAL_FRACTION)
    masks = {k: torch.zeros(n, dtype=torch.bool) for k in ('train', 'val', 'test')}
    masks['train'][order[:train_end]] = True
    masks['val'][order[train_end:val_end]] = True
    masks['test'][order[val_end:]] = True
    train_np = masks['train'].numpy()
    print(f"   -> Chronological split: train/val/test = {masks['train'].sum().item():,}/"
          f"{masks['val'].sum().item():,}/{masks['test'].sum().item():,}")

    # 4. Edge features: past-only history, standardised on the training period
    print("2. Engineering past-only edge features...")
    features, names = build_past_only_features(df, src, dst)
    features = _zscore(features, features[train_np]).astype(np.float32)
    edge_attr = torch.tensor(features, dtype=torch.float)
    print(f"   -> edge_attr dimension: {edge_attr.size(1)}")

    y = torch.tensor(df['Is Laundering'].values, dtype=torch.float)

    # 5. Node features from TRAINING edges only
    print("3. Engineering node features from training-period transactions...")
    train_df = pd.DataFrame({'s': src[train_np], 'r': dst[train_np], 'amt': df['Amount Paid'].to_numpy()[train_np]})
    sent = train_df.groupby('s')['amt'].agg(out_degree='count', total_sent='sum', avg_sent='mean')
    recv = train_df.groupby('r')['amt'].agg(in_degree='count', total_received='sum')
    node_features = pd.DataFrame(index=range(num_nodes)).join(sent).join(recv).fillna(0.0)
    cols = ['out_degree', 'total_sent', 'avg_sent', 'in_degree', 'total_received']
    logged = np.log1p(node_features[cols].to_numpy(dtype=float))
    X = torch.tensor(_zscore(logged, logged), dtype=torch.float)
    print(f"   -> Node feature dimension: {X.size(1)}")

    # 6. Build and save. Message passing uses training edges only.
    data = Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=masks['train'],
        val_mask=masks['val'],
        test_mask=masks['test'],
        mp_edge_index=edge_index[:, masks['train']],
    )
    data.edge_feature_names = list(names)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    torch.save(data, PROCESSED_DATA_PATH)
    print(f"4. ✅ IBM AML Graph Data saved successfully to: {PROCESSED_DATA_PATH}")
    return data


if __name__ == '__main__':
    if not os.path.exists(RAW_TRANS_PATH):
        print(f"Error: Raw data not found at {RAW_TRANS_PATH}. Download HI-Small_Trans.csv from Kaggle first.")
    else:
        preprocess_ibm_data()
