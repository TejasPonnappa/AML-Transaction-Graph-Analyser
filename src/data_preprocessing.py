import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
RAW_DATA_PATH = 'data/raw/amlsim_transactions.csv'
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'

def load_data(path):
    """Loads the CSV and selects/renames required columns."""
    print("1. Loading raw data...")
    try:
        df = pd.read_csv(path)
        
        # Standardize the column names
        df.columns = [col.upper().replace('-', '_') for col in df.columns]

        # Select the columns we need (and ignore TX_ID and ALERT_ID)
        # Ensure the column names exactly match your CSV
        df = df[[
            'SENDER_ACCOUNT_ID', 
            'RECEIVER_ACCOUNT_ID', 
            'TX_TYPE', 
            'TX_AMOUNT', 
            'IS_FRAUD' 
        ]].copy()
        
        # Ensure the label is binary (1 for fraud, 0 otherwise)
        df['IS_FRAUD'] = df['IS_FRAUD'].astype(int)
        
        print(f"   -> Data loaded with {len(df)} transactions.")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please download the CSV.")
        return None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None

def preprocess_and_build_graph(df):
    """Converts the DataFrame into a PyG Data object, incorporating TX_TYPE features."""
    print("2. Preprocessing features and building edge index...")
    
    # --- 2.1 Node Encoding (SENDER/RECEIVER IDs) ---
    all_accounts = pd.concat([df['SENDER_ACCOUNT_ID'], df['RECEIVER_ACCOUNT_ID']]).unique()
    le = LabelEncoder()
    le.fit(all_accounts)
    
    df['source_id'] = le.transform(df['SENDER_ACCOUNT_ID'])
    df['target_id'] = le.transform(df['RECEIVER_ACCOUNT_ID'])
    
    num_nodes = len(all_accounts)
    print(f"   -> Total unique accounts (Nodes): {num_nodes}")
    
    # --- 2.2 Edge Features (TX_AMOUNT and TX_TYPE) ---
    
    # Numeric Feature: TX_AMOUNT (Log-transform and scale)
    amount_log = np.log1p(df['TX_AMOUNT'].values).reshape(-1, 1)
    scaler = StandardScaler()
    amount_scaled = scaler.fit_transform(amount_log)
    
    # Categorical Feature: TX_TYPE (One-Hot Encoding)
    # Ensure TX_TYPE column is treated as strings to avoid issues
    df['TX_TYPE'] = df['TX_TYPE'].astype(str) 
    tx_type_onehot = pd.get_dummies(df['TX_TYPE'], prefix='tx_type')
    
    # Combine all edge features
    edge_features_df = pd.DataFrame(amount_scaled, columns=['amount_scaled']).join(tx_type_onehot)
    
    # --- FIX: Ensure numpy array is numeric before converting to torch tensor ---
    edge_features_numpy = edge_features_df.values.astype(np.float32)
    
    # Handle NaNs that might arise from converting non-numeric data to float
    edge_features_numpy = np.nan_to_num(edge_features_numpy, nan=0.0) 

    # Convert to PyTorch Tensor
    edge_attr = torch.tensor(edge_features_numpy, dtype=torch.float)
    print(f"   -> Edge features created (Dimension: {edge_attr.size(1)}).")

    # --- 2.3 Edge Index and Labels ---
    source_nodes = torch.tensor(df['source_id'].values, dtype=torch.long)
    target_nodes = torch.tensor(df['target_id'].values, dtype=torch.long)
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    edge_labels = torch.tensor(df['IS_FRAUD'].values, dtype=torch.float)
    
    # --- 2.4 Dummy Node Features ---
    x = torch.ones(num_nodes, 1, dtype=torch.float) 
    
    # --- 2.5 PyTorch Geometric Data object ---
    data = Data(
        x=x,                        
        edge_index=edge_index,      
        edge_attr=edge_attr,        
        y=edge_labels               # Edge labels
    )

    # --- 2.6 Train/Val/Test Split for Edges (Stratified) ---
    num_edges = data.num_edges
    indices = np.arange(num_edges)
    
    # Stratified split is critical for imbalanced data like fraud
    train_indices, temp_indices = train_test_split(indices, test_size=0.4, stratify=df['IS_FRAUD'].values, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=df.iloc[temp_indices]['IS_FRAUD'].values, random_state=42)
    
    data.train_mask = torch.zeros(num_edges, dtype=torch.bool)
    data.val_mask = torch.zeros(num_edges, dtype=torch.bool)
    data.test_mask = torch.zeros(num_edges, dtype=torch.bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True
    
    print(f"   -> Train/Val/Test Split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)} edges.")
    print(f"   -> Fraud ratio (Train): {edge_labels[train_indices].sum() / len(train_indices):.4f}")
    
    return data

def save_data(data, path):
    """Saves the PyTorch Geometric Data object."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    print(f"3. ✅ Graph Data saved successfully to: {path}")

if __name__ == "__main__":
    df_transactions = load_data(RAW_DATA_PATH)
    
    if df_transactions is not None and not df_transactions.empty:
        graph_data = preprocess_and_build_graph(df_transactions)
        save_data(graph_data, PROCESSED_DATA_PATH)
        print("\nPipeline stage 1 COMPLETE. Ready for model training!")