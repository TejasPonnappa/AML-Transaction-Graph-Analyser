"""
UI-only data layer. (The processed graph files are loaded with torch.load(weights_only=False), the same as
the training scripts do; they are generated locally by src/ and must not come from an untrusted source.)
 The dashboard never reads the large prediction CSVs directly:
it builds a compact Parquet file once (only the columns the pages need, plus a
train/val/test split column and, where available, a timestamp) and loads that
with st.cache_resource. Raw data and src/ are only read, never modified.

Files are written to outputs/.graph_cache/ui_data/ (already gitignored).
Rebuild by hand with:  python -m ui.data_store
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

CACHE_DIR = 'outputs/.graph_cache/ui_data'

EDGE_PROJECTS = {
    'ibm': dict(
        scores='outputs/ibm_predictions_rf.csv',      # RandomForest model of record (src/predict_ibm_rf.py)
        graph='data/processed/graph_data_ibm.pt',
        raw='data/raw/ibm_aml/HI-Small_Trans.csv',
        raw_time='Timestamp',
        time_kind='datetime',
    ),
}
ELLIPTIC = dict(
    scores='outputs/bitcoin_predictions.csv',
    graph='data/processed/graph_data_bitcoin.pt',
    edges='data/raw/elliptic_txs_edgelist.csv',
)
SPLITS = ['train', 'val', 'test']


def parquet_path(name):
    return os.path.join(CACHE_DIR, f'{name}.parquet')


def _is_stale(target, *sources):
    if not os.path.exists(target):
        return True
    mtime = os.path.getmtime(target)
    return any(os.path.exists(s) and os.path.getmtime(s) > mtime for s in sources)


def _split_column(graph_path, n_rows):
    """Reads the train/val/test masks from the processed graph; rows in none of them are 'unlabeled'."""
    import torch
    data = torch.load(graph_path, weights_only=False)
    codes = np.full(n_rows, -1, dtype=np.int8)
    for i, name in enumerate(SPLITS):
        mask = getattr(data, f'{name}_mask').cpu().numpy()
        if len(mask) != n_rows:
            raise ValueError(f"{graph_path}: {name}_mask has {len(mask)} rows but the scores file has {n_rows}.")
        codes[mask] = i
    return pd.Categorical.from_codes(codes, categories=SPLITS) if (codes >= 0).all() else \
        pd.Categorical.from_codes(np.where(codes < 0, 3, codes), categories=SPLITS + ['unlabeled'])


def build_edge_parquet(project):
    cfg = EDGE_PROJECTS[project]
    cols = ['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']
    df = pd.read_csv(cfg['scores'], usecols=cols,
                     dtype={'SENDER_ACCOUNT_ID': str, 'RECEIVER_ACCOUNT_ID': str, 'IS_FRAUD': str,
                            'TX_TYPE': str, 'SUSPICION_SCORE': np.float32})
    n = len(df)

    accounts = pd.Index(pd.unique(np.concatenate([df['SENDER_ACCOUNT_ID'].unique(),
                                                  df['RECEIVER_ACCOUNT_ID'].unique()])))
    df['SENDER_ACCOUNT_ID'] = pd.Categorical(df['SENDER_ACCOUNT_ID'], categories=accounts)
    df['RECEIVER_ACCOUNT_ID'] = pd.Categorical(df['RECEIVER_ACCOUNT_ID'], categories=accounts)
    df['TX_TYPE'] = df['TX_TYPE'].astype('category')
    df['IS_FRAUD'] = df['IS_FRAUD'].isin(['1', 'True', 'true']).astype(np.int8)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)

    if os.path.exists(cfg['raw']):
        raw_time = pd.read_csv(cfg['raw'], usecols=[cfg['raw_time']])[cfg['raw_time']]
        if len(raw_time) == n:
            try:
                if cfg['time_kind'] == 'datetime':
                    raw_time = pd.to_datetime(raw_time, format='%Y/%m/%d %H:%M')
                df['TIMESTAMP'] = raw_time.values
            except ValueError:
                pass  # unexpected timestamp format: the date filter is simply not offered

    if 'TIMESTAMP' in df.columns and cfg['time_kind'] == 'datetime':
        from src.ibm_features import low_volume_tail_mask
        df['TAIL'] = low_volume_tail_mask(pd.Series(df['TIMESTAMP']))     # days with almost no volume but ~60% laundering

    if os.path.exists(cfg['graph']):
        df['SPLIT'] = _split_column(cfg['graph'], n)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(parquet_path(project), index=False)
    return parquet_path(project)


def ensure_edge_parquet(project):
    cfg = EDGE_PROJECTS[project]
    path = parquet_path(project)
    if _is_stale(path, cfg['scores'], cfg['graph'], cfg['raw']):
        build_edge_parquet(project)
    return path


def build_elliptic_parquet():
    df = pd.read_csv(ELLIPTIC['scores'])
    df['class'] = df['class'].astype('category')
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1).astype(np.float32)
    if os.path.exists(ELLIPTIC['graph']):
        df['SPLIT'] = _split_column(ELLIPTIC['graph'], len(df))
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(parquet_path('elliptic'), index=False)
    return parquet_path('elliptic')


def ensure_elliptic_parquet():
    path = parquet_path('elliptic')
    if _is_stale(path, ELLIPTIC['scores'], ELLIPTIC['graph']):
        build_elliptic_parquet()
    return path


# --- Streamlit loaders (shared, read-only objects: callers must copy before mutating) ---
def _mtimes(*paths):
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0.0 for p in paths)


@st.cache_resource(show_spinner="Preparing data (first run only)...")
def _load_edges(project, mtimes):
    return pd.read_parquet(ensure_edge_parquet(project))


def load_edges(project):
    """The cache key includes the source files' mtimes, so new predictions are picked up without a restart."""
    cfg = EDGE_PROJECTS[project]
    return _load_edges(project, _mtimes(cfg['scores'], cfg['graph'], cfg['raw']))


@st.cache_resource(show_spinner="Preparing data (first run only)...")
def _load_elliptic(mtimes):
    return pd.read_parquet(ensure_elliptic_parquet())


def load_elliptic():
    return _load_elliptic(_mtimes(ELLIPTIC['scores'], ELLIPTIC['graph']))


@st.cache_resource(show_spinner=False)
def load_elliptic_edges():
    edges = pd.read_csv(ELLIPTIC['edges'])
    return edges.rename(columns={'txId1': 'source', 'txId2': 'target'})


def predictions_older_than(scores_path, *newer_paths):
    """True if the predictions file predates any of the given files (e.g. metrics from a later training run)."""
    if not os.path.exists(scores_path):
        return False
    mtime = os.path.getmtime(scores_path)
    return any(os.path.exists(p) and os.path.getmtime(p) > mtime for p in newer_paths)


def available(project):
    if project == 'elliptic':
        return os.path.exists(ELLIPTIC['scores'])
    return os.path.exists(EDGE_PROJECTS[project]['scores'])


if __name__ == '__main__':
    for proj in EDGE_PROJECTS:
        if os.path.exists(EDGE_PROJECTS[proj]['scores']):
            print(proj, '->', build_edge_parquet(proj))
    if os.path.exists(ELLIPTIC['scores']):
        print('elliptic ->', build_elliptic_parquet())
    sys.exit(0)
