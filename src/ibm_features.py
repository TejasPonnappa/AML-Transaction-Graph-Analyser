"""
Past-only transaction features for the IBM AML data, shared by the GNN preprocessing and the tree baselines
so both models see exactly the same inputs.

Every history feature for a transaction at time t is computed from transactions with a STRICTLY EARLIER
timestamp (same-minute rows, including the row itself, are excluded). Labels are never used. A deployed
system knows the full past, so these features are available at scoring time; nothing from the future leaks in.

Returned rows are in the same order as the input frame.
"""
import numpy as np
import pandas as pd

PAYMENT_FORMATS = ['Cheque', 'Credit Card', 'ACH', 'Wire', 'Cash', 'Bitcoin', 'Reinvestment']
WINDOWS = (3600, 86400)   # seconds: 1 hour, 1 day

STATIC_NAMES = ['log_paid', 'log_received', 'currency_mismatch', 'self_loop', 'bank_mismatch', 'hour',
                'day_of_week'] + [f'format_{f}' for f in PAYMENT_FORMATS]


def _role_names(prefix):
    return [f'{prefix}_prior_count', f'{prefix}_prior_sum', f'{prefix}_secs_since_last',
            f'{prefix}_count_1h', f'{prefix}_count_1d']


FEATURE_NAMES = (STATIC_NAMES + _role_names('sender_out') + _role_names('receiver_in')
                 + ['sender_in_prior_count', 'sender_in_prior_sum', 'receiver_out_prior_count',
                    'receiver_out_prior_sum', 'pair_prior_count', 'pair_secs_since_last'])


def past_stats(data_group, data_t, data_amount, query_group, query_t, windows=WINDOWS):
    """
    For every query row, statistics over data rows of the same group with a strictly earlier time:
    (count, amount sum, seconds since the latest such row or the full time span if none, [count per window]).
    """
    data_group = np.asarray(data_group, dtype=np.int64)
    query_group = np.asarray(query_group, dtype=np.int64)
    data_t = np.asarray(data_t, dtype=np.int64)
    query_t = np.asarray(query_t, dtype=np.int64)
    span = int(max(data_t.max(), query_t.max())) + 1

    keys = data_group * span + data_t
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    prefix = np.concatenate([[0.0], np.cumsum(np.asarray(data_amount, dtype=float)[order])])

    query_keys = query_group * span + query_t
    pos = np.searchsorted(sorted_keys, query_keys, side='left')           # rows strictly before the query
    start = np.searchsorted(sorted_keys, query_group * span, side='left')  # first row of the query's group
    count = pos - start
    total = prefix[pos] - prefix[start]
    last = sorted_keys[np.maximum(pos - 1, 0)] - query_group * span
    since = np.where(count > 0, query_t - last, span)
    per_window = []
    for w in windows:
        lo = np.searchsorted(sorted_keys, query_group * span + np.maximum(query_t - w, 0), side='left')
        per_window.append(pos - lo)
    return count, total, since, per_window


def build_past_only_features(df, src, dst):
    """
    df: IBM transactions with 'Timestamp', 'Amount Paid', 'Payment Format' (optional: 'Amount Received',
    'Payment Currency', 'Receiving Currency', 'From Bank', 'To Bank').
    src, dst: integer account ids per row. Returns (float32 matrix in df order, FEATURE_NAMES).
    """
    n = len(df)
    ts = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    t = ((ts - ts.min()).dt.total_seconds()).to_numpy(dtype=np.int64)
    paid = df['Amount Paid'].to_numpy(dtype=float)
    received = df['Amount Received'].to_numpy(dtype=float) if 'Amount Received' in df.columns else paid

    if 'Payment Currency' in df.columns and 'Receiving Currency' in df.columns:
        mismatch = (df['Payment Currency'] != df['Receiving Currency']).to_numpy(dtype=float)
    else:
        mismatch = np.zeros(n)
    if 'From Bank' in df.columns and 'To Bank' in df.columns:
        bank_mismatch = (df['From Bank'].astype(str) != df['To Bank'].astype(str)).to_numpy(dtype=float)
    else:
        bank_mismatch = np.zeros(n)

    cols = [np.log1p(paid), np.log1p(received), mismatch, (src == dst).astype(float), bank_mismatch,
            ts.dt.hour.to_numpy(dtype=float), ts.dt.dayofweek.to_numpy(dtype=float)]
    cols += [(df['Payment Format'] == f).to_numpy(dtype=float) for f in PAYMENT_FORMATS]

    def role(prefix_cols, count, total, since, windows=None):
        out = [np.log1p(count), np.log1p(total), ]
        if since is not None:
            out.append(np.log1p(since))
        if windows is not None:
            out += [np.log1p(w) for w in windows]
        prefix_cols += out

    # sender as sender / receiver as receiver (with recency and burst windows)
    c, s, since, w = past_stats(src, t, paid, src, t)
    role(cols, c, s, since, w)
    c, s, since, w = past_stats(dst, t, paid, dst, t)
    role(cols, c, s, since, w)
    # reverse roles: how much the sender has received, how much the receiver has sent
    c, s, _, _ = past_stats(dst, t, paid, src, t, windows=())
    role(cols, c, s, None)
    c, s, _, _ = past_stats(src, t, paid, dst, t, windows=())
    role(cols, c, s, None)
    # the ordered pair
    pair, _ = pd.factorize(src.astype(np.int64) * (int(max(src.max(), dst.max())) + 1) + dst.astype(np.int64))
    c, _, since, _ = past_stats(pair, t, paid, pair, t, windows=())
    cols += [np.log1p(c), np.log1p(since)]

    X = np.column_stack(cols).astype(np.float32)
    assert X.shape[1] == len(FEATURE_NAMES), (X.shape, len(FEATURE_NAMES))
    return X, FEATURE_NAMES
