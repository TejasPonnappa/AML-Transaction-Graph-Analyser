"""
Tree-model baselines on the same splits as the GNNs, to show how much a graph model adds
and how much of each GNN score comes from leakage or trivial separability.

    python src/baselines.py amlsim   [--seeds 3]
    python src/baselines.py ibm      [--seeds 3] [--neg-frac 0.1]
    python src/baselines.py elliptic [--seeds 3]

Leak-free by construction: account aggregates (counts, sums, pair counts) are computed from
TRAINING rows only and frozen; for training rows the row itself is subtracted (leave-one-out) so
train and evaluation features are distributed alike. Thresholds are tuned on validation (max F1
on the precision-recall curve) and applied once to test with `>=`, as in predict*.py.

Reads raw data and the processed graphs (for the train/val/test masks); writes only
outputs/baselines_<dataset>.json. Nothing in the GNN pipeline is modified.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from training_utils import tune_threshold

AMLSIM = dict(raw='data/raw/amlsim_transactions.csv', graph='data/processed/graph_data.pt',
              metrics='outputs/metrics.json')
IBM = dict(raw='data/raw/ibm_aml/HI-Small_Trans.csv', graph='data/processed/graph_data_ibm.pt',
           metrics='outputs/metrics_ibm.json')
ELLIPTIC = dict(features='data/raw/elliptic_txs_features.csv', edges='data/raw/elliptic_txs_edgelist.csv',
                graph='data/processed/graph_data_bitcoin.pt', metrics='outputs/metrics_bitcoin.json',
                predictions='outputs/bitcoin_predictions.csv')
PAYMENT_FORMATS = ['Cheque', 'Credit Card', 'ACH', 'Wire', 'Cash', 'Bitcoin', 'Reinvestment']
TEST_STEPS = range(39, 50)
SHUTDOWN_STEP = 43   # Elliptic dark-market shutdown (from the dataset's literature; unverified here)


# --------------------------------------------------------------------------- metrics
def evaluate(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    n_pos = int((y == 1).sum())
    k = n_pos
    top = np.argsort(-p)[:k] if k else np.array([], dtype=int)
    return dict(
        pr_auc=float(average_precision_score(y, p)) if n_pos else float('nan'),
        roc_auc=float(roc_auc_score(y, p)) if 0 < n_pos < len(y) else float('nan'),
        precision=tp / max(int(pred.sum()), 1),
        recall=tp / max(n_pos, 1),
        f1=2 * tp / max(int(pred.sum()) + n_pos, 1),
        precision_at_k=float((y[top] == 1).mean()) if k else float('nan'),
        threshold=thr, n_test=int(len(y)), n_pos=n_pos,
    )


def summarise(runs):
    """Mean and std over seeds for each metric."""
    out = {}
    for key in runs[0]:
        vals = np.array([r[key] for r in runs], dtype=float)
        out[key] = dict(mean=float(np.nanmean(vals)), std=float(np.nanstd(vals)))
    return out


# --------------------------------------------------------------------------- features
def account_aggregate_features(src, dst, amount, train_mask, n_accounts):
    """
    Per-edge features from account aggregates over TRAINING rows only. For training rows the
    row's own contribution is subtracted (leave-one-out). Returns an (n_edges, 12) float32 array.
    """
    w = train_mask
    out_cnt = np.bincount(src[w], minlength=n_accounts).astype(float)
    in_cnt = np.bincount(dst[w], minlength=n_accounts).astype(float)
    out_sum = np.bincount(src[w], weights=amount[w], minlength=n_accounts)
    in_sum = np.bincount(dst[w], weights=amount[w], minlength=n_accounts)
    me = train_mask.astype(float)
    loop = (src == dst).astype(float)

    s_out_c, s_out_s = out_cnt[src] - me, out_sum[src] - me * amount
    s_in_c, s_in_s = in_cnt[src] - me * loop, in_sum[src] - me * loop * amount
    r_in_c, r_in_s = in_cnt[dst] - me, in_sum[dst] - me * amount
    r_out_c, r_out_s = out_cnt[dst] - me * loop, out_sum[dst] - me * loop * amount

    cols = []
    for cnt, tot in ((s_out_c, s_out_s), (s_in_c, s_in_s), (r_in_c, r_in_s), (r_out_c, r_out_s)):
        cnt, tot = np.maximum(cnt, 0), np.maximum(tot, 0)
        cols += [np.log1p(cnt), np.log1p(tot), np.log1p(tot / np.maximum(cnt, 1))]
    return np.column_stack(cols).astype(np.float32)


def pair_count_feature(src, dst, train_mask, n_accounts):
    """log(1 + number of TRAINING transactions between the same ordered pair), leave-one-out for train rows."""
    key = src.astype(np.int64) * n_accounts + dst.astype(np.int64)
    counts = pd.Series(key[train_mask]).value_counts()
    c = pd.Series(key).map(counts).fillna(0).to_numpy(dtype=float) - train_mask.astype(float)
    return np.log1p(np.maximum(c, 0)).astype(np.float32)[:, None]


def factorize_accounts(*columns):
    codes, _ = pd.factorize(pd.concat([pd.Series(c) for c in columns], ignore_index=True))
    n = len(columns[0])
    return [codes[i * n:(i + 1) * n] for i in range(len(columns))], int(codes.max()) + 1


def load_masks(graph_path, n_rows):
    import torch
    data = torch.load(graph_path, weights_only=False)
    masks = {k: getattr(data, f'{k}_mask').cpu().numpy() for k in ('train', 'val', 'test')}
    for k, m in masks.items():
        if len(m) != n_rows:
            raise ValueError(f"{graph_path}: {k}_mask has {len(m)} rows, expected {n_rows}.")
    return masks, data.y.cpu().numpy()


def temporal_masks(order_key, fractions=(0.7, 0.85)):
    """Chronological 70/15/15 split by a sortable key (stable, so ties keep file order)."""
    n = len(order_key)
    order = np.argsort(np.asarray(order_key), kind='stable')
    cut1, cut2 = int(n * fractions[0]), int(n * fractions[1])
    masks = {k: np.zeros(n, dtype=bool) for k in ('train', 'val', 'test')}
    masks['train'][order[:cut1]] = True
    masks['val'][order[cut1:cut2]] = True
    masks['test'][order[cut2:]] = True
    return masks


def build_amlsim():
    df = pd.read_csv(AMLSIM['raw'])
    df.columns = [c.upper() for c in df.columns]
    (src, dst), n_acc = factorize_accounts(df['SENDER_ACCOUNT_ID'], df['RECEIVER_ACCOUNT_ID'])
    amount = df['TX_AMOUNT'].to_numpy(dtype=float)
    y = df['IS_FRAUD'].astype(str).str.lower().isin(['1', 'true']).to_numpy().astype(int)
    splits = {'random (as GNN)': load_masks(AMLSIM['graph'], len(df))[0],
              'temporal': temporal_masks(df['TIMESTAMP'].to_numpy())}

    def features(masks):
        base = np.column_stack([np.log1p(amount), (amount < 20).astype(float)]).astype(np.float32)
        return np.hstack([base, account_aggregate_features(src, dst, amount, masks['train'], n_acc),
                          pair_count_feature(src, dst, masks['train'], n_acc)])
    return y, splits, features


def build_ibm():
    df = pd.read_csv(IBM['raw'])
    df.columns = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received',
                  'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']
    (src, dst), n_acc = factorize_accounts(df['Account'], df['Account.1'])
    paid = df['Amount Paid'].to_numpy(dtype=float)
    y = df['Is Laundering'].to_numpy().astype(int)
    ts = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    base = [np.log1p(paid), np.log1p(df['Amount Received'].to_numpy(dtype=float)),
            (df['Payment Currency'] != df['Receiving Currency']).to_numpy(float),
            (src == dst).astype(float), (df['From Bank'] != df['To Bank']).to_numpy(float),
            ts.dt.hour.to_numpy(float), ts.dt.dayofweek.to_numpy(float)]
    base += [(df['Payment Format'] == f).to_numpy(float) for f in PAYMENT_FORMATS]
    base = np.column_stack(base).astype(np.float32)
    splits = {'chronological (as GNN)': load_masks(IBM['graph'], len(df))[0]}

    def features(masks):
        return np.hstack([base, account_aggregate_features(src, dst, paid, masks['train'], n_acc),
                          pair_count_feature(src, dst, masks['train'], n_acc)])
    return y, splits, features


def build_elliptic():
    import scipy.sparse as sp
    feats = pd.read_csv(ELLIPTIC['features'], header=None)
    tx_ids = feats[0].to_numpy()
    steps = feats[1].to_numpy()
    values = feats.iloc[:, 2:].to_numpy(dtype=np.float32)          # time_step deliberately excluded
    local, aggregated = values[:, :93], values[:, 93:]
    masks, y = load_masks(ELLIPTIC['graph'], len(feats))
    edges = pd.read_csv(ELLIPTIC['edges'])
    index = pd.Series(np.arange(len(tx_ids)), index=tx_ids)
    s, t = index.reindex(edges.iloc[:, 0]).to_numpy(), index.reindex(edges.iloc[:, 1]).to_numpy()
    ok = ~(np.isnan(s) | np.isnan(t))
    s, t = s[ok].astype(int), t[ok].astype(int)
    n = len(tx_ids)
    adj = sp.coo_matrix((np.ones(len(s)), (s, t)), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float32)                   # undirected, binary
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)
    neigh = np.asarray(adj @ local) / deg[:, None]                 # mean of neighbours' local features; no labels
    sets = {'A local only': local, 'B all 165 (no time_step)': np.hstack([local, aggregated]),
            'C local + neighbour mean': np.hstack([local, neigh])}
    return y, masks, sets, steps


# --------------------------------------------------------------------------- models
def make_model(name, seed):
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight='balanced',
                                              random_state=seed)
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced_subsample',
                                  n_jobs=-1, random_state=seed)


def fit_predict(name, seed, X, y, masks, neg_frac=1.0):
    """Fit on train (optionally negative-subsampled with weights), tune on val, return val/test probabilities."""
    rng = np.random.default_rng(seed)
    train = np.flatnonzero(masks['train'])
    weight = None
    if neg_frac < 1.0:
        keep = (y[train] == 1) | (rng.random(len(train)) < neg_frac)
        weight = np.where(y[train][keep] == 1, 1.0, 1.0 / neg_frac)
        train = train[keep]
    model = make_model(name, seed)
    t0 = time.time()
    model.fit(X[train], y[train], sample_weight=weight)
    fit_s = time.time() - t0
    val, test = np.flatnonzero(masks['val']), np.flatnonzero(masks['test'])
    return model.predict_proba(X[val])[:, 1], model.predict_proba(X[test])[:, 1], val, test, fit_s


def run_edge_dataset(name, y, splits, features, seeds, models, neg_frac):
    results = {}
    for split_name, masks in splits.items():
        X = features(masks)
        print(f"\n[{name}] split: {split_name}  rows: train {masks['train'].sum():,} / val "
              f"{masks['val'].sum():,} / test {masks['test'].sum():,}  features: {X.shape[1]}", flush=True)
        for model_name in models:
            runs, times = [], []
            for seed in range(seeds):
                p_val, p_test, val, test, fit_s = fit_predict(model_name, seed, X, y, masks, neg_frac)
                thr = tune_threshold(y[val], p_val)
                runs.append(evaluate(y[test], p_test, thr))
                times.append(fit_s)
                print(f"   {model_name} seed {seed}: fit {fit_s:.1f}s  PR-AUC {runs[-1]['pr_auc']:.3f}  "
                      f"F1 {runs[-1]['f1']:.3f}", flush=True)
            results[f"{model_name} | {split_name}"] = dict(summary=summarise(runs), fit_seconds=times, seeds=seeds)
    return results


def run_elliptic(seeds, models):
    y, masks, sets, steps = build_elliptic()
    results, per_step = {}, {}
    for set_name, X in sets.items():
        for model_name in models:
            runs, step_runs = [], []
            for seed in range(seeds):
                p_val, p_test, val, test, fit_s = fit_predict(model_name, seed, X, y, masks)
                thr = tune_threshold(y[val], p_val)
                runs.append(evaluate(y[test], p_test, thr))
                rows = {}
                for step in TEST_STEPS:
                    m = steps[test] == step
                    if m.any():
                        rows[step] = evaluate(y[test][m], p_test[m], thr)
                step_runs.append(rows)
                print(f"   {model_name} on {set_name}, seed {seed}: fit {fit_s:.1f}s  F1 {runs[-1]['f1']:.3f}",
                      flush=True)
            key = f"{model_name} | {set_name}"
            results[key] = dict(summary=summarise(runs), seeds=seeds)
            per_step[key] = {step: {k: float(np.nanmean([r[step][k] for r in step_runs if step in r]))
                                    for k in ('precision', 'recall', 'f1')} | {'n_pos': step_runs[0][step]['n_pos']}
                             for step in step_runs[0]}
    gnn = gnn_per_step()
    if gnn:
        per_step['GNN (existing predictions)'] = gnn
    return results, per_step


def gnn_per_step():
    """Per-time-step test metrics of the existing GNN predictions at its saved threshold (no retraining)."""
    if not (os.path.exists(ELLIPTIC['predictions']) and os.path.exists(ELLIPTIC['metrics'])):
        return None
    pred = pd.read_csv(ELLIPTIC['predictions'])
    thr = json.load(open(ELLIPTIC['metrics'])).get('best_threshold', 0.5)
    out = {}
    for step in TEST_STEPS:
        m = (pred['TimeStep'] == step) & pred['class'].isin(['Illicit', 'Licit'])
        if m.any():
            r = evaluate((pred.loc[m, 'class'] == 'Illicit').to_numpy().astype(int),
                         pred.loc[m, 'SUSPICION_SCORE'].to_numpy(), thr)
            out[step] = dict(precision=r['precision'], recall=r['recall'], f1=r['f1'], n_pos=r['n_pos'])
    return out


def reported_gnn(path):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    return dict(pr_auc=m.get('test_pr_auc'), roc_auc=m.get('test_roc_auc'), precision=m.get('test_precision'),
                recall=m.get('test_recall'), f1=m.get('test_f1_tuned_thresh', m.get('test_f1')),
                threshold=m.get('best_threshold'))


def print_table(results, gnn):
    print(f"\n{'model | split':58s} {'PR-AUC':>13s} {'ROC-AUC':>13s} {'precision':>13s} {'recall':>13s} "
          f"{'F1':>13s} {'P@k':>13s}")
    for key, res in results.items():
        s = res['summary']
        cells = [f"{s[k]['mean']:.3f}+/-{s[k]['std']:.3f}" for k in
                 ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'precision_at_k')]
        print(f"{key:58s} " + " ".join(f"{c:>13s}" for c in cells))
    if gnn:
        cells = [f"{gnn[k]:.3f}" if gnn.get(k) is not None else "n/a" for k in
                 ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1')]
        print(f"{'GNN, as reported in its metrics file (leak-prone)':58s} " + " ".join(f"{c:>13s}" for c in cells))


def print_per_step(per_step):
    print("\nElliptic per-time-step test F1 (precision / recall / F1; n illicit) at each model's validation-tuned threshold")
    steps = list(TEST_STEPS)
    print(f"{'model':40s} " + " ".join(f"{s:>6d}" for s in steps))
    for key, rows in per_step.items():
        print(f"{key:40s} " + " ".join(f"{rows[s]['f1']:6.2f}" if s in rows else f"{'-':>6s}" for s in steps))
    anyrows = next(iter(per_step.values()))
    print(f"{'n illicit in step':40s} " + " ".join(f"{anyrows[s]['n_pos']:6d}" if s in anyrows else f"{'-':>6s}" for s in steps))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=['amlsim', 'ibm', 'elliptic'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--models', default='hgb,rf', help='comma list from: hgb, rf')
    parser.add_argument('--neg-frac', type=float, default=1.0,
                        help='fraction of negative TRAIN rows used for fitting (weighted back); use <1 for IBM')
    args = parser.parse_args()
    models = args.models.split(',')
    t0 = time.time()
    os.makedirs('outputs', exist_ok=True)

    if args.dataset == 'elliptic':
        results, per_step = run_elliptic(args.seeds, models)
        print_table(results, reported_gnn(ELLIPTIC['metrics']))
        print_per_step(per_step)
        payload = dict(results=results, per_step=per_step)
    else:
        builder, cfg = (build_amlsim, AMLSIM) if args.dataset == 'amlsim' else (build_ibm, IBM)
        y, splits, features = builder()
        results = run_edge_dataset(args.dataset, y, splits, features, args.seeds, models, args.neg_frac)
        print_table(results, reported_gnn(cfg['metrics']))
        payload = dict(results=results)

    payload.update(dataset=args.dataset, seeds=args.seeds, neg_frac=args.neg_frac,
                   runtime_seconds=time.time() - t0, gnn_reported=reported_gnn(
                       {'amlsim': AMLSIM, 'ibm': IBM, 'elliptic': ELLIPTIC}[args.dataset]['metrics']))
    out = f'outputs/baselines_{args.dataset}.json'
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nTotal runtime {time.time() - t0:.0f}s. Results written to {out}")


if __name__ == '__main__':
    sys.exit(main())
