"""
Builds what the dashboard's explanation views need for the IBM RandomForest, in one pass (about 10 minutes, about 3 GB):

    python src/explain_ibm.py

Reads the transactions, the benchmark pattern file, the fitted forest and its scores; rebuilds the 43 past-only features;
and writes to outputs/.graph_cache/ui_data/ibm_explain/:
  tree.json         a depth-8 decision tree that imitates the forest's alert decisions, with its fidelity
  quantiles.json    per-feature reference quantiles on normal-volume days, for "unusual signal" percentiles
  detail.parquet    features, forest votes, benchmark scheme and detector flags for the rows worth explaining
  metrics.json      detector precision/recall per scheme, model recall per scheme, analyst-workload figures

The surrogate tree is an approximation of the forest, not the forest: its agreement with the forest is reported.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from baselines import factorize_accounts, load_masks
from ibm_detectors import ALL_NAMES, DETECTORS, detector_scores, evaluate_detectors, run_detectors
from ibm_features import (FEATURE_LABELS, build_past_only_features, build_past_only_graph_features,
                          low_volume_tail_mask)
from ibm_patterns import SCHEMES, UNATTRIBUTED, label_rows, parse_patterns

RAW_TRANS_PATH = 'data/raw/ibm_aml/HI-Small_Trans.csv'
PATTERNS_PATH = 'data/raw/ibm_aml/HI-Small_Patterns.txt'
GRAPH_PATH = 'data/processed/graph_data_ibm.pt'
SCORES_PATH = 'outputs/ibm_predictions_rf.csv'
METRICS_PATH = 'outputs/metrics_ibm_rf.json'
MODEL_PATH = 'outputs/model_ibm_rf.joblib'
OUT_DIR = 'outputs/.graph_cache/ui_data/ibm_explain'
COLUMNS = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency',
           'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']
UNSCALED = {'currency_mismatch', 'self_loop', 'bank_mismatch', 'hour', 'day_of_week'}
LOG_FEATURES = [n for n in ALL_NAMES if n not in UNSCALED and not n.startswith('format_')]
DETAIL_MIN_SCORE = 0.05      # rows scoring at least this (or labelled laundering) get a detail record
VOTES_MIN_SCORE = 0.20       # per-tree votes are computed for rows scoring at least this
QUANTILE_GRID = np.r_[np.arange(0, 100), 99.5, 99.9, 100.0] / 100.0
WORKLOAD_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05)


def export_tree(clf, X, alert, y, names):
    """Plain-dict export of a fitted sklearn tree, with unweighted counts of rows, alerts and laundering per node."""
    t = clf.tree_
    path = clf.decision_path(X)
    n = np.asarray(path.sum(axis=0)).ravel()
    alerts = np.asarray(path.T @ alert.astype(float)).ravel()
    laundering = np.asarray(path.T @ y.astype(float)).ravel()
    nodes = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        feat = None if leaf else names[t.feature[i]]
        nodes.append(dict(
            id=i, left=int(t.children_left[i]), right=int(t.children_right[i]), leaf=bool(leaf),
            feature=feat, label=None if leaf else FEATURE_LABELS.get(feat, feat),
            log1p=bool(feat in LOG_FEATURES), threshold=None if leaf else float(t.threshold[i]),
            n=int(n[i]), alert_frac=float(alerts[i] / max(n[i], 1)), laundering_rate=float(laundering[i] / max(n[i], 1)),
            predict_alert=bool(leaf and alerts[i] / max(n[i], 1) >= 0.5)))
    return dict(max_depth=int(clf.get_depth()), n_nodes=int(t.node_count), nodes=nodes)


def account_workload(src, dst, score, y, mask, n_accounts, fractions=WORKLOAD_FRACTIONS):
    """
    Account-level view of the rows in `mask`: an account's score is the highest score among its transactions (as sender
    or receiver); it is a laundering account if it takes part in any laundering transaction. Returns, for the top
    fraction of accounts by score, the share of laundering accounts caught and the precision of that review list.
    """
    s, r = src[mask], dst[mask]
    sc, yy = score[mask], y[mask] == 1
    best = np.full(n_accounts, -1.0)
    np.maximum.at(best, s, sc)
    np.maximum.at(best, r, sc)
    bad = np.zeros(n_accounts, dtype=bool)
    bad[s[yy]] = True
    bad[r[yy]] = True
    present = best >= 0
    order = np.argsort(-best[present])
    flagged_bad = bad[present][order]
    total_bad = int(flagged_bad.sum())
    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * present.sum())))
        caught = int(flagged_bad[:k].sum())
        rows.append(dict(top_fraction=frac, accounts_reviewed=k, laundering_accounts_caught=caught,
                         share_of_laundering_accounts=caught / max(total_bad, 1), precision=caught / k))
    return dict(accounts=int(present.sum()), laundering_accounts=total_bad, rows=rows)


def alerts_per_day_for_recall(score, y, days, recalls=(0.5, 0.8)):
    """Alerts per day needed to reach a given transaction-level recall when reviewing highest scores first."""
    order = np.argsort(-score)
    hit = np.cumsum(y[order] == 1)
    total = max(int((y == 1).sum()), 1)
    out = {}
    for target in recalls:
        k = int(np.searchsorted(hit, target * total, side='left')) + 1
        out[str(target)] = dict(alerts=k, alerts_per_day=k / max(days, 1e-9), precision=float(hit[min(k, len(hit)) - 1] / k))
    return out


def scheme_recall(scheme, score, threshold, mask):
    """Share of each scheme's transactions that score at or above the threshold."""
    out = {}
    for name in SCHEMES + [UNATTRIBUTED]:
        m = mask & (scheme == name)
        if m.any():
            out[name] = dict(rows=int(m.sum()), recall=float((score[m] >= threshold).mean()),
                             mean_score=float(score[m].mean()))
    return out


def surrogate_fidelity(clf, X, alert, y, mask, normal):
    """Agreement between the tree's alert decision and the forest's, overall and on normal-volume rows."""
    out = {}
    for name, m in (('pooled', mask), ('normal-volume days', mask & normal)):
        pred = clf.predict(X[m])
        a = alert[m]
        tp = int((pred & a).sum())
        out[name] = dict(
            rows=int(m.sum()), forest_alerts=int(a.sum()), tree_alerts=int(pred.sum()),
            agreement=float((pred == a).mean()),
            alert_precision=tp / max(int(pred.sum()), 1), alert_recall=tp / max(int(a.sum()), 1),
            auc=float(roc_auc_score(a, clf.predict_proba(X[m])[:, 1])) if 0 < a.sum() < len(a) else float('nan'))
    return out


def forest_votes(model, X, chunk=20_000):
    """Per-tree positive-class probabilities for the given rows: (rows, trees)."""
    parts = []
    for i in range(0, len(X), chunk):
        block = X[i:i + chunk]
        parts.append(np.column_stack([est.predict_proba(block)[:, 1] for est in model.estimators_]).astype(np.float32))
    return np.vstack(parts)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_TRANS_PATH)
    df.columns = COLUMNS
    n = len(df)
    (src, dst), n_acc = factorize_accounts(df['Account'], df['Account.1'])
    y = df['Is Laundering'].to_numpy().astype(int)
    ts = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    t = ((ts - ts.min()).dt.total_seconds()).to_numpy(dtype=np.int64)
    paid = df['Amount Paid'].to_numpy(dtype=float)
    normal = ~low_volume_tail_mask(ts)
    masks, _ = load_masks(GRAPH_PATH, n)
    threshold = json.load(open(METRICS_PATH))['best_threshold']
    score = pd.read_csv(SCORES_PATH, usecols=['SUSPICION_SCORE'])['SUSPICION_SCORE'].to_numpy(dtype=np.float32)
    assert len(score) == n, "predictions and transactions are out of step"
    print(f"1. loaded {n:,} rows, threshold {threshold:.4f} ({time.time() - t0:.0f}s)", flush=True)

    scheme, attempt = label_rows(df, parse_patterns(PATTERNS_PATH))
    print("   scheme labels:", pd.Series(scheme[y == 1]).value_counts().to_dict(), flush=True)

    past, _ = build_past_only_features(df, src, dst)
    graph, _ = build_past_only_graph_features(df, src, dst)
    X = np.hstack([past, graph])
    del past, graph
    print(f"2. features {X.shape} ({time.time() - t0:.0f}s)", flush=True)

    # detectors: thresholds from the training period only
    det_scores = detector_scores(X, src, dst, t, paid)
    flags = run_detectors(det_scores, scheme, y, masks['train'])
    detectors = {split: evaluate_detectors(flags, scheme, y, masks[split]) for split in ('train', 'val', 'test')}
    detectors['test_normal_volume'] = evaluate_detectors(flags, scheme, y, masks['test'] & normal)

    # model recall per scheme, workload, alerts per day
    test, test_normal = masks['test'], masks['test'] & normal
    day = ts.dt.floor('D')
    days_all = day[test].nunique()
    days_normal = day[test_normal].nunique()
    metrics = dict(
        threshold=float(threshold), schemes_in_data={k: int(v) for k, v in pd.Series(scheme[y == 1]).value_counts().items()},
        detectors=detectors,
        model_recall_by_scheme=dict(test=scheme_recall(scheme, score, threshold, test),
                                    test_normal_volume=scheme_recall(scheme, score, threshold, test_normal)),
        workload=dict(
            accounts_test=account_workload(src, dst, score, y, test, n_acc),
            accounts_test_normal_volume=account_workload(src, dst, score, y, test_normal, n_acc),
            alerts_per_day_at_threshold=dict(
                test=float((score[test] >= threshold).sum() / max(days_all, 1)),
                test_normal_volume=float((score[test_normal] >= threshold).sum() / max(days_normal, 1))),
            alerts_per_day_for_recall=dict(
                test=alerts_per_day_for_recall(score[test], y[test], days_all),
                test_normal_volume=alerts_per_day_for_recall(score[test_normal], y[test_normal], days_normal)),
            test_days=int(days_all), test_normal_volume_days=int(days_normal)))
    print(f"3. detectors and metrics ({time.time() - t0:.0f}s)", flush=True)

    # surrogate tree on validation rows, fidelity on test rows
    alert = score >= threshold
    fit = masks['val']
    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight={False: 1, True: 10}, random_state=0)  # leaves of >= 10 rows
    clf.fit(X[fit], alert[fit])
    tree = export_tree(clf, X[fit], alert[fit], y[fit], ALL_NAMES)
    tree['fidelity'] = surrogate_fidelity(clf, X, alert, y, test, normal)
    tree['fit_rows'] = int(fit.sum())
    tree['feature_labels'] = FEATURE_LABELS
    with open(os.path.join(OUT_DIR, 'tree.json'), 'w') as f:
        json.dump(tree, f)
    metrics['surrogate_fidelity'] = tree['fidelity']
    print(f"4. surrogate tree, nodes {tree['n_nodes']} ({time.time() - t0:.0f}s)", flush=True)

    # reference quantiles on normal-volume rows of the evaluation periods
    ref = (masks['val'] | masks['test']) & normal
    quant = np.quantile(X[ref].astype(np.float64), QUANTILE_GRID, axis=0)
    with open(os.path.join(OUT_DIR, 'quantiles.json'), 'w') as f:
        json.dump(dict(grid=QUANTILE_GRID.tolist(), log1p=LOG_FEATURES,
                       values={name: quant[:, i].tolist() for i, name in enumerate(ALL_NAMES)}), f)

    # detail rows: everything scoring above a floor plus all laundering rows
    sel = np.flatnonzero((score >= DETAIL_MIN_SCORE) | (y == 1))
    detail = pd.DataFrame(X[sel], columns=ALL_NAMES)
    detail.insert(0, 'row_id', sel.astype(np.int32))
    detail['scheme'] = pd.Categorical(scheme[sel])
    for name in DETECTORS:
        detail[f'det_{name}'] = flags[name][1][sel]
    votes_rows = np.flatnonzero(score[sel] >= VOTES_MIN_SCORE)
    p = forest_votes(joblib.load(MODEL_PATH), X[sel[votes_rows]])
    votes = np.full(len(sel), -1, dtype=np.int16)
    votes[votes_rows] = (p >= 0.5).sum(axis=1)
    hist = np.full((len(sel), 10), -1, dtype=np.int16)
    hist[votes_rows] = np.stack([np.histogram(r, bins=10, range=(0, 1))[0] for r in p])
    detail['forest_votes'] = votes
    detail['n_trees'] = np.int16(p.shape[1])
    for b in range(10):
        detail[f'tree_hist_{b}'] = hist[:, b]
    detail.to_parquet(os.path.join(OUT_DIR, 'detail.parquet'), index=False)
    metrics['detail_rows'] = int(len(sel))
    metrics['vote_rows'] = int(len(votes_rows))
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"5. wrote {len(sel):,} detail rows ({len(votes_rows):,} with votes) to {OUT_DIR} ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == '__main__':
    main()
