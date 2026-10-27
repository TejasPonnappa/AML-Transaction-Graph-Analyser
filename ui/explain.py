"""
Explanation helpers for the IBM page: loads what src/explain_ibm.py wrote and turns it into things the page can show
(unusual signals, the path a transaction takes through the surrogate tree, detector badges).
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

EXPLAIN_DIR = 'outputs/.graph_cache/ui_data/ibm_explain'
BINARY_PREFIXES = ('format_',)
BINARY = {'currency_mismatch', 'self_loop', 'bank_mismatch'}
STATIC = {'hour', 'day_of_week', 'log_paid', 'log_received', 'currency_mismatch', 'self_loop', 'bank_mismatch'}
BADGES = {'fan_out': 'Fan-out', 'fan_in': 'Fan-in', 'cycle': 'Cycle', 'pass_through': 'Pass-through'}
BADGE_MIN_VAL_PRECISION = 0.20      # a detector is shown as a badge only if it was at least this precise on validation


def is_binary(feature):
    return feature in BINARY or (feature or '').startswith(BINARY_PREFIXES)


@st.cache_resource(show_spinner="Loading explanations...")
def load_explain(mtime):
    """Returns the explanation bundle, or None when src/explain_ibm.py has not been run. `mtime` keys the cache."""
    needed = ('tree.json', 'quantiles.json', 'detail.parquet', 'metrics.json')
    if not all(os.path.exists(os.path.join(EXPLAIN_DIR, f)) for f in needed):
        return None
    tree = json.load(open(os.path.join(EXPLAIN_DIR, 'tree.json')))
    log_set = set(json.load(open(os.path.join(EXPLAIN_DIR, 'quantiles.json')))['log1p'])
    for node in tree['nodes']:
        node['binary'] = is_binary(node['feature'])
        if not node['leaf']:                                   # readable split text for the picture
            node['text'] = (f"{node['label']}?" if node['binary'] else
                            f"{node['label']} <= {display_value(node['feature'], node['threshold'], node['feature'] in log_set)}")
    quant = json.load(open(os.path.join(EXPLAIN_DIR, 'quantiles.json')))
    quant['values'] = {k: np.asarray(v) for k, v in quant['values'].items()}
    detail = pd.read_parquet(os.path.join(EXPLAIN_DIR, 'detail.parquet')).set_index('row_id')
    metrics = json.load(open(os.path.join(EXPLAIN_DIR, 'metrics.json')))
    val = metrics['detectors']['val']
    active = {k for k, v in val.items() if v['precision_any_laundering'] >= BADGE_MIN_VAL_PRECISION}
    return dict(tree=tree, quant=quant, detail=detail, metrics=metrics, active_badges=active)


def explain_bundle():
    path = os.path.join(EXPLAIN_DIR, 'tree.json')
    return load_explain(os.path.getmtime(path) if os.path.exists(path) else 0.0)


def display_value(feature, stored, log1p):
    """Human-readable value of a stored feature (counts, sums and durations were stored as log1p)."""
    v = float(np.expm1(stored)) if log1p else float(stored)
    if 'secs' in feature:
        if v >= 86400 * 2:
            return f"{v / 86400:.1f} days"
        if v >= 3600:
            return f"{v / 3600:.1f} h"
        return f"{v / 60:.0f} min" if v >= 60 else f"{v:.0f} s"
    if 'sum' in feature or feature in ('log_paid', 'log_received'):
        return f"{v:,.0f}"
    if feature in ('hour', 'day_of_week'):
        return f"{v:.0f}"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-6 else f"{v:,.1f}"


def percentile(quant, feature, stored):
    """Where a stored value sits among normal-volume transactions of the evaluation period (0-100)."""
    grid = np.asarray(quant['grid'])
    vals = quant['values'][feature]
    return float(100 * grid[min(int(np.searchsorted(vals, stored, side='right')) - 1, len(grid) - 1)]) if stored >= vals[0] else 0.0


def unusual_signals(bundle, row, labels, top=6, minimum=95.0):
    """The row's features that are most extreme compared with normal days: [{label, value, percentile}]."""
    quant, found = bundle['quant'], []
    log_set = set(quant['log1p'])
    for feature in quant['values']:
        if feature in STATIC or feature.startswith('format_'):
            continue
        stored = float(row[feature])
        if 'secs' in feature and stored >= 0.99 * quant['values'][feature][-1]:
            continue                                            # the "no earlier transaction" placeholder is not a signal
        p = percentile(quant, feature, stored)
        if p >= minimum:
            found.append(dict(label=labels.get(feature, feature), value=display_value(feature, stored, feature in log_set),
                              percentile=p))
    return sorted(found, key=lambda d: -d['percentile'])[:top]


def tree_path(bundle, row):
    """The path of this transaction through the surrogate tree: [{id, text}], one entry per node visited."""
    nodes = {n['id']: n for n in bundle['tree']['nodes']}
    log_set = set(bundle['quant']['log1p'])
    steps, current = [], 0
    while True:
        n = nodes[current]
        if n['leaf']:
            verdict = 'alert' if n['predict_alert'] else 'no alert'
            steps.append(dict(id=current, text=f"Leaf: {n['alert_frac'] * 100:.0f}% of {n['n']:,} similar transactions "
                                               f"are alerts, so the tree says {verdict}"))
            return steps
        stored = float(row[n['feature']])
        goes_left = stored <= n['threshold']
        if n['binary']:
            steps.append(dict(id=current, text=f"{n['label']}? {'No' if goes_left else 'Yes'}"))
        else:
            thr = n['threshold']
            shown_thr = display_value(n['feature'], thr, n['feature'] in log_set)
            value = display_value(n['feature'], stored, n['feature'] in log_set)
            steps.append(dict(id=current, text=f"{n['label']} is {value}, which is {'at most' if goes_left else 'above'} "
                                               f"{shown_thr}"))
        current = n['left'] if goes_left else n['right']


def badges_for(detail_row, active=None):
    """Detector badges for a detail row, in a fixed order. `active` limits them to detectors precise enough to show."""
    return [label for key, label in BADGES.items()
            if (active is None or key in active) and bool(detail_row.get(f'det_{key}', False))]


def badge_text(row, active=None):
    labels = badges_for(row, active)
    return ' | '.join(labels) if labels else ''
