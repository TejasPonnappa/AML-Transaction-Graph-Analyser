"""Elliptic (node classification) dashboard, laid out like the IBM page: KPI row and four views, only the selected one computed."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ui import theme
from ui.components import network
from ui.data_store import ELLIPTIC, load_elliptic, load_elliptic_edges, predictions_older_than
from ui.metrics_loader import get_threshold, has_tuned_threshold, load_metrics

METRICS_PATH = 'outputs/metrics_bitcoin.json'
KEY = 'ell'
VIEWS = ["Overview", "Alerts", "Investigate", "Evaluation"]
SPLIT_CHOICES = {
    "Test (held out)": ["test"],
    "Validation": ["val"],
    "Training": ["train"],
    "Unlabelled": ["unlabeled"],
    "All nodes": ["train", "val", "test", "unlabeled"],
}
TABLE_COLUMNS = ["txId", "TimeStep", "class", "SUSPICION_SCORE", "SPLIT"]
HERO_MAX_NODES = 100


def indicative_typology(account_id, edges, has_alert):
    """Rule-based reading of the node's degree pattern. Indicative only; not a model output."""
    if not has_alert:
        return "No alert on this transaction"
    in_degree = int((edges['target'] == account_id).sum())
    out_degree = int((edges['source'] == account_id).sum())
    patterns = []
    is_mule = abs(in_degree - out_degree) > 2
    if is_mule:
        if in_degree > out_degree:
            patterns.append("Mule-like (consolidation)")
            if in_degree > 5:
                patterns.append("Smurfing inputs suspected")
        else:
            patterns.append("Mule-like (dispersal)")
    if not is_mule and in_degree >= 1 and out_degree >= 1:
        patterns.append("Layering / pass-through")
    return ", ".join(patterns) if patterns else "Alert present; no rule-based typology matched"


# --------------------------------------------------------------------------- filters and graph builders
def _sidebar_filters(df, metrics):
    st.sidebar.markdown("**Filters**")
    has_split = 'SPLIT' in df.columns
    choices = list(SPLIT_CHOICES) if has_split else ["All nodes"]
    split_label = st.sidebar.selectbox("Data split", choices, index=0, key=f"{KEY}_split",
                                       help="Labels of test nodes were never used for training or threshold tuning.")
    if not has_split:
        st.sidebar.caption("Split information is unavailable (processed graph not found), so every view "
                           "includes training nodes.")
    if not has_tuned_threshold(metrics):
        st.sidebar.caption("No tuned threshold found in the metrics file; using 0.50.")
    threshold = st.sidebar.slider("Alert threshold (model score)", 0.0, 1.0,
                                  float(min(max(get_threshold(metrics), 0.0), 1.0)), 0.005, key=f"{KEY}_thr",
                                  help="Defaults to the threshold tuned on validation data.")
    lo, hi = int(df['TimeStep'].min()), int(df['TimeStep'].max())
    mask = df['SPLIT'].isin(SPLIT_CHOICES[split_label]) if has_split else pd.Series(True, index=df.index)
    if hi > lo:
        a, b = st.sidebar.slider("Time step", lo, hi, (lo, hi), key=f"{KEY}_steps")
        mask &= df['TimeStep'].between(a, b)
    return df[mask], threshold, split_label


def _graph_dicts(sub_edges, scores, labels, threshold, centre=None):
    """Node and edge dicts for the network component. scores / labels map txId -> value."""
    ids = sorted(set(sub_edges['source']) | set(sub_edges['target']))
    nodes = []
    for tx in ids:
        s = float(scores.get(tx, 0.0))
        label = str(labels.get(tx, 'Unknown'))
        nodes.append(dict(id=str(tx), label=str(tx), score=s, alert=bool(s >= threshold),
                          size=float(34 if tx == centre else 13 + 20 * s), center=bool(tx == centre),
                          labelled=label == 'Illicit', badges=[],
                          tip=f"Transaction {tx}\nModel score {s:.4f}\nLabel {label}"))
    edges = [dict(id=f"{u}>{v}", source=str(u), target=str(v), score=float(max(scores.get(u, 0.0), scores.get(v, 0.0))),
                  alert=bool(max(scores.get(u, 0.0), scores.get(v, 0.0)) >= threshold), kind='normal', w=1.4,
                  tip=f"{u} to {v}") for u, v in zip(sub_edges['source'], sub_edges['target'])]
    return nodes, edges


def _alert_clusters(edges, alert_ids, max_nodes=HERO_MAX_NODES):
    """Edges among alert nodes, limited to the largest connected clusters. Returns an edge frame (possibly empty)."""
    among = edges[edges['source'].isin(alert_ids) & edges['target'].isin(alert_ids)]
    if among.empty:
        return among
    codes, inv = np.unique(np.concatenate([among['source'].to_numpy(), among['target'].to_numpy()]), return_inverse=True)
    n = len(codes)
    si, ti = inv[:len(among)], inv[len(among):]
    ncomp, labels = connected_components(coo_matrix((np.ones(len(si)), (si, ti)), shape=(n, n)), directed=False)
    size = np.bincount(labels, minlength=ncomp)
    keep, total = [], 0
    for comp in np.argsort(-size):
        if total and total + size[comp] > max_nodes:
            continue
        keep.append(comp)
        total += int(size[comp])
        if total >= max_nodes:
            break
    ok = np.isin(labels, keep)
    return among[ok[si]]


def _handle_click(result, nonce_key):
    if isinstance(result, dict) and result.get('type') == 'node' and st.session_state.get(nonce_key) != result.get('nonce'):
        st.session_state[nonce_key] = result.get('nonce')
        st.session_state['_ell_goto'] = result['id']
        st.rerun()


# --------------------------------------------------------------------------- views
def _overview(view, edges, alerts, threshold, metrics, is_alert):
    left, right = st.columns([5, 2])
    with left:
        theme.section("Largest alert clusters",
                      "Alert transactions linked to each other, drawn as a network. Click a transaction to open its case.")
        alert_ids = set(alerts['txId'])
        sub = _alert_clusters(edges, alert_ids)
        if sub.empty:
            theme.empty_state("No alert transactions are linked to each other at this threshold.")
        else:
            scores = view.set_index('txId')['SUSPICION_SCORE'].to_dict()
            labels = view.set_index('txId')['class'].astype(str).to_dict()
            nodes, links = _graph_dicts(sub, scores, labels, threshold)
            result = network(nodes, links, key="ell_hero", sig=f"hero-{len(view)}-{threshold:.4f}-{len(nodes)}",
                             layout="cose", height=560)
            _handle_click(result, "_ell_hero_nonce")
            st.caption(f"{len(nodes)} transactions in the largest clusters, from {len(alerts):,} alerts in this view.")
    with right:
        theme.section("Score distribution", "Log scale, so the alert tail stays visible.")
        theme.score_histogram(view['SUSPICION_SCORE'], threshold)
    theme.section("Alerts by time step", "Number of alerts per time step in the current view.")
    per_step = view[is_alert].groupby('TimeStep').size().reindex(
        range(int(view['TimeStep'].min()), int(view['TimeStep'].max()) + 1), fill_value=0)
    fig = go.Figure(go.Bar(x=per_step.index, y=per_step.values, marker_color=theme.NEUTRAL,
                           hovertemplate="step %{x}<br>%{y:,} alerts<extra></extra>"))
    fig.update_layout(height=240, xaxis_title="Time step", yaxis_title="Alerts")
    st.plotly_chart(fig, use_container_width=True)


def _alerts(alerts, threshold):
    theme.section("Alert queue", f"Transactions at or above the alert threshold, highest model score first. "
                                 f"{theme.fmt_int(len(alerts))} in view.")
    if alerts.empty:
        theme.empty_state("No alerts at this threshold. Lower the threshold in the sidebar to see more.")
        return
    table = alerts[TABLE_COLUMNS].copy()
    table['txId'] = table['txId'].astype(str)
    theme.data_table(table, key="ell_alerts", file_name="elliptic_alerts.csv", threshold=threshold)


def _investigate(df, edges, alerts, threshold):
    theme.section("Case investigation", "Review one transaction and its direct connections. Scores of training "
                                        "nodes are optimistic; the case header does not distinguish splits.")
    top_ids = alerts['txId'].head(25).astype(str).tolist() if len(alerts) else []
    left, right = st.columns(2)
    left.selectbox("Highest-scoring alerts in the current view", [""] + top_ids, key=f"{KEY}_suggest")
    typed = right.text_input("Or enter a transaction ID", key=f"{KEY}_account").strip()
    raw = typed or st.session_state.get(f"{KEY}_suggest", "")
    if not raw:
        theme.empty_state("Select or enter a transaction ID, or click a transaction in the Overview network.")
        return
    if not raw.isascii() or not raw.isdigit():
        st.error("Transaction IDs are whole numbers, for example 230425980.")
        return
    tx = int(raw)
    node = df[df['txId'] == tx]
    if node.empty:
        st.error(f"Transaction {tx} was not found in the scored data.")
        return

    score = float(node['SUSPICION_SCORE'].iloc[0])
    case_edges = edges[(edges['source'] == tx) | (edges['target'] == tx)]
    neighbours = (set(case_edges['source']) | set(case_edges['target'])) - {tx}
    has_alert = score >= threshold
    theme.kpi_row([
        {"label": "Model score", "value": f"{score:.4f}", "alert": has_alert},
        {"label": "Status", "value": str(theme.risk_band([score], threshold)[0])},
        {"label": "Label", "value": str(node['class'].iloc[0])},
        {"label": "Time step", "value": str(int(node['TimeStep'].iloc[0]))},
        {"label": "Direct connections", "value": theme.fmt_int(len(case_edges)), "help": f"{len(neighbours):,} neighbours"},
    ])
    st.markdown(f"**Indicative typology (rule-based):** {indicative_typology(tx, edges, has_alert)}")

    theme.section("Connection network", "Click a transaction to re-centre on it.")
    local = df[df['txId'].isin(neighbours | {tx})]
    scores = local.set_index('txId')['SUSPICION_SCORE'].to_dict()
    labels = local.set_index('txId')['class'].astype(str).to_dict()
    shown = case_edges.assign(_s=case_edges['source'].map(scores).fillna(0) + case_edges['target'].map(scores).fillna(0)) \
        .nlargest(120, '_s')
    nodes, links = _graph_dicts(shown, scores, labels, threshold, centre=tx)
    result = network(nodes, links, key="ell_ego", sig=f"ego-{tx}-{threshold:.4f}", layout="concentric", height=520)
    _handle_click(result, "_ell_ego_nonce")

    theme.section("Connections")
    rows = case_edges.merge(df[['txId', 'class', 'SUSPICION_SCORE']].rename(
        columns={'txId': 'source', 'class': 'From label', 'SUSPICION_SCORE': 'From score'}), on='source', how='left')
    rows = rows.merge(df[['txId', 'class', 'SUSPICION_SCORE']].rename(
        columns={'txId': 'target', 'class': 'To label', 'SUSPICION_SCORE': 'To score'}), on='target', how='left')
    rows = rows.sort_values('To score', ascending=False)
    for col in ('source', 'target', 'From label', 'To label'):
        rows[col] = rows[col].astype(str)
    rows = rows.rename(columns={'source': 'From transaction', 'target': 'To transaction'})
    st.dataframe(rows, hide_index=True, use_container_width=True, column_config={
        'From score': st.column_config.ProgressColumn('From score', min_value=0.0, max_value=1.0, format="%.3f"),
        'To score': st.column_config.ProgressColumn('To score', min_value=0.0, max_value=1.0, format="%.3f"),
    })
    st.download_button("Export CSV", rows.to_csv(index=False).encode('utf-8'),
                       file_name=f"elliptic_case_{tx}.csv", mime="text/csv", key="dl_ell_case")


def _evaluation(metrics):
    theme.section("Held-out test evaluation", "From the last training run (metrics file). Scores are not calibrated probabilities.")
    left, right = st.columns([2, 3])
    with left:
        theme.evaluation_table(metrics)
    with right:
        theme.regime_table(metrics)
        st.caption("Results are split at time step 43. The time step itself is not a model input.")


# --------------------------------------------------------------------------- entry point
def render_elliptic():
    df = load_elliptic()
    edges = load_elliptic_edges()
    metrics = load_metrics(METRICS_PATH)

    goto = st.session_state.pop('_ell_goto', None)
    if goto:
        st.session_state[f"{KEY}_view"] = "Investigate"
        st.session_state[f"{KEY}_account"] = str(goto)

    theme.page_header("Elliptic Bitcoin Transactions",
                      "Node classification on the Elliptic transaction graph. Each node is a transaction scored "
                      "for illicit activity.")
    theme.stale_warning(predictions_older_than(ELLIPTIC['scores'], METRICS_PATH, ELLIPTIC['graph']),
                        'python src/predict_bitcoin.py')
    view, threshold, split_label = _sidebar_filters(df, metrics)
    if view.empty:
        theme.empty_state("No transactions match the current filters.")
        return

    is_alert = view['SUSPICION_SCORE'] >= threshold
    labelled = view['class'].isin(['Illicit', 'Licit'])
    illicit = view['class'] == 'Illicit'
    n_alerts, n_illicit = int(is_alert.sum()), int(illicit.sum())
    true_alerts = int((is_alert & illicit).sum())
    alerts_labelled = int((is_alert & labelled).sum())
    theme.kpi_row([
        {"label": "Transactions in view", "value": theme.fmt_int(len(view)), "help": split_label},
        {"label": "Alerts", "value": theme.fmt_int(n_alerts), "alert": n_alerts > 0,
         "help": f"score of {threshold:.2f} or higher"},
        {"label": "Alert rate", "value": theme.fmt_pct(n_alerts / len(view), 2)},
        {"label": "Alert precision", "value": theme.fmt_pct(true_alerts / alerts_labelled if alerts_labelled else None),
         "help": "of labelled alerts, share that are illicit"},
        {"label": "Alert recall", "value": theme.fmt_pct(true_alerts / n_illicit if n_illicit else None),
         "help": f"{theme.fmt_int(n_illicit)} labelled illicit in view"},
    ])
    if 'SPLIT' not in view.columns or split_label in ("Training", "All nodes"):
        st.caption("This view includes training nodes the model has already seen, so precision and recall "
                   "here are optimistic. Use the test split for an honest read.")
    if split_label in ("Unlabelled", "All nodes"):
        st.caption("Unlabelled nodes have no ground truth; precision and recall use labelled nodes only.")

    alerts = view[is_alert].sort_values('SUSPICION_SCORE', ascending=False)
    selected = st.radio("View", VIEWS, horizontal=True, key=f"{KEY}_view", label_visibility="collapsed")
    if selected == "Overview":
        _overview(view, edges, alerts, threshold, metrics, is_alert)
    elif selected == "Alerts":
        _alerts(alerts, threshold)
    elif selected == "Investigate":
        _investigate(df, edges, alerts, threshold)
    else:
        _evaluation(metrics)
