"""Elliptic (node classification) dashboard, laid out like the edge projects."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import theme
from ui.data_store import ELLIPTIC, load_elliptic, load_elliptic_edges, predictions_older_than
from ui.graph_display_bitcoin import display_network_graph
from ui.metrics_loader import load_metrics, get_threshold, has_tuned_threshold

METRICS_PATH = 'outputs/metrics_bitcoin.json'
SPLIT_CHOICES = {
    "Test (held out)": ["test"],
    "Validation": ["val"],
    "Training": ["train"],
    "Unlabelled": ["unlabeled"],
    "All nodes": ["train", "val", "test", "unlabeled"],
}
TABLE_COLUMNS = ["txId", "TimeStep", "class", "SUSPICION_SCORE", "SPLIT"]
NETWORK_DEFAULT_N = 300
NETWORK_MAX_N = 1000
MAX_NETWORK_EDGES = 2000


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


def _sidebar_filters(df, metrics):
    st.sidebar.markdown("**Filters**")
    has_split = 'SPLIT' in df.columns
    choices = list(SPLIT_CHOICES) if has_split else ["All nodes"]
    split_label = st.sidebar.selectbox("Data split", choices, index=0, key="ell_split",
                                       help="Labels of test nodes were never used for training or threshold tuning.")
    if not has_split:
        st.sidebar.caption("Split information is unavailable (processed graph not found), so every view "
                           "includes training nodes.")
    if not has_tuned_threshold(metrics):
        st.sidebar.caption("No tuned threshold found in the metrics file; using 0.50.")
    default_thr = min(max(get_threshold(metrics), 0.0), 1.0)
    threshold = st.sidebar.slider("Alert threshold (model score)", 0.0, 1.0, float(default_thr), 0.005,
                                  key="ell_thr", help="Defaults to the threshold tuned on validation data.")
    lo, hi = int(df['TimeStep'].min()), int(df['TimeStep'].max())
    mask = df['SPLIT'].isin(SPLIT_CHOICES[split_label]) if has_split else pd.Series(True, index=df.index)
    if hi > lo:
        a, b = st.sidebar.slider("Time step", lo, hi, (lo, hi), key="ell_steps")
        mask &= df['TimeStep'].between(a, b)
    return df[mask], threshold, split_label


def _investigate(df, edges, alerts, threshold):
    theme.section("Case investigation", "Review one transaction and its direct connections. Scores of training "
                                        "nodes are optimistic; the case header does not distinguish splits.")
    top_ids = alerts['txId'].head(25).astype(str).tolist() if len(alerts) else []
    left, right = st.columns(2)
    suggested = left.selectbox("Highest-scoring alerts in the current view", [""] + top_ids, key="ell_suggest")
    typed = right.text_input("Or enter a transaction ID", key="ell_account").strip()
    raw = typed or suggested
    if not raw:
        theme.empty_state("Select or enter a transaction ID to open a case.")
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

    theme.section("Connection network")
    nodes = df[df['txId'].isin(neighbours | {tx})]
    display_network_graph(case_edges, nodes, view_key="ell_case", threshold=threshold)

    theme.section("Connections")
    rows = case_edges.merge(df[['txId', 'class', 'SUSPICION_SCORE']].rename(
        columns={'txId': 'source', 'class': 'From label', 'SUSPICION_SCORE': 'From score'}), on='source', how='left')
    rows = rows.merge(df[['txId', 'class', 'SUSPICION_SCORE']].rename(
        columns={'txId': 'target', 'class': 'To label', 'SUSPICION_SCORE': 'To score'}), on='target', how='left')
    rows = rows.sort_values('To score', ascending=False)
    for col in ('source', 'target'):
        rows[col] = rows[col].astype(str)
    for col in ('From label', 'To label'):
        rows[col] = rows[col].astype(str)
    rows = rows.rename(columns={'source': 'From transaction', 'target': 'To transaction'})
    st.dataframe(rows, hide_index=True, use_container_width=True, column_config={
        'From score': st.column_config.ProgressColumn('From score', min_value=0.0, max_value=1.0, format="%.3f"),
        'To score': st.column_config.ProgressColumn('To score', min_value=0.0, max_value=1.0, format="%.3f"),
    })
    st.download_button("Export CSV", rows.to_csv(index=False).encode('utf-8'),
                       file_name=f"elliptic_case_{tx}.csv", mime="text/csv", key="dl_ell_case")


def render_elliptic():
    df = load_elliptic()
    edges = load_elliptic_edges()
    metrics = load_metrics(METRICS_PATH)

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

    tab_overview, tab_alerts, tab_network, tab_case = st.tabs(["Overview", "Alerts", "Network", "Investigate"])
    alerts = view[is_alert].sort_values('SUSPICION_SCORE', ascending=False)

    with tab_overview:
        left, right = st.columns([3, 2])
        with left:
            theme.section("Model score distribution", theme.SCORE_NOTE)
            theme.score_histogram(view['SUSPICION_SCORE'], threshold)
        with right:
            theme.section("Held-out test evaluation", "From the last training run (metrics file).")
            theme.evaluation_table(metrics)
            theme.regime_table(metrics)
        theme.section("Alerts by time step", "Number of alerts per time step in the current view.")
        per_step = view[is_alert].groupby('TimeStep').size().reindex(
            range(int(view['TimeStep'].min()), int(view['TimeStep'].max()) + 1), fill_value=0)
        fig = go.Figure(go.Bar(x=per_step.index, y=per_step.values, marker_color=theme.NEUTRAL,
                               hovertemplate="step %{x}<br>%{y:,} alerts<extra></extra>"))
        fig.update_layout(height=260, xaxis_title="Time step", yaxis_title="Alerts")
        st.plotly_chart(fig, use_container_width=True)

    with tab_alerts:
        theme.section("Alert queue", f"Transactions at or above the alert threshold, highest model score first. "
                                     f"{theme.fmt_int(n_alerts)} in view.")
        if alerts.empty:
            theme.empty_state("No alerts at this threshold. Lower the threshold in the sidebar to see more.")
        else:
            table = alerts[TABLE_COLUMNS].copy()
            table['txId'] = table['txId'].astype(str)
            theme.data_table(table, key="ell_alerts", file_name="elliptic_alerts.csv", threshold=threshold)

    with tab_network:
        theme.section("Transaction network",
                      "The highest-scoring transactions in the current view and their direct connections. "
                      "Red nodes are alerts.")
        n = st.slider("Transactions to plot", 50, NETWORK_MAX_N, NETWORK_DEFAULT_N, 50, key="ell_netn")
        top_ids = set(view.nlargest(n, 'SUSPICION_SCORE')['txId'])
        sub = edges[edges['source'].isin(top_ids) | edges['target'].isin(top_ids)]
        if len(sub) > MAX_NETWORK_EDGES:
            score_of = df.set_index('txId')['SUSPICION_SCORE']
            edge_score = sub['source'].map(score_of).fillna(0).combine(sub['target'].map(score_of).fillna(0), max)
            sub = sub.loc[edge_score.nlargest(MAX_NETWORK_EDGES).index]
            st.caption(f"Showing the {MAX_NETWORK_EDGES:,} highest-scoring connections.")
        involved = set(sub['source']) | set(sub['target'])
        if sub.empty:
            theme.empty_state("The selected transactions have no recorded connections.")
        else:
            display_network_graph(sub, df[df['txId'].isin(involved)], view_key="ell_network", threshold=threshold)

    with tab_case:
        _investigate(df, edges, alerts, threshold)
