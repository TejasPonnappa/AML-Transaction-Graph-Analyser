"""
One dashboard for the two edge-classification projects (AMLSim, IBM AML), so both
look and behave identically. Layout: header, KPI row, then tabs
(Overview, Alerts, Network, Investigate). The sidebar holds filters only.
"""
import pandas as pd
import streamlit as st

from ui import theme
from ui.data_store import SPLITS, EDGE_PROJECTS, predictions_older_than
from ui.graph_display_amlsim import display_network_graph
from ui.metrics_loader import get_threshold, has_tuned_threshold

SPLIT_CHOICES = {
    "Test (held out)": ["test"],
    "Validation": ["val"],
    "Training": ["train"],
    "All rows": SPLITS,
}
TABLE_COLUMNS = ["TIMESTAMP", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TX_TYPE",
                 "SUSPICION_SCORE", "IS_FRAUD", "SPLIT"]
NETWORK_DEFAULT_N = 300
NETWORK_MAX_N = 1000


def _sidebar_filters(df, cfg, metrics):
    key = cfg['key']
    st.sidebar.markdown("**Filters**")
    has_split = 'SPLIT' in df.columns
    choices = list(SPLIT_CHOICES) if has_split else ["All rows"]
    split_label = st.sidebar.selectbox("Data split", choices, index=0, key=f"{key}_split",
                                       help="Labels of test rows were never used for training or threshold tuning.")
    if not has_split:
        st.sidebar.caption("Split information is unavailable (processed graph not found), so every view "
                           "includes training rows.")
    if not has_tuned_threshold(metrics):
        st.sidebar.caption("No tuned threshold found in the metrics file; using 0.50.")
    default_thr = min(max(get_threshold(metrics), 0.0), 1.0)
    threshold = st.sidebar.slider("Alert threshold (model score)", 0.0, 1.0, float(default_thr), 0.005,
                                  key=f"{key}_thr",
                                  help="Defaults to the threshold tuned on validation data during training.")

    mask = pd.Series(True, index=df.index)
    if has_split:
        mask &= df['SPLIT'].isin(SPLIT_CHOICES[split_label])

    if 'TAIL' in df.columns:
        regime = st.sidebar.radio(
            "Volume regime", ["Normal-volume days", "All days", "Low-volume tail days"], key=f"{key}_regime",
            help="The last days of the IBM data have almost no volume but about 60% laundering. They lift pooled "
                 "PR-AUC and precision, so the default view leaves them out.")
        if regime == "Normal-volume days":
            mask &= ~df['TAIL']
        elif regime == "Low-volume tail days":
            mask &= df['TAIL']

    types = list(df['TX_TYPE'].cat.categories) if 'TX_TYPE' in df.columns else []
    if len(types) > 1:
        chosen = st.sidebar.multiselect("Transaction type", types, default=[], placeholder="All types",
                                        key=f"{key}_types")
        if chosen:
            mask &= df['TX_TYPE'].isin(chosen)

    if cfg['time_kind'] == 'datetime' and 'TIMESTAMP' in df.columns:
        lo, hi = df['TIMESTAMP'].min().date(), df['TIMESTAMP'].max().date()
        picked = st.sidebar.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi,
                                       key=f"{key}_dates")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
            if (start, end) != (pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)):
                mask &= (df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)
    elif 'TIMESTAMP' in df.columns:
        lo, hi = int(df['TIMESTAMP'].min()), int(df['TIMESTAMP'].max())
        if hi > lo:
            a, b = st.sidebar.slider("Simulation step", lo, hi, (lo, hi), key=f"{key}_steps")
            if (a, b) != (lo, hi):
                mask &= df['TIMESTAMP'].between(a, b)

    return df[mask], threshold, split_label


def _accounts_with_alerts(alerts):
    """One row per account: alert count, highest score and total alert amount (as sender or receiver)."""
    parts = []
    for col in ('SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID'):
        part = alerts[[col, 'SUSPICION_SCORE', 'TX_AMOUNT']].copy()
        part.columns = ['ACCOUNT_ID', 'SUSPICION_SCORE', 'TX_AMOUNT']
        parts.append(part)
    both = pd.concat(parts)
    both['ACCOUNT_ID'] = both['ACCOUNT_ID'].astype(str)
    out = both.groupby('ACCOUNT_ID', observed=True).agg(
        ALERTS=('SUSPICION_SCORE', 'size'),
        MAX_SCORE=('SUSPICION_SCORE', 'max'),
        ALERT_AMOUNT=('TX_AMOUNT', 'sum'),
    ).reset_index()
    return out.sort_values(['ALERTS', 'MAX_SCORE'], ascending=False)


def indicative_typology(df_account, account_id, has_alert):
    """Rule-based reading of the account's flows. Indicative only; not a model output."""
    if not has_alert:
        return "No alert on this account"

    sent = df_account[df_account['SENDER_ACCOUNT_ID'].astype(str) == account_id]
    received = df_account[df_account['RECEIVER_ACCOUNT_ID'].astype(str) == account_id]
    in_degree, out_degree = len(received), len(sent)
    total_sent, total_received = sent['TX_AMOUNT'].sum(), received['TX_AMOUNT'].sum()
    net_flow = total_received - total_sent
    volume = total_sent + total_received

    patterns = []
    is_mule = False
    if volume > 0 and abs(net_flow) > 0.5 * volume:
        is_mule = True
        patterns.append("Mule account (consolidation target)" if net_flow > 0 else "Mule account (dispersal source)")
    if is_mule and net_flow > 0 and in_degree > 5:
        patterns.append("Smurfing target (many deposits)")
    if in_degree >= 2 and out_degree >= 2 and total_received > 0 and total_sent >= 0.25 * total_received:
        if volume == 0 or abs(net_flow) < 0.20 * volume:
            patterns.append("Layering / pass-through node")
    return ", ".join(patterns) if patterns else "Alert present; no rule-based typology matched"


def _investigate(df, view, alerts, cfg, threshold):
    key = cfg['key']
    theme.section("Case investigation",
                  "Review every transaction of one account, across all data splits. Scores on training rows "
                  "are optimistic, so check the Split column before relying on an alert.")
    top_accounts = _accounts_with_alerts(alerts)['ACCOUNT_ID'].head(25).tolist() if len(alerts) else []
    left, right = st.columns(2)
    suggested = left.selectbox("Accounts with the most alerts in the current view", [""] + top_accounts,
                               key=f"{key}_suggest")
    typed = right.text_input("Or enter an account ID", key=f"{key}_account").strip()
    account = typed or suggested
    if not account:
        theme.empty_state("Select or enter an account ID to open a case.")
        return

    case = df[(df['SENDER_ACCOUNT_ID'] == account) | (df['RECEIVER_ACCOUNT_ID'] == account)]
    if case.empty:
        st.error(f"Account {account} was not found in the scored transactions.")
        return
    case = case.sort_values('SUSPICION_SCORE', ascending=False)

    sent = case.loc[case['SENDER_ACCOUNT_ID'] == account, 'TX_AMOUNT'].sum()
    received = case.loc[case['RECEIVER_ACCOUNT_ID'] == account, 'TX_AMOUNT'].sum()
    n_alerts = int((case['SUSPICION_SCORE'] >= threshold).sum())
    theme.kpi_row([
        {"label": "Transactions", "value": theme.fmt_int(len(case))},
        {"label": "Alerts", "value": theme.fmt_int(n_alerts), "alert": n_alerts > 0},
        {"label": "Highest model score", "value": f"{case['SUSPICION_SCORE'].max():.3f}"},
        {"label": "Total sent", "value": theme.fmt_amount(sent), "help": cfg['amount_note']},
        {"label": "Total received", "value": theme.fmt_amount(received), "help": cfg['amount_note']},
    ])
    st.markdown(f"**Indicative typology (rule-based):** {indicative_typology(case, account, n_alerts > 0)}")

    theme.section("Counterparty network", f"Up to {NETWORK_DEFAULT_N} highest-scoring transactions of this account.")
    display_network_graph(case.head(NETWORK_DEFAULT_N), view_key=f"{key}_case", threshold=threshold)

    theme.section("Transactions")
    theme.data_table(case[[c for c in TABLE_COLUMNS if c in case.columns]], key=f"{key}_case",
                     file_name=f"{key}_case_{account}.csv", threshold=threshold,
                     currency=cfg['currency'], time_kind=cfg['time_kind'])


def render_edge_project(cfg):
    key = cfg['key']
    df = cfg['loader'](key)
    metrics = cfg['metrics']()

    theme.page_header(cfg['title'], cfg['subtitle'])
    theme.stale_warning(
        predictions_older_than(EDGE_PROJECTS[key]['scores'], cfg['metrics_path'], EDGE_PROJECTS[key]['graph']),
        cfg['predict_cmd'])
    view, threshold, split_label = _sidebar_filters(df, cfg, metrics)
    if view.empty:
        theme.empty_state("No transactions match the current filters.")
        return

    scores = view['SUSPICION_SCORE']
    is_alert = scores >= threshold
    labelled = view['IS_FRAUD'] == 1
    n_alerts, n_labelled = int(is_alert.sum()), int(labelled.sum())
    true_alerts = int((is_alert & labelled).sum())

    theme.kpi_row([
        {"label": "Transactions in view", "value": theme.fmt_int(len(view)), "help": split_label},
        {"label": "Alerts", "value": theme.fmt_int(n_alerts), "alert": n_alerts > 0,
         "help": f"score of {threshold:.2f} or higher"},
        {"label": "Alert rate", "value": theme.fmt_pct(n_alerts / len(view), 2)},
        {"label": "Alert precision", "value": theme.fmt_pct(true_alerts / n_alerts if n_alerts else None),
         "help": "alerts that carry a laundering label"},
        {"label": "Alert recall", "value": theme.fmt_pct(true_alerts / n_labelled if n_labelled else None),
         "help": f"{theme.fmt_int(n_labelled)} labelled transactions in view"},
    ])
    if 'SPLIT' not in view.columns or split_label in ("Training", "All rows"):
        st.caption("This view includes training rows the model has already seen, so precision and recall "
                   "here are optimistic. Use the test split for an honest read.")

    tab_overview, tab_alerts, tab_network, tab_case = st.tabs(["Overview", "Alerts", "Network", "Investigate"])
    alerts = view[is_alert].sort_values('SUSPICION_SCORE', ascending=False)
    columns = [c for c in TABLE_COLUMNS if c in view.columns]

    with tab_overview:
        left, right = st.columns([3, 2])
        with left:
            theme.section("Model score distribution", theme.SCORE_NOTE)
            theme.score_histogram(scores, threshold)
        with right:
            theme.section("Held-out test evaluation", "From the last training run (metrics file).")
            theme.evaluation_table(metrics)
            theme.regime_table(metrics)
        theme.section("Accounts with the most alerts", "Ranked by alert count, then highest model score.")
        if alerts.empty:
            theme.empty_state("No alerts at this threshold.")
        else:
            theme.data_table(_accounts_with_alerts(alerts), key=f"{key}_accounts", height=300,
                             file_name=f"{key}_accounts_with_alerts.csv", currency=cfg['currency'],
                             time_kind=cfg['time_kind'], max_rows=200)

    with tab_alerts:
        theme.section("Alert queue", f"Transactions at or above the alert threshold, highest model score first. "
                                     f"{theme.fmt_int(n_alerts)} in view.")
        if alerts.empty:
            theme.empty_state("No alerts at this threshold. Lower the threshold in the sidebar to see more.")
        else:
            theme.data_table(alerts[columns], key=f"{key}_alerts", file_name=f"{key}_alerts.csv",
                             threshold=threshold, currency=cfg['currency'], time_kind=cfg['time_kind'],
                             label_yes="Laundering")

    with tab_network:
        theme.section("Transaction network",
                      "The highest-scoring transactions in the current view. Red edges are alerts.")
        n = st.slider("Transactions to plot", 50, NETWORK_MAX_N, NETWORK_DEFAULT_N, 50, key=f"{key}_netn")
        display_network_graph(view.nlargest(n, 'SUSPICION_SCORE'), view_key=f"{key}_network", threshold=threshold)

    with tab_case:
        _investigate(df, view, alerts, cfg, threshold)
