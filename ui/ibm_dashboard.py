"""
IBM AML dashboard: header, KPI row and five views (Overview, Alerts, Investigate, How it decides, Evaluation).
Only the selected view is computed, so switching is fast; the sidebar holds filters only. The network and the animated
decision tree are the custom components in ui/components.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import theme
from ui.components import decision_tree, network
from ui.data_store import EDGE_PROJECTS, load_edges, predictions_older_than
from ui.explain import badge_text, badges_for, explain_bundle, tree_path, unusual_signals
from ui.graph_data import ego_graph, hero_graph, load_arrays
from ui.metrics_loader import get_threshold, has_tuned_threshold, load_metrics

KEY = 'ibm'
METRICS_PATH = 'outputs/metrics_ibm_rf.json'
PREDICT_CMD = 'python src/predict_ibm_rf.py'
VIEWS = ["Overview", "Alerts", "Investigate", "How it decides", "Evaluation"]
SPLIT_CHOICES = {"Test (held out)": "test", "Validation": "val", "Training": "train", "All rows": None}
TABLE_COLUMNS = ["TIMESTAMP", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TX_TYPE", "SUSPICION_SCORE",
                 "PATTERN", "IS_FRAUD", "SPLIT"]


# --------------------------------------------------------------------------- filters
def _filters(df, arr, metrics):
    st.sidebar.markdown("**Filters**")
    split_label = st.sidebar.selectbox("Data split", list(SPLIT_CHOICES), index=0, key=f"{KEY}_split",
                                       help="Labels of test rows were never used for training or threshold tuning.")
    if not has_tuned_threshold(metrics):
        st.sidebar.caption("No tuned threshold found in the metrics file; using 0.50.")
    threshold = st.sidebar.slider("Alert threshold (model score)", 0.0, 1.0,
                                  float(min(max(get_threshold(metrics), 0.0), 1.0)), 0.005, key=f"{KEY}_thr",
                                  help="Defaults to the threshold tuned on validation data during training.")
    mask = np.ones(len(arr['score']), dtype=bool)
    wanted = SPLIT_CHOICES[split_label]
    if arr['split'] is not None and wanted is not None:
        mask &= arr['split'] == arr['split_names'].index(wanted)
    if arr['tail'] is not None:
        regime = st.sidebar.radio(
            "Volume regime", ["Normal-volume days", "All days", "Low-volume tail days"], key=f"{KEY}_regime",
            help="The last days of the IBM data have almost no volume but about 60% laundering. They lift pooled "
                 "PR-AUC and precision, so the default view leaves them out.")
        if regime == "Normal-volume days":
            mask &= ~arr['tail']
        elif regime == "Low-volume tail days":
            mask &= arr['tail']
    chosen = st.sidebar.multiselect("Transaction type", arr['tx_type_names'], default=[], placeholder="All types",
                                    key=f"{KEY}_types")
    if chosen:
        mask &= np.isin(arr['tx_type'], [arr['tx_type_names'].index(c) for c in chosen])
    if arr['t'] is not None:
        lo, hi = pd.Timestamp(arr['t'].min()), pd.Timestamp(arr['t'].max())
        picked = st.sidebar.date_input("Date range", value=(lo.date(), hi.date()), min_value=lo.date(), max_value=hi.date(),
                                       key=f"{KEY}_dates")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start = pd.Timestamp(picked[0]).value
            end = (pd.Timestamp(picked[1]) + pd.Timedelta(days=1)).value
            if (start, end) != (lo.floor('D').value, (hi.floor('D') + pd.Timedelta(days=1)).value):
                mask &= (arr['t'] >= start) & (arr['t'] < end)
    return np.flatnonzero(mask), threshold, split_label


# --------------------------------------------------------------------------- helpers
def _top_accounts(arr, alert_idx, limit=25):
    """Accounts ranked by how many alert transactions they take part in (as sender or receiver)."""
    if len(alert_idx) == 0:
        return pd.DataFrame(columns=['ACCOUNT_ID', 'ALERTS', 'MAX_SCORE', 'ALERT_AMOUNT'])
    frame = pd.DataFrame({
        'code': np.concatenate([arr['s'][alert_idx], arr['r'][alert_idx]]),
        'score': np.tile(arr['score'][alert_idx], 2), 'amount': np.tile(arr['amount'][alert_idx], 2)})
    out = frame.groupby('code').agg(ALERTS=('score', 'size'), MAX_SCORE=('score', 'max'), ALERT_AMOUNT=('amount', 'sum')).reset_index()
    out['ACCOUNT_ID'] = arr['names'].take(out['code'].to_numpy()).astype(str)
    return out.sort_values(['ALERTS', 'MAX_SCORE'], ascending=False).head(limit)[['ACCOUNT_ID', 'ALERTS', 'MAX_SCORE', 'ALERT_AMOUNT']]


def _rows_frame(df, arr, rows, bundle):
    """Display frame for the given row positions, with the detector pattern column."""
    view = df.iloc[rows][[c for c in TABLE_COLUMNS if c in df.columns and c != 'PATTERN']].copy()
    pattern = np.full(len(rows), '', dtype=object)
    if bundle is not None:
        detail = bundle['detail']
        known = detail.index.intersection(rows)
        if len(known):
            labels = {int(i): badge_text(detail.loc[i], bundle['active_badges']) for i in known}
            pattern = np.array([labels.get(int(r), '') for r in rows], dtype=object)
    view.insert(view.columns.get_loc('SUSPICION_SCORE') + 1, 'PATTERN', pattern)
    return view[[c for c in TABLE_COLUMNS if c in view.columns]]


def _handle_click(result, nonce_key):
    """A click on a node sends the account to the Investigate view."""
    if isinstance(result, dict) and result.get('type') == 'node' and st.session_state.get(nonce_key) != result.get('nonce'):
        st.session_state[nonce_key] = result.get('nonce')
        st.session_state['_ibm_goto'] = result['id']
        st.rerun()


def _votes_chart(row):
    hist = [int(row[f'tree_hist_{b}']) for b in range(10)]
    fig = go.Figure(go.Bar(x=[f"{b / 10:.1f}-{(b + 1) / 10:.1f}" for b in range(10)], y=hist, marker_color=theme.NEUTRAL,
                           hovertemplate="trees giving %{x}: %{y}<extra></extra>"))
    fig.update_layout(height=230, xaxis_title="Probability given by an individual tree", yaxis_title="Trees", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- views
def _overview(df, arr, idx, alert_idx, threshold, bundle, metrics):
    left, right = st.columns([5, 2])
    with left:
        theme.section("Largest alert clusters",
                      "Alert transactions that share accounts, drawn as a network. Click an account to open its case.")
        built = hero_graph(arr, idx, threshold, bundle['detail'] if bundle else None,
                           active=bundle['active_badges'] if bundle else None)
        if built is None:
            theme.empty_state("No alerts at this threshold. Lower the threshold in the sidebar to see more.")
        else:
            nodes, edges, info = built
            sig = f"hero-{len(idx)}-{threshold:.4f}-{len(nodes)}-{len(edges)}"
            result = network(nodes, edges, key="ibm_hero", sig=sig, layout="cose", height=580)
            _handle_click(result, "_ibm_hero_nonce")
            st.caption(f"{info['clusters']} largest clusters, {info['accounts']} accounts, from {info['alerts']:,} alert "
                       f"transactions in this view.")
    with right:
        theme.section("Score distribution", "Log scale, so the alert tail stays visible.")
        theme.score_histogram(arr['score'][idx], threshold)
        theme.section("Accounts with the most alerts")
        top = _top_accounts(arr, alert_idx, 8)
        if top.empty:
            theme.empty_state("No alerts at this threshold.")
        else:
            theme.data_table(top, key="ibm_top", file_name="ibm_accounts_with_alerts.csv", height=330, max_rows=8,
                             compact=True)


def _alerts(df, arr, idx, alert_idx, threshold, bundle):
    theme.section("Alert queue", f"Transactions at or above the alert threshold, highest model score first. "
                                 f"{theme.fmt_int(len(alert_idx))} in view. The pattern column shows which detectors fired.")
    if len(alert_idx) == 0:
        theme.empty_state("No alerts at this threshold. Lower the threshold in the sidebar to see more.")
        return
    ordered = alert_idx[np.argsort(-arr['score'][alert_idx])][:1000]
    theme.data_table(_rows_frame(df, arr, ordered, bundle), key="ibm_alerts", file_name="ibm_alerts.csv",
                     threshold=threshold, currency='localized', time_kind='datetime', label_yes="Laundering")


def _investigate(df, arr, idx, alert_idx, threshold, bundle):
    theme.section("Case investigation", "Search an account to see its network, why its transactions were flagged and how "
                                        "the decision tree handles them. Scores on training rows are optimistic.")
    top = _top_accounts(arr, alert_idx, 25)['ACCOUNT_ID'].tolist() if len(alert_idx) else []
    c1, c2 = st.columns(2)
    c1.selectbox("Accounts with the most alerts in the current view", [""] + top, key=f"{KEY}_suggest")
    typed = c2.text_input("Or enter an account ID", key=f"{KEY}_account").strip()
    account = typed or st.session_state.get(f"{KEY}_suggest", "")
    if not account:
        theme.empty_state("Select or enter an account ID, or click an account in the Overview network.")
        return
    built = ego_graph(arr, bundle['detail'] if bundle else None, account, threshold,
                      active=bundle['active_badges'] if bundle else None)
    if built is None:
        st.error(f"Account {account} was not found in the scored transactions.")
        return
    nodes, edges, info = built
    names = arr['names']
    a = int(names.get_loc(account))
    rows = np.flatnonzero((arr['s'] == a) | (arr['r'] == a))
    rows = rows[np.argsort(-arr['score'][rows])]
    sent = arr['amount'][arr['s'] == a].sum() if len(rows) < 1 else arr['amount'][rows][arr['s'][rows] == a].sum()
    received = arr['amount'][rows][arr['r'][rows] == a].sum()
    theme.kpi_row([
        {"label": "Transactions", "value": theme.fmt_int(info['transactions'])},
        {"label": "Alerts", "value": theme.fmt_int(info['alerts']), "alert": info['alerts'] > 0},
        {"label": "Highest model score", "value": f"{arr['score'][rows].max():.3f}"},
        {"label": "Total sent", "value": theme.fmt_amount(sent), "help": "Mixed currencies, as paid"},
        {"label": "Total received", "value": theme.fmt_amount(received), "help": "Mixed currencies, as paid"},
    ])
    found = []
    if info['badges']:
        found.append("detector patterns: " + ", ".join(info['badges']))
    if info['cycles']:
        found.append(f"{info['cycles']} cycle(s) through this account")
    if info['pass_through_pairs']:
        found.append(f"{info['pass_through_pairs']} pass-through pair(s)")
    if info['benchmark_schemes']:
        found.append("benchmark label: " + ", ".join(info['benchmark_schemes']))
    st.markdown("**Patterns:** " + ("; ".join(found) if found else "none detected around this account"))

    theme.section("Network", f"{info['neighbours']} counterparties and {info['second_hop']} of theirs. "
                             "Click an account to re-centre on it.")
    sig = f"ego-{account}-{threshold:.4f}"
    result = network(nodes, edges, key="ibm_ego", sig=sig, layout="concentric", height=520)
    _handle_click(result, "_ibm_ego_nonce")

    theme.section("Transactions")
    shown = rows[:300]
    frame = _rows_frame(df, arr, shown, bundle)
    theme.data_table(frame, key="ibm_case", file_name=f"ibm_case_{account}.csv", threshold=threshold, currency='localized',
                     time_kind='datetime', max_rows=300, label_yes="Laundering")

    if bundle is None:
        theme.empty_state("Explanations are not built yet. Run python src/explain_ibm.py.")
        return
    theme.section("Why was it flagged, and how does the tree decide?",
                  "Pick one of this account's transactions. Detail is stored for transactions scoring at least 0.05 or "
                  "labelled as laundering.")
    detail = bundle['detail']
    explained = [int(r) for r in shown if int(r) in detail.index][:40]
    if not explained:
        theme.empty_state("None of this account's top transactions has stored detail (all score below 0.05).")
        return
    options = {f"#{r}  score {arr['score'][r]:.3f}  amount {arr['amount'][r]:,.0f}  {df['TX_TYPE'].iat[r]}": r for r in explained}
    choice = st.selectbox("Transaction", list(options), key=f"{KEY}_tx")
    row_id = options[choice]
    row = detail.loc[row_id]
    labels = bundle['tree']['feature_labels']
    left, right = st.columns([3, 2])
    with left:
        signals = unusual_signals(bundle, row, labels)
        st.markdown("**Unusual signals** (compared with normal-volume days)")
        if signals:
            table = pd.DataFrame(signals).rename(columns={'label': 'Signal', 'value': 'Value', 'percentile': 'Percentile'})
            st.dataframe(table, hide_index=True, use_container_width=True, column_config={
                'Percentile': st.column_config.ProgressColumn('Percentile', min_value=0, max_value=100, format="%.1f")})
        else:
            st.caption("Nothing in this transaction is unusually extreme on its own.")
        verdict = badges_for(row, bundle['active_badges'])
        if verdict:
            st.markdown("**Detectors that fired:** " + ", ".join(verdict))
    with right:
        votes = int(row['forest_votes'])
        if votes >= 0:
            n_trees = int(row['n_trees'])
            st.markdown(f"**The forest:** {votes} of {n_trees} trees give this transaction at least a 50% laundering "
                        f"probability. Final score {arr['score'][row_id]:.3f}.")
            _votes_chart(row)
        else:
            st.caption("Per-tree votes are stored for transactions scoring at least 0.20.")
    path = tree_path(bundle, row)
    st.caption("This tree is a depth-8 approximation of the forest, trained to imitate its alert decisions. It can "
               "disagree with the forest on individual transactions.")
    decision_tree(bundle['tree'], path, key="ibm_tree_case", sig=f"tree-{row_id}", height=680)


def _how_it_decides(df, arr, idx, alert_idx, threshold, bundle):
    theme.section("How the model decides", "A forest of 200 deep trees makes the real decision. This depth-8 tree was "
                                           "trained to imitate it, so you can watch a transaction travel through the rules.")
    if bundle is None:
        theme.empty_state("Explanations are not built yet. Run python src/explain_ibm.py.")
        return
    fid = bundle['tree']['fidelity']
    cols = st.columns(3)
    cols[0].metric("Agreement with the forest (test)", f"{fid['pooled']['agreement'] * 100:.2f}%")
    cols[1].metric("Alerts the tree also flags", f"{fid['pooled']['alert_recall'] * 100:.0f}%")
    cols[2].metric("Tree alerts the forest agrees with", f"{fid['pooled']['alert_precision'] * 100:.0f}%")
    detail = bundle['detail']
    candidates = [int(r) for r in alert_idx[np.argsort(-arr['score'][alert_idx])][:300] if int(r) in detail.index][:25]
    if not candidates:
        decision_tree(bundle['tree'], [], key="ibm_tree_global", sig="tree-empty", height=680)
        return
    options = {f"#{r}  score {arr['score'][r]:.3f}  amount {arr['amount'][r]:,.0f}": r for r in candidates}
    choice = st.selectbox("Watch an alert travel the tree", list(options), key=f"{KEY}_tree_pick")
    row_id = options[choice]
    decision_tree(bundle['tree'], tree_path(bundle, detail.loc[row_id]), key="ibm_tree_global", sig=f"tree-{row_id}", height=680)


def _evaluation(metrics, bundle):
    theme.section("Held-out test evaluation", "From the last training run (metrics file). Scores are not calibrated probabilities.")
    left, right = st.columns([2, 3])
    with left:
        theme.evaluation_table(metrics)
    with right:
        theme.regime_table(metrics)
        st.caption("Low-volume tail days (the last days of the data) have about 60% laundering and lift pooled numbers.")
    if bundle is None:
        theme.empty_state("Per-scheme and workload results are not built yet. Run python src/explain_ibm.py.")
        return
    m = bundle['metrics']
    theme.section("Recall by laundering scheme",
                  "Share of each benchmark scheme's transactions the model flags (test, normal-volume days). "
                  "Rows labelled UNATTRIBUTED are laundering rows the benchmark's pattern file does not list.")
    rec = m['model_recall_by_scheme']['test_normal_volume']
    st.dataframe(pd.DataFrame([dict(Scheme=k, Transactions=v['rows'], Recall=v['recall'], MeanScore=v['mean_score'])
                               for k, v in rec.items()]), hide_index=True, use_container_width=True, column_config={
        'Recall': st.column_config.ProgressColumn('Recall', min_value=0, max_value=1, format="%.2f"),
        'MeanScore': st.column_config.NumberColumn('Mean score', format="%.3f")})
    theme.section("Pattern detectors", "Rule-based detectors with thresholds tuned on the training period only. Precision is "
                                       "against the laundering label; recall is against each detector's own schemes (test). "
                                       "Only detectors at least 20% precise on validation appear as badges.")
    det = m['detectors']['test']
    st.dataframe(pd.DataFrame([dict(Detector=v['label'], Flagged=v['flagged'], Precision=v['precision_any_laundering'],
                                    Recall=v['recall_own_schemes'], Threshold=v['threshold']) for v in det.values()]),
                 hide_index=True, use_container_width=True, column_config={
        'Precision': st.column_config.ProgressColumn('Precision', min_value=0, max_value=1, format="%.2f"),
        'Recall': st.column_config.ProgressColumn('Recall', min_value=0, max_value=1, format="%.2f"),
        'Threshold': st.column_config.NumberColumn(format="%.2f")})
    theme.section("Analyst workload", "Account-level, test period, normal-volume days: review the highest-scoring accounts "
                                      "first and see how much laundering is caught.")
    work = m['workload']['accounts_test_normal_volume']
    st.dataframe(pd.DataFrame([dict(Reviewed=f"top {r['top_fraction'] * 100:g}% of accounts", Accounts=r['accounts_reviewed'],
                                    Caught=r['laundering_accounts_caught'], Share=r['share_of_laundering_accounts'],
                                    Precision=r['precision']) for r in work['rows']]),
                 hide_index=True, use_container_width=True, column_config={
        'Share': st.column_config.ProgressColumn('Share of laundering accounts caught', min_value=0, max_value=1, format="%.2f"),
        'Precision': st.column_config.NumberColumn(format="%.3f")})
    per_day = m['workload']['alerts_per_day_at_threshold']['test_normal_volume']
    st.caption(f"At the alert threshold the model raises about {per_day:,.0f} alerts per day on normal-volume days "
               f"({work['laundering_accounts']:,} laundering accounts among {work['accounts']:,} active accounts).")


# --------------------------------------------------------------------------- entry point
def render_ibm():
    df = load_edges(KEY)
    cfg = EDGE_PROJECTS[KEY]
    mtimes = tuple(os.path.getmtime(p) if os.path.exists(p) else 0.0 for p in (cfg['scores'], cfg['graph'], cfg['raw']))
    arr = load_arrays(KEY, mtimes, df)
    metrics = load_metrics(METRICS_PATH)
    bundle = explain_bundle()

    goto = st.session_state.pop('_ibm_goto', None)         # set by a click on a network node
    if goto:
        st.session_state[f"{KEY}_view"] = "Investigate"
        st.session_state[f"{KEY}_account"] = goto

    theme.page_header("IBM AML Transactions",
                      "A RandomForest on past-only transaction history and graph-structure features scores each "
                      "transaction for money-laundering activity.")
    theme.stale_warning(predictions_older_than(cfg['scores'], METRICS_PATH, cfg['graph']), PREDICT_CMD)
    idx, threshold, split_label = _filters(df, arr, metrics)
    if len(idx) == 0:
        theme.empty_state("No transactions match the current filters.")
        return

    scores, labelled = arr['score'][idx], arr['y'][idx] == 1
    is_alert = scores >= threshold
    alert_idx = idx[is_alert]
    n_alerts, n_labelled = int(is_alert.sum()), int(labelled.sum())
    true_alerts = int((is_alert & labelled).sum())
    theme.kpi_row([
        {"label": "Transactions in view", "value": theme.fmt_int(len(idx)), "help": split_label},
        {"label": "Alerts", "value": theme.fmt_int(n_alerts), "alert": n_alerts > 0, "help": f"score of {threshold:.2f} or higher"},
        {"label": "Alert rate", "value": theme.fmt_pct(n_alerts / len(idx), 2)},
        {"label": "Alert precision", "value": theme.fmt_pct(true_alerts / n_alerts if n_alerts else None),
         "help": "alerts that carry a laundering label"},
        {"label": "Alert recall", "value": theme.fmt_pct(true_alerts / n_labelled if n_labelled else None),
         "help": f"{theme.fmt_int(n_labelled)} labelled transactions in view"},
    ])
    if arr['split'] is None or split_label in ("Training", "All rows"):
        st.caption("This view includes training rows the model has already seen, so precision and recall here are optimistic.")

    view = st.radio("View", VIEWS, horizontal=True, key=f"{KEY}_view", label_visibility="collapsed")
    if view == "Overview":
        _overview(df, arr, idx, alert_idx, threshold, bundle, metrics)
    elif view == "Alerts":
        _alerts(df, arr, idx, alert_idx, threshold, bundle)
    elif view == "Investigate":
        _investigate(df, arr, idx, alert_idx, threshold, bundle)
    elif view == "How it decides":
        _how_it_decides(df, arr, idx, alert_idx, threshold, bundle)
    else:
        _evaluation(metrics, bundle)
