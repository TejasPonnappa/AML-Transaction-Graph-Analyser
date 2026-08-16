import streamlit as st
import pandas as pd
import os
from ui.dashboard_components import animated_metric_row, score_distribution_chart, top_risk_ticker, risk_gauge
from ui.graph_display_amlsim import display_network_graph  # reused, same schema now

# --- Configuration ---
SCORES_PATH = 'outputs/ibm_predictions.csv'
METRICS_PATH = 'outputs/metrics_ibm.json'


@st.cache_data
def load_scores_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Output scores file not found at {path}. Please run 'python src/predict_ibm.py' first.")
        return None
    df = pd.read_csv(path)
    df['SENDER_ACCOUNT_ID'] = df['SENDER_ACCOUNT_ID'].astype(str)
    df['RECEIVER_ACCOUNT_ID'] = df['RECEIVER_ACCOUNT_ID'].astype(str)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df


@st.cache_data
def load_metrics(path):
    import json
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def render_leaderboards(df_scores):
    st.subheader("📊 Top 10 High-Risk Account Leaderboards")
    RISK_THRESHOLD = 0.5
    df_risky = df_scores[df_scores['SUSPICION_SCORE'] >= RISK_THRESHOLD].copy()

    if df_risky.empty:
        st.info(f"No transactions meet the high-risk threshold of {RISK_THRESHOLD:.4f}.")
        return

    st.markdown("#### 🥇 Top 10 Risky Accounts (by Average Suspicion Score)")
    senders = df_risky[['SENDER_ACCOUNT_ID', 'SUSPICION_SCORE']].rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
    receivers = df_risky[['RECEIVER_ACCOUNT_ID', 'SUSPICION_SCORE']].rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})
    all_risky_accounts = pd.concat([senders, receivers])
    avg_score_leaderboard = all_risky_accounts.groupby('ACCOUNT_ID')['SUSPICION_SCORE'].mean().sort_values(ascending=False).head(10).reset_index()
    avg_score_leaderboard.columns = ['Account ID', 'Average Suspicion Score']
    st.dataframe(avg_score_leaderboard, use_container_width=True, hide_index=True)


def draw_ibm_dashboard():
    df_scores = load_scores_data(SCORES_PATH)
    metrics = load_metrics(METRICS_PATH)

    st.title("🏦 IBM AML Transaction Analyzer")
    st.markdown("### GraphSAGE Edge Classification Results (HI-Small)")

    if df_scores is not None:
        animated_metric_row([
            {'label': 'Total Transactions', 'value': f"{len(df_scores):,}"},
            {'label': 'Test ROC-AUC', 'value': f"{metrics.get('test_roc_auc', 0):.4f}"},
            {'label': 'Test Recall', 'value': f"{metrics.get('test_recall', 0):.4f}"},
            {'label': 'Test PR-AUC', 'value': f"{metrics.get('test_pr_auc', 0):.4f}"},
        ])
        st.markdown("---")

        st.markdown("#### 🔴 Highest Risk Transactions Right Now")
        top_risk_ticker(df_scores)
        st.markdown("---")

        st.subheader("Suspicion Score Distribution")
        score_distribution_chart(df_scores)
        st.markdown("---")

        render_leaderboards(df_scores)
        st.markdown("---")

        st.subheader("Interactive Suspicious Network Map")
        best_thresh = metrics.get('best_threshold', 0.5)
        suspicion_threshold = st.slider(
            "Suspicion Score Threshold",
            min_value=float(df_scores['SUSPICION_SCORE'].min()),
            max_value=float(df_scores['SUSPICION_SCORE'].max()),
            value=float(best_thresh),
            step=0.001,
            format="%.4f"
        )
        df_filtered = df_scores[df_scores['SUSPICION_SCORE'] >= suspicion_threshold].copy()
        st.write(f"Transactions above threshold: {len(df_filtered):,}")

        if len(df_filtered) > 2000:
            st.warning("Large number of matches — showing top 2000 by score to keep the graph responsive.")
            df_filtered = df_filtered.nlargest(2000, 'SUSPICION_SCORE')

        display_network_graph(df_filtered, view_key="ibm_global_dashboard")

        st.markdown("---")
        st.subheader(f"Transaction Table (Filtered: {len(df_filtered):,} edges)")
        st.dataframe(
            df_filtered[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
            .sort_values(by='SUSPICION_SCORE', ascending=False)
            .head(100)
        )


def draw_ibm_deep_dive():
    df_scores = load_scores_data(SCORES_PATH)

    st.title("Investigate a Single IBM AML Account")

    if 'ibm_display_id' not in st.session_state:
        st.session_state.ibm_display_id = ''

    with st.form(key='ibm_search_form'):
        search_id_input = st.text_input("Enter Account ID:", value=st.session_state.ibm_display_id, key='ibm_search_box')
        submitted = st.form_submit_button("Analyze Account")
        if submitted:
            st.session_state.ibm_display_id = search_id_input.strip()
            st.rerun()

    if st.session_state.ibm_display_id:
        account_id = st.session_state.ibm_display_id
        st.markdown(f"### Deep Dive: `{account_id}`")

        df_account = df_scores[
            (df_scores['SENDER_ACCOUNT_ID'] == account_id) |
            (df_scores['RECEIVER_ACCOUNT_ID'] == account_id)
        ].copy()

        if df_account.empty:
            st.error("Account ID not found.")
            st.session_state.ibm_display_id = ''
            return

        total_sent = df_account[df_account['SENDER_ACCOUNT_ID'] == account_id]['TX_AMOUNT'].sum()
        total_received = df_account[df_account['RECEIVER_ACCOUNT_ID'] == account_id]['TX_AMOUNT'].sum()
        avg_score = df_account['SUSPICION_SCORE'].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total $ Sent", f"${total_sent:,.2f}")
        col2.metric("Total $ Received", f"${total_received:,.2f}")
        risk_gauge(avg_score)

        st.markdown("#### Local Transaction Graph (Egonet)")
        display_network_graph(df_account, view_key="ibm_deep_dive")

        st.markdown("#### Full Transaction History")
        st.dataframe(
            df_account[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
            .sort_values(by='SUSPICION_SCORE', ascending=False)
        )


def draw_ibm_router():
    st.sidebar.title("🏦 IBM AML Pages")
    mode = st.sidebar.radio("Choose view:", ["Global Dashboard", "Account Deep Dive"])
    df_scores = load_scores_data(SCORES_PATH)
    if df_scores is None:
        st.error("IBM AML data is required to view this project.")
        return
    if mode == "Global Dashboard":
        draw_ibm_dashboard()
    else:
        draw_ibm_deep_dive()