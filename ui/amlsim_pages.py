import streamlit as st
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from ui.graph_display_amlsim import display_network_graph  # now defaults to physics_profile="repulsion"

# --- Configuration (Using RELATIVE paths for reliability) ---
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'
SCORES_PATH = 'outputs/suspicion_scores.csv'
BEST_MODEL_AUC = 0.5649
FRAUD_RATIO = 0.0013

# --- Function to Load Graph Data for Global Metrics ---
@st.cache_data
def load_graph_data(path):
    """Loads the processed graph data object for dimension extraction."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception:
        return None

# --- Function to Load Suspicion Scores CSV ---
@st.cache_data
def load_scores_data(path):
    """Loads the processed suspicion scores."""
    if not os.path.exists(path):
        st.error(f"Error: Output scores file not found at {path}. Please run 'python src/predict.py' first.")
        return None
    df = pd.read_csv(path)
    df['SENDER_ACCOUNT_ID'] = df['SENDER_ACCOUNT_ID'].astype(str)
    df['RECEIVER_ACCOUNT_ID'] = df['RECEIVER_ACCOUNT_ID'].astype(str)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df

# --- Function to Render Leaderboards ---
def render_leaderboards(df_scores):
    st.subheader("📊 Top 10 High-Risk Account Leaderboards")

    RISK_THRESHOLD = 0.5075
    df_risky = df_scores[df_scores['SUSPICION_SCORE'] >= RISK_THRESHOLD].copy()

    if df_risky.empty:
        st.info(f"No transactions meet the high-risk threshold of {RISK_THRESHOLD:.4f} for leaderboard calculation.")
        return

    st.markdown("#### 🥇 Top 10 Risky Accounts (by Average Suspicion Score)")
    senders = df_risky[['SENDER_ACCOUNT_ID', 'SUSPICION_SCORE']].rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
    receivers = df_risky[['RECEIVER_ACCOUNT_ID', 'SUSPICION_SCORE']].rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})
    all_risky_accounts = pd.concat([senders, receivers])
    avg_score_leaderboard = all_risky_accounts.groupby('ACCOUNT_ID')['SUSPICION_SCORE'].mean().sort_values(ascending=False).head(10).reset_index()
    avg_score_leaderboard.columns = ['Account ID', 'Average Suspicion Score']
    avg_score_leaderboard['Average Suspicion Score'] = avg_score_leaderboard['Average Suspicion Score'].map('{:.5f}'.format)
    st.dataframe(avg_score_leaderboard, use_container_width=True, hide_index=True)

    st.markdown("#### 🥈 Top 10 Risky Hubs (by Count of Suspicious Incoming Transactions)")
    hub_leaderboard = df_risky.groupby('RECEIVER_ACCOUNT_ID').size().sort_values(ascending=False).head(10).reset_index()
    hub_leaderboard.columns = ['Account ID', 'Incoming Suspicious Count']
    st.dataframe(hub_leaderboard, use_container_width=True, hide_index=True)

    st.markdown("#### 🥉 Top 10 Risky Volume Targets (by Total $ Received)")
    volume_leaderboard = df_risky.groupby('RECEIVER_ACCOUNT_ID')['TX_AMOUNT'].sum().sort_values(ascending=False).head(10).reset_index()
    volume_leaderboard.columns = ['Account ID', 'Total Suspicious $ Received']
    volume_leaderboard['Total Suspicious $ Received'] = volume_leaderboard['Total Suspicious $ Received'].map('${:,.2f}'.format)
    st.dataframe(volume_leaderboard, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# --- PATTERN ANALYSIS FUNCTION (FINAL HEURISTICS WITH SCORE CHECK) ---
# ----------------------------------------------------------------------

def analyze_aml_patterns(df_account, net_flow, avg_score):
    """Heuristically determines the most likely AML pattern based on local graph metrics."""

    if avg_score < 0.5:
        return "Likely Safe (Low GNN Score)"

    all_involved_ids = set(df_account['SENDER_ACCOUNT_ID']) | set(df_account['RECEIVER_ACCOUNT_ID'])
    central_account_id = pd.concat([df_account['SENDER_ACCOUNT_ID'], df_account['RECEIVER_ACCOUNT_ID']]).mode()[0] if not df_account.empty else None

    if central_account_id is None:
        return "Not enough data for analysis."

    in_degree = df_account[df_account['RECEIVER_ACCOUNT_ID'] == central_account_id].shape[0]
    out_degree = df_account[df_account['SENDER_ACCOUNT_ID'] == central_account_id].shape[0]

    total_sent = df_account[df_account['SENDER_ACCOUNT_ID'] == central_account_id]['TX_AMOUNT'].sum()
    total_received = df_account[df_account['RECEIVER_ACCOUNT_ID'] == central_account_id]['TX_AMOUNT'].sum()
    net_flow = total_received - total_sent
    total_volume = total_sent + total_received

    HIGH_DEGREE_THRESHOLD = 5
    MULE_FLOW_RATIO = 0.5
    MIN_DISPERSAL_RATIO = 0.25

    patterns = []

    is_mule = False
    if total_volume > 0 and abs(net_flow) > MULE_FLOW_RATIO * total_volume:
        is_mule = True
        if net_flow > 0:
            patterns.append("Mule Account (Consolidation Target)")
        else:
            patterns.append("Mule Account (Dispersal Source)")

    if is_mule and net_flow > 0 and in_degree > HIGH_DEGREE_THRESHOLD:
        patterns.append("Smurfing Target (Many Deposits)")

    if in_degree >= 2 and out_degree >= 2 and total_received > 0 and total_sent >= MIN_DISPERSAL_RATIO * total_received:
        if total_volume == 0 or abs(net_flow) < 0.20 * total_volume:
            patterns.append("Layering/Pass-Through Node")

    if not patterns:
        return "Suspicious Activity Detected (GNN Score High)"

    return ", ".join(patterns)

# ----------------------------------------------------------------------
# --- DRAW AMLSIM DASHBOARD (Global View) ---
# ----------------------------------------------------------------------

def draw_amlsim_dashboard():
    df_scores = load_scores_data(SCORES_PATH)
    data = load_graph_data(PROCESSED_DATA_PATH)
    total_unique_accounts = data.x.size(0) if data is not None else 0

    st.title("🌎 AMLSim Banking Transaction Analyzer")
    st.markdown("### GraphSAGE Edge Classification Results")

    if df_scores is not None:

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions Analyzed", f"{len(df_scores):,}")
        col2.metric("Total Unique Accounts", f"{total_unique_accounts:,}")
        col3.metric("AML Fraud Ratio", f"{FRAUD_RATIO * 100:.2f}%")
        col4.metric("Model ROC-AUC (Test)", f"{BEST_MODEL_AUC:.4f}")
        st.markdown("---")

        st.subheader("Transactions by Suspicion Score Range (0.05 steps)")
        bins = [round(x * 0.05, 2) for x in range(21)]
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
        if 'SUSPICION_SCORE' in df_scores.columns:
            df_scores['SCORE_RANGE'] = pd.cut(df_scores['SUSPICION_SCORE'], bins=bins, labels=labels, include_lowest=True, right=False)
            range_counts = df_scores['SCORE_RANGE'].value_counts().sort_index()
            st.bar_chart(range_counts)
        else:
            st.warning("Suspicion score data not available for histogram.")
        st.markdown("---")

        render_leaderboards(df_scores)
        st.markdown("---")

        st.subheader("Interactive Suspicious Network Map (Global)")
        max_score = df_scores['SUSPICION_SCORE'].max()

        FIXED_DEFAULT_THRESHOLD = 0.51805

        suspicion_threshold = st.slider(
            "Suspicion Score Threshold (Show Transactions Above This Score)",
            min_value=float(df_scores['SUSPICION_SCORE'].min()),
            max_value=float(max_score),
            value=float(FIXED_DEFAULT_THRESHOLD),
            step=0.00001,
            format="%.5f"
        )

        df_filtered = df_scores[df_scores['SUSPICION_SCORE'] >= suspicion_threshold].copy()
        st.write(f"Transactions above threshold {suspicion_threshold:.5f}: {len(df_filtered):,}")

        # physics_profile omitted -> defaults to "repulsion" (spread-out, matches the look you want)
        display_network_graph(df_filtered, view_key="global_dashboard")

        st.markdown("---")
        st.subheader(f"Transaction Table (Filtered: {len(df_filtered):,} edges)")
        st.dataframe(
            df_filtered[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
            .sort_values(by='SUSPICION_SCORE', ascending=False)
            .head(100)
        )

# ----------------------------------------------------------------------
# --- DRAW AMLSIM DEEP DIVE VIEW ---
# ----------------------------------------------------------------------
def draw_amlsim_deep_dive():
    df_scores = load_scores_data(SCORES_PATH)

    st.title("Investigate a Single Account")

    if 'amlsim_display_id' not in st.session_state:
        st.session_state.amlsim_display_id = ''

    with st.form(key='amlsim_search_form'):
        search_id_input = st.text_input(
            "Enter Account ID to investigate:",
            value=st.session_state.amlsim_display_id,
            key='amlsim_search_box'
        )
        submitted = st.form_submit_button("Analyze Account")

        if submitted:
            st.session_state.amlsim_display_id = search_id_input.strip()
            st.rerun()

    if st.session_state.amlsim_display_id:
        account_id = st.session_state.amlsim_display_id
        st.markdown(f"### Deep Dive Analysis for Account: `{account_id}`")

        df_account = df_scores[
            (df_scores['SENDER_ACCOUNT_ID'] == account_id) |
            (df_scores['RECEIVER_ACCOUNT_ID'] == account_id)
        ].copy()

        if df_account.empty:
            st.error("Account ID not found in transaction data.")
            st.session_state.amlsim_display_id = ''
            return

        tx_sent = df_account[df_account['SENDER_ACCOUNT_ID'] == account_id]
        tx_received = df_account[df_account['RECEIVER_ACCOUNT_ID'] == account_id]

        total_sent = tx_sent['TX_AMOUNT'].sum()
        total_received = tx_received['TX_AMOUNT'].sum()
        net_flow = total_received - total_sent
        avg_score = df_account['SUSPICION_SCORE'].mean()

        pattern_result = analyze_aml_patterns(df_account, net_flow, avg_score)

        st.markdown("#### Key Metrics and Pattern Classification")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Total $ Sent", f"${total_sent:,.2f}")
        stat_col2.metric("Total $ Received", f"${total_received:,.2f}")
        stat_col3.metric("Net $ Flow", f"${net_flow:,.2f}")
        stat_col4.metric("Avg. Suspicion Score", f"{avg_score:.4f}")

        st.markdown(f"**Predicted AML Patterns:** :red[{pattern_result}]")
        st.markdown("---")

        st.markdown("#### Local Transaction Graph (Egonet)")
        # physics_profile omitted -> defaults to "repulsion"
        display_network_graph(df_account, view_key="deep_dive")

        st.markdown("#### Full Transaction History")
        st.dataframe(
            df_account[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
            .sort_values(by='SUSPICION_SCORE', ascending=False)
        )

# ----------------------------------------------------------------------
# --- AMLSIM ROUTER FUNCTION ---
# ----------------------------------------------------------------------

def draw_amlsim_router():
    st.sidebar.title("🧠 AMLSim Pages")
    amlsim_mode = st.sidebar.radio(
        "Choose view:",
        ["Global Dashboard", "Account Deep Dive"]
    )
    df_scores = load_scores_data(SCORES_PATH)
    if df_scores is None:
        st.error("AMLSim data is required to view this project.")
        return

    if amlsim_mode == "Global Dashboard":
        draw_amlsim_dashboard()
    elif amlsim_mode == "Account Deep Dive":
        draw_amlsim_deep_dive()