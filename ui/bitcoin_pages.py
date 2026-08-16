import streamlit as st
import pandas as pd
import os
import plotly.express as px
from ui.dashboard_components import animated_metric_row, score_distribution_chart, top_risk_ticker, risk_gauge
from ui.graph_display_bitcoin import display_network_graph

# --- Configuration ---
PREDICTIONS_PATH = 'outputs/bitcoin_predictions.csv'
EDGES_PATH = 'data/raw/elliptic_txs_edgelist.csv'
TOTAL_TXS = 203769
TOTAL_LABELED = 4545 + 42019
TEST_F1_SCORE = 0.8497
RISK_THRESHOLD = 0.95
MAX_GRAPH_NODES = 10000  # safety ceiling only — NOT the default cap. The 50% logic below
                          # already produces 3000/6000, 7000/14000, etc. This just stops
                          # things from running away past 10k nodes on extreme inputs.

st.set_page_config(layout="wide", page_title="Elliptic Bitcoin Analyzer")


@st.cache_data
def load_bitcoin_predictions(path):
    if not os.path.exists(path):
        st.error(f"Error: Prediction file not found at {path}. Please run 'python src/predict_bitcoin.py' first.")
        return None
    df = pd.read_csv(path)
    df['txId'] = df['txId'].astype(str)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df


@st.cache_data
def load_edge_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Edge list not found at {path}. Cannot render graph.")
        return None
    df_edges = pd.read_csv(path)
    df_edges.rename(columns={'txId1': 'source', 'txId2': 'target'}, inplace=True)
    df_edges['source'] = df_edges['source'].astype(str)
    df_edges['target'] = df_edges['target'].astype(str)
    return df_edges


df_pred = load_bitcoin_predictions(PREDICTIONS_PATH)
df_edges = load_edge_data(EDGES_PATH)


def render_leaderboards(df_pred):
    st.subheader("🥇 Top Predicted High-Risk Addresses (Unseen)")

    df_risky = df_pred[
        (df_pred['SUSPICION_SCORE'] >= RISK_THRESHOLD) &
        (df_pred['class'] == 'Unlabeled')
    ].copy()

    if df_risky.empty:
        st.info(f"No unlabeled addresses meet the high-risk threshold of {RISK_THRESHOLD:.2f}.")
        return

    st.markdown("#### Top 10 High-Risk Unlabeled Addresses (by Illicit Score)")
    score_leaderboard = df_risky[['txId', 'SUSPICION_SCORE']].head(10).reset_index(drop=True)
    score_leaderboard.index += 1
    score_leaderboard.columns = ['Address ID', 'Illicit Score']
    st.dataframe(score_leaderboard.style.format({'Illicit Score': '{:.6f}'}), use_container_width=True)

    st.markdown("#### Top 10 Risky TimeSteps (Highest Count of Illicit Predictions)")
    hub_leaderboard = df_risky.groupby('TimeStep').size().sort_values(ascending=False).head(10).reset_index()
    hub_leaderboard.columns = ['TimeStep', 'Count of Risky Addresses']
    st.dataframe(hub_leaderboard, use_container_width=True, hide_index=True)


def analyze_bitcoin_patterns(account_id, df_edges, score):
    if score < 0.5:
        return "Likely Licit (Low GNN Score)"

    in_degree = df_edges[df_edges['target'] == account_id].shape[0]
    out_degree = df_edges[df_edges['source'] == account_id].shape[0]

    patterns = []
    HIGH_DEGREE_THRESHOLD = 5
    BALANCE_THRESHOLD = 2

    is_mule = False
    if abs(in_degree - out_degree) > BALANCE_THRESHOLD:
        is_mule = True
        if in_degree > out_degree:
            patterns.append("Mule-like (Consolidation Transaction)")
            if in_degree > HIGH_DEGREE_THRESHOLD:
                patterns.append("Smurfing Inputs Suspected")
        else:
            patterns.append("Mule-like (Dispersal Transaction)")

    if not is_mule and in_degree >= 1 and out_degree >= 1:
        patterns.append("Layering/Pass-Through Transaction")

    if not patterns:
        return "Suspicious Activity Detected (GNN Score High)"

    return ", ".join(patterns)


def draw_global_dashboard(df_pred):
    st.title("₿ Elliptic Bitcoin Transaction Analyzer")
    st.markdown("### GraphSAGE Node Classification Results")

    PREDICTED_UNLABELED = df_pred[(df_pred['SUSPICION_SCORE'] >= RISK_THRESHOLD) & (df_pred['class'] == 'Unlabeled')].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Addresses Analyzed", f"{TOTAL_TXS:,}")
    col2.metric("Total Labeled Addresses", f"{TOTAL_LABELED:,}")
    col3.metric("Model F1 Score (Test)", f"{TEST_F1_SCORE:.4f}")
    col4.metric(f"New Predictions (Risk ≥ {RISK_THRESHOLD:.2f})", f"{PREDICTED_UNLABELED:,}")

    st.markdown("#### 🔴 Highest Risk Addresses Right Now")
    top_risk_ticker(df_pred, id_col='txId', score_col='SUSPICION_SCORE')
    st.markdown("---")

    st.subheader("Distribution of Predicted Illicit Scores")
    fig_hist = px.histogram(
        df_pred, x='SUSPICION_SCORE', nbins=20,
        title='Illicit Probability Score Distribution Across All Addresses',
        template='plotly_dark', range_x=[0, 1]
    )
    fig_hist.update_layout(
        xaxis_title="Illicit Score (0.0 to 1.0)",
        yaxis_title="Number of Addresses",
        xaxis=dict(tickmode='linear', tick0=0.0, dtick=0.05),
        bargap=0.15
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")

    render_leaderboards(df_pred)
    st.markdown("---")

    st.subheader("Time-Series Risk Analysis (Average Suspicion Score per TimeStep)")
    df_time_risk = df_pred.groupby('TimeStep')['SUSPICION_SCORE'].mean().reset_index()
    df_time_risk.columns = ['TimeStep', 'Average Suspicion Score']
    fig_line = px.line(df_time_risk, x='TimeStep', y='Average Suspicion Score',
                        title='Average Illicit Probability by TimeStep', template='plotly_dark')
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("---")

    st.subheader("Interactive Suspicious Network Map (Global Top Risk)")
    max_score = df_pred['SUSPICION_SCORE'].max()
    FIXED_DEFAULT_THRESHOLD = 0.999

    suspicion_threshold = st.slider(
        "Filter Nodes by Suspicion Score Threshold:",
        min_value=float(df_pred['SUSPICION_SCORE'].min()),
        max_value=float(max_score),
        value=float(FIXED_DEFAULT_THRESHOLD),
        step=0.00001, format="%.5f", key='global_slider'
    )

    df_filtered_nodes = df_pred[df_pred['SUSPICION_SCORE'] >= suspicion_threshold].copy()

    # 50% scaling: this is what gives you 3000/6000, 7000/14000, etc. —
    # scales naturally with however many matches the threshold produces.
    total_matches = len(df_filtered_nodes)
    df_filtered_nodes = df_filtered_nodes.nlargest(total_matches // 2, 'SUSPICION_SCORE')

    # Ceiling only kicks in for genuinely huge inputs — does NOT override
    # the 50% figure above under normal conditions.
    if len(df_filtered_nodes) > MAX_GRAPH_NODES:
        st.warning(f"Showing top {MAX_GRAPH_NODES} of {total_matches:,} matches by score to keep the graph readable.")
        df_filtered_nodes = df_filtered_nodes.nlargest(MAX_GRAPH_NODES, 'SUSPICION_SCORE')

    nodes_to_keep = df_filtered_nodes['txId'].tolist()
    df_filtered_edges = df_edges[
        (df_edges['source'].isin(nodes_to_keep)) &
        (df_edges['target'].isin(nodes_to_keep))
    ].copy()
    all_involved_ids = set(df_filtered_edges['source']).union(set(df_filtered_edges['target']))
    df_filtered_nodes = df_pred[df_pred['txId'].isin(all_involved_ids)].copy()

    st.write(f"Showing connections between **{len(nodes_to_keep):,}** high-risk addresses, resulting in **{len(df_filtered_edges):,}** connections.")

    display_network_graph(df_filtered_edges, df_filtered_nodes, view_key="bitcoin_global")


def draw_bitcoin_deep_dive():
    st.title("🔎 Investigate a Single Bitcoin Transaction (Address)")
    st.markdown("Analyze the local network (Egonet) and risk metrics for any address.")

    search_id = st.text_input("Enter Transaction ID (e.g., from the table above):")

    if search_id:
        account_id = search_id.strip()
        node_data = df_pred[df_pred['txId'] == account_id]

        if node_data.empty:
            st.error("Transaction ID not found in prediction data.")
            return

        score = node_data['SUSPICION_SCORE'].iloc[0]
        ground_truth = node_data['class'].iloc[0]

        st.markdown(f"### Deep Dive Analysis for Transaction ID: `{account_id}`")

        df_account_edges = df_edges[
            (df_edges['source'] == account_id) | (df_edges['target'] == account_id)
        ].copy()

        neighbors = set(df_account_edges['source']).union(set(df_account_edges['target']))
        df_account_nodes = df_pred[df_pred['txId'].isin(neighbors)].copy()

        st.markdown("#### Key Risk Metrics and Pattern Classification")
        col1, col2, col3 = st.columns(3)
        col1.metric("Illicit Suspicion Score", f"{score:.6f}", delta_color="off", delta=f"Ground Truth: {ground_truth}")
        col2.metric("Total Neighbors (Transactions)", f"{len(neighbors) - 1:,}")
        col3.metric("Number of Direct Connections", f"{len(df_account_edges):,}")

        pattern_result = analyze_bitcoin_patterns(account_id, df_edges, score)
        st.markdown(f"**Predicted Structural Patterns:** :red[{pattern_result}]")
        st.markdown("---")

        st.markdown("#### Local Transaction Graph (Egonet)")
        display_network_graph(df_account_edges, df_account_nodes, view_key="bitcoin_deep_dive")

        st.markdown("---")
        st.markdown("#### Transaction Details (Egonet)")

        df_table = df_account_edges.merge(
            df_pred[['txId', 'class', 'SUSPICION_SCORE']].rename(columns={
                'txId': 'txId_source', 'class': 'class_sender', 'SUSPICION_SCORE': 'SUSPICION_SCORE_sender'
            }),
            left_on='source', right_on='txId_source', how='left'
        ).drop(columns=['txId_source'])

        df_table = df_table.merge(
            df_pred[['txId', 'class', 'SUSPICION_SCORE']].rename(columns={
                'txId': 'txId_target', 'class': 'class_receiver', 'SUSPICION_SCORE': 'SUSPICION_SCORE_receiver'
            }),
            left_on='target', right_on='txId_target', how='left'
        ).drop(columns=['txId_target'])

        df_display = df_table[['source', 'target', 'SUSPICION_SCORE_sender', 'class_sender', 'SUSPICION_SCORE_receiver', 'class_receiver']].copy()
        df_display.columns = ['Sender ID', 'Receiver ID', 'Score (Sender)', 'Class (Sender)', 'Score (Receiver)', 'Class (Receiver)']

        st.dataframe(
            df_display.style.format({'Score (Sender)': '{:.6f}', 'Score (Receiver)': '{:.6f}'}),
            use_container_width=True
        )


def draw_bitcoin_router():
    st.sidebar.title("₿ Elliptic Pages")
    app_mode = st.sidebar.radio("Choose view:", ["Global Dashboard", "Address Deep Dive"])

    if df_pred is not None and df_edges is not None:
        if app_mode == "Global Dashboard":
            draw_global_dashboard(df_pred)
        elif app_mode == "Address Deep Dive":
            draw_bitcoin_deep_dive()
    else:
        st.error("Failed to load necessary data files. Please ensure you have run data_preprocessing_bitcoin.py and predict_bitcoin.py.")