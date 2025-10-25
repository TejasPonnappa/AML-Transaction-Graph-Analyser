import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import matplotlib.colors as mcolors
import torch
from torch_geometric.data import Data  # Needed to correctly load graph data structure

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'
SCORES_PATH = 'outputs/suspicion_scores.csv'
BEST_MODEL_AUC = 0.5649 
FRAUD_RATIO = 0.0013 

# --- Function to Load Graph Data ---
@st.cache_data
def load_graph_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Processed graph data not found at {path}. Cannot display metrics.")
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        st.error(f"Failed to load graph data for metrics. Error: {e}")
        return None

# --- Global Data Loading ---
data = load_graph_data(PROCESSED_DATA_PATH)
total_unique_accounts = data.x.size(0) if data is not None else 0

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="AML Transaction Graph Analyzer")

# --- Function to Load Suspicion Scores CSV ---
@st.cache_data
def load_scores_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Output scores file not found at {path}. Please run 'python src/predict.py' first.")
        return None
    df = pd.read_csv(path)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df

# --- Function to Build and Display the Network ---
def display_network(df_filtered, threshold):
    if df_filtered.empty:
        st.info("No transactions meet the current suspicion threshold. Try lowering the slider.")
        return

    G = nx.from_pandas_edgelist(
        df_filtered, 
        source='SENDER_ACCOUNT_ID', 
        target='RECEIVER_ACCOUNT_ID', 
        edge_attr=['TX_AMOUNT', 'SUSPICION_SCORE', 'TX_TYPE', 'IS_FRAUD'],
        create_using=nx.DiGraph()
    )

    net = Network(height='600px', width='100%', directed=True, notebook=False, cdn_resources='in_line')
    net.set_options("""
        var options = {
          "physics": {
            "barnesHH": {
              "centralGravity": 0.2,
              "springLength": 100,
              "springConstant": 0.05,
              "damping": 0.9
            },
            "minVelocity": 0.75
          }
        }
    """)

    node_scores = {}
    for _, row in df_filtered.iterrows():
        node_scores[row['SENDER_ACCOUNT_ID']] = max(node_scores.get(row['SENDER_ACCOUNT_ID'], 0), row['SUSPICION_SCORE'])
        node_scores[row['RECEIVER_ACCOUNT_ID']] = max(node_scores.get(row['RECEIVER_ACCOUNT_ID'], 0), row['SUSPICION_SCORE'])

    fraud_accounts = set(df_filtered[df_filtered['IS_FRAUD'] == 1]['SENDER_ACCOUNT_ID']).union(
                     set(df_filtered[df_filtered['IS_FRAUD'] == 1]['RECEIVER_ACCOUNT_ID']))

    cmap = mcolors.LinearSegmentedColormap.from_list("suspicion_cmap", ["blue", "red"])

    for node in G.nodes():
        score = node_scores.get(node, 0)
        color_val = min(score * 1.5, 1.0) 
        hex_color = mcolors.to_hex(cmap(color_val))
        border_width = 3 if node in fraud_accounts else 1
        title_html = f"**Account ID:** {node}<br>**Max Edge Score:** {score:.4f}<br>**Known Fraud:** {'Yes' if node in fraud_accounts else 'No'}"
        net.add_node(
            n_id=node, 
            label=str(node), 
            title=title_html, 
            color={'border': '#000000' if node not in fraud_accounts else '#FF0000', 'background': hex_color},
            borderWidth=border_width
        )

    for source, target, data in G.edges(data=True):
        score = data['SUSPICION_SCORE']
        line_thickness = max(0.5, score * 10) 
        line_color = mcolors.to_hex(cmap(min(score * 1.5, 1.0)))
        title_html = (
            f"**Score:** {score:.4f}<br>"
            f"**Amount:** {data['TX_AMOUNT']:,.2f}<br>"
            f"**Type:** {data['TX_TYPE']}<br>"
            f"**Known Fraud:** {'YES' if data['IS_FRAUD'] == 1 else 'No'}"
        )
        net.add_edge(source=source, to=target, title=title_html, value=line_thickness, color=line_color)

    html_file_path = 'outputs/network_graph.html'
    html_str = net.generate_html()
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=650)
    st.markdown("---")

    st.subheader(f"Transaction Table (Filtered: {len(df_filtered):,} edges)")
    st.dataframe(
        df_filtered[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
        .sort_values(by='SUSPICION_SCORE', ascending=False)
        .head(100)
    )

# --- MAIN APP LOGIC ---
st.title("🧠 AML Transaction Graph Analyzer")
st.markdown("### GraphSAGE Model Results Dashboard")

df_scores = load_scores_data(SCORES_PATH)

if df_scores is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions Analyzed", f"{len(df_scores):,}")
    col2.metric("Total Unique Accounts", f"{total_unique_accounts:,}")
    col3.metric("AML Fraud Ratio", f"{FRAUD_RATIO * 100:.2f}%")
    col4.metric("Model ROC-AUC (Test)", f"{BEST_MODEL_AUC:.4f}")
    
    st.markdown("---")
    st.subheader("Transactions by Suspicion Score Range (0.05 steps)")

    # --- 0.05 bins ---
    bins = [round(x * 0.05, 2) for x in range(21)]  # 0.00 to 1.00
    labels = [f"{round(bins[i],2)}-{round(bins[i+1],2)}" for i in range(len(bins)-1)]
    df_scores['SCORE_RANGE'] = pd.cut(df_scores['SUSPICION_SCORE'], bins=bins, labels=labels, include_lowest=True)
    range_counts = df_scores['SCORE_RANGE'].value_counts().sort_index()
    st.bar_chart(range_counts)

    st.markdown("---")
    st.subheader("Interactive Suspicious Network Map")

    max_score = df_scores['SUSPICION_SCORE'].max()
    default_threshold = max(0.501, max_score * 0.99)
    suspicion_threshold = st.slider(
        "Suspicion Score Threshold (Show Transactions Above This Score)",
        min_value=df_scores['SUSPICION_SCORE'].min(),
        max_value=max_score,
        value=default_threshold,
        step=0.00001,
        format="%.5f"
    )

    df_filtered = df_scores[df_scores['SUSPICION_SCORE'] >= suspicion_threshold].copy()
    st.write(f"Transactions above threshold {suspicion_threshold:.5f}: {len(df_filtered):,}")

    display_network(df_filtered, suspicion_threshold)