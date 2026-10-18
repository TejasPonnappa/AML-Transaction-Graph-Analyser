"""Network view for the Elliptic (node classification) project."""
import streamlit as st

from ui.graph_view import render_graph


def display_network_graph(edges_df, nodes_df, view_key="bitcoin_global", threshold=0.5):
    """edges_df: source/target txIds. nodes_df: txId, SUSPICION_SCORE, class."""
    if edges_df.empty:
        st.info("No connections to display for this transaction.")
        return

    info = nodes_df.drop_duplicates('txId').set_index('txId')
    ids = set(edges_df['source']) | set(edges_df['target'])
    nodes = {}
    for tx in ids:
        if tx in info.index:
            row = info.loc[tx]
            score, label = float(row['SUSPICION_SCORE']), str(row['class'])
        else:
            score, label = 0.0, 'Unknown'
        nodes[tx] = {"score": score, "labelled": label == 'Illicit',
                     "tooltip": f"Transaction {tx}\nModel score {score:.4f}\nLabel {label}"}
    edges = [(s, t, nodes[s]["score"], f"{s} to {t}")
             for s, t in zip(edges_df['source'], edges_df['target'])]
    render_graph(nodes, edges, threshold)
