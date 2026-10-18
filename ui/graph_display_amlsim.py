"""Network view for the edge-classification projects (AMLSim, IBM AML)."""
import streamlit as st

from ui.graph_view import render_graph


def display_network_graph(df, view_key="global", threshold=0.5):
    """df: scored transactions (SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_AMOUNT, SUSPICION_SCORE, IS_FRAUD)."""
    if df.empty:
        st.info("No transactions to display for this view.")
        return

    work = df[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'SUSPICION_SCORE', 'IS_FRAUD']].copy()
    work['SENDER_ACCOUNT_ID'] = work['SENDER_ACCOUNT_ID'].astype(str)
    work['RECEIVER_ACCOUNT_ID'] = work['RECEIVER_ACCOUNT_ID'].astype(str)

    pairs = work.groupby(['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID'], sort=False).agg(
        score=('SUSPICION_SCORE', 'max'), count=('SUSPICION_SCORE', 'size'),
        amount=('TX_AMOUNT', 'sum'), labelled=('IS_FRAUD', 'max'),
    ).reset_index()

    nodes = {}
    for row in pairs.itertuples(index=False):
        for account in (row.SENDER_ACCOUNT_ID, row.RECEIVER_ACCOUNT_ID):
            info = nodes.setdefault(account, {"score": 0.0, "labelled": False, "count": 0})
            info["score"] = max(info["score"], float(row.score))
            info["labelled"] = info["labelled"] or bool(row.labelled)
            info["count"] += int(row.count)
    for account, info in nodes.items():
        info["tooltip"] = (f"Account {account}\nHighest model score {info['score']:.3f}\n"
                           f"Transactions plotted {info['count']:,}\nLabelled positive: {'yes' if info['labelled'] else 'no'}")

    edges = [
        (r.SENDER_ACCOUNT_ID, r.RECEIVER_ACCOUNT_ID, float(r.score),
         f"Model score {r.score:.3f}\nTransactions plotted {r.count:,}\nAmount plotted {r.amount:,.2f}\n"
         f"Labelled positive: {'yes' if r.labelled else 'no'}")
        for r in pairs.itertuples(index=False)
    ]
    render_graph(nodes, edges, threshold)
