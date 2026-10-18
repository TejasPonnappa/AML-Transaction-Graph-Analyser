import os

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="AML Transaction Analyser")

from ui import theme
from ui.metrics_loader import load_metrics

theme.apply_theme()

try:
    from ui.amlsim_pages import draw_amlsim_router
    from ui.bitcoin_pages import draw_bitcoin_router
    from ui.ibm_pages import draw_ibm_router
except ImportError as e:
    st.error("Could not load the dashboard modules.")
    st.error(f"Error details: {e}")
    st.stop()

PROJECTS = {
    "Overview": None,
    "AMLSim": dict(
        route=draw_amlsim_router, scores='outputs/suspicion_scores.csv', metrics='outputs/metrics.json',
        task="Edge classification", data="Simulated banking transfers"),
    "Elliptic": dict(
        route=draw_bitcoin_router, scores='outputs/bitcoin_predictions.csv', metrics='outputs/metrics_bitcoin.json',
        task="Node classification", data="Bitcoin transaction graph"),
    "IBM AML": dict(
        route=draw_ibm_router, scores='outputs/ibm_predictions.csv', metrics='outputs/metrics_ibm.json',
        task="Edge classification", data="IBM HI-Small synthetic banking data"),
}

st.sidebar.markdown("**AML Transaction Analyser**")
selection = st.sidebar.selectbox("Dataset", list(PROJECTS))
project = PROJECTS[selection]

if project is None:
    theme.page_header("AML Transaction Analyser",
                      "Graph neural network models that score transactions for money-laundering and "
                      "illicit-activity alerts, across three datasets.")
    rows = []
    for name, cfg in list(PROJECTS.items())[1:]:
        ready = os.path.exists(cfg['scores'])
        metrics = load_metrics(cfg['metrics'])
        rows.append({
            "Dataset": name,
            "Data": cfg['data'],
            "Task": cfg['task'],
            "Status": "Ready" if ready else "Run train and predict scripts",
            "Test PR-AUC": metrics.get('test_pr_auc'),
            "Test recall": metrics.get('test_recall'),
        })
    theme.section("Datasets", "Test metrics come from the held-out split of the last training run.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "Test PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Test recall": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption(theme.SCORE_NOTE)
elif not os.path.exists(project['scores']):
    st.error(f"No predictions found at {project['scores']}. Run the preprocessing, training and prediction "
             f"scripts for this dataset first.")
else:
    project['route']()
