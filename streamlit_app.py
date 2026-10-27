import os

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="AML Transaction Analyser")

from ui import theme
from ui.elliptic_dashboard import render_elliptic
from ui.ibm_dashboard import render_ibm
from ui.metrics_loader import load_metrics

theme.apply_theme()

PROJECTS = {
    "Overview": None,
    "IBM AML": dict(
        route=render_ibm, scores='outputs/ibm_predictions_rf.csv', metrics='outputs/metrics_ibm_rf.json',
        task="Edge classification, RandomForest", data="IBM HI-Small banking transactions"),
    "Elliptic": dict(
        route=render_elliptic, scores='outputs/bitcoin_predictions.csv', metrics='outputs/metrics_bitcoin.json',
        task="Node classification, GraphSAGE", data="Bitcoin transaction graph"),
}

st.sidebar.markdown("**AML Transaction Analyser**")
selection = st.sidebar.selectbox("Dataset", list(PROJECTS))
project = PROJECTS[selection]

if project is None:
    theme.page_header("AML Transaction Analyser",
                      "Graph-derived features and models that score transactions for money-laundering and "
                      "illicit-activity alerts, with case investigation on a transaction network.")
    rows = []
    for name, cfg in list(PROJECTS.items())[1:]:
        metrics = load_metrics(cfg['metrics'])
        rows.append({
            "Dataset": name, "Data": cfg['data'], "Model": cfg['task'],
            "Status": "Ready" if os.path.exists(cfg['scores']) else "Run the pipeline scripts",
            "Test PR-AUC": metrics.get('test_pr_auc'), "Test recall": metrics.get('test_recall'),
        })
    theme.section("Datasets", "Test metrics come from the held-out split of the last training run.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "Test PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Test recall": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption(theme.SCORE_NOTE)
elif not os.path.exists(project['scores']):
    st.error(f"No predictions found at {project['scores']}. Run the pipeline scripts for this dataset first.")
else:
    project['route']()
