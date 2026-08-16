import streamlit as st
import os
from ui.styling import apply_custom_style
from ui.dashboard_components import inject_component_css

apply_custom_style()
inject_component_css()

try:
    from ui.amlsim_pages import draw_amlsim_router
    from ui.bitcoin_pages import draw_bitcoin_router
    from ui.ibm_pages import draw_ibm_router  # UPDATED: new IBM page
except ImportError as e:
    st.error(f"FATAL ERROR: Could not find project modules. Ensure you have run file renaming commands.")
    st.error(f"Error details: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="Inferno GNN Analyzer")

st.sidebar.title("🔥 Inferno Project Selection")
st.sidebar.markdown("### GNN Financial Crime Analyzer")

amlsim_data_exists = os.path.exists('outputs/suspicion_scores.csv')
bitcoin_data_exists = os.path.exists('outputs/bitcoin_predictions.csv')
ibm_data_exists = os.path.exists('outputs/ibm_predictions.csv')  # UPDATED

project_options = {
    "Select Project": "landing",
    "1. AMLSim (Edge Classification)": "amlsim",
    "2. Elliptic (Node Classification)": "bitcoin",
    "3. IBM AML (Edge Classification)": "ibm"  # UPDATED
}

selection = st.sidebar.radio("Choose Project Model Type:", list(project_options.keys()))
project_mode = project_options[selection]

if project_mode == "landing":
    st.title("🔥 Inferno Project Portfolio")
    st.markdown("### Unified GNN Financial Crime Detection Dashboard")
    st.markdown("Welcome! Please select a project from the sidebar to begin analysis.")

    st.markdown("---")
    st.subheader("Project Status")

    st.metric("AMLSim Status (Edge Classification)", "Ready" if amlsim_data_exists else "Data Missing",
              delta="Run train/predict scripts" if not amlsim_data_exists else None)
    st.metric("Elliptic Status (Node Classification)", "Ready" if bitcoin_data_exists else "Data Missing",
              delta="Run train/predict scripts" if not bitcoin_data_exists else None)
    st.metric("IBM AML Status (Edge Classification)", "Ready" if ibm_data_exists else "Data Missing",  # UPDATED
              delta="Run train_ibm/predict_ibm scripts" if not ibm_data_exists else None)

elif project_mode == "amlsim":
    if amlsim_data_exists:
        draw_amlsim_router()
    else:
        st.error("AMLSim data not found. Please run the data_preprocessing.py, train.py, and predict.py scripts first.")

elif project_mode == "bitcoin":
    if bitcoin_data_exists:
        draw_bitcoin_router()
    else:
        st.error("Elliptic Bitcoin data not found. Please run the data_preprocessing_bitcoin.py, train_bitcoin.py, and predict_bitcoin.py scripts first.")

elif project_mode == "ibm":  # UPDATED
    if ibm_data_exists:
        draw_ibm_router()
    else:
        st.error("IBM AML data not found. Please run data_preprocessing_ibm.py, train_ibm.py, and predict_ibm.py first.")