import json
import os

import streamlit as st

DEFAULT_THRESHOLD = 0.5


@st.cache_data
def load_metrics(path):
    """Reads a metrics*.json written by the train scripts. Returns {} if it's missing."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def get_threshold(metrics, default=DEFAULT_THRESHOLD):
    """Decision threshold tuned on the validation set during training."""
    return float(metrics.get('best_threshold', default))
