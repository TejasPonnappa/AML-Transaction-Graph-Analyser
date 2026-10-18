import json
import os

import streamlit as st

DEFAULT_THRESHOLD = 0.5


@st.cache_data
def _read_metrics(path, mtime):
    with open(path, 'r') as f:
        return json.load(f)


def load_metrics(path):
    """Reads a metrics*.json written by the train scripts. Returns {} if it's missing.
    The cache is keyed on the file's modification time, so retraining shows up without a restart."""
    if not os.path.exists(path):
        return {}
    return _read_metrics(path, os.path.getmtime(path))


def get_threshold(metrics, default=DEFAULT_THRESHOLD):
    """Decision threshold tuned on the validation set during training."""
    return float(metrics.get('best_threshold', default))


def has_tuned_threshold(metrics):
    return 'best_threshold' in metrics
