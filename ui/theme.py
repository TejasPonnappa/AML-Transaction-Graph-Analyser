"""
Shared look and feel for every page: palette, Plotly template, CSS and the small
set of reusable components (header, KPI row, tables, legends).

Colour rules: one accent for interface chrome and neutral data; red is used only
for high-scoring items (alerts); amber marks the 'Review' band; everything else
is a neutral grey.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Palette -----------------------------------------------------------------
BG = "#0e1217"
SURFACE = "#151b23"
BORDER = "#252d38"
TEXT = "#e6e9ee"
MUTED = "#8b95a3"
ACCENT = "#4ea1c8"
HIGH = "#e5534b"      # reserved for alerts / high-scoring items
REVIEW = "#c9973d"
NEUTRAL = "#5c6773"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

SCORE_NOTE = (
    "Model score is the raw GNN output between 0 and 1. It is not calibrated, so it is "
    "not a probability or a percentage risk. An alert is a score at or above the "
    "alert threshold (by default the value tuned on validation data during training); Review is at least half of it."
)

# --- Plotly template -----------------------------------------------------------
_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT, size=12),
    colorway=[ACCENT, NEUTRAL, REVIEW, MUTED],
    margin=dict(t=16, b=40, l=56, r=16),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT)),
)
pio.templates["aml_dark"] = _template

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
#MainMenu, footer {{ visibility: hidden; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; font-weight: 600; }}
.page-header {{ margin-bottom: 1.1rem; }}
.page-header .title {{ font-size: 1.6rem; font-weight: 600; color: {TEXT}; line-height: 1.2; }}
.page-header .subtitle {{ color: {MUTED}; font-size: 0.92rem; margin-top: 0.25rem; }}
.kpi-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 0.8rem 1rem; }}
.kpi-label {{ color: {MUTED}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi-value {{ color: {TEXT}; font-size: 1.55rem; font-weight: 600; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }}
.kpi-value.alert {{ color: {HIGH}; }}
.kpi-help {{ color: {MUTED}; font-size: 0.75rem; margin-top: 0.15rem; }}
.section-label {{ color: {TEXT}; font-size: 1.02rem; font-weight: 600; margin: 1.4rem 0 0.15rem 0; }}
.section-caption {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 0.5rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; height: 2.6rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
</style>
"""


def apply_theme():
    """Call once per run, after st.set_page_config()."""
    pio.templates.default = "aml_dark"
    st.markdown(_CSS, unsafe_allow_html=True)


# --- Layout components -----------------------------------------------------------
def page_header(title, subtitle=None):
    sub = f'<div class="subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-header"><div class="title">{html.escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of dicts {label, value, help (optional), alert (optional bool)}."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        cls = "kpi-value alert" if item.get("alert") else "kpi-value"
        help_html = f'<div class="kpi-help">{html.escape(str(item["help"]))}</div>' if item.get("help") else ""
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{html.escape(str(item["label"]))}</div>'
            f'<div class="{cls}">{html.escape(str(item["value"]))}</div>{help_html}</div>',
            unsafe_allow_html=True,
        )


def section(title, caption=None):
    st.markdown(f'<div class="section-label">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)


def empty_state(message):
    st.info(message)


def stale_warning(is_stale, predict_cmd):
    if is_stale:
        st.warning("The predictions file is older than the latest training run, so the figures below and the "
                   f"held-out test evaluation may not describe the same model. Regenerate with `{predict_cmd}`.")


def fmt_int(n):
    return f"{int(n):,}"


def fmt_amount(x):
    """Compact amount for KPI cards (1.2M, 340.5K); full precision stays in tables."""
    x = float(x)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e4, "K")):
        if abs(x) >= limit:
            return f"{x / limit:,.1f}{suffix}"
    return f"{x:,.2f}"


def fmt_pct(x, digits=1):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


# --- Scores, bands and tables --------------------------------------------------------
def risk_band(scores, threshold):
    """'Alert' at or above the threshold, 'Review' at or above half of it, else 'Normal'."""
    s = np.asarray(scores, dtype=float)
    return np.where(s >= threshold, "Alert", np.where(s >= threshold * 0.5, "Review", "Normal"))


def _band_style(value):
    if value == "Alert":
        return f"color: {HIGH}; font-weight: 600"
    if value == "Review":
        return f"color: {REVIEW}"
    return f"color: {MUTED}"


COLUMN_LABELS = {
    "SENDER_ACCOUNT_ID": "From account",
    "RECEIVER_ACCOUNT_ID": "To account",
    "TX_AMOUNT": "Amount",
    "TX_TYPE": "Type",
    "TIMESTAMP": "Timestamp",
    "IS_FRAUD": "Labelled",
    "SUSPICION_SCORE": "Model score",
    "SPLIT": "Split",
    "STATUS": "Status",
    "txId": "Transaction ID",
    "TimeStep": "Time step",
    "class": "Label",
    "ACCOUNT_ID": "Account",
    "ALERTS": "Alerts",
    "MAX_SCORE": "Max model score",
    "ALERT_AMOUNT": "Alert amount",
}


def _column_config(columns, currency, time_kind):
    cfg = {}
    for col in columns:
        label = COLUMN_LABELS.get(col, col)
        if col == "SUSPICION_SCORE" or col == "MAX_SCORE":
            cfg[col] = st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.3f")
        elif col in ("TX_AMOUNT", "ALERT_AMOUNT"):
            cfg[col] = st.column_config.NumberColumn(label, format=currency)
        elif col == "TIMESTAMP" and time_kind == "datetime":
            cfg[col] = st.column_config.DatetimeColumn(label, format="YYYY-MM-DD HH:mm")
        elif col == "TIMESTAMP":
            cfg[col] = st.column_config.NumberColumn("Step", format="%d")
        elif col == "ALERTS":
            cfg[col] = st.column_config.NumberColumn(label, format="%d")
        else:
            cfg[col] = st.column_config.Column(label)
    return cfg


def data_table(df, *, key, file_name, threshold=None, currency="localized", time_kind="datetime",
               height=420, max_rows=1000, label_yes="Yes"):
    """Sortable table with formatted columns, a status badge and a CSV export.

    `df` should already be sorted the way it should appear. Only the first
    `max_rows` rows are rendered and exported.
    """
    view = df.head(max_rows).copy()
    if "IS_FRAUD" in view.columns:
        view["IS_FRAUD"] = np.where(view["IS_FRAUD"].astype(int) == 1, label_yes, "")
    if threshold is not None and "SUSPICION_SCORE" in view.columns:
        view.insert(view.columns.get_loc("SUSPICION_SCORE") + 1, "STATUS",
                    risk_band(view["SUSPICION_SCORE"], threshold))
    for col in view.columns:
        if isinstance(view[col].dtype, pd.CategoricalDtype):
            view[col] = view[col].astype(str)

    styled = view.style.map(_band_style, subset=["STATUS"]) if "STATUS" in view.columns else view
    st.dataframe(
        styled,
        column_config=_column_config(view.columns, currency, time_kind),
        hide_index=True,
        use_container_width=True,
        height=height,
    )
    export = view.rename(columns=COLUMN_LABELS)
    for col in export.columns:  # neutralise spreadsheet formulas in exported text
        if export[col].dtype == object:
            export[col] = export[col].map(
                lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    left, right = st.columns([5, 1])
    left.caption(f"Showing {len(view):,} of {len(df):,} rows. Click a column header to sort.")
    right.download_button(
        "Export CSV", export.to_csv(index=False).encode("utf-8"),
        file_name=file_name, mime="text/csv", key=f"dl_{key}", use_container_width=True,
    )


def score_histogram(scores, threshold, bins=50):
    """Counts per score bin on a log axis, so the rare high-score tail stays visible."""
    counts, edges = np.histogram(np.asarray(scores, dtype=float), bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, width=1.0 / bins * 0.92, marker_color=NEUTRAL,
                           hovertemplate="score %{x:.2f}<br>%{y:,} rows<extra></extra>"))
    fig.add_vline(x=threshold, line_dash="dash", line_color=TEXT,
                  annotation_text=f"Alert threshold {threshold:.2f}", annotation_font_color=TEXT,
                  annotation_position="top left")
    fig.update_layout(height=300, bargap=0.05, xaxis_title="Model score", yaxis_title="Rows (log scale)",
                      yaxis=dict(type="log", tickvals=[1, 10, 100, 1e3, 1e4, 1e5, 1e6],
                                  ticktext=["1", "10", "100", "1k", "10k", "100k", "1M"]),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


_EVAL_LABELS = [
    ("test_roc_auc", "ROC-AUC", "{:.3f}"),
    ("test_pr_auc", "PR-AUC", "{:.3f}"),
    ("test_precision", "Precision at threshold", "{:.3f}"),
    ("test_recall", "Recall at threshold", "{:.3f}"),
    ("test_f1_tuned_thresh", "F1 at threshold", "{:.3f}"),
    ("test_f1", "F1 at threshold", "{:.3f}"),
    ("best_threshold", "Alert threshold (validation-tuned)", "{:.2f}"),
]


def evaluation_table(metrics):
    """Held-out test metrics as read from metrics*.json, plus the confusion matrix."""
    if not metrics:
        empty_state("No evaluation metrics found. Run the training script to generate them.")
        return
    rows, seen = [], set()
    for key, label, fmt in _EVAL_LABELS:
        if metrics.get(key) is not None and label not in seen:
            rows.append((label, fmt.format(metrics[key])))
            seen.add(label)
    cm = metrics.get("confusion_matrix")
    if cm:
        rows += [("True positives", f"{cm['tp']:,}"), ("False positives", f"{cm['fp']:,}"),
                 ("False negatives", f"{cm['fn']:,}"), ("True negatives", f"{cm['tn']:,}")]
    st.dataframe(pd.DataFrame(rows, columns=["Test-set metric", "Value"]), hide_index=True,
                 use_container_width=True)
