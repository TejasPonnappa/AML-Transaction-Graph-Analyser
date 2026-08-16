"""
dashboard_components.py — reusable, genuinely interactive dashboard pieces.
Import these into amlsim_pages.py / bitcoin_pages.py to replace flat
st.bar_chart / st.metric calls with something that actually looks built,
not just default Streamlit with CSS on top.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def score_distribution_chart(df_scores, score_col='SUSPICION_SCORE', title="Suspicion Score Distribution"):
    """Replaces st.bar_chart with a styled, interactive Plotly histogram —
    hoverable, zoomable, with a gradient color scale tied to risk level."""
    fig = px.histogram(
        df_scores, x=score_col, nbins=40,
        color_discrete_sequence=['#6366f1']
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.15,
        margin=dict(t=50, b=30, l=30, r=30),
        font=dict(family="sans serif", size=13),
        showlegend=False
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#f59e0b",
                   annotation_text="Risk threshold", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)


def risk_gauge(score, label="Account Risk Score"):
    """A gauge chart — far more visually engaging than a plain st.metric
    for a single risk score, used in the deep-dive page."""
    color = "#dc2626" if score >= 0.8 else "#f59e0b" if score >= 0.5 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': '%', 'font': {'size': 36}},
        title={'text': label, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 50], 'color': '#10b98122'},
                {'range': [50, 80], 'color': '#f59e0b22'},
                {'range': [80, 100], 'color': '#dc262622'},
            ],
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e6e6e6'}
    )
    st.plotly_chart(fig, use_container_width=True)


def animated_metric_row(metrics: list):
    """
    metrics: list of dicts like {'label': 'Total Tx', 'value': '1,234', 'delta': '+12%'}
    Renders styled cards with a fade-in CSS animation instead of default st.metric flatness.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        delta_html = f'<div style="font-size:0.8rem; color:#10b981; margin-top:4px;">{m.get("delta", "")}</div>' if m.get('delta') else ''
        col.markdown(f"""
            <div class="metric-card-animated">
                <div style="font-size:0.78rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em;">{m['label']}</div>
                <div style="font-size:1.9rem; font-weight:800; margin-top:6px;">{m['value']}</div>
                {delta_html}
            </div>
        """, unsafe_allow_html=True)


def top_risk_ticker(df_scores, id_col='SENDER_ACCOUNT_ID', score_col='SUSPICION_SCORE', n=8):
    """A horizontal-scroll 'ticker' of the top-N riskiest accounts — gives the
    dashboard a live, active feel instead of static tables everywhere."""
    top = df_scores.nlargest(n, score_col)
    items_html = "".join([
        f'<div class="ticker-item"><span style="color:#dc2626; font-weight:700;">{row[score_col]:.3f}</span> '
        f'&nbsp;{row[id_col]}</div>'
        for _, row in top.iterrows()
    ])
    st.markdown(f"""
        <div class="ticker-wrap"><div class="ticker-track">{items_html}{items_html}</div></div>
    """, unsafe_allow_html=True)


def inject_component_css():
    """Call once alongside apply_custom_style()."""
    st.markdown("""
        <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .metric-card-animated {
            background: linear-gradient(145deg, #1a1f2b, #161a24);
            border: 1px solid #2d3344;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            animation: fadeInUp 0.4s ease-out;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card-animated:hover {
            transform: translateY(-2px);
            border-color: #6366f1;
        }

        .ticker-wrap {
            overflow: hidden;
            white-space: nowrap;
            border: 1px solid #2d3344;
            border-radius: 8px;
            background: #12161f;
            padding: 10px 0;
        }
        .ticker-track {
            display: inline-block;
            animation: ticker-scroll 25s linear infinite;
        }
        .ticker-item {
            display: inline-block;
            padding: 0 2rem;
            font-size: 0.9rem;
            font-family: monospace;
        }
        @keyframes ticker-scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
        </style>
    """, unsafe_allow_html=True)