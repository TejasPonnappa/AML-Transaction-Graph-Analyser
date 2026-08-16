"""
styling.py — custom CSS for the AML dashboard. Import and call apply_custom_style()
once at the top of streamlit_app.py, right after st.set_page_config().
"""
import streamlit as st


def apply_custom_style():
    st.markdown("""
        <style>
        /* Tighter, less default-Streamlit spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Metric cards: subtle border + background instead of flat default */
        div[data-testid="stMetric"] {
            background-color: #1a1f2b;
            border: 1px solid #2d3344;
            border-radius: 10px;
            padding: 1rem 1rem 0.5rem 1rem;
        }
        div[data-testid="stMetric"] label {
            color: #9ca3af !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }

        /* Sidebar accent */
        section[data-testid="stSidebar"] {
            background-color: #12161f;
            border-right: 1px solid #2d3344;
        }

        /* Headings */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        h2, h3 {
            font-weight: 600 !important;
            color: #e6e6e6 !important;
        }

        /* Dataframes: rounded corners, less boxy */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #2d3344;
        }

        /* Primary buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #dc2626;
        }

        /* Reduce default Streamlit top padding/branding clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


def risk_badge(score: float) -> str:
    """Returns an HTML-styled risk badge for a suspicion score, for use in st.markdown()."""
    if score >= 0.8:
        color, label = "#dc2626", "HIGH RISK"
    elif score >= 0.5:
        color, label = "#f59e0b", "MEDIUM RISK"
    else:
        color, label = "#10b981", "LOW RISK"
    return (
        f'<span style="background-color:{color}22; color:{color}; '
        f'border:1px solid {color}; padding:2px 10px; border-radius:12px; '
        f'font-size:0.8rem; font-weight:600;">{label} ({score:.3f})</span>'
    )