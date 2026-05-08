"""
ui.styles – CSS injection for the Streamlit app.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F1A 0%, #131726 100%);
    border-right: 1px solid #1E2130;
}
[data-testid="stSidebar"] .stRadio label {
    font-weight: 500;
    color: #A0AEC0;
    transition: color .2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #6C63FF;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1D2E 0%, #1E2130 100%);
    border: 1px solid #2D3150;
    border-radius: 12px;
    padding: 18px 20px !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700;
    color: #6C63FF !important;
}
[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #5A52E0 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all .25s ease;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(108,99,255,0.5);
}
.stButton > button:active {
    transform: translateY(0);
}

.stAlert {
    border-radius: 10px;
}

[data-testid="stDataFrameResizable"] {
    border: 1px solid #2D3150;
    border-radius: 10px;
}

h1 { color: #EAEAF5 !important; font-weight: 700; }
h2 { color: #C5C8E8 !important; font-weight: 600; }
h3 { color: #A0AEC0 !important; font-weight: 500; }

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg,#6C63FF22,#FF658422);
    border: 1px solid #6C63FF44;
    color: #6C63FF;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.feature-card {
    background: linear-gradient(135deg, #1A1D2E 0%, #1E2130 100%);
    border: 1px solid #2D3150;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
    transition: border-color .25s, box-shadow .25s;
}
.feature-card:hover {
    border-color: #6C63FF;
    box-shadow: 0 0 20px rgba(108,99,255,0.15);
}
.feature-card h4 {
    margin: 0 0 8px;
    color: #EAEAF5;
    font-size: 1rem;
}
.feature-card p {
    margin: 0;
    color: #A0AEC0;
    font-size: 0.88rem;
    line-height: 1.5;
}
</style>
"""


def inject_styles() -> None:
    """Inject the custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
