"""
Streamlit App - Placement Package Predictor
Deploy this on Streamlit Cloud by pointing to this file.
Requires: streamlit, joblib, numpy, scikit-learn
"""

import joblib
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Placement Package Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* hide streamlit default header / footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* card wrapper */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 40px 36px;
        margin: 40px auto;
        max-width: 520px;
        box-shadow: 0 24px 64px rgba(0,0,0,0.4);
    }

    /* badge */
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #7f53ac, #647dee);
        border-radius: 50px;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 4px 14px;
        margin-bottom: 16px;
        text-transform: uppercase;
        color: #fff;
    }

    h1.hero {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 8px;
        line-height: 1.25;
    }

    p.subtitle {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.55);
        margin-bottom: 32px;
    }

    /* result box */
    .result-box {
        margin-top: 24px;
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(127,83,172,0.25), rgba(100,125,238,0.25));
        border: 1px solid rgba(127,83,172,0.45);
        text-align: center;
    }

    .result-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 4px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #c471ed, #a2d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* slider label */
    .stSlider label, .stNumberInput label {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    /* Streamlit button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7f53ac, #647dee) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 0 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover { opacity: 0.88 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("placement_lr.pkl")

model = load_model()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="glass-card">
        <span class="badge">🎓 ML Model</span>
        <h1 class="hero">Placement Package Predictor</h1>
        <p class="subtitle">
            Enter your CGPA to estimate your placement package
            using a trained Simple Linear Regression model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input section
cgpa = st.slider(
    "Select your CGPA",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.01,
    format="%.2f",
)

st.markdown(f"**Selected CGPA:** `{cgpa}`")

predict_btn = st.button("Predict Package 🚀")

if predict_btn:
    prediction = model.predict(np.array([[cgpa]]))[0]
    prediction_rounded = round(float(prediction), 2)

    st.markdown(
        f"""
        <div class="result-box">
            <p class="result-label">Estimated Package</p>
            <div class="result-value">₹ {prediction_rounded} Lacs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer info
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:0.78rem; color:rgba(255,255,255,0.35);'>"
    "Model: Simple Linear Regression &nbsp;|&nbsp; "
    "Dataset: 200 student placement records &nbsp;|&nbsp; "
    "R² ≈ 0.78"
    "</p>",
    unsafe_allow_html=True,
)
