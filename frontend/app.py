"""
Streamlit frontend for the NER system.
Run: streamlit run frontend/app.py
"""

import os
import json
import requests
import streamlit as st
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NER Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── API URL ───────────────────────────────────────────────────────────────────
# Try environment variable first, then Streamlit secrets, then default
if "api_url_override" not in st.session_state:
    st.session_state.api_url_override = None

API_URL = st.session_state.api_url_override or os.getenv("API_URL") or st.secrets.get("api_url", "https://named-entity-recognition-system.onrender.com")

# Check API health silently
_api_available = False
try:
    _health = requests.get(f"{API_URL}/health", timeout=2)
    _api_available = _health.status_code == 200 and _health.json().get("model_loaded", False)
except:
    _api_available = False

# ── Label colours (modern gradient palette) ──────────────────────────────────
LABEL_COLORS = {
    "PER":  "#3b82f6",   # blue
    "ORG":  "#f59e0b",   # amber
    "LOC":  "#10b981",   # emerald
    "MISC": "#8b5cf6",   # violet
    "O":    "#6b7280",   # gray
}

def label_color(label: str) -> str:
    return LABEL_COLORS.get(label, "#cccccc")

# ── Custom CSS (modern design) ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main title with gradient */
    .main-title {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.6s ease-in;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }

    /* Input area styling */
    .stTextArea > div > div > textarea {
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* Token container */
    .token-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 20px;
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        min-height: 80px;
    }

    .token {
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .token:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .token-O {
        background: #f3f4f6;
        color: #6b7280;
        border-color: #d1d5db;
    }

    /* Entity card */
    .entity-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid;
        background: #f9fafb;
        transition: all 0.2s ease;
    }

    .entity-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    .entity-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 6px;
        color: #fff;
        white-space: nowrap;
    }

    .entity-text {
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #1f2937;
    }

    /* Stat box */
    .stat-box {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border: 1.5px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
        border-color: #d1d5db;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3) !important;
    }

    /* Example buttons */
    .example-button {
        background: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        color: #1f2937 !important;
        transition: all 0.2s ease !important;
    }

    .example-button:hover {
        background: #e5e7eb !important;
        border-color: #3b82f6 !important;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #e5e7eb;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f3f4f6;
        border-radius: 8px;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.4s ease-in;
    }

    hr {
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        border: none;
        height: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔖 Entity Types")
    cols = st.columns(2)
    for idx, (label, color) in enumerate([("PER", LABEL_COLORS["PER"]), 
                                          ("ORG", LABEL_COLORS["ORG"]),
                                          ("LOC", LABEL_COLORS["LOC"]),
                                          ("MISC", LABEL_COLORS["MISC"])]):
        with cols[idx % 2]:
            st.markdown(
                f'<div style="background:{color};color:#fff;padding:10px;'
                f'border-radius:8px;text-align:center;font-weight:600;margin:4px 0">'
                f'{label} — {"Person" if label=="PER" else "Organization" if label=="ORG" else "Location" if label=="LOC" else "Miscellaneous"}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    col1, col2 = st.columns(2)
    with col1:
        if _api_available:
            st.markdown('<div style="color:#10b981;font-weight:600">✓ API Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#ef4444;font-weight:600">✗ API Offline</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="color:#9ca3af;font-size:0.9rem">v1.0.0</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        **BiLSTM NER** — State-of-the-art Named Entity Recognition

        - **Dataset**: CoNLL-2003
        - **Model**: Bidirectional LSTM
        - **Framework**: PyTorch + FastAPI
        - **UI**: Streamlit
        """,
        unsafe_allow_html=False,
    )

    st.markdown("---")
    if st.checkbox("🔧 Debug Mode"):
        with st.expander("API Configuration"):
            st.info(f"**Current API URL**: `{API_URL}`")
            
            custom_url = st.text_input("Override API URL (leave blank to use default):", value="", key="api_override_input")
            if custom_url and custom_url.strip():
                st.session_state.api_url_override = custom_url.rstrip("/")
                st.rerun()
            elif st.session_state.api_url_override:
                if st.button("Clear Override"):
                    st.session_state.api_url_override = None
                    st.rerun()
            
            st.markdown("**Test Connection:**")
            if st.button("🔌 Test API", key="test_api_btn"):
                try:
                    resp = requests.get(f"{API_URL}/health", timeout=3)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.json(data)
                        if data.get("model_loaded"):
                            st.success("✅ API is working and model is loaded!")
                        else:
                            st.warning("⚠️ API works but model not loaded. Run training: `python -m backend.train`")
                    else:
                        st.error(f"API returned status {resp.status_code}")
                except requests.exceptions.Timeout:
                    st.error(f"Timeout: API not responding at {API_URL}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Connection refused. Check if API is running at {API_URL}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

# ── Main ──────────────────────────────────────────────────────────────────────
# Welcome banner
st.markdown('<div class="main-title">🔍 NER Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identify people, organizations, and locations in any text</div>', unsafe_allow_html=True)

# ── Sample Test Texts ─────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Tech Companies": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California. Today, Tim Cook serves as CEO and the company is headquartered in Cupertino.",
    "News Article": "Elon Musk founded Tesla in 2003 and SpaceX in 2002. Both companies are headquartered in California. Musk is also the founder of The Boring Company.",
    "Geography": "The United Nations headquarters is located in New York City. Paris is the capital of France. Mount Everest is in the Himalayas between Nepal and Tibet.",
    "Politics": "Barack Obama served as the 44th President of the United States from 2009 to 2017. He was born in Honolulu, Hawaii. Michelle Obama was the First Lady.",
    "Business": "Microsoft acquired LinkedIn for $26.2 billion in 2016. Bill Gates founded Microsoft in 1975 in Albuquerque, New Mexico. Satya Nadella is the current CEO.",
    "Entertainment": "Tom Hanks starred in Forrest Gump directed by Robert Zemeckis. The movie was filmed in various locations including Savannah, Georgia.",
}

if not _api_available:
    with st.warning("⚠️ Backend Connection Issue"):
        st.markdown("""
        The NER backend is not currently available. This could mean:
        
        1. **Local Testing**: Make sure FastAPI is running with: `uvicorn backend.api:app --reload`
        2. **Production**: Check if your Render/Railway deployment is live
        3. **API URL**: Use the 🔧 **Debug Mode** (bottom of sidebar) to verify/override the API URL
        
        📖 [See Deployment Guide](https://github.com/your-repo/blob/main/DEPLOY_BACKEND.md)
        """)

# Input section
st.markdown("### ✍️ Input Text")
col_input, col_examples = st.columns([3, 1])

with col_input:
    user_text = st.text_area(
        "Enter text to analyse",
        height=140,
        placeholder="Paste your text here and click Analyse...",
        label_visibility="collapsed"
    )

with col_examples:
    st.markdown("#### 📌 Samples")
    for label, text in list(SAMPLE_TEXTS.items())[:3]:
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state["example_text"] = text

# Show more samples in expander
with st.expander("➕ More Samples"):
    for label, text in list(SAMPLE_TEXTS.items())[3:]:
        if st.button(f"📌 {label}", key=f"more_{label}", use_container_width=True):
            st.session_state["example_text"] = text

if "example_text" in st.session_state:
    user_text = st.session_state.pop("example_text")
    st.rerun()

# Analyze button
st.markdown("###")
col_btn, col_space = st.columns([1, 4])
with col_btn:
    analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True, key="analyze_btn")

# ── Run prediction ────────────────────────────────────────────────────────────
if analyse_btn and user_text.strip():
    if not _api_available:
        st.error("⚠️ Cannot reach the API backend. Please ensure it's running.")
    else:
        with st.spinner("🔄 Analyzing text..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text},
                    timeout=15,
                )
                if resp.status_code == 200:
                    result = resp.json()
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    result = None
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the backend. Make sure it's deployed and running.")
                result = None
            except Exception as e:
                st.error(f"❌ Error: {e}")
                result = None

        if result:
            st.markdown("---")
            entities  = result["entities"]
            tokens    = result["tokens"]

            # ── Stats row ──────────────────────────────────────────────────────
            counts = {}
            for ent in entities:
                counts[ent["label"]] = counts.get(ent["label"], 0) + 1

            stat_cols = st.columns(5)
            labels_display = [("Total", len(entities), "#3b82f6"),
                              ("PER",  counts.get("PER",  0), LABEL_COLORS["PER"]),
                              ("ORG",  counts.get("ORG",  0), LABEL_COLORS["ORG"]),
                              ("LOC",  counts.get("LOC",  0), LABEL_COLORS["LOC"]),
                              ("MISC", counts.get("MISC", 0), LABEL_COLORS["MISC"])]

            for col, (name, num, color) in zip(stat_cols, labels_display):
                with col:
                    st.markdown(
                        f'<div class="stat-box">'
                        f'<div class="stat-number" style="color:{color}">{num}</div>'
                        f'<div class="stat-label">{name}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Token View ─────────────────────────────────────────────────────
            st.markdown("### 🏷️ Token Analysis")
            
            token_html = '<div class="token-container">'
            for tt in tokens:
                tag = tt["tag"]
                if tag == "O":
                    token_html += (
                        f'<span class="token token-O" title="O">{tt["token"]}</span>'
                    )
                else:
                    lbl   = tag.split("-")[-1] if "-" in tag else tag
                    color = label_color(lbl)
                    token_html += (
                        f'<span class="token" title="{tag}" '
                        f'style="background:{color}22;color:{color};border-color:{color}55;border:1.5px solid {color}55">'
                        f'{tt["token"]}'
                        f'<sup style="font-size:0.65em;margin-left:3px;opacity:0.9;font-weight:600">{lbl}</sup>'
                        f'</span>'
                    )
            token_html += "</div>"
            st.markdown(token_html, unsafe_allow_html=True)

            # ── Entities list ──────────────────────────────────────────────────
            if entities:
                st.markdown("### 📍 Extracted Entities")
                for ent in entities:
                    color = label_color(ent["label"])
                    st.markdown(
                        f'<div class="entity-card" style="background:{color}08;border-color:{color}">'
                        f'<span class="entity-label" style="background:{color}">{ent["label"]}</span>'
                        f'<span class="entity-text" style="color:{color}">{ent["text"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No named entities detected in this text.", icon="ℹ️")

            # ── Raw JSON toggle ────────────────────────────────────────────────
            with st.expander("📄 View Raw Response"):
                st.json(result)

elif analyse_btn:
    st.warning("Please enter some text first.", icon="⚠️")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.9rem'>"
    "Built with <strong>PyTorch</strong> · <strong>FastAPI</strong> · <strong>Streamlit</strong> | "
    "CoNLL-2003 Dataset"
    "</div>",
    unsafe_allow_html=True,
)
