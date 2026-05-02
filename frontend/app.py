"""
Streamlit frontend for the NER system.
Run: streamlit run frontend/app.py
"""

import os
import json
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NER Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API URL ───────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Label colours ─────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "PER":  "#4f8ef7",   # blue
    "ORG":  "#f7a44f",   # orange
    "LOC":  "#4fbe7c",   # green
    "MISC": "#b44ff7",   # purple
    "O":    "#888888",
}

def label_color(label: str) -> str:
    return LABEL_COLORS.get(label, "#cccccc")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #4f8ef7, #4fbe7c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Token chips */
    .token-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        padding: 1.2rem;
        background: #0e1117;
        border-radius: 12px;
        border: 1px solid #2a2a2a;
        font-family: 'IBM Plex Mono', monospace;
    }
    .token {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 400;
        border: 1px solid transparent;
        transition: transform 0.1s;
    }
    .token:hover { transform: translateY(-2px); }
    .token-O {
        background: #1e1e1e;
        color: #bbb;
        border-color: #333;
    }

    /* Entity cards */
    .entity-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .entity-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        color: #fff;
    }
    .entity-text {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
    }

    /* Stat box */
    .stat-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 700; }
    .stat-label  { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_url_input = st.text_input("API URL", value=API_URL)
    if api_url_input:
        API_URL = api_url_input.rstrip("/")

    st.markdown("---")
    st.markdown("### 🏷️ Entity Types")
    for label, color in LABEL_COLORS.items():
        if label == "O":
            continue
        st.markdown(
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:4px;font-size:0.85rem;font-weight:600">{label}</span>'
            f"&nbsp; {'Person' if label=='PER' else 'Organisation' if label=='ORG' else 'Location' if label=='LOC' else 'Miscellaneous'}",
            unsafe_allow_html=True,
        )
        st.write("")

    st.markdown("---")
    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            if data.get("model_loaded"):
                st.success("✅ API connected · Model loaded")
            else:
                st.warning("⚠️ API connected · Model not loaded\nRun training first.")
        else:
            st.error("❌ API returned an error")
    except Exception:
        st.error("❌ Cannot reach API\nMake sure FastAPI is running.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>BiLSTM · CoNLL-2003 · PyTorch</small>",
        unsafe_allow_html=True,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 NER Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Named Entity Recognition powered by BiLSTM deep learning</div>', unsafe_allow_html=True)

# Example sentences
EXAMPLES = [
    "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.",
    "The United Nations headquarters is located in New York City.",
    "Elon Musk's Tesla and SpaceX are headquartered in the United States.",
    "Barack Obama served as the 44th President of the United States.",
    "Google acquired YouTube in 2006 for $1.65 billion.",
    "The FIFA World Cup 2022 was held in Qatar.",
]

col1, col2 = st.columns([3, 1])
with col1:
    user_text = st.text_area(
        "Enter text to analyse",
        height=120,
        placeholder="Type or paste any text here…",
    )
with col2:
    st.markdown("**Quick examples**")
    for i, ex in enumerate(EXAMPLES[:4]):
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["example_text"] = ex

if "example_text" in st.session_state:
    user_text = st.session_state.pop("example_text")
    st.rerun()

analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=False)

# ── Run prediction ────────────────────────────────────────────────────────────
if analyse_btn and user_text.strip():
    with st.spinner("Running NER…"):
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
            st.error("Cannot connect to the FastAPI backend. Is it running?")
            result = None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            result = None

    if result:
        entities  = result["entities"]
        tokens    = result["tokens"]
        st.markdown("---")

        # ── Stats row ──────────────────────────────────────────────────────
        counts = {}
        for ent in entities:
            counts[ent["label"]] = counts.get(ent["label"], 0) + 1

        stat_cols = st.columns(5)
        labels_display = [("Total", len(entities), "#4f8ef7"),
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

        st.markdown("#### Token View")

        # Build token HTML
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
                    f'style="background:{color}22;color:{color};border-color:{color}55">'
                    f'{tt["token"]}'
                    f'<sup style="font-size:0.6em;margin-left:3px;opacity:0.8">{lbl}</sup>'
                    f'</span>'
                )
        token_html += "</div>"
        st.markdown(token_html, unsafe_allow_html=True)

        # ── Entities list ──────────────────────────────────────────────────
        if entities:
            st.markdown("#### Extracted Entities")
            for ent in entities:
                color = label_color(ent["label"])
                st.markdown(
                    f'<div class="entity-card" style="background:{color}11;border-color:{color}">'
                    f'<span class="entity-label" style="background:{color}">{ent["label"]}</span>'
                    f'<span class="entity-text" style="color:{color}">{ent["text"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No named entities detected in this text.")

        # ── Raw JSON toggle ────────────────────────────────────────────────
        with st.expander("📄 Raw JSON response"):
            st.json(result)

elif analyse_btn:
    st.warning("Please enter some text first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#555'>Built with PyTorch · FastAPI · Streamlit | CoNLL-2003 dataset</small>",
    unsafe_allow_html=True,
)
