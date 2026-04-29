"""
AgroAlert AI — Streamlit application.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import io
import os
import sys
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `utils/` importable when run with `streamlit run app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.classifier import LEVEL_META, LEVELS, SeverityClassifier
from utils.ner import ENTITY_LABELS, NERExtractor
from utils.pdf_export import build_report_pdf
from utils.rules import generate_recommendations


# =====================================================================
# Page config + global CSS (premium dark, glassmorphism, gradient bg)
# =====================================================================

st.set_page_config(
    page_title="AgroAlert AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-0: #060812;
    --bg-1: #0b1024;
    --bg-2: #0f1730;
    --text-0: #f1f5f9;
    --text-1: #cbd5e1;
    --text-2: #94a3b8;
    --accent: #22d3ee;
    --accent-2: #a78bfa;
    --accent-3: #34d399;
    --card: rgba(255,255,255,0.04);
    --card-border: rgba(255,255,255,0.08);
    --shadow: 0 20px 60px -20px rgba(15, 23, 42, 0.8);
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-0) !important;
}

.stApp {
    background:
      radial-gradient(1200px 700px at 12% -10%, rgba(34,211,238,0.18), transparent 60%),
      radial-gradient(1000px 800px at 110% 10%, rgba(167,139,250,0.18), transparent 60%),
      radial-gradient(900px 600px at 50% 110%, rgba(52,211,153,0.12), transparent 60%),
      linear-gradient(180deg, var(--bg-0), var(--bg-1) 60%, var(--bg-2));
    background-attachment: fixed;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1280px;}

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, rgba(8,12,28,0.96), rgba(8,12,28,0.85)) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
}
section[data-testid="stSidebar"] * {color: var(--text-0) !important;}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Hero */
.hero {
    border-radius: 24px;
    padding: 28px 32px;
    background:
      linear-gradient(135deg, rgba(34,211,238,0.16), rgba(167,139,250,0.16) 60%, rgba(52,211,153,0.10));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
}
.hero h1 {
    margin: 0; font-size: 2.05rem; font-weight: 700;
    background: linear-gradient(90deg, #67e8f9 0%, #c4b5fd 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero p {color: var(--text-1); margin: 6px 0 0; font-size: 0.98rem;}
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 0.78rem; color: var(--text-1);
}
.chip .dot {width:7px; height:7px; border-radius:50%; background:var(--accent-3); box-shadow:0 0 10px var(--accent-3);}

/* Glass card */
.glass {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 18px 20px;
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow);
}

/* Metric cards */
.metric-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 8px;}
.metric-card {
    position: relative;
    border-radius: 18px;
    padding: 16px 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
    transition: transform .25s ease, border-color .25s ease;
}
.metric-card:hover {transform: translateY(-2px); border-color: rgba(255,255,255,0.18);}
.metric-card .label {font-size: .78rem; color: var(--text-2); text-transform: uppercase; letter-spacing: .08em;}
.metric-card .value {font-family:'Space Grotesk'; font-size: 1.85rem; font-weight: 700; margin-top: 4px;}
.metric-card .unit {font-size: .85rem; color: var(--text-2); margin-left: 4px;}
.metric-card .icon {position:absolute; top:14px; right:16px; font-size: 1.25rem; opacity:.85;}

.metric-card.tmax {background: linear-gradient(135deg, rgba(248,113,113,0.18), rgba(248,113,113,0.04));}
.metric-card.tmin {background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(96,165,250,0.04));}
.metric-card.rain {background: linear-gradient(135deg, rgba(34,211,238,0.18), rgba(34,211,238,0.04));}
.metric-card.hum  {background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(167,139,250,0.04));}
.metric-card.wind {background: linear-gradient(135deg, rgba(52,211,153,0.18), rgba(52,211,153,0.04));}

/* Alert badge */
.alert-badge {
    display:inline-flex; align-items:center; gap:10px;
    padding: 10px 18px; border-radius: 14px; font-weight:700; font-size: 1rem;
    letter-spacing: .04em;
}
.alert-badge .pulse {
    width: 10px; height: 10px; border-radius: 50%;
    box-shadow: 0 0 0 0 currentColor;
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,255,255,.6); }
    70%{ box-shadow: 0 0 0 12px rgba(255,255,255,0); }
    100%{ box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}

/* Highlighted entities in advisory text */
.advisory-render {
    line-height: 1.85; font-size: 1rem; color: var(--text-0);
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--card-border);
    border-radius: 14px; padding: 18px 22px;
    white-space: pre-wrap; word-wrap: break-word;
}
.ent {
    padding: 2px 8px; border-radius: 8px; font-weight: 600;
    border: 1px solid transparent; margin: 0 2px; display: inline-block;
    line-height: 1.4;
}
.ent .tag {
    font-size: .65rem; margin-left: 6px; padding: 1px 6px; border-radius: 6px;
    background: rgba(0,0,0,0.30); color: rgba(255,255,255,0.85); font-weight: 700;
    letter-spacing: .04em;
}
.ent.TEMP_MAX {background: rgba(248,113,113,0.22); color:#fecaca; border-color: rgba(248,113,113,0.4);}
.ent.TEMP_MIN {background: rgba(96,165,250,0.22); color:#bfdbfe; border-color: rgba(96,165,250,0.4);}
.ent.RAINFALL_LEVEL {background: rgba(34,211,238,0.22); color:#a5f3fc; border-color: rgba(34,211,238,0.4);}
.ent.HUMIDITY {background: rgba(167,139,250,0.22); color:#ddd6fe; border-color: rgba(167,139,250,0.4);}
.ent.WIND_SPEED {background: rgba(52,211,153,0.22); color:#a7f3d0; border-color: rgba(52,211,153,0.4);}

/* Recommendation cards */
.rec-card {
    display:flex; gap: 14px; align-items:flex-start;
    padding: 14px 16px; border-radius: 14px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px; transition: border-color .2s ease;
}
.rec-card:hover {border-color: rgba(255,255,255,0.18);}
.rec-card .check {
    flex-shrink: 0; width: 22px; height: 22px; border-radius: 6px;
    border: 2px solid var(--accent-3);
    display:flex; align-items:center; justify-content:center;
    color: var(--accent-3); font-size: .85rem; margin-top: 2px;
}
.rec-card .title {font-weight: 600; color: var(--text-0); margin: 0; font-size: 1rem;}
.rec-card .meta {color: var(--text-2); font-size: .78rem; margin-top: 2px;}
.rec-card .detail {color: var(--text-1); font-size: .92rem; margin-top: 6px; line-height: 1.5;}
.priority-badge {
    display:inline-block; padding: 2px 9px; font-size:.68rem; font-weight: 700;
    letter-spacing:.06em; border-radius: 999px; margin-right:6px;
}
.priority-CRITICAL {background: rgba(239,68,68,0.18); color:#fecaca;}
.priority-HIGH     {background: rgba(249,115,22,0.18); color:#fed7aa;}
.priority-MEDIUM   {background: rgba(234,179,8,0.18); color:#fde68a;}
.priority-LOW      {background: rgba(52,211,153,0.18); color:#a7f3d0;}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {gap: 6px; background: transparent; border-bottom: 1px solid rgba(255,255,255,0.08);}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03); padding: 10px 18px;
    border-radius: 12px 12px 0 0; color: var(--text-2);
    border: 1px solid rgba(255,255,255,0.06); border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(34,211,238,0.10) !important;
    color: var(--text-0) !important; border-color: rgba(34,211,238,0.3) !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #06b6d4, #8b5cf6) !important;
    color: white !important; border: 0 !important;
    border-radius: 12px !important; padding: 10px 18px !important;
    font-weight: 600 !important; transition: transform .15s ease, box-shadow .2s ease;
    box-shadow: 0 10px 24px -10px rgba(139,92,246,0.55);
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px -10px rgba(139,92,246,0.7);
}
.stButton>button:focus, .stDownloadButton>button:focus {box-shadow: 0 0 0 3px rgba(34,211,238,0.4) !important;}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-0) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    color: var(--text-0) !important;
}

/* Translation card */
.lang-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.10), rgba(34,211,238,0.06));
    border: 1px solid rgba(167,139,250,0.30);
    border-radius: 16px; padding: 18px 20px;
    font-size: 1.05rem; line-height: 1.75;
}

/* Section title */
.section-title {
    display:flex; align-items:center; gap:10px;
    font-family:'Space Grotesk'; font-size: 1.15rem; font-weight: 600;
    color: var(--text-0); margin: 4px 0 12px;
}
.section-title .bar {width: 4px; height: 18px; border-radius: 4px; background: linear-gradient(180deg, var(--accent), var(--accent-2));}

/* Sidebar nav */
.nav-item {
    display:flex; align-items:center; gap:10px; padding: 10px 12px;
    border-radius: 10px; color: var(--text-1); font-weight: 500; font-size:.95rem;
}
.brand {
    display:flex; align-items:center; gap:10px; padding: 6px 4px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px;
}
.brand .logo {
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, #22d3ee, #a78bfa);
    display:flex; align-items:center; justify-content:center;
    color: white; font-weight:800; font-family:'Space Grotesk';
    box-shadow: 0 8px 20px -6px rgba(34,211,238,0.6);
}
.brand .name {font-family:'Space Grotesk'; font-weight:700; font-size: 1.05rem; color: var(--text-0);}
.brand .sub {font-size:.7rem; color: var(--text-2); letter-spacing:.1em; text-transform:uppercase;}

.kw-pill {
    display:inline-block; padding: 4px 10px; border-radius:999px;
    background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10);
    color: var(--text-1); font-size:.78rem; margin: 3px 4px 3px 0;
}

.muted {color: var(--text-2); font-size: .85rem;}
hr {border-color: rgba(255,255,255,0.06) !important;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# Cached resource loaders
# =====================================================================


@st.cache_resource(show_spinner="Loading NER model…")
def load_ner() -> NERExtractor:
    return NERExtractor()


@st.cache_resource(show_spinner="Loading severity classifier…")
def load_classifier() -> SeverityClassifier:
    return SeverityClassifier()



def extract_pdf_text(file_bytes: bytes):
    """Extract text from a PDF file. Returns None on failure."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(p.strip() for p in pages if p and p.strip())
        return text or None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_sample_advisory() -> str:
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sample_data",
        "sample_advisory.txt",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# =====================================================================
# Sidebar
# =====================================================================

NAV_ITEMS = [
    ("Dashboard", "🛰️"),
    ("Weather Extraction", "🔬"),
    ("Alert & Recommendations", "⚠️"),
]


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="logo">A</div>
                <div>
                    <div class="name">AgroAlert AI</div>
                    <div class="sub">IMD Advisory Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            options=[name for name, _ in NAV_ITEMS],
            format_func=lambda n: f"{dict(NAV_ITEMS)[n]}  {n}",
            label_visibility="collapsed",
        )


    return page


# =====================================================================
# Input section (shared across pages via session_state)
# =====================================================================


def input_panel():
    st.markdown("<div class='section-title'><span class='bar'></span>Advisory Input</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("📋 Load sample IMD advisory", use_container_width=True):
            st.session_state["advisory_text"] = load_sample_advisory()
    with c2:
        uploaded = st.file_uploader(
            "Upload .txt or .pdf",
            type=["txt", "pdf"],
            label_visibility="collapsed",
            help="Upload a plain text (.txt) or PDF (.pdf) advisory.",
        )
        if uploaded is not None:
            file_bytes = uploaded.read()
            name = (uploaded.name or "").lower()
            text = None
            if name.endswith(".pdf") or uploaded.type == "application/pdf":
                with st.spinner("Extracting text from PDF..."):
                    text = extract_pdf_text(file_bytes)
                if not text:
                    st.error(
                        "Could not extract text from this PDF. "
                        "It may be a scanned image — try a text-based PDF or paste the advisory manually."
                    )
            else:
                try:
                    text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = file_bytes.decode("latin-1")
                    except Exception:
                        st.error("Could not read file as text.")
            if text:
                st.session_state["advisory_text"] = text
                st.success(f"Loaded {uploaded.name} ({len(text):,} chars).")
    with c3:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["advisory_text"] = ""

    text = st.text_area(
        "Paste IMD agromet advisory text here",
        value=st.session_state.get("advisory_text", ""),
        height=180,
        placeholder=(
            "Example: Maximum temperature is likely to range between 36 to 38 "
            "degrees Celsius… Heavy rainfall of about 75 mm is expected…"
        ),
        label_visibility="collapsed",
    )
    st.session_state["advisory_text"] = text


# =====================================================================
# Analysis runner (cached on text)
# =====================================================================


def run_analysis(text: str):
    if not text or not text.strip():
        return None

    ner = load_ner()
    clf = load_classifier()

    entities = ner.extract(text)
    summary = ner.to_summary(entities)
    cls = clf.predict(text, entity_summary=summary)
    recs = generate_recommendations(summary, alert_level=cls.label)

    return {
        "entities": entities,
        "summary": summary,
        "classification": cls,
        "recommendations": recs,
    }


# =====================================================================
# Renderers
# =====================================================================


def render_metric_card(css_class: str, icon: str, label: str, value, unit: str = ""):
    if value is None or value == "":
        value_html = "<span class='muted'>—</span>"
    else:
        value_html = f"{value}<span class='unit'>{unit}</span>"
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="icon">{icon}</div>
            <div class="label">{label}</div>
            <div class="value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_badge(level: str):
    meta = LEVEL_META.get(level, LEVEL_META["GREEN"])
    st.markdown(
        f"""
        <div class="alert-badge" style="background:{meta['bg']}; color:{meta['color']};
                                        border:1px solid {meta['color']};">
            <span class="pulse" style="background:{meta['color']};"></span>
            <span>{level} · {meta['label'].upper()}</span>
        </div>
        <div class='muted' style='margin-top:8px;'>{meta['description']}</div>
        """,
        unsafe_allow_html=True,
    )


def render_highlighted_text(text: str, entities) -> str:
    if not entities:
        return f"<div class='advisory-render'>{html.escape(text)}</div>"

    spans = sorted([e for e in entities], key=lambda e: e.start)
    out: List[str] = []
    cursor = 0
    for e in spans:
        if e.start < cursor:
            continue  # skip overlapping
        out.append(html.escape(text[cursor:e.start]))
        out.append(
            f"<span class='ent {e.label}' title='{e.label} · {e.confidence:.2f}'>"
            f"{html.escape(text[e.start:e.end])}"
            f"<span class='tag'>{e.label}</span>"
            f"</span>"
        )
        cursor = e.end
    out.append(html.escape(text[cursor:]))
    return f"<div class='advisory-render'>{''.join(out)}</div>"


def render_probability_chart(probs: dict):
    levels = ["GREEN", "YELLOW", "ORANGE", "RED"]
    colors = [LEVEL_META[l]["color"] for l in levels]
    values = [round(probs.get(l, 0) * 100, 2) for l in levels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=levels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, max(values + [10]) * 1.2],
            gridcolor="rgba(255,255,255,0.08)", color="#94a3b8", title=""
        ),
        yaxis=dict(color="#cbd5e1", title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_recommendation(rec: dict, idx: int):
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="check">✓</div>
            <div style="flex:1;">
                <div class="title">{idx}. {html.escape(rec['title'])}</div>
                <div class="meta">
                    <span class="priority-badge priority-{rec['priority']}">{rec['priority']}</span>
                    <span>{html.escape(rec['category'])} · triggered by {", ".join(rec['triggered_by']) or "general advisory"}</span>
                </div>
                <div class="detail">{html.escape(rec['detail'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def copy_to_clipboard_button(text: str, key: str, label: str = "📋 Copy to clipboard"):
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    component = f"""
    <div style="display:flex; gap:8px; align-items:center;">
      <button id="copy-{key}"
        style="background: linear-gradient(135deg, #06b6d4, #8b5cf6); color:#fff;
               border:0; border-radius:12px; padding:10px 18px; font-weight:600;
               cursor:pointer; font-family:'Inter',sans-serif;">
        {label}
      </button>
      <span id="copy-status-{key}" style="color:#94a3b8; font-size:.85rem;"></span>
    </div>
    <script>
      const btn = document.getElementById("copy-{key}");
      const status = document.getElementById("copy-status-{key}");
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(`{safe}`);
          status.textContent = "Copied!";
          setTimeout(()=>status.textContent="", 1800);
        }} catch (e) {{
          status.textContent = "Copy failed";
        }}
      }});
    </script>
    """
    st.components.v1.html(component, height=60)


# =====================================================================
# Pages
# =====================================================================


def hero_block():
    st.markdown(
        """
        <div class="hero">
            <h1>AgroAlert AI</h1>
            <p>Turn raw IMD agromet advisories into structured weather signals,
               severity alerts and crop-action plans — in one fluent workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard(result):
    hero_block()
    input_panel()

    if result is None:
        st.info("📥  Paste an advisory above (or load the sample) to see the dashboard come alive.")
        return

    summary = result["summary"]
    cls = result["classification"]

    st.markdown("<div class='section-title'><span class='bar'></span>Live Weather Signals</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    spec = [
        ("tmax", "🔥", "Max Temp", summary.get("TEMP_MAX")),
        ("tmin", "❄️", "Min Temp", summary.get("TEMP_MIN")),
        ("rain", "🌧️", "Rainfall", summary.get("RAINFALL_LEVEL")),
        ("hum",  "💧", "Humidity", summary.get("HUMIDITY")),
        ("wind", "💨", "Wind Speed", summary.get("WIND_SPEED")),
    ]
    for col, (cls_n, icon, label, info) in zip(cols, spec):
        with col:
            v = info["value"] if info else None
            u = info["unit"] if info else ""
            render_metric_card(cls_n, icon, label, v, u or "")

    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'><span class='bar'></span>Alert Status</div>", unsafe_allow_html=True)
        render_alert_badge(cls.label)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'><span class='bar'></span>Severity Distribution</div>", unsafe_allow_html=True)
        render_probability_chart(cls.probabilities)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><span class='bar'></span>Top Recommendations</div>", unsafe_allow_html=True)
    for i, r in enumerate(result["recommendations"][:3], 1):
        render_recommendation(r.to_dict(), i)
    st.markdown("</div>", unsafe_allow_html=True)


def page_extraction(result):
    hero_block()
    input_panel()

    if result is None:
        st.info("📥  Add advisory text to extract weather entities.")
        return

    text = st.session_state["advisory_text"]
    entities = result["entities"]

    st.markdown("<div class='section-title'><span class='bar'></span>NER-highlighted Advisory</div>", unsafe_allow_html=True)
    st.markdown(render_highlighted_text(text, entities), unsafe_allow_html=True)

    legend_html = " ".join(
        f"<span class='ent {l}' style='font-size:.78rem;'>{l}</span>"
        for l in ENTITY_LABELS
    )
    st.markdown(f"<div style='margin-top:10px;'>{legend_html}</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><span class='bar'></span>Extracted Entity Table</div>", unsafe_allow_html=True)

    if not entities:
        st.warning("No entities detected.")
    else:
        rows = []
        for e in entities:
            rows.append({
                "Label": e.label,
                "Text": e.text,
                "Value": e.value,
                "Unit": e.unit,
                "Confidence": f"{e.confidence * 100:.1f}%",
                "Span": f"[{e.start}:{e.end}]",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def page_alert(result):
    hero_block()
    input_panel()

    if result is None:
        st.info("📥  Add advisory text to see severity alert and recommendations.")
        return

    cls = result["classification"]

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'><span class='bar'></span>Predicted Alert</div>", unsafe_allow_html=True)
        render_alert_badge(cls.label)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'><span class='bar'></span>Class Probabilities</div>", unsafe_allow_html=True)
        render_probability_chart(cls.probabilities)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><span class='bar'></span>Recommended Actions</div>", unsafe_allow_html=True)
    if not result["recommendations"]:
        st.success("No specific actions required. Conditions look favourable.")
    for i, r in enumerate(result["recommendations"], 1):
        render_recommendation(r.to_dict(), i)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><span class='bar'></span>Why this alert? (Explainability)</div>",
        unsafe_allow_html=True,
    )

    triggers: List[str] = []
    for r in result["recommendations"]:
        triggers.extend(r.triggered_by)
    triggers = list(dict.fromkeys(triggers))[:8]

    if triggers:
        st.markdown(
            "<div class='muted'>Triggers detected from the advisory:</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            " ".join(f"<span class='kw-pill'>📌 {html.escape(t)}</span>" for t in triggers),
            unsafe_allow_html=True,
        )

    if cls.top_keywords:
        st.markdown(
            "<div class='muted' style='margin-top:14px;'>Top severity keywords found in text:</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            " ".join(
                f"<span class='kw-pill'>🔑 {html.escape(k)}</span>"
                for k in cls.top_keywords
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='muted' style='margin-top:14px;'>"
        "The classifier weighs both these textual cues and the numeric weather "
        "values extracted by the NER model when assigning a severity level."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><span class='bar'></span>Export</div>",
        unsafe_allow_html=True,
    )
    pdf_bytes = build_report_pdf(
        advisory_text=st.session_state.get("advisory_text", ""),
        entity_summary=result["summary"],
        alert_level=result["classification"].label,
        alert_probabilities=result["classification"].probabilities,
        recommendations=[r.to_dict() for r in result["recommendations"]],
    )
    st.download_button(
        label="📥 Download report as PDF",
        data=pdf_bytes,
        file_name="agroalert_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# =====================================================================
# Main
# =====================================================================


def main():
    if "advisory_text" not in st.session_state:
        st.session_state["advisory_text"] = ""

    page = sidebar()

    text = st.session_state.get("advisory_text", "")
    result = run_analysis(text)

    if page == "Dashboard":
        page_dashboard(result)
    elif page == "Weather Extraction":
        page_extraction(result)
    elif page == "Alert & Recommendations":
        page_alert(result)


if __name__ == "__main__":
    main()
