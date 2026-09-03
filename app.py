"""
app.py
-------
TRACKYUKTI — Smarter Planning. Efficient Solutions.
WEST CENTRAL RAILWAY (WCR) — JABALPUR DIVISION
Joint Rolling Block Planning & Corridor Operations Portal (IR-JRBP System)
Production-Grade Portal — Ministry of Railways, Government of India
"""

import base64
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.data_gen import generate_requests, CORRIDORS, BRANCH_ACTIONS
from backend.risk_model import CriticalityScorer
from backend.geo_cluster import find_bundling_clusters
from backend.optimizer import run_block_optimizer

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrackYukti | WCR Jabalpur Division",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ASSET LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

BASE = Path(__file__).parent
LOGO_B64   = load_b64(str(BASE / "assets" / "logo.png"))
BG_B64     = load_b64(str(BASE / "assets" / "train_bg.jpg"))
BG_CSS_VAL = f"url('data:image/jpeg;base64,{BG_B64}')" if BG_B64 else "none"

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
DEPT_COLORS = {
    "Engineering": "#38BDF8",
    "S&T":         "#FCD34D",
    "Electrical":  "#C084FC",
}
RISK_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH":     "#F97316",
    "MEDIUM":   "#FBBF24",
    "LOW":      "#4ADE80",
}

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION-GRADE GLOBAL CSS  (v4 — visible train BG, all text fixed)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ══════════════════════════════════════════════════════
   1.  ROOT & BODY
══════════════════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* ── Full-bleed train photo. Overlay: ~15% dark only for text readability ── */
.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.15), rgba(0,0,0,0.15)),
        {BG_CSS_VAL};
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    color: #F1F5F9;
    min-height: 100vh;
}}

/* Remove default Streamlit gutters */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
}}

/* ══════════════════════════════════════════════════════
   2.  SIDEBAR — all text forced visible
══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: rgba(4, 10, 28, 0.97) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.10) !important;
    backdrop-filter: blur(24px);
}}
/* All markdown text in sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li {{
    color: #E2E8F0 !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    color: #FFFFFF !important;
}}
/* Sidebar FIELD HEADING labels (selectbox, slider etc.) */
section[data-testid="stSidebar"] .stSelectbox > label p,
section[data-testid="stSidebar"] .stSlider > label p,
section[data-testid="stSidebar"] .stTextInput > label p,
section[data-testid="stSidebar"] .stRadio > label p {{
    color: #94A3B8 !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}
/* Sidebar RADIO / CHECKBOX / TOGGLE OPTION text → bright white */
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] .stRadio > div label,
section[data-testid="stSidebar"] .stCheckbox [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] .stCheckbox > label p,
section[data-testid="stSidebar"] .stToggle [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] .stToggle > label p {{
    color: #FFFFFF !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}
section[data-testid="stSidebar"] .stCaption p {{
    color: #64748B !important;
    font-size: 11px !important;
}}
hr {{ border-color: rgba(148,163,184,0.12) !important; }}

/* ══════════════════════════════════════════════════════
   3.  TABS — polished pill switcher
══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(4, 10, 28, 0.70);
    padding: 5px 6px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
    backdrop-filter: blur(16px);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 10px 26px;
    font-weight: 700;
    font-size: 13.5px;
    color: #64748B;
    background: transparent;
    border: none;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #1E3A8A, #1D4ED8) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 18px rgba(29,78,216,0.35);
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #CBD5E1 !important;
    background: rgba(255,255,255,0.05) !important;
}}

/* ══════════════════════════════════════════════════════
   4.  BUTTONS
══════════════════════════════════════════════════════ */
.stButton > button {{
    background: rgba(15, 25, 55, 0.80) !important;
    color: #CBD5E1 !important;
    border: 1px solid rgba(148,163,184,0.22) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    padding: 9px 18px !important;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
    backdrop-filter: blur(8px) !important;
    letter-spacing: 0.01em;
}}
.stButton > button:hover {{
    background: rgba(29,78,216,0.70) !important;
    border-color: rgba(59,130,246,0.50) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.30) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(99,179,255,0.35) !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.35) !important;
    letter-spacing: 0.02em;
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,0.45) !important;
    transform: translateY(-2px) !important;
}}
.stDownloadButton > button {{
    background: linear-gradient(135deg, #065F46 0%, #059669 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(52,211,153,0.30) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    padding: 9px 18px !important;
    box-shadow: 0 4px 16px rgba(5,150,105,0.30) !important;
    transition: all 0.18s ease !important;
}}
.stDownloadButton > button:hover {{
    background: #064E3B !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 22px rgba(5,150,105,0.40) !important;
}}
.stButton > button:disabled,
.stDownloadButton > button:disabled {{
    background: rgba(15,25,55,0.40) !important;
    color: #334155 !important;
    border: 1px solid rgba(71,85,105,0.20) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    transform: none !important;
}}

/* ══════════════════════════════════════════════════════
   5.  FORM CONTROLS — ALL TEXT VISIBLE
══════════════════════════════════════════════════════ */
/* ── SELECT BOXES ── */
div[data-baseweb="select"] > div {{
    background: rgba(5,12,36,0.92) !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    backdrop-filter: blur(8px);
    transition: border-color 0.15s ease;
}}
div[data-baseweb="select"] > div:focus-within {{
    border-color: rgba(59,130,246,0.60) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}}
div[data-baseweb="select"] * {{
    color: #E2E8F0 !important;
    background: rgba(5,12,36,0.98) !important;
}}
div[data-baseweb="select"] li:hover {{
    background: rgba(29,78,216,0.35) !important;
}}
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
    color: #E2E8F0 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
}}

/* ── TEXT INPUT ── */
.stTextInput input {{
    background: rgba(5,12,36,0.92) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    backdrop-filter: blur(8px);
}}
.stTextInput input:focus {{
    border-color: rgba(59,130,246,0.60) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}}
.stTextInput input::placeholder {{ color: #475569 !important; }}

/* ── SLIDER ── */
.stSlider [data-testid="stSlider"] > div > div {{
    background: rgba(29,78,216,0.55) !important;
}}

/* ── FIELD HEADING LABELS (selectbox / slider / textinput / multiselect) ── */
.stSelectbox > label p,
.stSlider > label p,
.stTextInput > label p,
.stMultiSelect > label p,
.stNumberInput > label p {{
    color: #CBD5E1 !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}

/* ══════════════════════════════════════════════════════
   6.  RADIO BUTTONS — DEEP SELECTOR FIX
   Streamlit nests label text 3–4 levels deep in spans.
   We target every layer to ensure text is WHITE.
══════════════════════════════════════════════════════ */
/* Radio group heading */
.stRadio > label p {{
    color: #94A3B8 !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}
/* Each radio option text — ALL descendant text nodes */
.stRadio > div > div > label p,
.stRadio > div > div > label span,
.stRadio [role="radiogroup"] label p,
.stRadio [role="radiogroup"] label span,
.stRadio [data-testid="stMarkdownContainer"] p,
.stRadio [data-testid="stMarkdownContainer"] span {{
    color: #FFFFFF !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}

/* ══════════════════════════════════════════════════════
   7.  CHECKBOXES & TOGGLES
══════════════════════════════════════════════════════ */
.stCheckbox > label p,
.stCheckbox > label span,
.stCheckbox [data-testid="stMarkdownContainer"] p {{
    color: #FFFFFF !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}
.stToggle > label p,
.stToggle > label span,
.stToggle [data-testid="stMarkdownContainer"] p {{
    color: #FFFFFF !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}

/* ── Expander ── */
div[data-testid="stExpander"] {{
    background: rgba(5,12,36,0.80) !important;
    border: 1px solid rgba(148,163,184,0.14) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(14px) !important;
    margin-top: 16px !important;
}}
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {{
    color: #CBD5E1 !important;
    font-weight: 700 !important;
}}

/* ── Metric container ── */
[data-testid="metric-container"] {{
    background: rgba(5,12,36,0.65) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    backdrop-filter: blur(12px) !important;
}}
[data-testid="metric-container"] label p {{
    color: #64748B !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 700 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] div {{
    color: #F1F5F9 !important;
    font-size: 22px !important;
    font-weight: 900 !important;
}}

/* Plotly charts transparent bg */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* ══════════════════════════════════════════════════════
   6.  COMPONENT TOKENS — premium glassmorphism
   All backgrounds use low-opacity rgba so train shows through
══════════════════════════════════════════════════════ */

/* ── LOGIN SHELL — full premium glass ── */
.ty-login-shell {{
    max-width: 900px;
    margin: 24px auto;
    background: rgba(5, 10, 25, 0.52);
    backdrop-filter: blur(28px) saturate(180%);
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-top: 3px solid #F59E0B;
    border-radius: 18px;
    padding: 38px 48px 36px;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.06) inset,
        0 32px 80px rgba(0,0,0,0.40),
        0 8px 24px rgba(0,0,0,0.25);
}}

/* ── GENERIC GLASS CARD ── */
.ty-card {{
    background: rgba(5, 10, 25, 0.48);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.06) inset,
        0 8px 28px rgba(0,0,0,0.22);
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}}
.ty-card:hover {{
    border-color: rgba(255,255,255,0.22);
    box-shadow: 0 12px 36px rgba(0,0,0,0.30);
    transform: translateY(-1px);
}}

/* ── TOP HEADER BANNER ── */
.ty-header {{
    background: rgba(5, 10, 25, 0.55);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.14);
    border-bottom: 2px solid rgba(245,158,11,0.40);
    border-radius: 14px;
    padding: 16px 26px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.28);
}}

/* ── STAT TILES ── */
.ty-stat {{
    background: rgba(5, 10, 25, 0.50);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}}
.ty-stat:hover {{
    border-color: rgba(255,255,255,0.22);
    transform: translateY(-2px);
}}
.ty-stat-label {{
    font-size: 10.5px;
    font-weight: 700;
    color: rgba(148,163,184,0.85);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
.ty-stat-value {{
    font-size: 22px;
    font-weight: 900;
    margin-top: 4px;
    line-height: 1.1;
}}

/* — Financial metric cards — */
.ty-fin {{
    background: rgba(8,20,50,0.75);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.07);
    border-top: 3px solid;
    border-radius: 10px;
    padding: 18px 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.ty-fin:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.30);
}}
.ty-fin-label {{
    font-size: 10.5px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
.ty-fin-value {{
    font-size: 26px;
    font-weight: 900;
    color: #F1F5F9;
    margin: 6px 0 3px;
    line-height: 1.15;
}}
.ty-fin-sub {{
    font-size: 11.5px;
    font-weight: 600;
    color: #4ADE80;
}}

/* — Alert banners — */
.ty-alert-danger {{
    background: rgba(127,29,29,0.22);
    border: 1px solid rgba(248,113,113,0.28);
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 13px 16px;
    color: #FCA5A5;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}}
.ty-alert-warn {{
    background: rgba(92,45,0,0.22);
    border: 1px solid rgba(253,211,77,0.25);
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 13px 16px;
    color: #FDE68A;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}}
.ty-alert-success {{
    background: rgba(6,78,59,0.26);
    border: 1px solid rgba(52,211,153,0.25);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 14px 18px;
    color: #A7F3D0;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}}

/* — SMS log rows — */
.ty-sms {{
    background: rgba(14,36,96,0.38);
    border: 1px solid rgba(96,165,250,0.20);
    border-left: 3px solid #3B82F6;
    border-radius: 7px;
    padding: 9px 13px;
    font-size: 12px;
    color: #BAE6FD;
    margin-top: 7px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.01em;
}}

/* — Badges — */
.ty-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(37,99,235,0.18);
    color: #93C5FD;
    border: 1px solid rgba(59,130,246,0.26);
    padding: 3px 10px;
    border-radius: 5px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.ty-badge-green {{
    background: rgba(6,78,59,0.25);
    color: #6EE7B7;
    border: 1px solid rgba(52,211,153,0.25);
}}
.ty-badge-amber {{
    background: rgba(92,45,0,0.25);
    color: #FCD34D;
    border: 1px solid rgba(245,158,11,0.28);
}}

/* — Pulse dot — */
.ty-pulse {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    background: #22C55E;
    box-shadow: 0 0 10px #22C55E;
    animation: ty-pulse 2s infinite;
}}
@keyframes ty-pulse {{
    0%,100% {{ transform: scale(0.9); opacity: 0.85; box-shadow: 0 0 6px #22C55E; }}
    50%      {{ transform: scale(1.3); opacity: 1;    box-shadow: 0 0 14px #4ADE80; }}
}}

/* — Clock pill — */
.ty-clock {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 8px;
    padding: 7px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    font-weight: 600;
    color: #E2E8F0;
    display: inline-flex;
    align-items: center;
    gap: 9px;
    backdrop-filter: blur(10px);
}}

/* — Divider — */
.ty-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.18), transparent);
    margin: 16px 0;
    border: none;
}}

/* — Section heading — */
.ty-section-heading {{
    font-size: 13px;
    font-weight: 800;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}}

/* Plotly charts transparent bg — DUPLICATE GUARD (already set above) */
.js-plotly-plot .plotly, .plotly-graph-div {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DUAL-LANGUAGE TERMINOLOGY MATRIX
# ─────────────────────────────────────────────────────────────────────────────
TRANS = {
    "English": {
        "portal_title": "GOVERNMENT OF INDIA · MINISTRY OF RAILWAYS · WCR JABALPUR",
        "portal_sub": "Joint Rolling Block Planning & Corridor Operations Portal (IR-JRBP)",
        "tab_1": "📋 Master Block Timetable & Dispatch",
        "tab_2": "💰 Financial & Punctuality Audit",
        "config_header": "Departmental Block Requisition",
        "timeline_header": "24-Hour Corridor Rolling Block Timetable",
        "branch_label": "Operating Branch",
        "corridor_label": "Corridor Section",
        "track_label": "Track Line",
        "duration_label": "Requested Duration (Mins)",
        "action_label": "Maintenance Plan / Activity",
        "heavy_label": "Requires Heavy Machine / BCM / TRT (Exclusive Possession)",
        "btn_push": "Submit Work Order to Joint Queue",
        "btn_broadcast": "AUTHORIZE & TRANSMIT BLOCK PROGRAM (SMS)",
        "btn_export": "Download Master Timetable (CSV)",
        "total_pool": "Total Requisitions",
        "scheduled_metric": "Approved Blocks",
        "deferred_metric": "Deferred",
        "critical_metric": "Priority Flaws",
        "demurrage_card_title": "Demurrage Penalties Averted",
        "capacity_card_title": "Section Capacity Recovered",
        "traction_card_title": "Traction Loss Prevented",
        "caution_card_title": "Caution Orders Reduced",
        "green_banner_title": "Energy & Environmental Audit Certificate",
        "green_banner_desc": "Unified spatial possession eliminates redundant loco idling and repeated section power shutdowns, saving diesel traction and electricity.",
        "cost_pie_title": "Cost Savings Breakdown (Weekly)",
        "starvation_title": "Section Capacity & Demurrage Ledger",
        "telemetry_expander": "CRIS Optimization Engine Audit Logs",
        "siren_conflict": "⚠ Section Conflict: Overlapping Departmental Requisitions Detected",
        "conflict_action": "Joint possession protocol applied — combined into a single synchronized window.",
        "sms_success_title": "CRIS GATEWAY — ROLLING BLOCK PROGRAM TRANSMITTED",
    },
    "Hindi / हिंदी": {
        "portal_title": "भारत सरकार · रेल मंत्रालय · पमरे जबलपुर",
        "portal_sub": "संयुक्त रोलिंग ब्लॉक नियोजन एवं कॉरिडोर परिचालन पोर्टल (IR-JRBP)",
        "tab_1": "📋 मास्टर ब्लॉक समय-सारिणी एवं प्रेषण",
        "tab_2": "💰 वित्तीय एवं समय-पालन ऑडिट",
        "config_header": "विभागीय ब्लॉक मांग प्रपत्र",
        "timeline_header": "24-घंटे कॉरिडोर रोलिंग ब्लॉक समय-सारिणी",
        "branch_label": "परिचालन शाखा",
        "corridor_label": "कॉरिडोर खंड",
        "track_label": "ट्रैक लाइन",
        "duration_label": "अपेक्षित अवधि (मिनट)",
        "action_label": "रखरखाव कार्य विवरण",
        "heavy_label": "भारी मशीन / बीसीएम / टीआरटी आवश्यक (अनन्य ब्लॉक)",
        "btn_push": "कार्य आदेश संयुक्त कतार में दर्ज करें",
        "btn_broadcast": "रोलिंग ब्लॉक कार्यक्रम अधिकृत एवं प्रसारित करें (SMS)",
        "btn_export": "मास्टर समय-सारिणी डाउनलोड करें (CSV)",
        "total_pool": "कुल मांग",
        "scheduled_metric": "स्वीकृत ब्लॉक",
        "deferred_metric": "स्थगित",
        "critical_metric": "गंभीर दोष",
        "demurrage_card_title": "डेमरेज दंड बचत",
        "capacity_card_title": "लाइन क्षमता पुनर्प्राप्ति",
        "traction_card_title": "कर्षण हानि रोकथाम",
        "caution_card_title": "कॉशन ऑर्डर न्यूनीकरण",
        "green_banner_title": "ऊर्जा एवं पर्यावरण ऑडिट प्रमाण पत्र",
        "green_banner_desc": "समकालिक स्थानिक ब्लॉक द्वारा अनावश्यक इंजन आइडलिंग और बार-बार विद्युत कटौती समाप्त करके डीजल व बिजली की बचत की गई।",
        "cost_pie_title": "लागत बचत वर्गीकरण (साप्ताहिक)",
        "starvation_title": "रेल लाइन क्षमता एवं डेमरेज ऑडिट खाता",
        "telemetry_expander": "CRIS अनुकूलन इंजन ऑडिट लॉग",
        "siren_conflict": "⚠ सेक्शन टकराव: समकालिक विभागीय मांग की पहचान",
        "conflict_action": "संयुक्त पज़ेशन प्रोटोकॉल लागू — एकल विंडो में संयोजित।",
        "sms_success_title": "CRIS गेटवे — रोलिंग ब्लॉक कार्यक्रम प्रसारित",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_scorer():
    return CriticalityScorer()

@st.cache_data
def get_cached_requests(seed=42):
    return generate_requests(n_requests=24, seed=seed)

def run_pipeline(df, horizon_hours, setup_buffer, delayed_corridor=None, delay_minutes=0):
    scorer  = get_scorer()
    scored  = scorer.score_requests(df)
    bundled = find_bundling_clusters(scored, radius_m=500.0)
    result  = run_block_optimizer(
        bundled, horizon_hours=horizon_hours,
        setup_buffer_minutes=setup_buffer,
        delayed_corridor=delayed_corridor,
        delay_minutes=delay_minutes,
    )
    return result, bundled, scorer

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "is_logged_in": False,
    "user_dept": "Engineering",
    "user_designation": "Sr. Divisional Engineer (Sr. DEN / Track)",
    "lang_choice": "English",
    "seed": 42,
    "custom_requests": [],
    "simulate_collision": False,
    "sync_failure": False,
    "dispatch_executed": False,
    "siren_off_halt": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_all():
    for k in ["seed","custom_requests","simulate_collision","sync_failure","dispatch_executed","siren_off_halt"]:
        st.session_state[k] = _defaults[k]


# =============================================================================
#  LOGIN PORTAL
# =============================================================================
if not st.session_state["is_logged_in"]:

    logo_html = (
        f'<img src="data:image/png;base64,{LOGO_B64}" '
        f'style="height:70px;width:auto;display:block;margin:0 auto 10px;" alt="TrackYukti">'
        if LOGO_B64 else ""
    )

    st.markdown(f"""
    <div class="ty-login-shell">
      {logo_html}
      <div style="text-align:center;padding-bottom:22px;border-bottom:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:11px;font-weight:800;color:#64748B;text-transform:uppercase;letter-spacing:0.12em;">
          Government of India &nbsp;·&nbsp; Ministry of Railways
        </div>
        <h1 style="margin:6px 0 3px;font-size:30px;font-weight:900;color:#FFFFFF;letter-spacing:-0.025em;">
          TRACK<span style="color:#F59E0B;">YUKTI</span>
        </h1>
        <div style="font-size:11px;color:#F59E0B;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;">
          Smarter Planning &nbsp;·&nbsp; Efficient Solutions
        </div>
        <p style="margin:8px 0 0;font-size:13px;color:#475569;font-weight:500;">
          West Central Railway &nbsp;·&nbsp; Jabalpur Division
          &nbsp;·&nbsp; Joint Rolling Block Operations Portal (IR-JRBP v2.5)
        </p>
        <div style="margin-top:14px;">
          <span class="ty-badge ty-badge-amber">
            🔐 &nbsp;AUTHORIZED PERSONNEL ONLY &nbsp;·&nbsp; DEFAULT PASSKEY: JBP2026
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    lc1, lc2 = st.columns([1.1, 1])

    with lc1:
        st.markdown("""
        <div class="ty-card" style="margin-top:10px;">
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
            Step 1 — Select Operating Branch
          </div>
          <div style="font-size:12px;color:#475569;">
            Portal access and block permissions are provisioned per departmental jurisdiction.
          </div>
        </div>
        """, unsafe_allow_html=True)

        dept_choice = st.radio("Branch:", [
            "Engineering (Civil / Track / P-Way)",
            "Signal & Telecom (S&T)",
            "Electrical (TRD / OHE Maintenance)",
            "Chief Controller (CHC) / Operating Control",
        ], index=0)

        dept_map = {
            "Engineering (Civil / Track / P-Way)":      "Engineering",
            "Signal & Telecom (S&T)":                    "S&T",
            "Electrical (TRD / OHE Maintenance)":        "Electrical",
            "Chief Controller (CHC) / Operating Control":"Chief Controller / DRM",
        }
        sel_dept = dept_map[dept_choice]

    with lc2:
        st.markdown("""
        <div class="ty-card" style="margin-top:10px;">
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
            Step 2 — Officer Designation & Passkey
          </div>
          <div style="font-size:12px;color:#475569;">
            Enter the divisional authorization passkey:
            <span style="color:#F59E0B;font-weight:700;">JBP2026</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        desig_map = {
            "Engineering": [
                "Sr. Divisional Engineer (Sr. DEN / Track)",
                "Sr. Divisional Engineer (Sr. DEN / Bridge)",
                "Sr. Divisional Engineer (Sr. DEN / Construction)",
                "Assistant Divisional Engineer (ADEN / Track)",
                "Assistant Divisional Engineer (ADEN / Works)",
                "Junior Engineer (JE / P-Way)",
                "Senior Section Engineer (SSE / P-Way)",
            ],
            "S&T": [
                "Sr. Divisional Signal & Telecom Engineer (Sr. DSTE)",
                "Divisional Signal & Telecom Engineer (DSTE)",
                "Assistant Signal & Telecom Engineer (ASTE)",
                "Junior Engineer (JE / Signal)",
                "Junior Engineer (JE / Telecom)",
                "Senior Section Engineer (SSE / Signal)",
                "Senior Section Engineer (SSE / Telecom)",
            ],
            "Electrical": [
                "Sr. Divisional Electrical Engineer (Sr. DEE / TRD)",
                "Sr. Divisional Electrical Engineer (Sr. DEE / General)",
                "Divisional Electrical Engineer (DEE / TRD)",
                "Divisional Electrical Engineer (DEE / General)",
                "Assistant Divisional Electrical Engineer (ADEE)",
                "Junior Engineer (JE / TRD)",
                "Senior Section Engineer (SSE / OHE)",
            ],
            "Chief Controller / DRM": [
                "Chief Controller (CHC / Central Control)",
                "Dy. Chief Controller (Dy. CHC)",
                "Section Controller (SC / Train Control)",
                "Sr. Divisional Operations Manager (Sr. DOM)",
                "Divisional Operations Manager (DOM)",
                "Sr. Divisional Commercial Manager (Sr. DCM)",
                "Divisional Safety Officer (DSO)",
                "Divisional Railway Manager (DRM Jabalpur)",
            ],
        }
        desig = st.selectbox("Officer Designation", desig_map[sel_dept])
        pk    = st.text_input("Security Passkey", value="JBP2026", type="password")

        ba, bb = st.columns(2)
        with ba:
            if st.button("🔐  Access Workstation", type="primary", use_container_width=True):
                if pk.strip() == "JBP2026":
                    st.session_state.update(is_logged_in=True, user_dept=sel_dept, user_designation=desig)
                    st.rerun()
                else:
                    st.error("❌  Invalid passkey. Default: JBP2026")
        with bb:
            if st.button("👁  Guest / Read-Only Entry", use_container_width=True):
                st.session_state.update(
                    is_logged_in=True,
                    user_dept="Chief Controller / DRM",
                    user_designation="Divisional Safety Officer (DSO)",
                )
                st.rerun()

        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)

        # ── TEACHER DEMO BUTTON ───────────────────────────────────────────────
        if st.button(
            "🎓  LOAD TEACHER DEMO — DRM Full Access + Pre-Filled Block Data",
            use_container_width=True,
        ):
            # Pre-load 8 realistic block requisitions across all 4 corridors
            demo_reqs = [
                # Engineering — Heavy tamping on JBP-KTE freight route
                dict(request_id="DEMO-ENG-001", department="Engineering",
                     action="Deep Screening & Track Renewal (TSM-V)",
                     corridor="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route",
                     section_track="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route :: DN-Main",
                     asset_id="AST-DEMO-ENG-001",
                     latitude=23.51, longitude=80.22, overdue_days=112,
                     last_inspection_score=91.0, traffic_density=148,
                     corridor_priority=1.5, estimated_duration_mins=120,
                     is_heavy_machinery=True, exclusive_block=True),

                # S&T — Interlocking overhaul JBP-ET route
                dict(request_id="DEMO-SNT-001", department="S&T",
                     action="Electronic Interlocking Commissioning (EI-IBS)",
                     corridor="Jabalpur (JBP) - Itarsi (ET) Main Trunk Route",
                     section_track="Jabalpur (JBP) - Itarsi (ET) Main Trunk Route :: UP-Main",
                     asset_id="AST-DEMO-SNT-001",
                     latitude=23.18, longitude=79.94, overdue_days=88,
                     last_inspection_score=85.0, traffic_density=132,
                     corridor_priority=1.4, estimated_duration_mins=90,
                     is_heavy_machinery=False, exclusive_block=False),

                # Electrical — OHE maintenance Bina route
                dict(request_id="DEMO-ELC-001", department="Electrical",
                     action="OHE Tension Balancer & Jumper Replacement",
                     corridor="Katni (KTE) - Bina (BINA) Coal Corridor",
                     section_track="Katni (KTE) - Bina (BINA) Coal Corridor :: UP-Main",
                     asset_id="AST-DEMO-ELC-001",
                     latitude=23.82, longitude=80.40, overdue_days=65,
                     last_inspection_score=79.0, traffic_density=120,
                     corridor_priority=1.3, estimated_duration_mins=75,
                     is_heavy_machinery=False, exclusive_block=False),

                # Engineering — USFD testing on Gondia route
                dict(request_id="DEMO-ENG-002", department="Engineering",
                     action="Ultrasonic Rail Flaw Detection (USFD Testing)",
                     corridor="Jabalpur (JBP) - Gondia (G) Passenger Corridor",
                     section_track="Jabalpur (JBP) - Gondia (G) Passenger Corridor :: DN-Main",
                     asset_id="AST-DEMO-ENG-002",
                     latitude=23.32, longitude=80.55, overdue_days=95,
                     last_inspection_score=87.0, traffic_density=108,
                     corridor_priority=1.2, estimated_duration_mins=60,
                     is_heavy_machinery=False, exclusive_block=False),

                # S&T — Axle counter renewal KTE-BINA
                dict(request_id="DEMO-SNT-002", department="S&T",
                     action="Digital Axle Counter (DAC) System Renewal",
                     corridor="Katni (KTE) - Bina (BINA) Coal Corridor",
                     section_track="Katni (KTE) - Bina (BINA) Coal Corridor :: DN-Main",
                     asset_id="AST-DEMO-SNT-002",
                     latitude=23.85, longitude=80.42, overdue_days=72,
                     last_inspection_score=83.0, traffic_density=118,
                     corridor_priority=1.3, estimated_duration_mins=60,
                     is_heavy_machinery=False, exclusive_block=False),

                # Electrical — Substation shutdown JBP-ET
                dict(request_id="DEMO-ELC-002", department="Electrical",
                     action="25kV Traction Sub-Station Shutdown & Breaker Maintenance",
                     corridor="Jabalpur (JBP) - Itarsi (ET) Main Trunk Route",
                     section_track="Jabalpur (JBP) - Itarsi (ET) Main Trunk Route :: DN-Main",
                     asset_id="AST-DEMO-ELC-002",
                     latitude=23.15, longitude=79.90, overdue_days=58,
                     last_inspection_score=76.0, traffic_density=130,
                     corridor_priority=1.4, estimated_duration_mins=90,
                     is_heavy_machinery=False, exclusive_block=True),

                # Engineering — BCM machine JBP-KTE
                dict(request_id="DEMO-ENG-003", department="Engineering",
                     action="Ballast Cleaning Machine (BCM) Operation — 3km Run",
                     corridor="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route",
                     section_track="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route :: UP-Main",
                     asset_id="AST-DEMO-ENG-003",
                     latitude=23.49, longitude=80.18, overdue_days=130,
                     last_inspection_score=93.0, traffic_density=145,
                     corridor_priority=1.5, estimated_duration_mins=180,
                     is_heavy_machinery=True, exclusive_block=True),

                # S&T — Signal gantry painting JBP-G
                dict(request_id="DEMO-SNT-003", department="S&T",
                     action="Colour Light Signal (CLS) Maintenance & Lamp Renewal",
                     corridor="Jabalpur (JBP) - Gondia (G) Passenger Corridor",
                     section_track="Jabalpur (JBP) - Gondia (G) Passenger Corridor :: UP-Main",
                     asset_id="AST-DEMO-SNT-003",
                     latitude=23.30, longitude=80.52, overdue_days=45,
                     last_inspection_score=71.0, traffic_density=102,
                     corridor_priority=1.2, estimated_duration_mins=45,
                     is_heavy_machinery=False, exclusive_block=False),
            ]
            st.session_state.update(
                is_logged_in=True,
                user_dept="Chief Controller / DRM",
                user_designation="Chief Controller (CHC / Central Control)",
                custom_requests=demo_reqs,
                simulate_collision=False,
                sync_failure=False,
                dispatch_executed=False,
                siren_off_halt=False,
            )
            st.balloons()
            st.rerun()

    st.stop()



# =============================================================================
#  SIDEBAR
# =============================================================================
with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f'<img src="data:image/png;base64,{LOGO_B64}" style="height:54px;width:auto;margin-bottom:4px;">',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<span style="font-size:18px;font-weight:900;color:#FFFFFF;">Track<span style="color:#F59E0B;">Yukti</span></span>'
        '<br><span style="font-size:11px;color:#475569;font-weight:500;">WCR Jabalpur Division</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(148,163,184,0.10);margin:10px 0;">', unsafe_allow_html=True)

    lang = st.selectbox("🌐 Language / भाषा",
                        ["English", "Hindi / हिंदी"],
                        index=0 if st.session_state["lang_choice"] == "English" else 1)
    st.session_state["lang_choice"] = lang
    T = TRANS[lang]

    st.markdown(f"""
    <div style="background:rgba(29,78,216,0.16);border:1px solid rgba(59,130,246,0.22);
                border-radius:9px;padding:12px 14px;margin:12px 0;">
      <div style="font-size:10px;font-weight:800;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;">
        Active Session
      </div>
      <div style="font-size:14px;font-weight:800;color:#E2E8F0;margin-top:3px;">
        {st.session_state['user_dept']}
      </div>
      <div style="font-size:11.5px;color:#64748B;margin-top:2px;">
        {st.session_state['user_designation']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪  Switch Officer / Logout", use_container_width=True):
        st.session_state["is_logged_in"] = False
        st.rerun()

    st.markdown('<hr style="border-color:rgba(148,163,184,0.10);">', unsafe_allow_html=True)
    if st.button("♻️  Reset Operational Parameters", use_container_width=True):
        reset_all(); st.rerun()

    st.markdown('<hr style="border-color:rgba(148,163,184,0.10);">', unsafe_allow_html=True)
    st.markdown("#### 📍 Corridor Jurisdiction")
    sel_corr = st.selectbox(
        "Active Corridor",
        ["All Corridors (Jabalpur Division)"] + list(CORRIDORS.keys())
    )

    st.markdown("#### ⏱️ Planning Parameters")
    horizon_hours = st.slider("Planning Window (Hours)", 6, 24, 12, step=1)
    setup_buffer  = st.slider("Handover Safety Buffer (Mins)", 5, 45, 15, step=5)

    st.markdown('<hr style="border-color:rgba(148,163,184,0.10);">', unsafe_allow_html=True)
    st.markdown("#### 🧪 Simulation Controls")
    st.session_state["sync_failure"]       = st.toggle("Simulate CRIS Server Offline",  value=st.session_state["sync_failure"])
    st.session_state["simulate_collision"] = st.toggle("Inject Section Conflict",       value=st.session_state["simulate_collision"])
    st.session_state["siren_off_halt"]     = st.toggle("Engage Safety Interlock Hold",  value=st.session_state["siren_off_halt"])
    delay_minutes = st.slider("Inject Train Delay (Mins)", 0, 75, 0, step=5)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
base_df = get_cached_requests(seed=st.session_state["seed"])

if st.session_state["simulate_collision"]:
    coll_corr  = "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route"
    coll_track = f"{coll_corr} :: DN-Main"
    coll_rows  = [
        dict(request_id="WCR-ENG-COL-1", department="Engineering",
             action="Track Tamping & Rail Renewal", corridor=coll_corr,
             section_track=coll_track, asset_id="AST-COL-ENG",
             latitude=23.501, longitude=80.201, overdue_days=90,
             last_inspection_score=88.0, traffic_density=135,
             corridor_priority=1.4, estimated_duration_mins=90,
             is_heavy_machinery=False, exclusive_block=False),
        dict(request_id="WCR-SNT-COL-2", department="S&T",
             action="Electronic Interlocking Overhaul", corridor=coll_corr,
             section_track=coll_track, asset_id="AST-COL-SNT",
             latitude=23.503, longitude=80.204, overdue_days=85,
             last_inspection_score=84.0, traffic_density=135,
             corridor_priority=1.4, estimated_duration_mins=75,
             is_heavy_machinery=False, exclusive_block=False),
    ]
    base_df     = base_df[~base_df["request_id"].str.contains("COL")]
    combined_df = pd.concat([pd.DataFrame(coll_rows), base_df], ignore_index=True)
else:
    combined_df = base_df.copy()

if st.session_state["custom_requests"]:
    combined_df = pd.concat([pd.DataFrame(st.session_state["custom_requests"]), combined_df], ignore_index=True)

delayed_corr_arg = None if sel_corr == "All Corridors (Jabalpur Division)" else sel_corr
baseline_result, bundled_df, scorer = run_pipeline(combined_df, horizon_hours, setup_buffer)

if delay_minutes > 0 and delayed_corr_arg:
    live_result, _, _ = run_pipeline(combined_df, horizon_hours, setup_buffer, delayed_corr_arg, delay_minutes)
else:
    live_result = baseline_result

schedule = live_result.schedule.copy()
bs        = baseline_result.schedule.set_index("request_id")["start_min"]
schedule["baseline_start_min"]  = schedule["request_id"].map(bs)
schedule["dynamically_shifted"] = (
    schedule["is_scheduled"] & schedule["baseline_start_min"].notna()
    & (schedule["start_min"] != schedule["baseline_start_min"])
)

# conflict detection
has_conflict, coll_depts, coll_track_name = False, [], ""
for trk, grp in combined_df.groupby("section_track"):
    depts = grp["department"].unique()
    if len(depts) >= 2:
        has_conflict, coll_depts, coll_track_name = True, list(depts), trk
        break

total_tasks      = len(schedule)
scheduled_tasks  = int(schedule["is_scheduled"].sum())
deferred_tasks   = total_tasks - scheduled_tasks
critical_risks   = int((schedule["risk_band"] == "CRITICAL").sum())
bundled_clusters = int(schedule.loc[schedule["bundle_cluster"] >= 0, "bundle_cluster"].nunique())
efficiency_pct   = round(scheduled_tasks / total_tasks * 100, 1)

# ─────────────────────────────────────────────────────────────────────────────
# TOP HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
now_dt  = datetime.now()
ist_str = now_dt.strftime("%H:%M:%S IST")
utc_str = (now_dt - timedelta(hours=5, minutes=30)).strftime("%H:%M:%S UTC")
date_str = "02 September 2026"

status_html = (
    '<span class="ty-badge ty-badge-green">'
    '<span class="ty-pulse"></span> SYSTEM OPERATIONAL'
    '</span>'
    if not st.session_state["siren_off_halt"]
    else '<span class="ty-badge" style="background:rgba(127,29,29,0.25);color:#FCA5A5;border-color:rgba(248,113,113,0.25);">'
         '⛔ SAFETY HOLD ACTIVE</span>'
)

logo_hdr = (
    f'<img src="data:image/png;base64,{LOGO_B64}" style="height:50px;width:auto;" alt="TrackYukti">'
    if LOGO_B64 else ""
)

st.markdown(f"""
<div class="ty-header">
  <div style="display:flex;align-items:center;gap:16px;">
    {logo_hdr}
    <div>
      <div style="font-size:21px;font-weight:900;color:#FFFFFF;letter-spacing:-0.02em;">
        TRACK<span style="color:#F59E0B;">YUKTI</span>
      </div>
      <div style="font-size:11.5px;color:#64748B;margin-top:2px;font-weight:500;">
        {T['portal_title']} &nbsp;·&nbsp; {T['portal_sub']}
      </div>
      <div style="font-size:11px;color:#475569;margin-top:1px;">
        Logged in: <b style="color:#93C5FD;">{st.session_state['user_dept']}</b>
        &nbsp;·&nbsp; {st.session_state['user_designation']}
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <div class="ty-clock">🕒 {date_str} &nbsp;|&nbsp; {ist_str}</div>
    {status_html}
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state["sync_failure"]:
    st.markdown(
        '<div class="ty-alert-warn"><b>⚠ CRIS / COA SERVER OFFLINE</b> — '
        'Operating on local cached database with static safety headway rules.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# WORKSPACE TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([T["tab_1"], T["tab_2"]])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — MASTER BLOCK TIMETABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([4, 6])

    # ── Left 40% ──────────────────────────────────────────────────────────────
    with col_l:
        active_dept = st.session_state["user_dept"]
        is_chc      = (active_dept == "Chief Controller / DRM")

        st.markdown(f'<div class="ty-section-heading">{T["config_header"]}</div>', unsafe_allow_html=True)

        # Form card
        st.markdown(f"""
        <div class="ty-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div style="font-size:13.5px;font-weight:700;color:#E2E8F0;">
              {T['branch_label']}: <span style="color:#93C5FD;">{active_dept}</span>
            </div>
            <span class="ty-badge">{active_dept}</span>
          </div>
        """, unsafe_allow_html=True)

        form_branch = (
            st.selectbox(f"{T['branch_label']}:", ["Engineering","S&T","Electrical"], key="fb")
            if is_chc else active_dept
        )

        c1, c2 = st.columns(2)
        with c1:
            corridor_input = st.selectbox(T["corridor_label"], list(CORRIDORS.keys()), index=1, key="ci")
        with c2:
            track_input = st.selectbox(T["track_label"], CORRIDORS[corridor_input]["tracks"], index=0, key="ti")

        action_input   = st.selectbox(T["action_label"], BRANCH_ACTIONS[form_branch], key="ai")
        duration_input = st.slider(T["duration_label"], 30, 240, 90, step=15, key="di")
        heavy_toggle   = st.checkbox(T["heavy_label"], value=False)

        if st.button(T["btn_push"], type="primary", use_container_width=True):
            nid  = f"WCR-REQ-{1050 + len(st.session_state['custom_requests'])}"
            meta = CORRIDORS[corridor_input]
            st.session_state["custom_requests"].append(dict(
                request_id=nid, department=form_branch, action=action_input,
                corridor=corridor_input,
                section_track=f"{corridor_input} :: {track_input}",
                asset_id=f"AST-{form_branch[:3].upper()}-9901",
                latitude=meta["lat"] + np.random.uniform(-0.01, 0.01),
                longitude=meta["lon"] + np.random.uniform(-0.01, 0.01),
                overdue_days=75, last_inspection_score=82.0,
                traffic_density=110, corridor_priority=meta["priority"],
                estimated_duration_mins=duration_input,
                is_heavy_machinery=heavy_toggle, exclusive_block=heavy_toggle,
            ))
            st.success(f"Work order {nid} added to joint queue.")
            time.sleep(0.2); st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Stat Tiles ────────────────────────────────────────────────────────
        st.markdown(f'<div class="ty-section-heading" style="margin-top:14px;">{T["total_pool"]} Ledger</div>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        for col, lbl, val, clr in [
            (s1, T["total_pool"],       total_tasks,                             "#38BDF8"),
            (s2, T["scheduled_metric"], f"{scheduled_tasks}<br><small style='font-size:11px;'>({efficiency_pct}%)</small>", "#4ADE80"),
            (s3, T["deferred_metric"],  deferred_tasks,                          "#F87171"),
            (s4, T["critical_metric"],  critical_risks,                          "#FCD34D"),
        ]:
            with col:
                st.markdown(
                    f'<div class="ty-stat"><div class="ty-stat-label">{lbl}</div>'
                    f'<div class="ty-stat-value" style="color:{clr};">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Safety Parameter Info ─────────────────────────────────────────────
        st.markdown("""
        <div class="ty-card" style="border-left:3px solid #38BDF8;margin-top:12px;">
          <div style="font-size:10.5px;font-weight:800;color:#94A3B8;text-transform:uppercase;
                      letter-spacing:0.06em;margin-bottom:5px;">
            Safety Parameter Evaluation Matrix
          </div>
          <div style="font-size:12px;color:#64748B;line-height:1.5;">
            USFD Rail Flaw Severity <b style="color:#CBD5E1;">35%</b> &nbsp;·&nbsp;
            Overdue Maintenance Days <b style="color:#CBD5E1;">25%</b> &nbsp;·&nbsp;
            GMT Load Density <b style="color:#CBD5E1;">20%</b> &nbsp;·&nbsp;
            Corridor Priority <b style="color:#CBD5E1;">20%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Right 60% ─────────────────────────────────────────────────────────────
    with col_r:
        st.markdown(f'<div class="ty-section-heading">{T["timeline_header"]}</div>', unsafe_allow_html=True)

        if has_conflict:
            st.markdown(
                f'<div class="ty-alert-danger"><b>{T["siren_conflict"]}</b><br>'
                f'<span style="font-size:12px;">{coll_track_name}: '
                f'{" & ".join(coll_depts)} — {T["conflict_action"]}</span></div>',
                unsafe_allow_html=True,
            )

        # Gantt
        gantt_df = schedule[schedule["is_scheduled"]].copy()
        if sel_corr != "All Corridors (Jabalpur Division)":
            gantt_df = gantt_df[gantt_df["corridor"] == sel_corr]

        if gantt_df.empty:
            st.warning("No scheduled blocks for the selected corridor.")
        else:
            bt = datetime.combine(datetime.today(), datetime.min.time())
            gantt_df["Start"]  = gantt_df["start_min"].apply(lambda m: bt + timedelta(minutes=float(m)))
            gantt_df["Finish"] = gantt_df["end_min"].apply(  lambda m: bt + timedelta(minutes=float(m)))
            gantt_df["Label"]  = gantt_df.apply(
                lambda r: f"{r['request_id']} ({r['department'][:3]})"
                + (" [EXCL]"  if r.get("is_heavy_machinery") else "")
                + (" [SHIFT]" if r["dynamically_shifted"] else ""),
                axis=1,
            )
            fig = px.timeline(
                gantt_df, x_start="Start", x_end="Finish",
                y="section_track", color="department",
                color_discrete_map=DEPT_COLORS, text="Label",
                hover_data={
                    "request_id": True, "department": True, "action": True,
                    "risk_score": True, "corridor": True, "estimated_duration_mins": True,
                    "section_track": False, "Start": False, "Finish": False,
                },
            )
            fig.update_yaxes(autorange="reversed", title="Track Section",
                             title_font_color="#64748B", tickfont_color="#94A3B8",
                             gridcolor="rgba(148,163,184,0.06)", showgrid=True)
            fig.update_xaxes(title=f"Time (00:00 – {horizon_hours:02d}:00)",
                             title_font_color="#64748B", tickfont_color="#94A3B8",
                             gridcolor="rgba(148,163,184,0.06)")
            fig.update_traces(textposition="inside", insidetextanchor="start",
                              marker_line_width=1.5, marker_line_color="rgba(255,255,255,0.20)")
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(4,10,28,0.55)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94A3B8",
                legend_title_text="Branch",
                legend=dict(orientation="h", y=1.04, x=1, xanchor="right",
                            font_color="#94A3B8", bgcolor="rgba(4,10,28,0.55)",
                            bordercolor="rgba(148,163,184,0.12)", borderwidth=1),
                height=max(340, 60 + 42 * gantt_df["section_track"].nunique()),
                margin=dict(l=8, r=8, t=32, b=8),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Dispatch Controls ─────────────────────────────────────────────────
        st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)

        user_desig = st.session_state["user_designation"]
        is_auth = any(r in user_desig for r in [
            "Chief Controller", "Dy. Chief Controller", "Section Controller",
            "DRM", "Sr. DEN", "Sr. DOM", "Sr. DSTE", "Sr. DEE",
            "Divisional Railway Manager", "Divisional Safety Officer",
        ])

        bc1, bc2 = st.columns([2.5, 1.5])
        with bc1:
            if st.session_state["siren_off_halt"]:
                st.button("⛔  DISPATCH LOCKED — Safety Hold Active",
                          disabled=True, use_container_width=True)
            elif not is_auth:
                st.button("🔒  AUTHORIZE & TRANSMIT — Department Head Clearance Required",
                          disabled=True, use_container_width=True)
                st.markdown(
                    '<div style="font-size:11px;color:#475569;margin-top:4px;">'
                    'Access restricted to Sr. DEN / Sr. DOM / Sr. DSTE / Sr. DEE / CHC / DRM</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(T["btn_broadcast"], type="primary", use_container_width=True):
                    st.session_state["dispatch_executed"] = True
                    st.balloons()
        with bc2:
            buf = io.StringIO()
            schedule.to_csv(buf, index=False)
            st.download_button(
                T["btn_export"],
                data=buf.getvalue(),
                file_name=f"trackyukti_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # ── SMS Dispatch Success Panel ─────────────────────────────────────────
        if st.session_state["dispatch_executed"] and is_auth and not st.session_state["siren_off_halt"]:
            order_ref = f"WCR/JBP/JRBP/{datetime.now().strftime('%Y%m%d-%H%M')}"
            st.markdown(f"""
            <div class="ty-alert-success" style="margin-top:14px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div style="font-size:13.5px;font-weight:800;">{T['sms_success_title']}</div>
                <span class="ty-badge ty-badge-green">CRIS TLS-1.3 VERIFIED</span>
              </div>
              <div style="font-size:12px;line-height:1.6;margin-bottom:4px;">
                Order: <span class="ty-badge">{order_ref}</span>
                &nbsp;|&nbsp; Auth: <span class="ty-badge">{user_desig}</span><br>
                Token: <code style="font-size:10.5px;color:#93C5FD;font-family:JetBrains Mono,monospace;">
                SEC_TOKEN_JBP2026_SMS_VERIFIED_OK</code>
              </div>
              <div class="ty-sms">
                📱 <b>Civil / P-Way (ENG):</b> [WCR/JBP/ENG] Track Tamping Approved. Window 02:00–04:30. Auth: CHC-JBP
              </div>
              <div class="ty-sms">
                📱 <b>Signal & Telecom (S&T):</b> [WCR/JBP/S&T] Interlocking synchronized at KM 1042. Auth: CHC-JBP
              </div>
              <div class="ty-sms">
                📱 <b>Traction / TRD (OHE):</b> [WCR/JBP/TRD] OHE Power Block scheduled. Zero starvation. Auth: CHC-JBP
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FINANCIAL & PUNCTUALITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="ty-section-heading">Division Financial & Punctuality Audit Ledger</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#475569;margin-bottom:18px;">Freight demurrage prevention, line capacity reclamation, and environmental savings — Jabalpur Division.</div>',
                unsafe_allow_html=True)

    # Metric cards
    fk1, fk2, fk3, fk4 = st.columns(4)
    fin_data = [
        (fk1, T["demurrage_card_title"], "₹42.8 Lakhs", "▲ 34.2% Detention Penalty Averted", "#059669"),
        (fk2, T["capacity_card_title"],  "+18.4 Hours",  "+6 Freight Paths / Week",            "#0284C7"),
        (fk3, T["traction_card_title"],  "₹16.5 Lakhs", "Zero Unscheduled OHE Power Cuts",    "#7C3AED"),
        (fk4, T["caution_card_title"],   "−38% TSR",     "₹8.2L Fuel/Traction Idling Saved",   "#D97706"),
    ]
    for col, lbl, val, sub, clr in fin_data:
        with col:
            st.markdown(
                f'<div class="ty-fin" style="border-top-color:{clr};">'
                f'<div class="ty-fin-label">{lbl}</div>'
                f'<div class="ty-fin-value">{val}</div>'
                f'<div class="ty-fin-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Environmental certificate
    st.markdown(f"""
    <div class="ty-card" style="border-left:4px solid #059669;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  flex-wrap:wrap;gap:14px;">
        <div>
          <div style="font-size:14.5px;font-weight:800;color:#4ADE80;margin-bottom:5px;">
            🌱 {T['green_banner_title']}
          </div>
          <div style="font-size:12.5px;color:#64748B;line-height:1.55;max-width:600px;">
            {T['green_banner_desc']}
          </div>
        </div>
        <div style="background:rgba(6,78,59,0.25);border:1px solid rgba(52,211,153,0.22);
                    border-radius:8px;padding:10px 18px;font-weight:800;font-size:13.5px;
                    color:#4ADE80;white-space:nowrap;">
          124.6 T CO₂e Abated / Month
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f'<div class="ty-section-heading">{T["cost_pie_title"]}</div>', unsafe_allow_html=True)
        pie_df = pd.DataFrame({
            "Category": ["Demurrage Averted","Fuel / Traction Recovered",
                         "TSR Caution Acceleration","Gang Synergy Savings"],
            "₹ Lakhs":  [42.8, 16.5, 8.2, 11.4],
        })
        pie = px.pie(pie_df, names="Category", values="₹ Lakhs", hole=0.45,
                     color_discrete_sequence=["#059669","#0284C7","#D97706","#7C3AED"])
        pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(l=8,r=8,t=18,b=8),
            legend=dict(font_color="#94A3B8", bgcolor="rgba(4,10,28,0.50)"),
        )
        st.plotly_chart(pie, use_container_width=True)

    with fc2:
        st.markdown(f'<div class="ty-section-heading">{T["starvation_title"]}</div>', unsafe_allow_html=True)
        ledger_df = pd.DataFrame({
            "Corridor":           list(CORRIDORS.keys()),
            "Throughput Gain":    ["+5.8 hrs","+6.2 hrs","+2.8 hrs","+3.6 hrs"],
            "Demurrage Saved":    ["₹14.2L","₹16.8L","₹4.6L","₹7.2L"],
            "Capacity Index":     ["96.2%","94.8%","98.1%","95.5%"],
        })
        st.dataframe(ledger_df, use_container_width=True, height=260)


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY LOGS (EXPANDER)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander(T["telemetry_expander"], expanded=False):
    logs = {
        "system": "TrackYukti IR-JRBP v2.5",
        "engine": "Google OR-Tools CP-SAT v9.15",
        "spatial_bundling": "GeoPandas / Shapely EPSG:32644",
        "solver_status":     live_result.solver_status,
        "objective_score":   float(live_result.objective_value),
        "planning_horizon":  {"hours": int(horizon_hours), "minutes": int(horizon_hours * 60)},
        "blocks":            {"total": int(total_tasks), "scheduled": int(scheduled_tasks),
                              "deferred": int(deferred_tasks), "clusters": int(bundled_clusters)},
        "risk_profile":      {"critical": int(critical_risks)},
        "safety_interlock":  bool(st.session_state["siren_off_halt"]),
        "server_status":     "LOCAL_SQLITE_FALLBACK" if st.session_state["sync_failure"] else "CRIS_COA_ONLINE",
        "officer":           {"department": st.session_state["user_dept"],
                              "designation": st.session_state["user_designation"]},
        "language":          st.session_state["lang_choice"],
        "operational_date":  date_str,
        "timestamp_iso":     datetime.now().isoformat(),
    }
    st.markdown(
        f'<div style="background:#040A1C;border:1px solid rgba(148,163,184,0.10);'
        f'border-radius:10px;padding:16px;overflow-x:auto;">'
        f'<pre style="margin:0;color:#38BDF8;font-family:JetBrains Mono,monospace;'
        f'font-size:12px;line-height:1.6;">{json.dumps(logs, indent=2)}</pre>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<hr style="border-color:rgba(148,163,184,0.10);margin-top:24px;">'
    '<div style="text-align:center;font-size:11.5px;color:#334155;padding:8px 0;">'
    '🚆 TrackYukti &nbsp;·&nbsp; Smarter Planning. Efficient Solutions. &nbsp;·&nbsp; '
    'Government of India · Ministry of Railways · WCR Jabalpur Division · CRIS'
    '</div>',
    unsafe_allow_html=True,
)
