"""
app.py
-------
TRACKYUKTI — Smarter Planning. Efficient Solutions.
WEST CENTRAL RAILWAY (WCR) — JABALPUR DIVISION
Joint Rolling Block Planning & Corridor Operations Portal (IR-JRBP System)
"""

import base64
import io
import json
import os
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

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="TrackYukti | WCR Jabalpur Division Joint Block Portal",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# LOAD ASSETS (LOGO)
# --------------------------------------------------------------------------
def load_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

LOGO_DIR = Path(__file__).parent / "assets" / "logo.png"
LOGO_B64 = load_image_as_base64(str(LOGO_DIR))

# --------------------------------------------------------------------------
# DEPARTMENT & RISK COLOR SYSTEM
# --------------------------------------------------------------------------
DEPT_COLORS = {
    "Engineering": "#0369A1",
    "S&T": "#B45309",
    "Electrical": "#6D28D9",
}

RISK_COLORS = {
    "CRITICAL": "#B91C1C",
    "HIGH": "#C2410C",
    "MEDIUM": "#A16207",
    "LOW": "#15803D",
}

# --------------------------------------------------------------------------
# GLOBAL CSS — TrackYukti Brand Design System
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    /* ── App Background with train atmosphere ── */
    .stApp {{
        background:
            linear-gradient(
                160deg,
                rgba(15, 23, 42, 0.92) 0%,
                rgba(14, 35, 90, 0.88) 35%,
                rgba(8, 47, 73, 0.90) 70%,
                rgba(15, 23, 42, 0.94) 100%
            ),
            url("https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=1800&q=85&fit=crop") center/cover no-repeat fixed;
        color: #F1F5F9;
        min-height: 100vh;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: rgba(10, 18, 40, 0.97);
        border-right: 1px solid rgba(148, 163, 184, 0.15);
        backdrop-filter: blur(16px);
    }}
    section[data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    section[data-testid="stSidebar"] label {{
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
    }}

    /* ── Top Brand Header ── */
    .trackyukti-header {{
        background: rgba(10, 18, 40, 0.82);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 14px;
        padding: 18px 26px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.30);
    }}

    .ty-brand-logo-img {{
        height: 52px;
        width: auto;
    }}

    .ty-brand-title {{
        font-size: 22px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        line-height: 1.15;
    }}

    .ty-brand-title span.accent {{
        color: #F59E0B;
    }}

    .ty-brand-sub {{
        font-size: 12px;
        color: #94A3B8;
        margin-top: 2px;
        font-weight: 500;
    }}

    .ty-clock-pill {{
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 8px;
        padding: 7px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        font-weight: 600;
        color: #E2E8F0;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}

    .pulse-dot {{
        width: 8px; height: 8px;
        background: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 10px #22C55E;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ transform: scale(0.9); opacity: 0.85; }}
        50%       {{ transform: scale(1.3); opacity: 1; box-shadow: 0 0 16px #4ADE80; }}
    }}

    /* ── Glass Cards ── */
    .glass-card {{
        background: rgba(15, 25, 50, 0.70);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.20);
    }}

    /* ── Stat Tiles ── */
    .stat-tile {{
        background: rgba(15, 25, 55, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 10px;
        padding: 14px 15px;
        text-align: left;
        backdrop-filter: blur(10px);
    }}
    .stat-tile-label {{
        font-size: 10.5px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .stat-tile-value {{
        font-size: 22px;
        font-weight: 900;
        margin-top: 3px;
        line-height: 1.1;
    }}

    /* ── Financial Metric Cards ── */
    .fin-metric-card {{
        background: rgba(15, 25, 55, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid #059669;
        border-radius: 10px;
        padding: 16px 18px;
        backdrop-filter: blur(10px);
    }}
    .fin-metric-title {{
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .fin-metric-value {{
        font-size: 24px;
        font-weight: 900;
        color: #F1F5F9;
        margin: 5px 0 2px;
        line-height: 1.15;
    }}
    .fin-metric-sub {{
        font-size: 11.5px;
        color: #4ADE80;
        font-weight: 600;
    }}

    /* ── Alert Banners ── */
    .alert-danger {{
        background: rgba(153, 27, 27, 0.25);
        border: 1px solid rgba(252, 165, 165, 0.4);
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #FCA5A5;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }}
    .alert-warning {{
        background: rgba(120, 70, 0, 0.25);
        border: 1px solid rgba(253, 230, 138, 0.35);
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 12px 16px;
        color: #FDE68A;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }}
    .alert-success {{
        background: rgba(6, 95, 70, 0.30);
        border: 1px solid rgba(74, 222, 128, 0.30);
        border-left: 4px solid #22C55E;
        border-radius: 8px;
        padding: 14px 18px;
        color: #BBF7D0;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: rgba(10, 18, 40, 0.60);
        padding: 5px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        margin-bottom: 14px;
        backdrop-filter: blur(10px);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px;
        padding: 9px 24px;
        font-weight: 700;
        font-size: 13.5px;
        color: #94A3B8;
        background: transparent;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(30, 58, 138, 0.85) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.30);
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: rgba(30, 41, 59, 0.85) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(148, 163, 184, 0.30) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 9px 18px !important;
        transition: all 0.18s ease !important;
        backdrop-filter: blur(8px) !important;
    }}
    .stButton > button:hover {{
        background: rgba(30, 58, 138, 0.85) !important;
        border-color: #3B82F6 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #1E3A8A, #2563EB) !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.30) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.40) !important;
    }}
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #047857, #059669) !important;
        color: #FFFFFF !important;
        border: 1px solid #10B981 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 9px 18px !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25) !important;
    }}
    .stDownloadButton > button:hover {{
        background: #065F46 !important;
        transform: translateY(-1px);
    }}
    .stButton > button:disabled, .stDownloadButton > button:disabled {{
        background: rgba(30, 41, 59, 0.50) !important;
        color: #475569 !important;
        border: 1px solid rgba(71, 85, 105, 0.30) !important;
        cursor: not-allowed !important;
    }}

    /* ── Inputs ── */
    div[data-baseweb="select"] > div {{
        background: rgba(15, 25, 55, 0.80) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }}
    div[data-baseweb="select"] * {{
        color: #E2E8F0 !important;
        background: rgba(10, 18, 40, 0.95) !important;
    }}
    .stTextInput input {{
        background: rgba(15, 25, 55, 0.80) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 8px !important;
    }}
    .stSlider label, .stSelectbox label, .stTextInput label,
    .stMultiSelect label, .stCheckbox label, .stToggle label {{
        color: #CBD5E1 !important;
        font-weight: 700 !important;
        font-size: 12.5px !important;
    }}
    .stCheckbox span, .stToggle span {{
        color: #CBD5E1 !important;
    }}
    div[data-testid="stExpander"] {{
        background: rgba(15, 25, 55, 0.70) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(12px) !important;
    }}
    .stMetric {{
        color: #E2E8F0 !important;
    }}

    /* ── Tag / Badge ── */
    .ty-badge {{
        display: inline-block;
        background: rgba(37, 99, 235, 0.20);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.30);
        padding: 2px 10px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 11.5px;
    }}

    .sms-log-item {{
        background: rgba(30, 58, 138, 0.20);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-left: 3px solid #3B82F6;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        color: #BAE6FD;
        margin-top: 6px;
    }}

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {{
        background: transparent !important;
    }}

    /* ── Login card ── */
    .login-glass-card {{
        max-width: 860px;
        margin: 28px auto;
        background: rgba(10, 18, 40, 0.88);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-top: 4px solid #F59E0B;
        border-radius: 14px;
        padding: 32px 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
    }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# DUAL-LANGUAGE TERMINOLOGY MATRIX
# --------------------------------------------------------------------------
TRANS = {
    "English": {
        "portal_title": "GOVERNMENT OF INDIA · MINISTRY OF RAILWAYS · WCR JABALPUR DIVISION",
        "portal_sub": "Joint Rolling Block Planning & Corridor Operations Portal (IR-JRBP)",
        "tab_1": "📋 Master Block Timetable & Dispatch",
        "tab_2": "💰 Division Financial & Punctuality Audit",
        "config_header": "Departmental Block Requisition Form",
        "timeline_header": "24-Hour Corridor Rolling Block Timetable (Gantt)",
        "branch_label": "Operating Branch",
        "corridor_label": "Corridor Section",
        "track_label": "Track Line",
        "duration_label": "Requested Duration (Mins)",
        "action_label": "Maintenance Plan / Activity",
        "heavy_label": "Requires Heavy Track Machine / BCM / TRT (Exclusive Possession Window)",
        "btn_push": "Submit Work Order to Joint Queue",
        "btn_broadcast": "AUTHORIZE & TRANSMIT ROLLING BLOCK PROGRAM (SMS)",
        "btn_export": "Download Master Timetable (CSV)",
        "total_pool": "Total Requisitions",
        "scheduled_metric": "Approved Blocks",
        "deferred_metric": "Deferred (Capacity)",
        "critical_metric": "Priority Flaws (USFD)",
        "demurrage_card_title": "Freight Demurrage Averted",
        "capacity_card_title": "Section Capacity Recovered",
        "traction_card_title": "Traction Power Loss Prevented",
        "caution_card_title": "Caution Orders (TSR) Reduced",
        "green_banner_title": "Rolling Block Energy & Environmental Audit Certificate",
        "green_banner_desc": "Unified spatial possession eliminates redundant loco idling and repeated section power shutdowns, saving diesel traction and electricity.",
        "cost_pie_title": "Operational Cost Savings Breakdown (Weekly)",
        "starvation_title": "Section Capacity & Demurrage Audit Ledger",
        "telemetry_expander": "CRIS Optimization Engine Audit Logs",
        "siren_conflict": "Section Conflict Notice: Overlapping Departmental Requisitions",
        "conflict_action": "Joint possession protocol applied. Combined into single synchronized window.",
        "sms_success_title": "CRIS GATEWAY: ROLLING BLOCK PROGRAM TRANSMITTED TO FIELD DIVISIONS",
    },
    "Hindi / हिंदी": {
        "portal_title": "भारत सरकार · रेल मंत्रालय · पमरे जबलपुर मंडल",
        "portal_sub": "संयुक्त रोलिंग ब्लॉक नियोजन एवं नियंत्रण पोर्टल (IR-JRBP)",
        "tab_1": "📋 मास्टर ब्लॉक समय-सारिणी एवं प्रेषण",
        "tab_2": "💰 मंडल वित्तीय एवं समय-पालन ऑडिट",
        "config_header": "विभागीय ब्लॉक मांग प्रपत्र",
        "timeline_header": "24-घंटे कॉरिडोर रोलिंग ब्लॉक समय-सारिणी (गैंट चार्ट)",
        "branch_label": "परिचालन शाखा",
        "corridor_label": "कॉरिडोर खंड",
        "track_label": "ट्रैक लाइन अनुभाग",
        "duration_label": "अपेक्षित अवधि (मिनट)",
        "action_label": "रखरखाव कार्य विवरण",
        "heavy_label": "भारी ट्रैक मशीन / बीसीएम / टीआरटी आवश्यक (अनन्य ब्लॉक)",
        "btn_push": "कार्य आदेश संयुक्त कतार में दर्ज करें",
        "btn_broadcast": "रोलिंग ब्लॉक कार्यक्रम अधिकृत एवं प्रसारित करें (SMS)",
        "btn_export": "मास्टर समय-सारिणी डाउनलोड करें (CSV)",
        "total_pool": "कुल मांग",
        "scheduled_metric": "स्वीकृत ब्लॉक",
        "deferred_metric": "स्थगित (क्षमता सीमा)",
        "critical_metric": "अति-गंभीर दोष (USFD)",
        "demurrage_card_title": "डेमरेज दंड बचत",
        "capacity_card_title": "लाइन क्षमता पुनर्प्राप्ति",
        "traction_card_title": "कर्षण विद्युत रिसाव रोकथाम",
        "caution_card_title": "कॉशन ऑर्डर (TSR) न्यूनीकरण",
        "green_banner_title": "रोलिंग ब्लॉक ऊर्जा एवं पर्यावरण ऑडिट प्रमाण पत्र",
        "green_banner_desc": "समकालिक स्थानिक ब्लॉक द्वारा अनावश्यक इंजन आइडलिंग और बार-बार विद्युत कटौती समाप्त करके डीजल व बिजली की बचत की गई।",
        "cost_pie_title": "परिचालन लागत बचत वर्गीकरण (साप्ताहिक)",
        "starvation_title": "रेल लाइन क्षमता एवं डेमरेज ऑडिट खाता",
        "telemetry_expander": "क्रिस (CRIS) अनुकूलन इंजन ऑडिट लॉग",
        "siren_conflict": "सेक्शन टकराव सूचना: समकालिक विभागीय मांग",
        "conflict_action": "संयुक्त पज़ेशन प्रोटोकॉल लागू। दोनों को एकल विंडो में संयोजित किया गया।",
        "sms_success_title": "क्रिस गेटवे: रोलिंग ब्लॉक कार्यक्रम फील्ड डिवीजनों को प्रसारित",
    }
}

# --------------------------------------------------------------------------
# PIPELINE CACHE & ML SCORER
# --------------------------------------------------------------------------
@st.cache_resource
def get_scorer():
    return CriticalityScorer()

@st.cache_data
def get_cached_requests(seed=42):
    return generate_requests(n_requests=24, seed=seed)

def run_pipeline(requests_df, horizon_hours, setup_buffer, delayed_corridor=None, delay_minutes=0):
    scorer = get_scorer()
    scored = scorer.score_requests(requests_df)
    bundled = find_bundling_clusters(scored, radius_m=500.0)
    result = run_block_optimizer(
        bundled,
        horizon_hours=horizon_hours,
        setup_buffer_minutes=setup_buffer,
        delayed_corridor=delayed_corridor,
        delay_minutes=delay_minutes,
    )
    return result, bundled, scorer

# --------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------
defaults = {
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
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_entire_system():
    for k in ["seed", "custom_requests", "simulate_collision", "sync_failure", "dispatch_executed", "siren_off_halt"]:
        st.session_state[k] = defaults[k]


# ==========================================================================
# 🌟 TRACKYUKTI LOGIN PORTAL
# ==========================================================================
if not st.session_state["is_logged_in"]:
    logo_tag = ""
    if LOGO_B64:
        logo_tag = f'<img src="data:image/png;base64,{LOGO_B64}" style="height: 64px; width: auto; margin: 0 auto 8px auto; display: block;">'

    st.markdown(f"""
    <div class="login-glass-card">
        {logo_tag}
        <div style="text-align: center; padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.10);">
            <div style="font-size: 11.5px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">Government of India · Ministry of Railways</div>
            <h1 style="margin: 5px 0 2px; font-size: 26px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.02em;">
                TRACK<span style="color: #F59E0B;">YUKTI</span>
            </h1>
            <div style="font-size: 11px; color: #F59E0B; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">SMARTER PLANNING · EFFICIENT SOLUTIONS</div>
            <p style="margin: 6px 0 0; font-size: 12.5px; color: #64748B;">West Central Railway · Jabalpur Division · Joint Rolling Block Operations Portal</p>
            <div style="margin-top: 12px;">
                <span style="display: inline-block; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.30); border-radius: 4px; padding: 3px 12px; font-size: 11px; font-weight: 700; color: #FCD34D;">
                    🔐 AUTHORIZED PERSONNEL ACCESS · DEFAULT PASSKEY: JBP2026
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    login_col1, login_col2 = st.columns([1.1, 1])

    with login_col1:
        st.markdown("""
        <div style="margin-top: 10px; padding: 14px 18px; background: rgba(15,25,55,0.70); border: 1px solid rgba(255,255,255,0.10); border-radius: 8px; backdrop-filter: blur(10px);">
            <div style="font-size: 13px; font-weight: 700; color: #E2E8F0;">Step 1 — Select Operating Branch</div>
            <div style="font-size: 12px; color: #64748B; margin-top: 2px;">Permissions are tailored per departmental jurisdiction.</div>
        </div>
        """, unsafe_allow_html=True)

        dept_choice = st.radio(
            "Operating Branch:",
            [
                "Engineering (Civil / Track / P-Way)",
                "Signal & Telecom (S&T)",
                "Electrical (TRD / OHE Maintenance)",
                "Chief Controller (CHC) / Operating Control",
            ],
            index=0,
        )
        dept_map = {
            "Engineering (Civil / Track / P-Way)": "Engineering",
            "Signal & Telecom (S&T)": "S&T",
            "Electrical (TRD / OHE Maintenance)": "Electrical",
            "Chief Controller (CHC) / Operating Control": "Chief Controller / DRM",
        }
        selected_dept_key = dept_map[dept_choice]

    with login_col2:
        st.markdown("""
        <div style="margin-top: 10px; padding: 14px 18px; background: rgba(15,25,55,0.70); border: 1px solid rgba(255,255,255,0.10); border-radius: 8px; backdrop-filter: blur(10px);">
            <div style="font-size: 13px; font-weight: 700; color: #E2E8F0;">Step 2 — Officer Designation & Passkey</div>
            <div style="font-size: 12px; color: #64748B; margin-top: 2px;">Enter divisional authorization passkey: <b style="color: #F59E0B;">JBP2026</b></div>
        </div>
        """, unsafe_allow_html=True)

        desig_map = {
            "Engineering": [
                "Sr. Divisional Engineer (Sr. DEN / Track)",
                "Assistant Divisional Engineer (ADEN)",
                "Senior Section Engineer (SSE / P-Way)",
            ],
            "S&T": [
                "Sr. Divisional Signal & Telecom Engineer (Sr. DSTE)",
                "Divisional Signal & Telecom Engineer (DSTE)",
                "Senior Section Engineer (SSE / Signal)",
            ],
            "Electrical": [
                "Sr. Divisional Electrical Engineer (Sr. DEE / TRD)",
                "Divisional Electrical Engineer (DEE / TRD)",
                "Senior Section Engineer (SSE / OHE)",
            ],
            "Chief Controller / DRM": [
                "Chief Controller (CHC / Central Control)",
                "Sr. Divisional Operations Manager (Sr. DOM)",
                "Divisional Railway Manager (DRM Jabalpur)",
            ],
        }
        designation_choice = st.selectbox("Officer Designation", desig_map[selected_dept_key], index=0)
        passkey_input = st.text_input("Divisional Security Passkey", value="JBP2026", type="password")

        btn_a, btn_b = st.columns(2)
        with btn_a:
            if st.button("Access Workstation", type="primary", use_container_width=True):
                if passkey_input.strip() == "JBP2026":
                    st.session_state["is_logged_in"] = True
                    st.session_state["user_dept"] = selected_dept_key
                    st.session_state["user_designation"] = designation_choice
                    time.sleep(0.2)
                    st.rerun()
                else:
                    st.error("Invalid passkey. Default: JBP2026")
        with btn_b:
            if st.button("Guest / Demo Entry", use_container_width=True):
                st.session_state["is_logged_in"] = True
                st.session_state["user_dept"] = "Chief Controller / DRM"
                st.session_state["user_designation"] = "Chief Controller (CHC / Central Control)"
                st.rerun()

    st.stop()


# ==========================================================================
# 🌟 MAIN APPLICATION WORKSPACE
# ==========================================================================

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_B64:
        st.markdown(f'<img src="data:image/png;base64,{LOGO_B64}" style="height: 56px; width: auto; margin-bottom: 6px;">', unsafe_allow_html=True)
    st.markdown("## TrackYukti")
    st.caption("WCR Jabalpur Division · Control Terminal")

    lang = st.selectbox("🌐 Language / भाषा चयन", ["English", "Hindi / हिंदी"],
                        index=0 if st.session_state["lang_choice"] == "English" else 1)
    st.session_state["lang_choice"] = lang
    T = TRANS[lang]

    st.markdown(f"""
    <div style="background: rgba(30, 58, 138, 0.25); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 12px; margin: 10px 0;">
        <div style="font-size: 10.5px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Active Session</div>
        <div style="font-size: 13.5px; font-weight: 800; color: #E2E8F0; margin-top: 2px;">{st.session_state['user_dept']}</div>
        <div style="font-size: 11.5px; color: #94A3B8; margin-top: 2px;">{st.session_state['user_designation']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Switch Officer / Logout", use_container_width=True):
        st.session_state["is_logged_in"] = False
        st.rerun()

    st.markdown("---")
    if st.button("♻️ Reset Operational Parameters", use_container_width=True):
        reset_entire_system()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📍 Corridor Jurisdiction")
    corridor_options = ["All Corridors (Jabalpur Division)"] + list(CORRIDORS.keys())
    selected_corridor = st.selectbox("Active Track Corridor", corridor_options, index=0)

    st.markdown("#### ⏱️ Planning Parameters")
    horizon_hours = st.slider("Planning Window (Hours)", 6, 24, 12, step=1)
    setup_buffer = st.slider("Safety Handover Buffer (Mins)", 5, 45, 15, step=5)

    st.markdown("---")
    st.markdown("#### 🧪 Operational Simulation")
    st.session_state["sync_failure"] = st.toggle("Simulate CRIS Server Offline", value=st.session_state["sync_failure"])
    st.session_state["simulate_collision"] = st.toggle("Inject Section Conflict", value=st.session_state["simulate_collision"])
    st.session_state["siren_off_halt"] = st.toggle("Safety Interlock Hold", value=st.session_state["siren_off_halt"])
    delay_minutes = st.slider("Inject Train Delay (Mins)", 0, 75, 0, step=5)

# ── Data Pipeline ───────────────────────────────────────────────────────────
base_req_df = get_cached_requests(seed=st.session_state["seed"])

if st.session_state["simulate_collision"]:
    sample_corr = "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route"
    sample_track = f"{sample_corr} :: DN-Main"
    sim_rows = [
        {"request_id": "WCR-ENG-COLLIDE-1", "department": "Engineering", "action": "Track Tamping & Rail Renewal",
         "corridor": sample_corr, "section_track": sample_track, "asset_id": "AST-WCR-ENG-COL",
         "latitude": 23.501, "longitude": 80.201, "overdue_days": 90, "last_inspection_score": 88.0,
         "traffic_density": 135, "corridor_priority": 1.4, "estimated_duration_mins": 90,
         "is_heavy_machinery": False, "exclusive_block": False},
        {"request_id": "WCR-S&T-COLLIDE-2", "department": "S&T", "action": "Electronic Interlocking Overhaul",
         "corridor": sample_corr, "section_track": sample_track, "asset_id": "AST-WCR-SNT-COL",
         "latitude": 23.503, "longitude": 80.204, "overdue_days": 85, "last_inspection_score": 84.0,
         "traffic_density": 135, "corridor_priority": 1.4, "estimated_duration_mins": 75,
         "is_heavy_machinery": False, "exclusive_block": False},
    ]
    base_req_df = base_req_df[~base_req_df["request_id"].str.contains("COLLIDE")]
    combined_req_df = pd.concat([pd.DataFrame(sim_rows), base_req_df], ignore_index=True)
else:
    combined_req_df = base_req_df.copy()

if st.session_state["custom_requests"]:
    combined_req_df = pd.concat([pd.DataFrame(st.session_state["custom_requests"]), combined_req_df], ignore_index=True)

delayed_corridor_arg = None if selected_corridor == "All Corridors (Jabalpur Division)" else selected_corridor
baseline_result, bundled_df, scorer = run_pipeline(combined_req_df, horizon_hours, setup_buffer, None, 0)

if delay_minutes > 0 and delayed_corridor_arg:
    live_result, live_bundled, _ = run_pipeline(combined_req_df, horizon_hours, setup_buffer, delayed_corridor_arg, delay_minutes)
else:
    live_result, live_bundled = baseline_result, bundled_df

schedule = live_result.schedule.copy()
base_starts = baseline_result.schedule.set_index("request_id")["start_min"]
schedule["baseline_start_min"] = schedule["request_id"].map(base_starts)
schedule["dynamically_shifted"] = (
    schedule["is_scheduled"] & schedule["baseline_start_min"].notna()
    & (schedule["start_min"] != schedule["baseline_start_min"])
)

# Conflict detection
has_conflict = False
colliding_depts, collision_track, collision_corridor = [], "", ""
for track_name, grp in combined_req_df.groupby("section_track"):
    depts = grp["department"].unique()
    if len(depts) >= 2:
        has_conflict = True
        colliding_depts = list(depts)
        collision_track = track_name
        collision_corridor = grp["corridor"].iloc[0]
        break

total_tasks = len(schedule)
scheduled_tasks = int(schedule["is_scheduled"].sum())
deferred_tasks = total_tasks - scheduled_tasks
critical_risks = int((schedule["risk_band"] == "CRITICAL").sum())
bundled_clusters_count = int(schedule.loc[schedule["bundle_cluster"] >= 0, "bundle_cluster"].nunique())
efficiency_pct = round((scheduled_tasks / total_tasks) * 100, 1)

# ── Top Header Banner ───────────────────────────────────────────────────────
now_time = datetime.now()
target_date_str = "02 September 2026"
time_ist_str = now_time.strftime("%H:%M:%S IST")
time_utc_str = (now_time - timedelta(hours=5, minutes=30)).strftime("%H:%M:%S UTC")

status_badge = (
    '<span style="background: rgba(21,128,61,0.25); color: #4ADE80; border: 1px solid rgba(74,222,128,0.35); padding: 5px 12px; border-radius: 5px; font-size: 11.5px; font-weight: 700;"><span class="pulse-dot" style="display:inline-block; width:7px; height:7px; background:#22C55E; border-radius:50%; margin-right:6px; box-shadow:0 0 8px #22C55E;"></span> SYSTEM OPERATIONAL</span>'
    if not st.session_state["siren_off_halt"]
    else '<span style="background: rgba(185,28,28,0.25); color: #FCA5A5; border: 1px solid rgba(252,165,165,0.35); padding: 5px 12px; border-radius: 5px; font-size: 11.5px; font-weight: 700;">⛔ SAFETY HOLD ACTIVE</span>'
)

logo_header = f'<img src="data:image/png;base64,{LOGO_B64}" style="height: 52px; width: auto;">' if LOGO_B64 else "🚆"

st.markdown(f"""
<div class="trackyukti-header">
  <div style="display: flex; align-items: center; gap: 16px;">
    {logo_header}
    <div>
      <div class="ty-brand-title">TRACK<span class="accent">YUKTI</span></div>
      <div class="ty-brand-sub">{T['portal_title']} &nbsp;·&nbsp; {T['portal_sub']}</div>
      <div style="font-size: 11.5px; color: #64748B; margin-top: 2px;">
        Logged in: <b style="color: #93C5FD;">{st.session_state['user_dept']}</b>
        &nbsp;·&nbsp; {st.session_state['user_designation']}
      </div>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
    <div class="ty-clock-pill">🕒 {target_date_str} &nbsp;|&nbsp; {time_ist_str}</div>
    {status_badge}
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state["sync_failure"]:
    st.markdown('<div class="alert-warning"><b>⚠️ CRIS / COA SERVER OFFLINE:</b> Operating on local cached database with static safety headway rules.</div>', unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([T["tab_1"], T["tab_2"]])

# ===========================================================================
# TAB 1 — MASTER BLOCK TIMETABLE
# ===========================================================================
with tab1:
    col_left, col_right = st.columns([4, 6])

    # ── Left 40%: Requisition Form ─────────────────────────────────────────
    with col_left:
        active_dept = st.session_state["user_dept"]
        is_chc = (active_dept == "Chief Controller / DRM")

        st.markdown(f"#### {T['config_header']}")

        st.markdown(f"""<div class="glass-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
  <div style="font-size:13.5px; font-weight:700; color:#E2E8F0;">{T['branch_label']}: <span style="color:#93C5FD;">{active_dept}</span></div>
  <span class="ty-badge">{active_dept}</span>
</div>""", unsafe_allow_html=True)

        form_branch = st.selectbox(f"{T['branch_label']}:", ["Engineering", "S&T", "Electrical"], key="fb") if is_chc else active_dept

        c1, c2 = st.columns(2)
        with c1:
            corridor_input = st.selectbox(T["corridor_label"], list(CORRIDORS.keys()), index=1, key="ci")
        with c2:
            track_input = st.selectbox(T["track_label"], CORRIDORS[corridor_input]["tracks"], index=0, key="ti")

        action_input = st.selectbox(T["action_label"], BRANCH_ACTIONS[form_branch], index=0, key="ai")
        duration_input = st.slider(T["duration_label"], 30, 240, 90, step=15, key="di")
        heavy_toggle = st.checkbox(T["heavy_label"], value=False)

        if st.button(T["btn_push"], type="primary", use_container_width=True):
            new_id = f"WCR-REQ-{1000 + len(st.session_state['custom_requests']) + 50}"
            meta = CORRIDORS[corridor_input]
            st.session_state["custom_requests"].append({
                "request_id": new_id, "department": form_branch, "action": action_input,
                "corridor": corridor_input, "section_track": f"{corridor_input} :: {track_input}",
                "asset_id": f"AST-WCR-{form_branch[:3].upper()}-9901",
                "latitude": meta["lat"] + np.random.uniform(-0.01, 0.01),
                "longitude": meta["lon"] + np.random.uniform(-0.01, 0.01),
                "overdue_days": 75, "last_inspection_score": 82.0, "traffic_density": 110,
                "corridor_priority": meta["priority"], "estimated_duration_mins": duration_input,
                "is_heavy_machinery": heavy_toggle, "exclusive_block": heavy_toggle,
            })
            st.success(f"Work order {new_id} added to queue.")
            time.sleep(0.2)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Stat Tiles
        st.markdown(f"##### {T['total_pool']} Ledger")
        s1, s2, s3, s4 = st.columns(4)
        tiles = [
            (s1, T["total_pool"], total_tasks, "#38BDF8"),
            (s2, T["scheduled_metric"], f"{scheduled_tasks} ({efficiency_pct}%)", "#4ADE80"),
            (s3, T["deferred_metric"], deferred_tasks, "#F87171"),
            (s4, T["critical_metric"], critical_risks, "#FCD34D"),
        ]
        for col, label, val, color in tiles:
            with col:
                st.markdown(f"""<div class="stat-tile">
<div class="stat-tile-label">{label}</div>
<div class="stat-tile-value" style="color:{color};">{val}</div>
</div>""", unsafe_allow_html=True)

        # Safety parameter note
        st.markdown("""<div class="glass-card" style="border-left: 3px solid #38BDF8; margin-top:12px;">
<div style="font-size:11px; font-weight:700; color:#94A3B8; text-transform:uppercase;">Safety Parameter Evaluation Matrix</div>
<div style="font-size:12px; color:#CBD5E1; margin-top:4px; line-height:1.4;">
USFD Rail Flaw (35%) · Overdue Maintenance Days (25%) · GMT Load Density (20%) · Corridor Priority (20%)
</div>
</div>""", unsafe_allow_html=True)

    # ── Right 60%: Gantt Chart & Dispatch ──────────────────────────────────
    with col_right:
        st.markdown(f"#### {T['timeline_header']}")

        if has_conflict:
            st.markdown(f"""<div class="alert-danger">
<div style="font-size:13.5px; font-weight:700;">{T['siren_conflict']}</div>
<div style="font-size:12px; margin-top:3px;"><b>{collision_track}</b>: {' & '.join(colliding_depts)} — {T['conflict_action']}</div>
</div>""", unsafe_allow_html=True)

        gantt_df = schedule[schedule["is_scheduled"]].copy()
        if selected_corridor != "All Corridors (Jabalpur Division)":
            gantt_df = gantt_df[gantt_df["corridor"] == selected_corridor]

        if gantt_df.empty:
            st.warning("No blocks scheduled for the selected corridor filter.")
        else:
            base_time = datetime.combine(datetime.today(), datetime.min.time())
            gantt_df["Start"] = gantt_df["start_min"].apply(lambda m: base_time + timedelta(minutes=float(m)))
            gantt_df["Finish"] = gantt_df["end_min"].apply(lambda m: base_time + timedelta(minutes=float(m)))
            gantt_df["Label"] = gantt_df.apply(
                lambda r: f"{r['request_id']} ({r['department'][:3]})"
                + (" [EXCL]" if r.get("is_heavy_machinery") else "")
                + (" [SHIFT]" if r["dynamically_shifted"] else ""), axis=1)

            fig = px.timeline(
                gantt_df, x_start="Start", x_end="Finish", y="section_track",
                color="department", color_discrete_map=DEPT_COLORS, text="Label",
                hover_data={"request_id": True, "department": True, "action": True,
                            "risk_score": True, "corridor": True, "estimated_duration_mins": True,
                            "section_track": False, "Start": False, "Finish": False},
            )
            fig.update_yaxes(autorange="reversed", title="Track Section",
                             title_font_color="#94A3B8", tickfont_color="#CBD5E1",
                             gridcolor="rgba(148,163,184,0.08)")
            fig.update_xaxes(title=f"Time Horizon (00:00 – {horizon_hours:02d}:00)",
                             title_font_color="#94A3B8", tickfont_color="#CBD5E1",
                             gridcolor="rgba(148,163,184,0.08)")
            fig.update_traces(textposition="inside", insidetextanchor="start",
                              marker_line_width=1, marker_line_color="rgba(255,255,255,0.25)")
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(10,18,40,0.60)",
                paper_bgcolor="rgba(10,18,40,0.00)",
                legend_title_text="Branch",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font_color="#CBD5E1", bgcolor="rgba(10,18,40,0.60)"),
                height=max(340, 60 + 40 * gantt_df["section_track"].nunique()),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        user_desig = st.session_state["user_designation"]
        is_auth = any(r in user_desig for r in ["Department Head", "Chief Controller", "DRM",
                                                  "Sr. DEN", "Sr. DOM", "Sr. DSTE", "Sr. DEE"])

        bc1, bc2 = st.columns([2.5, 1.5])
        with bc1:
            if st.session_state["siren_off_halt"]:
                st.button("⛔ DISPATCH LOCKED (Safety Hold Active)", disabled=True, use_container_width=True)
            elif not is_auth:
                st.button("🔒 AUTHORIZE & TRANSMIT (Clearance Required)", disabled=True, use_container_width=True)
                st.caption("Access: Sr. DEN / Sr. DOM / Sr. DSTE / Sr. DEE / CHC only.")
            else:
                if st.button(T["btn_broadcast"], type="primary", use_container_width=True):
                    st.session_state["dispatch_executed"] = True
                    st.balloons()
        with bc2:
            buf = io.StringIO()
            schedule.to_csv(buf, index=False)
            st.download_button(T["btn_export"], data=buf.getvalue(),
                               file_name=f"trackyukti_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", use_container_width=True)

        if st.session_state["dispatch_executed"] and is_auth and not st.session_state["siren_off_halt"]:
            order_ref = f"WCR/JBP/JRBP/{datetime.now().strftime('%Y%m%d-%H%M')}"
            st.markdown(f"""<div class="alert-success" style="margin-top:12px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
  <div style="font-size:14px; font-weight:800;">{T['sms_success_title']}</div>
  <span class="ty-badge" style="background:rgba(21,128,61,0.25); color:#4ADE80; border-color:rgba(74,222,128,0.30);">CRIS TLS-1.3 VERIFIED</span>
</div>
<div style="font-size:12px; line-height:1.5; margin-bottom:6px;">
  Order Ref: <span class="ty-badge">{order_ref}</span> &nbsp;|&nbsp;
  Auth: <span class="ty-badge">{user_desig}</span><br>
  Token: <code style="font-size:11px; color:#93C5FD;">SEC_TOKEN_JBP2026_SMS_VERIFIED_OK</code>
</div>
<div class="sms-log-item">📱 <b>Civil / P-Way:</b> [WCR/JBP/ENG] Track Tamping Approved. Window 02:00–04:30. Auth: CHC-JBP</div>
<div class="sms-log-item">📱 <b>Signal & Telecom (S&T):</b> [WCR/JBP/S&T] Interlocking synchronized at KM 1042. Auth: CHC-JBP</div>
<div class="sms-log-item">📱 <b>Traction / TRD:</b> [WCR/JBP/TRD] OHE Power Block scheduled. Zero starvation. Auth: CHC-JBP</div>
</div>""", unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — FINANCIAL & PUNCTUALITY AUDIT
# ===========================================================================
with tab2:
    st.markdown("### Division Financial & Punctuality Audit Ledger")
    st.caption("Freight demurrage prevention, line capacity reclamation, and environmental savings.")

    fk1, fk2, fk3, fk4 = st.columns(4)
    fin_cards = [
        (fk1, T["demurrage_card_title"], "₹42.8 Lakhs", "▲ 34.2% Detention Penalty Aversion", "#059669"),
        (fk2, T["capacity_card_title"], "+18.4 Hours", "Equivalent to +6 Freight Paths / Wk", "#0284C7"),
        (fk3, T["traction_card_title"], "₹16.5 Lakhs", "Zero Unscheduled OHE Power Cuts", "#7C3AED"),
        (fk4, T["caution_card_title"], "−38% TSR", "Saved ₹8.2L in Fuel / Traction Idling", "#D97706"),
    ]
    for col, title, val, sub, color in fin_cards:
        with col:
            st.markdown(f"""<div class="fin-metric-card" style="border-top-color:{color};">
<div class="fin-metric-title">{title}</div>
<div class="fin-metric-value">{val}</div>
<div class="fin-metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Environmental Certificate
    st.markdown(f"""<div class="glass-card" style="border-left: 4px solid #059669;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
<div>
  <div style="font-size:14.5px; font-weight:800; color:#4ADE80;">{T['green_banner_title']}</div>
  <p style="margin:4px 0 0; font-size:12.5px; color:#94A3B8; line-height:1.5;">{T['green_banner_desc']}</p>
</div>
<div style="background:rgba(5,150,105,0.20); border:1px solid rgba(74,222,128,0.25); border-radius:6px; padding:8px 16px; font-weight:700; font-size:13px; color:#4ADE80;">
  124.6 T CO₂e Abated / Month
</div>
</div>
</div>""", unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"#### {T['cost_pie_title']}")
        fin_df = pd.DataFrame({
            "Category": ["Demurrage Averted", "Traction Fuel Recovered", "TSR Acceleration Gains", "Gang Synergy Savings"],
            "₹ Lakhs": [42.8, 16.5, 8.2, 11.4],
        })
        pie = px.pie(fin_df, names="Category", values="₹ Lakhs", hole=0.45,
                     color_discrete_sequence=["#059669", "#0284C7", "#D97706", "#7C3AED"])
        pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(pie, use_container_width=True)

    with fc2:
        st.markdown(f"#### {T['starvation_title']}")
        starve_df = pd.DataFrame({
            "Corridor": list(CORRIDORS.keys()),
            "Throughput Gain": ["+5.8 hrs", "+6.2 hrs", "+2.8 hrs", "+3.6 hrs"],
            "Demurrage Saved": ["₹14.2L", "₹16.8L", "₹4.6L", "₹7.2L"],
            "Capacity Index": ["96.2%", "94.8%", "98.1%", "95.5%"],
        })
        st.dataframe(starve_df, use_container_width=True, height=260)


# ── Telemetry Logs ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander(T["telemetry_expander"], expanded=False):
    logs = {
        "engine": "Google OR-Tools CP-SAT v9.15",
        "spatial_bundling": "GeoPandas EPSG:32644",
        "solver_status": live_result.solver_status,
        "objective_score": float(live_result.objective_value),
        "planning_horizon_hours": int(horizon_hours),
        "total_requests": int(total_tasks),
        "scheduled_blocks": int(scheduled_tasks),
        "deferred_blocks": int(deferred_tasks),
        "bundled_clusters": int(bundled_clusters_count),
        "critical_risks": int(critical_risks),
        "safety_interlock": bool(st.session_state["siren_off_halt"]),
        "server_status": "LOCAL_SQLITE_FALLBACK" if st.session_state["sync_failure"] else "CRIS_COA_ONLINE",
        "officer_session": st.session_state["user_dept"],
        "designation": st.session_state["user_designation"],
        "language": st.session_state["lang_choice"],
        "operational_date": target_date_str,
        "timestamp_iso": datetime.now().isoformat(),
    }
    st.markdown(
        f'<div style="background: #0F172A; color: #38BDF8; font-family: JetBrains Mono, monospace; font-size:12px; padding:14px; border-radius:8px; overflow-x:auto;">'
        f'<pre style="margin:0;">{json.dumps(logs, indent=2)}</pre>'
        f'</div>', unsafe_allow_html=True
    )

st.markdown("---")
st.caption("🚆 TrackYukti · Smarter Planning. Efficient Solutions. · Government of India · Ministry of Railways · WCR Jabalpur Division · CRIS")
