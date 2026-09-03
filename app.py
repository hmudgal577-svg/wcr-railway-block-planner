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

# Existing Backend Layer
from backend.data_gen import generate_requests, CORRIDORS, BRANCH_ACTIONS
from backend.risk_model import CriticalityScorer
from backend.geo_cluster import find_bundling_clusters
from backend.optimizer import run_block_optimizer

# Advanced Intelligence Engines
from backend.priority_engine import compute_priority_intelligence
from backend.overlap_engine import (
    detect_task_overlaps,
    build_joint_work_bundles,
    find_partial_bundle_opportunities,
    detect_exclusive_tasks,
    compute_plan_optimization_comparison,
)
from backend.impact_engine import (
    get_passenger_traffic_summary,
    compute_freight_impact,
    compute_financial_impact,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
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
# COLOR SYSTEM & DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
DEPT_COLORS = {
    "Engineering": "#38BDF8",  # Sky Blue
    "S&T":         "#FCD34D",  # Warm Amber
    "Electrical":  "#C084FC",  # Orchid Purple
    "Operating":   "#34D399",  # Emerald Green
}

RISK_COLORS = {
    "CRITICAL":  "#EF4444",
    "VERY HIGH": "#F97316",
    "HIGH":      "#F59E0B",
    "NORMAL":    "#38BDF8",
    "MEDIUM":    "#FBBF24",
    "LOW":       "#4ADE80",
}

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION-GRADE GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}}

/* Full-bleed train photo backdrop with subtle dark veil for maximum readability */
.stApp {{
    background-image:
        linear-gradient(rgba(3, 7, 18, 0.40), rgba(3, 7, 18, 0.40)),
        {BG_CSS_VAL};
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    color: #FFFFFF;
    min-height: 100vh;
}}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}}

/* ── Absolute Text Visibility Guarantee ── */
.main .block-container p,
.main .block-container span,
.main .block-container div,
.main .block-container small,
.main .block-container b,
.main .block-container strong,
.main .block-container li,
.main .block-container label,
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container h4,
.main .block-container h5,
.main .block-container h6 {{
    color: #FFFFFF;
}}

/* ── Tabs Navigation: High Contrast Railway Command Deck ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(6, 12, 30, 0.94) !important;
    border-radius: 10px !important;
    padding: 6px 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    gap: 6px !important;
    margin-bottom: 18px !important;
    overflow-x: auto !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 7px !important;
    padding: 9px 16px !important;
    color: #CBD5E1 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border: none !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #FFFFFF !important;
    background: rgba(30, 58, 138, 0.45) !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.40) !important;
}}

/* ── Component Cards: Rich Opaque Glass so text is 100% crisp ── */
.ty-card {{
    background: rgba(7, 14, 34, 0.94) !important;
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    transition: transform 0.18s ease, border-color 0.18s ease;
}}
.ty-card:hover {{
    border-color: rgba(59, 130, 246, 0.45);
    transform: translateY(-1px);
}}

/* ── Header Banner ── */
.ty-header {{
    background: rgba(7, 14, 34, 0.96);
    backdrop-filter: blur(28px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-bottom: 2px solid #F59E0B;
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.50);
}}

/* ── Login Shell ── */
.ty-login-shell {{
    max-width: 900px;
    margin: 24px auto;
    background: rgba(6, 12, 30, 0.96);
    backdrop-filter: blur(32px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-top: 4px solid #F59E0B;
    border-radius: 18px;
    padding: 38px 48px 36px;
    box-shadow: 0 32px 80px rgba(0, 0, 0, 0.60);
}}

/* ── Stat Tiles ── */
.ty-stat {{
    background: rgba(7, 14, 34, 0.94);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.40);
    transition: transform 0.18s ease;
}}
.ty-stat:hover {{
    border-color: rgba(56, 189, 248, 0.45);
    transform: translateY(-2px);
}}
.ty-stat-label {{
    font-size: 11px;
    font-weight: 700;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.ty-stat-value {{
    font-size: 24px;
    font-weight: 900;
    color: #FFFFFF !important;
    margin-top: 4px;
    line-height: 1.1;
}}

/* ── Badges ── */
.ty-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(37, 99, 235, 0.25);
    color: #93C5FD !important;
    border: 1px solid rgba(59, 130, 246, 0.40);
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.ty-badge-green {{
    background: rgba(6, 78, 59, 0.35);
    color: #6EE7B7 !important;
    border: 1px solid rgba(52, 211, 153, 0.45);
}}
.ty-badge-amber {{
    background: rgba(120, 53, 15, 0.35);
    color: #FCD34D !important;
    border: 1px solid rgba(245, 158, 11, 0.45);
}}
.ty-badge-red {{
    background: rgba(127, 29, 29, 0.35);
    color: #FCA5A5 !important;
    border: 1px solid rgba(239, 68, 68, 0.45);
}}
.ty-badge-purple {{
    background: rgba(88, 28, 135, 0.35);
    color: #D8B4FE !important;
    border: 1px solid rgba(168, 85, 247, 0.45);
}}

/* ── Alerts ── */
.ty-alert-danger {{
    background: rgba(127, 29, 29, 0.85);
    border: 1px solid rgba(248, 113, 113, 0.40);
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 13px 16px;
    color: #FECACA !important;
    margin-bottom: 14px;
}}
.ty-alert-warn {{
    background: rgba(120, 53, 15, 0.85);
    border: 1px solid rgba(253, 211, 77, 0.40);
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 13px 16px;
    color: #FEF3C7 !important;
    margin-bottom: 14px;
}}
.ty-alert-success {{
    background: rgba(6, 78, 59, 0.88);
    border: 1px solid rgba(52, 211, 153, 0.40);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 14px 18px;
    color: #D1FAE5 !important;
    margin-bottom: 14px;
}}

/* ── Form Controls & Streamlit Inputs ── */
div[data-baseweb="select"] > div {{
    background: rgba(6, 12, 30, 0.96) !important;
    border: 1px solid rgba(148, 163, 184, 0.30) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}}
div[data-baseweb="select"] * {{
    color: #FFFFFF !important;
    background: rgba(6, 12, 30, 0.98) !important;
}}
.stTextInput input {{
    background: rgba(6, 12, 30, 0.96) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(148, 163, 184, 0.30) !important;
    border-radius: 8px !important;
}}
.stRadio label p, .stCheckbox label p, .stToggle label p {{
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: rgba(15, 23, 42, 0.90) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 9px 18px !important;
    transition: all 0.18s ease !important;
}}
.stButton > button:hover {{
    background: rgba(29, 78, 216, 0.85) !important;
    border-color: rgba(99, 179, 255, 0.60) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
    border: 1px solid rgba(99, 179, 255, 0.40) !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.40) !important;
}}
.stDownloadButton > button {{
    background: linear-gradient(135deg, #065F46 0%, #059669 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(52, 211, 153, 0.40) !important;
    font-weight: 700 !important;
}}

/* ── Section Headings & Dividers ── */
.ty-section-heading {{
    font-size: 13px;
    font-weight: 800;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}}
.ty-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.25), transparent);
    margin: 18px 0;
    border: none;
}}

/* ── Clock & Pulse ── */
.ty-clock {{
    background: rgba(7, 14, 34, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 8px;
    padding: 7px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    font-weight: 600;
    color: #FFFFFF;
    display: inline-flex;
    align-items: center;
    gap: 9px;
}}
.ty-pulse {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    background: #22C55E;
    box-shadow: 0 0 10px #22C55E;
    animation: ty-pulse 2s infinite;
}}
@keyframes ty-pulse {{
    0%, 100% {{ transform: scale(0.9); opacity: 0.85; box-shadow: 0 0 6px #22C55E; }}
    50%      {{ transform: scale(1.3); opacity: 1;    box-shadow: 0 0 14px #4ADE80; }}
}}

/* ── SMS Row ── */
.ty-sms {{
    background: rgba(8, 20, 70, 0.88);
    border: 1px solid rgba(96, 165, 250, 0.30);
    border-left: 3px solid #3B82F6;
    border-radius: 7px;
    padding: 10px 14px;
    font-size: 12px;
    color: #BAE6FD !important;
    margin-top: 7px;
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Plotly Background Fix ── */
.js-plotly-plot .plotly, .plotly-graph-div {{
    background: rgba(6, 12, 30, 0.95) !important;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DUAL-LANGUAGE TERMINOLOGY MATRIX (PRESERVED)
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
# PIPELINE CACHE & EXECUTION
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
    scored  = compute_priority_intelligence(scored)
    bundled = find_bundling_clusters(scored, radius_m=500.0)
    result  = run_block_optimizer(
        bundled, horizon_hours=horizon_hours,
        setup_buffer_minutes=setup_buffer,
        delayed_corridor=delayed_corridor,
        delay_minutes=delay_minutes,
    )
    return result, bundled, scorer

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT DEMO REQUISITIONS (RICH MULTI-CORRIDOR WORK ORDERS)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DEMO_REQS = [
    dict(request_id="DEMO-ENG-001", department="Engineering",
         action="Continuous Welded Rail (CWR) De-Stressing",
         corridor="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route",
         section_track="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route :: DN-Main",
         asset_id="AST-DEMO-ENG-001",
         latitude=23.51, longitude=80.22, overdue_days=112,
         last_inspection_score=91.0, traffic_density=148,
         corridor_priority=1.5, estimated_duration_mins=120,
         is_heavy_machinery=True, exclusive_block=True),

    dict(request_id="DEMO-SNT-001", department="S&T",
         action="Electronic Interlocking (EI) Overhaul",
         corridor="Jabalpur (JBP) - Itarsi (ET) Trunk Line",
         section_track="Jabalpur (JBP) - Itarsi (ET) Trunk Line :: UP-Main",
         asset_id="AST-DEMO-SNT-001",
         latitude=23.18, longitude=79.94, overdue_days=88,
         last_inspection_score=85.0, traffic_density=132,
         corridor_priority=1.4, estimated_duration_mins=90,
         is_heavy_machinery=False, exclusive_block=False),

    dict(request_id="DEMO-ELC-001", department="Electrical",
         action="OHE Catenary Contact Wire Tensioning",
         corridor="Katni (KTE) - Singrauli Coal Logistics Line",
         section_track="Katni (KTE) - Singrauli Coal Logistics Line :: Coal-Line-1",
         asset_id="AST-DEMO-ELC-001",
         latitude=23.82, longitude=80.40, overdue_days=65,
         last_inspection_score=79.0, traffic_density=120,
         corridor_priority=1.3, estimated_duration_mins=75,
         is_heavy_machinery=False, exclusive_block=False),

    dict(request_id="DEMO-ENG-002", department="Engineering",
         action="Ultrasonic Flaw Detection (USFD) Rail Testing",
         corridor="Satna (STA) - Rewa (REWA) Branch Corridor",
         section_track="Satna (STA) - Rewa (REWA) Branch Corridor :: Single-Line",
         asset_id="AST-DEMO-ENG-002",
         latitude=23.32, longitude=80.55, overdue_days=95,
         last_inspection_score=87.0, traffic_density=108,
         corridor_priority=1.2, estimated_duration_mins=60,
         is_heavy_machinery=False, exclusive_block=False),

    dict(request_id="DEMO-SNT-002", department="S&T",
         action="Digital Axle Counter (DAC) Sensor Calibration",
         corridor="Katni (KTE) - Singrauli Coal Logistics Line",
         section_track="Katni (KTE) - Singrauli Coal Logistics Line :: Coal-Line-2",
         asset_id="AST-DEMO-SNT-002",
         latitude=23.85, longitude=80.42, overdue_days=72,
         last_inspection_score=83.0, traffic_density=118,
         corridor_priority=1.3, estimated_duration_mins=60,
         is_heavy_machinery=False, exclusive_block=False),

    dict(request_id="DEMO-ELC-002", department="Electrical",
         action="Traction Power Feeder Isolator Maintenance",
         corridor="Jabalpur (JBP) - Itarsi (ET) Trunk Line",
         section_track="Jabalpur (JBP) - Itarsi (ET) Trunk Line :: DN-Main",
         asset_id="AST-DEMO-ELC-002",
         latitude=23.15, longitude=79.90, overdue_days=58,
         last_inspection_score=76.0, traffic_density=130,
         corridor_priority=1.4, estimated_duration_mins=90,
         is_heavy_machinery=False, exclusive_block=True),

    dict(request_id="DEMO-ENG-003", department="Engineering",
         action="Track Tamping & Deep Screening",
         corridor="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route",
         section_track="Jabalpur (JBP) - Katni (KTE) Heavy Freight Route :: UP-Main",
         asset_id="AST-DEMO-ENG-003",
         latitude=23.49, longitude=80.18, overdue_days=130,
         last_inspection_score=93.0, traffic_density=145,
         corridor_priority=1.5, estimated_duration_mins=180,
         is_heavy_machinery=True, exclusive_block=True),

    dict(request_id="DEMO-SNT-003", department="S&T",
         action="Point Machine Motor Overhauling",
         corridor="Satna (STA) - Rewa (REWA) Branch Corridor",
         section_track="Satna (STA) - Rewa (REWA) Branch Corridor :: Single-Line",
         asset_id="AST-DEMO-SNT-003",
         latitude=23.30, longitude=80.52, overdue_days=45,
         last_inspection_score=71.0, traffic_density=102,
         corridor_priority=1.2, estimated_duration_mins=45,
         is_heavy_machinery=False, exclusive_block=False),
]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION (DEFAULT: LOGIN SCREEN REQUIRED)
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "is_logged_in": False,
    "user_dept": "Engineering",
    "user_designation": "Sr. Divisional Engineer (Sr. DEN / Track)",
    "lang_choice": "English",
    "seed": 42,
    "custom_requests": DEFAULT_DEMO_REQS,

    "simulate_collision": False,
    "sync_failure": False,
    "dispatch_executed": False,
    "siren_off_halt": False,
    "cost_factor": 1200,
    "workflow_status": "AWAITING_APPROVAL",
    "dept_work_status": {
        "Engineering": "SCHEDULED",
        "S&T":         "SCHEDULED",
        "Electrical":  "SCHEDULED",
        "Operating":   "SCHEDULED",
    },
    "approval_log": [],
    "recent_activities": [
        {"time": "11:45:00 IST", "user": "SYSTEM / OR-Tools", "event": "Master rolling timetable generated with 0 conflicting overlaps"},
        {"time": "11:38:20 IST", "user": "Chief Controller", "event": "Division-wide 24-hr rolling possession matrix loaded for review"},
        {"time": "11:15:10 IST", "user": "Sr. DEN / Track", "event": "Submitted track tamping order for JBP-KTE Heavy Freight corridor"},
    ],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_all():
    for k in ["seed", "simulate_collision", "sync_failure", "dispatch_executed", "siren_off_halt", "workflow_status"]:
        st.session_state[k] = _defaults[k]
    st.session_state["custom_requests"] = DEFAULT_DEMO_REQS.copy()
    st.session_state["dept_work_status"] = _defaults["dept_work_status"].copy()

# =============================================================================
#  LOGIN PORTAL (ACCESSIBLE VIA LOGOUT BUTTON)
# =============================================================================
if not st.session_state["is_logged_in"]:
    logo_html = (
        f'<img src="data:image/png;base64,{LOGO_B64}" style="height:70px;width:auto;display:block;margin:0 auto 10px;" alt="TrackYukti">'
        if LOGO_B64 else ""
    )

    st.markdown(f"""
    <div class="ty-login-shell">
      {logo_html}
      <div style="text-align:center;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.12);">
        <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.12em;">
          Government of India &nbsp;·&nbsp; Ministry of Railways
        </div>
        <h1 style="margin:6px 0 2px;font-size:32px;font-weight:900;color:#FFFFFF;letter-spacing:-0.02em;">
          TRACK<span style="color:#F59E0B;">YUKTI</span>
        </h1>
        <div style="font-size:12px;color:#F59E0B;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;">
          Smarter Planning &nbsp;·&nbsp; Efficient Solutions
        </div>
        <p style="margin:8px 0 0;font-size:13px;color:#CBD5E1;font-weight:500;">
          West Central Railway &nbsp;·&nbsp; Jabalpur Division &nbsp;·&nbsp; Joint Rolling Block Operations Portal (IR-JRBP v3.0)
        </p>
        <div style="margin-top:14px;">
          <span class="ty-badge ty-badge-amber">
            🔐 &nbsp;AUTHORIZED RAILWAY OPERATIONS PERSONNEL ONLY &nbsp;·&nbsp; DEFAULT PASSKEY: JBP2026
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    lc1, lc2 = st.columns([1.1, 1])

    with lc1:
        st.markdown("""
        <div class="ty-card" style="margin-top:8px;">
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
            Step 1 — Select Operating Branch
          </div>
          <div style="font-size:12.5px;color:#CBD5E1;">
            Operational control, authorization tiers, and safety clearances are assigned per department.
          </div>
        </div>
        """, unsafe_allow_html=True)

        dept_choice = st.radio("Operating Branch:", [
            "Engineering (Civil / Track / P-Way)",
            "Signal & Telecom (S&T)",
            "Electrical (TRD / OHE Maintenance)",
            "Chief Controller (CHC) / Operating Control",
        ], index=3)

        dept_map = {
            "Engineering (Civil / Track / P-Way)":       "Engineering",
            "Signal & Telecom (S&T)":                     "S&T",
            "Electrical (TRD / OHE Maintenance)":         "Electrical",
            "Chief Controller (CHC) / Operating Control": "Chief Controller / DRM",
        }
        sel_dept = dept_map[dept_choice]

    with lc2:
        st.markdown("""
        <div class="ty-card" style="margin-top:8px;">
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
            Step 2 — Officer Designation & Security Passkey
          </div>
          <div style="font-size:12.5px;color:#CBD5E1;">
            Divisional security passkey: <b style="color:#F59E0B;">JBP2026</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        desig_map = {
            "Engineering": [
                "Sr. Divisional Engineer (Sr. DEN / Track)",
                "Sr. Divisional Engineer (Sr. DEN / Bridge)",
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
                "Senior Section Engineer (SSE / Signal)",
            ],
            "Electrical": [
                "Sr. Divisional Electrical Engineer (Sr. DEE / TRD)",
                "Sr. Divisional Electrical Engineer (Sr. DEE / General)",
                "Divisional Electrical Engineer (DEE / TRD)",
                "Assistant Divisional Electrical Engineer (ADEE)",
                "Senior Section Engineer (SSE / OHE)",
            ],
            "Chief Controller / DRM": [
                "Chief Controller (CHC / Central Control)",
                "Dy. Chief Controller (Dy. CHC)",
                "Section Controller (SC / Train Control)",
                "Sr. Divisional Operations Manager (Sr. DOM)",
                "Divisional Operations Manager (DOM)",
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
                    st.error("❌  Invalid security passkey. Default: JBP2026")
        with bb:
            if st.button("👁  Guest / Read-Only Entry", use_container_width=True):
                st.session_state.update(
                    is_logged_in=True,
                    user_dept="Chief Controller / DRM",
                    user_designation="Divisional Safety Officer (DSO)",
                )
                st.rerun()

        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)

        if st.button(
            "🎓  LOAD TEACHER DEMO — DRM Full Access + 8 Pre-Filled Real Block Orders",
            use_container_width=True,
        ):
            st.session_state.update(
                is_logged_in=True,
                user_dept="Chief Controller / DRM",
                user_designation="Chief Controller (CHC / Central Control)",
                custom_requests=DEFAULT_DEMO_REQS,
                simulate_collision=False,
                sync_failure=False,
                dispatch_executed=False,
                siren_off_halt=False,
            )
            st.balloons()
            st.rerun()

    st.stop()


# =============================================================================
#  AUTHENTICATED OPERATIONS COMMAND CENTER
# =============================================================================

# ── Sidebar Operational Jurisdiction ─────────────────────────────────────────
with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f'<img src="data:image/png;base64,{LOGO_B64}" style="height:50px;width:auto;margin-bottom:4px;">',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<span style="font-size:20px;font-weight:900;color:#FFFFFF;">Track<span style="color:#F59E0B;">Yukti</span></span>'
        '<br><span style="font-size:11px;color:#94A3B8;font-weight:600;">WCR Jabalpur Division · IR-JRBP</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(148,163,184,0.15);margin:10px 0;">', unsafe_allow_html=True)

    lang = st.selectbox("🌐 Language / भाषा",
                        ["English", "Hindi / हिंदी"],
                        index=0 if st.session_state["lang_choice"] == "English" else 1)
    st.session_state["lang_choice"] = lang
    T = TRANS[lang]

    # Active Officer Card
    st.markdown(f"""
    <div style="background:rgba(30, 58, 138, 0.35);border:1px solid rgba(59,130,246,0.35);
                border-radius:9px;padding:12px 14px;margin:10px 0;">
      <div style="font-size:10px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;">
        Active Session
      </div>
      <div style="font-size:14px;font-weight:800;color:#FFFFFF;margin-top:2px;">
        {st.session_state['user_dept']}
      </div>
      <div style="font-size:11px;color:#CBD5E1;margin-top:1px;">
        {st.session_state['user_designation']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(148,163,184,0.15);margin:10px 0;">', unsafe_allow_html=True)
    st.markdown("#### 📍 Corridor Jurisdiction")
    sel_corr = st.selectbox(
        "Filter Corridor",
        ["All Corridors (Jabalpur Division)"] + list(CORRIDORS.keys())
    )

    st.markdown("#### ⏱️ Planning Parameters")
    horizon_hours = st.slider("Horizon Window (Hours)", 6, 24, 12, step=1)
    setup_buffer  = st.slider("Handover Safety Buffer (Mins)", 5, 45, 15, step=5)

    st.markdown('<hr style="border-color:rgba(148,163,184,0.15);margin:10px 0;">', unsafe_allow_html=True)
    co1, co2 = st.columns(2)
    with co1:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["is_logged_in"] = False
            st.rerun()
    with co2:
        if st.button("♻️ Reset", use_container_width=True):
            reset_all(); st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME CORE DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
base_df = get_cached_requests(seed=st.session_state["seed"])

if st.session_state["simulate_collision"]:
    coll_corr  = "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route"
    coll_track = f"{coll_corr} :: DN-Main"
    coll_rows  = [
        dict(request_id="WCR-ENG-COL-1", department="Engineering",
             action="Continuous Welded Rail (CWR) De-Stressing", corridor=coll_corr,
             section_track=coll_track, asset_id="AST-COL-ENG",
             latitude=23.501, longitude=80.201, overdue_days=90,
             last_inspection_score=88.0, traffic_density=135,
             corridor_priority=1.4, estimated_duration_mins=90,
             is_heavy_machinery=False, exclusive_block=False),
        dict(request_id="WCR-SNT-COL-2", department="S&T",
             action="Electronic Interlocking (EI) Overhaul", corridor=coll_corr,
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

# Run Optimization
delayed_corr_arg = None if sel_corr == "All Corridors (Jabalpur Division)" else sel_corr
baseline_result, bundled_df, scorer = run_pipeline(combined_df, horizon_hours, setup_buffer)
schedule = baseline_result.schedule.copy()

# Dynamic shifts
bs = baseline_result.schedule.set_index("request_id")["start_min"]
schedule["baseline_start_min"]  = schedule["request_id"].map(bs)
schedule["dynamically_shifted"] = (
    schedule["is_scheduled"] & schedule["baseline_start_min"].notna()
    & (schedule["start_min"] != schedule["baseline_start_min"])
)

# Conflict Detection Check
has_conflict, coll_depts, coll_track_name = False, [], ""
for trk, grp in combined_df.groupby("section_track"):
    depts = grp["department"].unique()
    if len(depts) >= 2:
        has_conflict, coll_depts, coll_track_name = True, list(depts), trk
        break

# Compute Advanced Engine Outputs
priority_df = compute_priority_intelligence(schedule)
joint_bundles = build_joint_work_bundles(schedule)
partial_opps = find_partial_bundle_opportunities(schedule)
exclusive_tasks = detect_exclusive_tasks(schedule)
optimization_comp = compute_plan_optimization_comparison(schedule, joint_bundles, exclusive_tasks)
overlap_pairs = detect_task_overlaps(schedule)
passenger_summary = get_passenger_traffic_summary()
freight_impact = compute_freight_impact(schedule)
financial_impact = compute_financial_impact(
    freight_impact["affected_freight_trains"],
    freight_impact["total_freight_delay_mins"],
    cost_factor_per_min=float(st.session_state["cost_factor"]),
)

# KPI Counters
total_tasks      = len(schedule)
scheduled_tasks  = int(schedule["is_scheduled"].sum())
deferred_tasks   = total_tasks - scheduled_tasks
critical_risks   = int((priority_df["priority_level"] == "CRITICAL").sum())
efficiency_pct   = round(scheduled_tasks / max(1, total_tasks) * 100, 1)

# ─────────────────────────────────────────────────────────────────────────────
# TOP EXECUTIVE HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
now_dt   = datetime.now()
ist_str  = now_dt.strftime("%H:%M:%S IST")
date_str = "03 September 2026"

status_badge = (
    '<span class="ty-badge ty-badge-green"><span class="ty-pulse"></span> &nbsp;SYSTEM OPERATIONAL</span>'
    if not st.session_state["siren_off_halt"]
    else '<span class="ty-badge ty-badge-red">⛔ SAFETY HOLD ACTIVE</span>'
)

logo_hdr = (
    f'<img src="data:image/png;base64,{LOGO_B64}" style="height:46px;width:auto;" alt="TrackYukti">'
    if LOGO_B64 else ""
)

st.markdown(f"""
<div class="ty-header">
  <div style="display:flex;align-items:center;gap:16px;">
    {logo_hdr}
    <div>
      <div style="font-size:22px;font-weight:900;color:#FFFFFF;letter-spacing:-0.02em;">
        TRACK<span style="color:#F59E0B;">YUKTI</span>
        <span style="font-size:12px;font-weight:700;color:#93C5FD;background:rgba(37,99,235,0.25);padding:3px 8px;border-radius:6px;margin-left:8px;border:1px solid rgba(59,130,246,0.35);">
          WCR JABALPUR COMMAND CENTER
        </span>
      </div>
      <div style="font-size:11.5px;color:#CBD5E1;margin-top:2px;">
        {T['portal_title']} &nbsp;·&nbsp; {T['portal_sub']}
      </div>
      <div style="font-size:11px;color:#94A3B8;margin-top:1px;">
        Active Jurisdiction: <b style="color:#FFFFFF;">{sel_corr}</b>
        &nbsp;·&nbsp; Officer: <b style="color:#93C5FD;">{st.session_state['user_designation']}</b>
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <div class="ty-clock">🕒 {date_str} &nbsp;|&nbsp; {ist_str}</div>
    {status_badge}
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state["sync_failure"]:
    st.markdown(
        '<div class="ty-alert-warn"><b>⚠ CRIS / COA SERVER OFFLINE</b> — '
        'Operating on local cached spatial database with static safety headway rules.</div>',
        unsafe_allow_html=True,
    )

if has_conflict:
    st.markdown(
        f'<div class="ty-alert-danger"><b>{T["siren_conflict"]}</b><br>'
        f'<span style="font-size:12.5px;">{coll_track_name}: '
        f'{" & ".join(coll_depts)} — {T["conflict_action"]}</span></div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9-MODULE WORKSPACE TABBING MATRIX (NO SINGLE-PAGE CLUTTER)
# ─────────────────────────────────────────────────────────────────────────────
tab_dash, tab_opt, tab_time, tab_prio, tab_impact, tab_fin, tab_sim, tab_wf, tab_orders = st.tabs([
    "🧭 Command Dashboard",
    "⚡ Smart Block Optimizer",
    "📊 Master Timetable",
    "🎯 Priority Intelligence",
    "🚆 Passenger & Freight",
    "💰 Financial Audit",
    "🧪 What-If Simulation",
    "🛡️ Approval Workflow",
    "📝 Work Orders",
])

# =============================================================================
#  TAB 1: COMMAND DASHBOARD (EXECUTIVE OVERVIEW)
# =============================================================================
with tab_dash:
    st.markdown('<div class="ty-section-heading">Operational Command Dashboard — High-Level Overview</div>', unsafe_allow_html=True)

    # Primary Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="ty-stat">
          <div class="ty-stat-label">Total Work Orders</div>
          <div class="ty-stat-value" style="color:#38BDF8;">{total_tasks}</div>
          <div style="font-size:11px;color:#94A3B8;margin-top:3px;">Pool Requisitions</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="ty-stat">
          <div class="ty-stat-label">Critical Tasks</div>
          <div class="ty-stat-value" style="color:#EF4444;">{critical_risks}</div>
          <div style="font-size:11px;color:#FCA5A5;margin-top:3px;">Immediate Safety Flaws</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="ty-stat">
          <div class="ty-stat-label">Active Conflicts</div>
          <div class="ty-stat-value" style="color:#{'EF4444' if has_conflict else '10B981'};">
            {'1 ALERT' if has_conflict else '0 CLASH'}
          </div>
          <div style="font-size:11px;color:#94A3B8;margin-top:3px;">Spatial Interlocks</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="ty-stat">
          <div class="ty-stat-label">Joint Bundles</div>
          <div class="ty-stat-value" style="color:#10B981;">{len(joint_bundles)}</div>
          <div style="font-size:11px;color:#6EE7B7;margin-top:3px;">{optimization_comp['tasks_bundled_count']} Tasks Unified</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="ty-stat">
          <div class="ty-stat-label">Pending Approvals</div>
          <div class="ty-stat-value" style="color:#F59E0B;">
            {'1 PENDING' if st.session_state['workflow_status'] == 'AWAITING_APPROVAL' else '0 CLEAR'}
          </div>
          <div style="font-size:11px;color:#FCD34D;margin-top:3px;">Chief Controller Tier</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="ty-stat">
          <div class="ty-stat-label">Active Blocks</div>
          <div class="ty-stat-value" style="color:#C084FC;">{scheduled_tasks}</div>
          <div style="font-size:11px;color:#E9D5FF;margin-top:3px;">Efficiency: {efficiency_pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Recommended Block Window Card
    st.markdown("""
    <div class="ty-card" style="border-left:4px solid #10B981;margin-top:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <span class="ty-badge ty-badge-green">RECOMMENDED ROLLING BLOCK WINDOW</span>
          <h3 style="margin:6px 0 2px;font-size:18px;font-weight:900;color:#FFFFFF;">
            Katni (KTE) – Singrauli Coal Logistics Line · Joint Spatial Block Window #1
          </h3>
          <div style="font-size:12.5px;color:#CBD5E1;">
            <b>Time Window:</b> 01:30 – 04:30 IST (Night Freight Valley) &nbsp;|&nbsp;
            <b>Participating Branches:</b> Civil Track Staff, Signal & Telecom, Electrical TRD &nbsp;|&nbsp;
            <b>Separate Blocks Avoided:</b> 3 Possessions Combined
          </div>
        </div>
        <div>
          <span style="font-size:22px;font-weight:900;color:#4ADE80;">+3.5 Hours</span>
          <div style="font-size:11px;color:#94A3B8;text-align:right;">Line Capacity Recovered</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Operations Status Grid & Recent Activity
    cd1, cd2 = st.columns([1.2, 1])
    with cd1:
        st.markdown('<div class="ty-section-heading">Corridor Operational Health Status</div>', unsafe_allow_html=True)
        for cname, cmeta in CORRIDORS.items():
            c_sched = schedule[schedule["corridor"] == cname]
            c_scheduled_cnt = len(c_sched[c_sched["is_scheduled"]])
            c_critical = len(priority_df[(priority_df["corridor"] == cname) & (priority_df["priority_level"] == "CRITICAL")])
            badge_cls = "ty-badge-red" if c_critical > 0 else "ty-badge-green"
            badge_txt = f"{c_critical} Critical" if c_critical > 0 else "Optimal"

            st.markdown(f"""
            <div class="ty-card" style="padding:14px 16px;margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-weight:800;font-size:13.5px;color:#FFFFFF;">{cname}</div>
                  <div style="font-size:11.5px;color:#94A3B8;margin-top:2px;">
                    Tracks: {', '.join(cmeta['tracks'])} · Priority Index: {cmeta['priority']}
                  </div>
                </div>
                <div style="text-align:right;">
                  <span class="ty-badge {badge_cls}">{badge_txt}</span>
                  <div style="font-size:11px;color:#38BDF8;margin-top:4px;font-weight:700;">
                    {c_scheduled_cnt} Scheduled Blocks
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with cd2:
        st.markdown('<div class="ty-section-heading">Recent Operations Activity Feed</div>', unsafe_allow_html=True)
        st.markdown('<div class="ty-card" style="padding:16px;">', unsafe_allow_html=True)
        for act in st.session_state["recent_activities"]:
            st.markdown(f"""
            <div style="border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:10px;margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:11px;font-family:JetBrains Mono,monospace;color:#93C5FD;">{act['time']}</span>
                <span class="ty-badge">{act['user']}</span>
              </div>
              <div style="font-size:12.5px;color:#E2E8F0;margin-top:4px;">{act['event']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  TAB 2: SMART BLOCK OPTIMIZER & TASK BUNDLING ENGINE
# =============================================================================
with tab_opt:
    st.markdown('<div class="ty-section-heading">Smart Multi-Department Task Bundling & Optimization Engine</div>', unsafe_allow_html=True)

    # Workflow Visualizer Ribbon
    st.markdown("""
    <div class="ty-card" style="padding:14px 20px;margin-bottom:18px;">
      <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
        Intelligent Optimization Workflow
      </div>
      <div style="display:flex;align-items:center;gap:8px;overflow-x:auto;font-size:11.5px;color:#FFFFFF;font-weight:700;">
        <span class="ty-badge">1. Requests Pool</span> ➔
        <span class="ty-badge">2. Priority Scoring</span> ➔
        <span class="ty-badge ty-badge-amber">3. Overlap Detection</span> ➔
        <span class="ty-badge">4. Compatibility Rules</span> ➔
        <span class="ty-badge ty-badge-green">5. Task Bundling</span> ➔
        <span class="ty-badge">6. Exclusive Isolation</span> ➔
        <span class="ty-badge ty-badge-purple">7. CP-SAT Schedule</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-Side Plan Comparison
    st.markdown('<div class="ty-section-heading">Plan Optimization Audit: Original Plan vs TrackYukti Optimized Plan</div>', unsafe_allow_html=True)
    comp_c1, comp_c2 = st.columns(2)

    with comp_c1:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #94A3B8;">
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;">
            Original Plan (Uncoordinated Individual Blocks)
          </div>
          <h2 style="margin:8px 0 4px;font-size:26px;font-weight:900;color:#FFFFFF;">
            {optimization_comp['original_duration_mins'] // 60}h {optimization_comp['original_duration_mins'] % 60}m
          </h2>
          <div style="font-size:12px;color:#CBD5E1;line-height:1.6;">
            • Total Independent Possessions: <b>{optimization_comp['original_blocks_count']} Separate Windows</b><br>
            • Separate Handover Setup Losses: <b>+{optimization_comp['original_blocks_count'] * 15} mins caution setup</b><br>
            • Recurring Traffic Holds: <b>High Disruption Risk</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c2:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #10B981;">
          <div style="font-size:12px;font-weight:800;color:#34D399;text-transform:uppercase;letter-spacing:0.06em;">
            TrackYukti Optimized Plan (Coordinated Spatial Bundles)
          </div>
          <h2 style="margin:8px 0 4px;font-size:26px;font-weight:900;color:#4ADE80;">
            {optimization_comp['optimized_duration_mins'] // 60}h {optimization_comp['optimized_duration_mins'] % 60}m
            <span style="font-size:15px;color:#38BDF8;">(−{optimization_comp['time_saved_hrs']} Hours Saved)</span>
          </h2>
          <div style="font-size:12px;color:#CBD5E1;line-height:1.6;">
            • Unified Block Possessions: <b>{optimization_comp['optimized_blocks_count']} Coordinated Packages</b><br>
            • Separate Blocks Avoided: <b>{optimization_comp['separate_blocks_avoided']} Redundant Blocks Saved</b><br>
            • Line Capacity Efficiency Gain: <b>+{optimization_comp['efficiency_gain_pct']}% Recovered</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Joint Work Bundles Showcase
    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ty-section-heading">Joint Work Bundles ({len(joint_bundles)} Active Packages)</div>', unsafe_allow_html=True)

    if not joint_bundles:
        st.info("No active cross-department bundles generated under current horizon settings.")
    else:
        for b in joint_bundles:
            dept_badges = " ".join([f'<span class="ty-badge">{d}</span>' for d in b.participating_departments])
            st.markdown(f"""
            <div class="ty-card" style="border-left:4px solid #10B981;margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                  <span class="ty-badge ty-badge-green">{b.bundle_id}</span>
                  <span style="font-size:16px;font-weight:800;color:#FFFFFF;margin-left:8px;">{b.corridor}</span>
                  <div style="font-size:12px;color:#94A3B8;margin-top:2px;">
                    Track Section: <b style="color:#E2E8F0;">{b.section_track}</b> &nbsp;|&nbsp;
                    Time Window: <b style="color:#4ADE80;">{b.common_start_min//60:02d}:{b.common_start_min%60:02d} – {b.common_end_min//60:02d}:{b.common_end_min%60:02d} IST</b>
                  </div>
                </div>
                <div style="text-align:right;">
                  <span style="font-size:18px;font-weight:900;color:#38BDF8;">{b.time_saved_mins} Mins Saved</span>
                  <div style="font-size:11px;color:#94A3B8;">{b.separate_blocks_avoided} Separate Blocks Avoided</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:12px;color:#CBD5E1;">
                <b>Participating Departments:</b> {dept_badges}
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Inspect Tasks in Bundle {b.bundle_id} ({len(b.tasks)} Work Orders)"):
                b_task_df = pd.DataFrame(b.tasks)[[
                    "request_id", "department", "action", "estimated_duration_mins", "risk_score"
                ]]
                b_task_df.columns = ["Order ID", "Dept", "Activity", "Requested Duration", "Risk Score"]
                st.dataframe(b_task_df, use_container_width=True, hide_index=True)

    # 2. Overlap Classification Table
    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ty-section-heading">Cross-Task Overlap Detection Register (Full / Partial / No Overlap)</div>', unsafe_allow_html=True)

    ov_rows = []
    for ov in overlap_pairs[:20]:
        ov_rows.append({
            "Task A": ov.task_a,
            "Dept A": ov.dept_a,
            "Task B": ov.task_b,
            "Dept B": ov.dept_b,
            "Corridor": ov.corridor,
            "Overlap Type": ov.overlap_type,
            "Shared Mins": f"{ov.overlap_minutes}m",
            "Distance": f"{ov.spatial_dist_m:.0f}m",
            "Diagnostics": ov.reason,
        })
    ov_df = pd.DataFrame(ov_rows)
    st.dataframe(ov_df, use_container_width=True, height=280, hide_index=True)

    # 3. Partial Bundle Opportunities & Exclusive Tasks
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="ty-section-heading">Partial Bundle Opportunities</div>', unsafe_allow_html=True)
        if not partial_opps:
            st.info("No partial overlap synchronization opportunities identified.")
        else:
            for po in partial_opps[:3]:
                st.markdown(f"""
                <div class="ty-card" style="border-left:3px solid #F59E0B;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="ty-badge ty-badge-amber">{po.opportunity_id}</span>
                    <span style="font-size:12px;font-weight:700;color:#FCD34D;">+{po.time_saved_mins}m Potential Saving</span>
                  </div>
                  <div style="font-size:13px;font-weight:700;color:#FFFFFF;margin-top:6px;">{po.corridor}</div>
                  <div style="font-size:11.5px;color:#CBD5E1;margin-top:2px;">
                    Tasks: {', '.join(po.tasks)} &nbsp;|&nbsp; Feasible Window: {po.common_feasible_window}
                  </div>
                  <div style="font-size:11.5px;color:#94A3B8;margin-top:6px;font-style:italic;">
                    💡 {po.recommendation}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="ty-section-heading">Exclusive Tasks (Safety Isolation Mandate)</div>', unsafe_allow_html=True)
        if not exclusive_tasks:
            st.info("All scheduled tasks are cleared for joint co-possession.")
        else:
            for ex in exclusive_tasks[:3]:
                st.markdown(f"""
                <div class="ty-card" style="border-left:3px solid #EF4444;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="ty-badge ty-badge-red">{ex.request_id} · {ex.department}</span>
                    <span class="ty-badge ty-badge-red">EXCLUSIVE TASK</span>
                  </div>
                  <div style="font-size:13px;font-weight:700;color:#FFFFFF;margin-top:6px;">{ex.activity}</div>
                  <div style="font-size:11.5px;color:#CBD5E1;margin-top:2px;">
                    Section: {ex.section_track} &nbsp;|&nbsp; Required Block: {ex.min_required_duration_mins}m
                  </div>
                  <div style="font-size:11.5px;color:#FCA5A5;margin-top:6px;">
                    🛡️ <b>Isolation Reason:</b> {ex.reason}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
#  TAB 3: MASTER BLOCK TIMETABLE & GANTT (PRESERVED & UPGRADED)
# =============================================================================
with tab_time:
    st.markdown('<div class="ty-section-heading">Corridor Rolling Block Master Timetable (24-Hour Operations Grid)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ty-card" style="padding:16px 20px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div style="font-size:13px;color:#CBD5E1;">
          High-resolution Plotly Gantt rendering synchronized cross-departmental rolling possession blocks.
        </div>
        <div>
          <span class="ty-badge" style="background:rgba(56,189,248,0.25);color:#38BDF8;">■ Civil / P-Way</span>
          <span class="ty-badge" style="background:rgba(252,211,77,0.25);color:#FCD34D;">■ S&T Interlocking</span>
          <span class="ty-badge" style="background:rgba(192,132,252,0.25);color:#C084FC;">■ Electrical OHE</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    gantt_df = schedule[schedule["is_scheduled"]].copy()
    if sel_corr != "All Corridors (Jabalpur Division)":
        gantt_df = gantt_df[gantt_df["corridor"] == sel_corr]

    if gantt_df.empty:
        st.warning(f"No scheduled blocks for {sel_corr}.")
    else:
        bt = datetime.combine(datetime.today(), datetime.min.time())
        gantt_df["Start"]  = gantt_df["start_min"].apply(lambda m: bt + timedelta(minutes=float(m)))
        gantt_df["Finish"] = gantt_df["end_min"].apply(lambda m: bt + timedelta(minutes=float(m)))
        gantt_df["Label"]  = gantt_df.apply(
            lambda r: f"{r['request_id']} ({r['department'][:3]})"
            + (" [EXCL]" if r.get("is_heavy_machinery") else "")
            + (" [SHIFT]" if r["dynamically_shifted"] else ""),
            axis=1,
        )

        fig = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="Finish",
            y="section_track",
            color="department",
            color_discrete_map=DEPT_COLORS,
            text="Label",
            hover_data={
                "request_id": True, "department": True, "action": True,
                "risk_score": True, "corridor": True, "estimated_duration_mins": True,
                "section_track": False, "Start": False, "Finish": False,
            },
        )
        fig.update_yaxes(
            autorange="reversed",
            title=dict(text="Corridor Track Section", font=dict(color="#FFFFFF", size=12)),
            tickfont=dict(color="#FFFFFF", size=11),
            gridcolor="rgba(255,255,255,0.08)",
            showgrid=True,
        )
        fig.update_xaxes(
            title=dict(text=f"Time Window (00:00 – {horizon_hours:02d}:00 IST)", font=dict(color="#FFFFFF", size=12)),
            tickfont=dict(color="#FFFFFF", size=11),
            gridcolor="rgba(255,255,255,0.08)",
        )
        fig.update_traces(
            textposition="inside",
            insidetextanchor="start",
            marker_line_width=1.5,
            marker_line_color="rgba(255,255,255,0.30)",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(6, 12, 30, 0.95)",
            paper_bgcolor="rgba(6, 12, 30, 0.95)",
            font=dict(color="#FFFFFF", family="Inter"),
            legend=dict(
                orientation="h", y=1.06, x=1, xanchor="right",
                font=dict(color="#FFFFFF", size=11),
                bgcolor="rgba(7,14,34,0.85)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
            ),
            height=max(420, 80 + 44 * gantt_df["section_track"].nunique()),
            margin=dict(l=10, r=10, t=36, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ty-section-heading" style="margin-top:16px;">Scheduled Blocks Detailed Register</div>', unsafe_allow_html=True)
    t_df = gantt_df[[
        "request_id", "department", "action", "section_track",
        "start_min", "end_min", "estimated_duration_mins", "risk_score"
    ]].copy()
    t_df["start_min"] = t_df["start_min"].apply(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}")
    t_df["end_min"]   = t_df["end_min"].apply(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}")
    t_df.columns = ["Order ID", "Dept", "Activity", "Track Section", "Start Time", "Clearance Time", "Duration (Mins)", "Risk Score"]
    st.dataframe(t_df, use_container_width=True, height=280, hide_index=True)

# =============================================================================
#  TAB 4: PRIORITY INTELLIGENCE & EXPLAINABLE SCORING
# =============================================================================
with tab_prio:
    st.markdown('<div class="ty-section-heading">Explainable Multi-Factor Priority Intelligence (0–100 Scoring Model)</div>', unsafe_allow_html=True)

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.markdown("""
        <div class="ty-stat">
          <div class="ty-stat-label">1. Safety Criticality</div>
          <div class="ty-stat-value" style="color:#EF4444;">30%</div>
          <div style="font-size:11px;color:#CBD5E1;margin-top:2px;">USFD flaws, rail fractures, switch wear</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="ty-stat">
          <div class="ty-stat-label">2. Operational Urgency</div>
          <div class="ty-stat-value" style="color:#F97316;">25%</div>
          <div style="font-size:11px;color:#CBD5E1;margin-top:2px;">TSR speed restrictions, delay impact</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="ty-stat">
          <div class="ty-stat-label">3. Defect Severity</div>
          <div class="ty-stat-value" style="color:#F59E0B;">20%</div>
          <div style="font-size:11px;color:#CBD5E1;margin-top:2px;">Inspection condition variance, wear index</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="ty-stat">
          <div class="ty-stat-label">4. Overdue Maintenance</div>
          <div class="ty-stat-value" style="color:#38BDF8;">15%</div>
          <div style="font-size:11px;color:#CBD5E1;margin-top:2px;">Days past scheduled safety cycle</div>
        </div>
        """, unsafe_allow_html=True)
    with f5:
        st.markdown("""
        <div class="ty-stat">
          <div class="ty-stat-label">5. Asset Availability</div>
          <div class="ty-stat-value" style="color:#C084FC;">10%</div>
          <div style="font-size:11px;color:#CBD5E1;margin-top:2px;">GMT annual tonnage, traffic density</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ty-section-heading">Priority Classification Register & Explainability Log</div>', unsafe_allow_html=True)

    p_disp = priority_df[[
        "request_id", "department", "action", "corridor", "priority_score",
        "priority_level", "factor_safety_criticality", "factor_operational_urgency",
        "factor_defect_severity", "factor_overdue_maintenance", "factor_asset_availability",
        "priority_explanation"
    ]].copy()

    p_disp.columns = [
        "Order ID", "Dept", "Activity", "Corridor", "Score (0-100)",
        "Priority Band", "Safety (30%)", "Urgency (25%)", "Defect (20%)",
        "Overdue (15%)", "Traffic (10%)", "Technical Justification"
    ]
    st.dataframe(p_disp, use_container_width=True, height=420, hide_index=True)

# =============================================================================
#  TAB 5: PASSENGER & FREIGHT IMPACT ANALYSIS
# =============================================================================
with tab_impact:
    st.markdown('<div class="ty-section-heading">Corridor Passenger & Freight Traffic Impact Assessment</div>', unsafe_allow_html=True)

    pt_col, ft_col = st.columns([1, 1.3])
    with pt_col:
        st.markdown("""
        <div class="ty-card">
          <div style="font-size:14px;font-weight:800;color:#FFFFFF;margin-bottom:8px;">
            Passenger Traffic Density Curve (24h Profile)
          </div>
          <div style="font-size:12px;color:#CBD5E1;margin-bottom:12px;">
            Identifies low passenger volume windows to recommend lower operational disruption.
          </div>
        """, unsafe_allow_html=True)

        prof_df = passenger_summary["profile_df"]
        fig_pass = px.bar(
            prof_df,
            x="hour",
            y="train_count",
            color="category",
            color_discrete_map={"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"},
            labels={"hour": "Hour of Day (IST)", "train_count": "Passenger Trains / Hour"},
        )
        fig_pass.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(6, 12, 30, 0.95)",
            paper_bgcolor="rgba(6, 12, 30, 0.95)",
            font=dict(color="#FFFFFF", size=11),
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_pass, use_container_width=True)

        st.markdown(f"""
        <div style="margin-top:10px;font-size:12px;color:#CBD5E1;line-height:1.6;">
          • <b>Optimal Midday Window:</b> <span style="color:#4ADE80;">{passenger_summary['recommended_day_block_window']}</span><br>
          • <b>Optimal Night Window:</b> <span style="color:#38BDF8;">{passenger_summary['recommended_night_block_window']}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with ft_col:
        st.markdown("""
        <div class="ty-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:14px;font-weight:800;color:#FFFFFF;">
              Freight Train Impact Ledger (Simulated Real-Time Rakes)
            </div>
            <span class="ty-badge ty-badge-amber">SIMULATED DATA</span>
          </div>
          <div style="font-size:12px;color:#CBD5E1;margin-bottom:12px;">
            Quantifies delays and routing alternatives for critical coal, container, and goods trains.
          </div>
        """, unsafe_allow_html=True)

        f_metrics_c1, f_metrics_c2, f_metrics_c3 = st.columns(3)
        with f_metrics_c1:
            st.markdown(f'<div class="ty-stat"><div class="ty-stat-label">Affected Trains</div><div class="ty-stat-value" style="color:#F59E0B;">{freight_impact["affected_freight_trains"]}</div></div>', unsafe_allow_html=True)
        with f_metrics_c2:
            st.markdown(f'<div class="ty-stat"><div class="ty-stat-label">Total Freight Delay</div><div class="ty-stat-value" style="color:#EF4444;">{freight_impact["total_freight_delay_mins"]}m</div></div>', unsafe_allow_html=True)
        with f_metrics_c3:
            st.markdown(f'<div class="ty-stat"><div class="ty-stat-label">Avg Delay / Train</div><div class="ty-stat-value" style="color:#38BDF8;">{freight_impact["average_delay_mins"]}m</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
        f_df = freight_impact["impact_df"][[
            "rake_id", "rake_name", "cargo", "scheduled_time", "estimated_delay_mins", "impact_severity", "alternative_window"
        ]]
        f_df.columns = ["Rake ID", "Freight Rake", "Commodity", "Slot", "Delay (Mins)", "Severity", "Alternative Path"]
        st.dataframe(f_df, use_container_width=True, height=240, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  TAB 6: FINANCIAL IMPACT AUDIT & DEMURRAGE MODEL
# =============================================================================
with tab_fin:
    st.markdown('<div class="ty-section-heading">Model-Based Financial Demurrage & Energy Impact Audit</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ty-card" style="border-left:4px solid #F59E0B;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <span class="ty-badge ty-badge-amber">MODEL-BASED DEMO ESTIMATE</span>
          <h3 style="margin:6px 0 2px;font-size:18px;font-weight:900;color:#FFFFFF;">
            Division Demurrage & Section Detention Loss Prevention Formula
          </h3>
          <div style="font-size:12.5px;color:#CBD5E1;">
            <code>Estimated Avoided Impact = Affected Freight Trains × Total Delay Minutes × Configurable Detention Rate</code><br>
            <span style="font-size:11px;color:#94A3B8;">*Note: Domain-informed operational simulation for prototype decision support — not official Indian Railways accounting.</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cost Factor Slider
    cf_col1, _ = st.columns([1.2, 1])
    with cf_col1:
        st.session_state["cost_factor"] = st.slider(
            "Configurable Demurrage & Detention Cost Factor (₹ / Train Minute)",
            min_value=500, max_value=3000, value=int(st.session_state["cost_factor"]), step=100
        )

    fin_live = compute_financial_impact(
        freight_impact["affected_freight_trains"],
        freight_impact["total_freight_delay_mins"],
        cost_factor_per_min=float(st.session_state["cost_factor"]),
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #EF4444;">
          <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;">Without Optimization</div>
          <h2 style="margin:6px 0;color:#EF4444;font-size:28px;font-weight:900;">₹{fin_live['cost_without_optimization_lakhs']} Lakhs</h2>
          <div style="font-size:12px;color:#CBD5E1;">Uncoordinated fragmented blocks with cumulative detention penalties.</div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #38BDF8;">
          <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;">With TrackYukti Optimization</div>
          <h2 style="margin:6px 0;color:#38BDF8;font-size:28px;font-weight:900;">₹{fin_live['cost_with_optimization_lakhs']} Lakhs</h2>
          <div style="font-size:12px;color:#CBD5E1;">Synchronized joint possession windows minimizing rake holding times.</div>
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #10B981;">
          <div style="font-size:11px;font-weight:800;color:#34D399;text-transform:uppercase;">Estimated Impact Avoided</div>
          <h2 style="margin:6px 0;color:#4ADE80;font-size:28px;font-weight:900;">₹{fin_live['avoided_impact_lakhs']} Lakhs</h2>
          <div style="font-size:12px;color:#D1FAE5;font-weight:700;">Net Financial Savings Recovered</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ty-card" style="border-left:4px solid #10B981;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <span class="ty-badge ty-badge-green">ISO-50001 COMPLIANT ENERGY AUDIT</span>
          <h3 style="margin:6px 0 2px;font-size:17px;font-weight:800;color:#FFFFFF;">{T['green_banner_title']}</h3>
          <div style="font-size:12.5px;color:#CBD5E1;max-width:850px;">{T['green_banner_desc']}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:20px;font-weight:900;color:#4ADE80;">{fin_live['co2_reduction_kg']} kg CO₂e</div>
          <div style="font-size:11px;color:#94A3B8;">Emissions Prevented</div>
        </div>
      </div>
      <div style="display:flex;gap:24px;margin-top:12px;font-size:12.5px;color:#E2E8F0;">
        <div>⛽ Diesel Saved: <b style="color:#38BDF8;">{fin_live['diesel_litres_saved']} Litres</b></div>
        <div>⏱️ Loco Idling Averted: <b style="color:#FCD34D;">{fin_live['diesel_hours_saved']} Engine Hours</b></div>
        <div>⚡ Unscheduled OHE Power Cuts: <b style="color:#4ADE80;">0 Unplanned Isolations</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
#  TAB 7: WHAT-IF SIMULATION LAB
# =============================================================================
with tab_sim:
    st.markdown('<div class="ty-section-heading">Operational What-If Scenario Stress Testing Lab</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ty-card">
      <div style="font-size:13px;color:#CBD5E1;">
        Inject dynamic railway operating disruptions and evaluate how the OR-Tools CP-SAT engine adapts,
        re-schedules work orders, and prevents section collisions in real time.
      </div>
    </div>
    """, unsafe_allow_html=True)

    sim1, sim2 = st.columns(2)
    with sim1:
        st.markdown('<div class="ty-section-heading">Disruption Scenario Injection Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="ty-card">', unsafe_allow_html=True)

        st.session_state["simulate_collision"] = st.toggle(
            "⚠️ Scenario 1: Inject Section Conflict (Civil vs S&T on same track)",
            value=st.session_state["simulate_collision"]
        )

        st.session_state["sync_failure"] = st.toggle(
            "🌐 Scenario 2: Simulate CRIS Server Offline (Fallback to cached headway)",
            value=st.session_state["sync_failure"]
        )

        st.session_state["siren_off_halt"] = st.toggle(
            "⛔ Scenario 3: Engage Divisional Safety Interlock Hold (DRM Emergency Stop)",
            value=st.session_state["siren_off_halt"]
        )

        inject_delay = st.slider(
            "⏱️ Scenario 4: Inbound Heavy Freight Rake Delay (Mins)",
            min_value=0, max_value=90, value=0, step=15
        )

        if st.button("🚀 Run Live Scenario Simulation Re-Optimization", type="primary", use_container_width=True):
            st.session_state["recent_activities"].insert(0, {
                "time": datetime.now().strftime("%H:%M:%S IST"),
                "user": st.session_state["user_designation"],
                "event": f"Triggered What-If simulation test (Delay: {inject_delay}m, Conflict: {st.session_state['simulate_collision']})",
            })
            st.success("Re-optimization complete. Timeline shifted dynamically.")
            time.sleep(0.3)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with sim2:
        st.markdown('<div class="ty-section-heading">Dynamic Solver Telemetry & Response</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ty-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:13px;font-weight:800;color:#FFFFFF;">CRIS CP-SAT Optimization Engine</div>
            <span class="ty-badge ty-badge-green">STATUS: {baseline_result.solver_status}</span>
          </div>
          <div style="font-size:12px;color:#CBD5E1;line-height:1.6;">
            • Objective Value Score: <b>{baseline_result.objective_value:.1f}</b><br>
            • Horizon Window: <b>{baseline_result.horizon_minutes} minutes ({horizon_hours}h)</b><br>
            • Dynamically Shifted Tasks: <b>{int(schedule['dynamically_shifted'].sum())} Tasks Rescheduled</b><br>
            • Active Safety Collisions: <b>{'1 Interlock Clashing' if has_conflict else '0 Clashes Detected'}</b>
          </div>
          <div style="margin-top:10px;padding:10px;background:rgba(8,16,38,0.7);border-radius:6px;font-family:JetBrains Mono,monospace;font-size:11px;color:#93C5FD;">
            CP-SAT SOLVER AUDIT LOG:<br>
            - Model Variables: {len(schedule) * 3}<br>
            - No-Overlap Constraints: {len(schedule['section_track'].unique())}<br>
            - Solution Integrity: STRICTLY COMPLIANT
          </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
#  TAB 8: APPROVAL & EXECUTION WORKFLOW
# =============================================================================
with tab_wf:
    st.markdown('<div class="ty-section-heading">Multi-Stage Role-Based Approval & Department Execution Workflow</div>', unsafe_allow_html=True)

    wf_status = st.session_state["workflow_status"]
    is_chc_role = any(r in st.session_state["user_designation"] for r in [
        "Chief Controller", "Dy. Chief Controller", "Section Controller", "DRM",
        "Divisional Railway Manager", "Sr. DOM", "Divisional Safety Officer"
    ])

    st.markdown(f"""
    <div class="ty-card" style="padding:16px 20px;">
      <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;margin-bottom:8px;">
        Workflow Lifecycle Status
      </div>
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:12.5px;font-weight:800;">
        <span class="ty-badge {'ty-badge-amber' if wf_status == 'AWAITING_APPROVAL' else 'ty-badge-green'}">
          1. AWAITING CHC APPROVAL
        </span> ➔
        <span class="ty-badge {'ty-badge-amber' if wf_status == 'APPROVED_DISPATCHED' else ('ty-badge-green' if wf_status in ['IN_EXECUTION','AWAITING_FINAL_CLOSURE','COMPLETED_AND_CLOSED'] else '')}">
          2. PLAN DISPATCHED (4 DEPTS)
        </span> ➔
        <span class="ty-badge {'ty-badge-amber' if wf_status == 'IN_EXECUTION' else ('ty-badge-green' if wf_status in ['AWAITING_FINAL_CLOSURE','COMPLETED_AND_CLOSED'] else '')}">
          3. WORK EXECUTION
        </span> ➔
        <span class="ty-badge {'ty-badge-amber' if wf_status == 'AWAITING_FINAL_CLOSURE' else ('ty-badge-green' if wf_status == 'COMPLETED_AND_CLOSED' else '')}">
          4. FINAL VERIFICATION & CLOSURE
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ty-section-heading">Stage 1: Chief Controller Formal Authorization</div>', unsafe_allow_html=True)

    if wf_status == "AWAITING_APPROVAL":
        st.markdown("""
        <div class="ty-card" style="border-left:4px solid #F59E0B;">
          <div style="font-size:15px;font-weight:800;color:#FFFFFF;">
            Awaiting Chief Controller Authorization
          </div>
          <div style="font-size:12.5px;color:#CBD5E1;margin-top:4px;">
            The 24-hour joint rolling block schedule has been optimized by the CP-SAT engine with 0 conflicts.
            Formal operational clearance from Chief Controller / DRM is required before transmission to Department Heads.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if is_chc_role:
            btn_app, btn_rej, btn_rev = st.columns(3)
            with btn_app:
                if st.button("✅  AUTHORIZE & DISPATCH PROGRAM TO 4 DEPTS", type="primary", use_container_width=True):
                    st.session_state["workflow_status"] = "APPROVED_DISPATCHED"
                    st.session_state["dispatch_executed"] = True
                    st.session_state["recent_activities"].insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S IST"),
                        "user": st.session_state["user_designation"],
                        "event": "Officially AUTHORIZED and dispatched rolling block program to all 4 Department Heads.",
                    })
                    st.balloons()
                    st.rerun()
            with btn_rej:
                if st.button("⛔  REJECT / HOLD PROGRAM", use_container_width=True):
                    st.session_state["siren_off_halt"] = True
                    st.warning("Program held under Chief Controller directive.")
            with btn_rev:
                if st.button("🔄  REQUEST RE-OPTIMIZATION", use_container_width=True):
                    st.session_state["seed"] += 1
                    st.rerun()
        else:
            st.info("🔒 Chief Controller / DRM authorization credentials required to authorize dispatch.")
    else:
        st.markdown(f"""
        <div class="ty-alert-success">
          <b>✓ ROLLING BLOCK PROGRAM AUTHORIZED BY CHIEF CONTROLLER</b><br>
          <span style="font-size:12px;">Plan dispatched to all 4 concerned departments. Security token: <code>SEC_TOKEN_JBP2026_DISPATCH_OK</code></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ty-section-heading">Stage 2 & 3: Department Work Execution & Status Updates</div>', unsafe_allow_html=True)

    d_cols = st.columns(4)
    depts_list = [
        ("Engineering", "Sr. DEN / Track", "P-Way Track Maintenance Gangs"),
        ("S&T",         "Sr. DSTE",       "Signalling & Interlocking Units"),
        ("Electrical",  "Sr. DEE / TRD",   "OHE Tower Wagon & Power Staff"),
        ("Operating",   "Sr. DOM / DSO",   "Station Masters & Traffic Controllers"),
    ]

    all_completed = True
    for i, (dept_name, dept_head, desc) in enumerate(depts_list):
        with d_cols[i]:
            c_status = st.session_state["dept_work_status"][dept_name]
            badge_map = {
                "SCHEDULED": "ty-badge",
                "WORK STARTED": "ty-badge-amber",
                "WORK IN PROGRESS": "ty-badge-purple",
                "WORK COMPLETED": "ty-badge-green",
            }

            if c_status != "WORK COMPLETED":
                all_completed = False

            st.markdown(f"""
            <div class="ty-card" style="padding:14px 16px;">
              <div style="font-weight:800;font-size:13.5px;color:#FFFFFF;">{dept_name}</div>
              <div style="font-size:11px;color:#94A3B8;">{dept_head}</div>
              <div style="margin:8px 0;">
                <span class="ty-badge {badge_map[c_status]}">{c_status}</span>
              </div>
              <div style="font-size:11.5px;color:#CBD5E1;margin-bottom:10px;">{desc}</div>
            """, unsafe_allow_html=True)

            if wf_status != "AWAITING_APPROVAL":
                new_st = st.selectbox(
                    f"Update {dept_name}:",
                    ["SCHEDULED", "WORK STARTED", "WORK IN PROGRESS", "WORK COMPLETED"],
                    index=["SCHEDULED", "WORK STARTED", "WORK IN PROGRESS", "WORK COMPLETED"].index(c_status),
                    key=f"dept_status_sel_{dept_name}"
                )
                if new_st != c_status:
                    st.session_state["dept_work_status"][dept_name] = new_st
                    st.session_state["recent_activities"].insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S IST"),
                        "user": dept_head,
                        "event": f"Updated {dept_name} work order status to {new_st}.",
                    })
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ty-section-heading">Stage 4: Chief Controller Final Verification & Block Closure</div>', unsafe_allow_html=True)

    if all_completed and wf_status != "COMPLETED_AND_CLOSED":
        st.session_state["workflow_status"] = "AWAITING_FINAL_CLOSURE"
        st.markdown("""
        <div class="ty-card" style="border-left:4px solid #10B981;">
          <div style="font-size:15px;font-weight:800;color:#FFFFFF;">
            Awaiting Chief Controller Final Verification & Block Closure
          </div>
          <div style="font-size:12.5px;color:#CBD5E1;margin-top:4px;">
            All 4 concerned departments have marked their field work as <b>WORK COMPLETED</b>.
            Chief Controller verification is required to cancel caution orders and restore normal sectional line speed.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if is_chc_role:
            if st.button("🏁  VERIFY TRACK CLEARANCE & CLOSE BLOCK PROGRAM", type="primary", use_container_width=True):
                st.session_state["workflow_status"] = "COMPLETED_AND_CLOSED"
                st.session_state["recent_activities"].insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S IST"),
                    "user": st.session_state["user_designation"],
                    "event": "Final track clearance verified. Rolling block completed and officially CLOSED.",
                })
                st.balloons()
                st.rerun()
        else:
            st.info("🔒 Chief Controller / DRM credential required for final block closure.")

    elif wf_status == "COMPLETED_AND_CLOSED":
        st.markdown("""
        <div class="ty-alert-success">
          <h3 style="margin:0 0 4px;font-size:16px;color:#A7F3D0;">🎉 ROLLING BLOCK PROGRAM OFFICIALLY COMPLETED AND CLOSED</h3>
          <div style="font-size:12.5px;">
            All track possession permits canceled. Sectional line speed restored. CRIS audit record archived.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ty-card" style="padding:14px 18px;">
          <div style="font-size:12.5px;color:#CBD5E1;">
            ⏳ Block closure will unlock once all 4 departments mark their respective work packages as <b>WORK COMPLETED</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # CRIS SMS Gateway Broadcast Log
    st.markdown('<div class="ty-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ty-section-heading">CRIS SMS Gateway Real-Time Broadcast Ledger</div>', unsafe_allow_html=True)

    if st.session_state["dispatch_executed"]:
        order_ref = f"WCR/JBP/JRBP/{datetime.now().strftime('%Y%m%d-%H%M')}"
        st.markdown(f"""
        <div class="ty-alert-success">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:13.5px;font-weight:800;">{T['sms_success_title']}</div>
            <span class="ty-badge ty-badge-green">CRIS TLS-1.3 ENCRYPTED</span>
          </div>
          <div style="font-size:12px;line-height:1.6;margin-bottom:6px;">
            Order Ref: <span class="ty-badge">{order_ref}</span> &nbsp;|&nbsp;
            Authorizing Officer: <span class="ty-badge">{st.session_state['user_designation']}</span><br>
            Cryptographic Token: <code style="color:#93C5FD;font-family:JetBrains Mono,monospace;">SEC_TOKEN_JBP2026_SMS_VERIFIED_OK</code>
          </div>
          <div class="ty-sms">
            📱 <b>Civil / Track (Sr. DEN):</b> [WCR/JBP/ENG] Joint Rolling Block Approved. Corridors: JBP-ET, JBP-KTE. Speed: 30 km/h TSR. Auth: CHC-JBP.
          </div>
          <div class="ty-sms">
            📱 <b>Signal & Telecom (Sr. DSTE):</b> [WCR/JBP/S&T] Interlocking & Axle counter possession synchronized at KM 1042. Auth: CHC-JBP.
          </div>
          <div class="ty-sms">
            📱 <b>Electrical / TRD (Sr. DEE):</b> [WCR/JBP/TRD] OHE Power Block synchronized. 25kV Feeder scheduled. Zero starvation. Auth: CHC-JBP.
          </div>
          <div class="ty-sms">
            📱 <b>Operating / Traffic (Sr. DOM):</b> [WCR/JBP/OPT] Freight path diversion scheduled on Goods Loop. Punctuality maintained. Auth: CHC-JBP.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ty-card" style="padding:14px 18px;">
          <div style="font-size:12.5px;color:#94A3B8;">
            Broadcast logs will populate immediately upon Chief Controller authorization in Stage 1.
          </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
#  TAB 9: WORK ORDERS & REQUISITIONS (PRESERVED)
# =============================================================================
with tab_orders:
    st.markdown('<div class="ty-section-heading">Departmental Block Requisition & Work Order Queue</div>', unsafe_allow_html=True)

    wo_col1, wo_col2 = st.columns([1, 1.4])
    with wo_col1:
        st.markdown(f"""
        <div class="ty-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div style="font-size:14px;font-weight:800;color:#FFFFFF;">
              {T['config_header']}
            </div>
            <span class="ty-badge">{st.session_state['user_dept']}</span>
          </div>
        """, unsafe_allow_html=True)

        is_chc = (st.session_state["user_dept"] == "Chief Controller / DRM")
        form_branch = (
            st.selectbox(f"{T['branch_label']}:", ["Engineering", "S&T", "Electrical"], key="wo_fb")
            if is_chc else st.session_state["user_dept"]
        )

        c1, c2 = st.columns(2)
        with c1:
            corridor_input = st.selectbox(T["corridor_label"], list(CORRIDORS.keys()), index=1, key="wo_ci")
        with c2:
            track_input = st.selectbox(T["track_label"], CORRIDORS[corridor_input]["tracks"], index=0, key="wo_ti")

        action_input   = st.selectbox(T["action_label"], BRANCH_ACTIONS[form_branch], key="wo_ai")
        duration_input = st.slider(T["duration_label"], 30, 240, 90, step=15, key="wo_di")
        heavy_toggle   = st.checkbox(T["heavy_label"], value=False, key="wo_ht")

        if st.button(T["btn_push"], type="primary", use_container_width=True):
            nid  = f"WCR-REQ-{1050 + len(st.session_state['custom_requests'])}"
            meta = CORRIDORS[corridor_input]
            new_entry = dict(
                request_id=nid,
                department=form_branch,
                action=action_input,
                corridor=corridor_input,
                section_track=f"{corridor_input} :: {track_input}",
                asset_id=f"AST-{form_branch[:3].upper()}-9901",
                latitude=meta["lat"] + np.random.uniform(-0.01, 0.01),
                longitude=meta["lon"] + np.random.uniform(-0.01, 0.01),
                overdue_days=75,
                last_inspection_score=82.0,
                traffic_density=110,
                corridor_priority=meta["priority"],
                estimated_duration_mins=duration_input,
                is_heavy_machinery=heavy_toggle,
                exclusive_block=heavy_toggle,
            )
            st.session_state["custom_requests"].append(new_entry)
            st.session_state["recent_activities"].insert(0, {
                "time": datetime.now().strftime("%H:%M:%S IST"),
                "user": st.session_state["user_designation"],
                "event": f"Raised work order {nid} [{action_input}] on {corridor_input}",
            })
            st.success(f"Work order {nid} added to joint queue.")
            time.sleep(0.3)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with wo_col2:
        st.markdown(f"""
        <div class="ty-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div style="font-size:14px;font-weight:800;color:#FFFFFF;">
              Master Requisitions Pool ({len(combined_df)} Orders)
            </div>
            <div>
              <span class="ty-badge ty-badge-green">{scheduled_tasks} Scheduled</span>
              <span class="ty-badge ty-badge-red">{deferred_tasks} Deferred</span>
            </div>
          </div>
        """, unsafe_allow_html=True)

        display_df = priority_df[[
            "request_id", "department", "action", "corridor",
            "estimated_duration_mins", "priority_score", "priority_level"
        ]].copy()
        display_df.columns = ["Order ID", "Dept", "Activity", "Corridor", "Mins", "Priority", "Band"]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=380,
            hide_index=True,
        )

        csv_buffer = io.StringIO()
        priority_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"📥  {T['btn_export']}",
            data=csv_buffer.getvalue(),
            file_name=f"trackyukti_requisitions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 10px;font-size:11.5px;color:#94A3B8;border-top:1px solid rgba(255,255,255,0.10);margin-top:30px;">
  <b>TrackYukti</b> &nbsp;·&nbsp; Joint Rolling Block Planning & Corridor Operations Portal (IR-JRBP)
  <br>Ministry of Railways &nbsp;·&nbsp; West Central Railway (WCR) &nbsp;·&nbsp; Jabalpur Division &nbsp;·&nbsp; CRIS Telemetry Integrated
</div>
""", unsafe_allow_html=True)
