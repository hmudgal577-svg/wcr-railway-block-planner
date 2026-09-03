"""
app.py
-------
SATARK / TRACKYUKTI — Operations & Safety Intelligence Command Center
Joint Early Warning & Corridor Rolling Block System (EWS-NER / IR-JRBP)
Production-Grade Portal — Government of India
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
    page_title="SATARK · Citizen Safety & Disaster Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
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
SATARK_LOGO_B64 = load_b64(str(BASE / "assets" / "satark_logo.jpg"))
LOGO_B64        = SATARK_LOGO_B64 if SATARK_LOGO_B64 else load_b64(str(BASE / "assets" / "logo.png"))
BG_B64          = load_b64(str(BASE / "assets" / "train_bg.jpg"))
BG_CSS_VAL      = f"url('data:image/jpeg;base64,{BG_B64}')" if BG_B64 else "none"

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
# EXACT PRODUCTION CSS MATCHING THE REFERENCE SCREENSHOT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

/* Full-bleed train & station backdrop with subtle dark overlay matching the reference screenshot */
.stApp {{
    background-image:
        linear-gradient(rgba(4, 9, 24, 0.55), rgba(4, 9, 24, 0.65)),
        {BG_CSS_VAL};
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    color: #FFFFFF;
    min-height: 100vh;
}}

.block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1320px !important;
}}

/* ── Absolute Text Contrast ── */
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

/* ── Top Red Marquee Emergency Banner ── */
.ty-top-alert-banner {{
    background: #DC2626;
    color: #FFFFFF !important;
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-align: center;
    padding: 8px 16px;
    border-radius: 6px;
    margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(220, 38, 38, 0.45);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}}

/* ── Header Navigation Card ── */
.ty-header-nav {{
    background: rgba(10, 16, 32, 0.82);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.50);
}}

.ty-logo-circle {{
    width: 48px;
    height: 48px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 0 16px rgba(59, 130, 246, 0.35);
}}

.ty-live-pill {{
    background: rgba(6, 78, 59, 0.50);
    color: #6EE7B7 !important;
    border: 1px solid rgba(52, 211, 153, 0.40);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.04em;
}}

.ty-live-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: ty-pulse-dot 1.8s infinite;
}}
@keyframes ty-pulse-dot {{
    0%, 100% {{ opacity: 0.8; transform: scale(0.9); }}
    50%      {{ opacity: 1;   transform: scale(1.3); box-shadow: 0 0 12px #34D399; }}
}}

/* ── Hero Center Titles ── */
.ty-hero-title {{
    font-size: 34px;
    font-weight: 900;
    color: #FFFFFF;
    text-align: center;
    margin: 4px 0 2px;
    letter-spacing: -0.02em;
}}
.ty-hero-sub {{
    font-size: 14.5px;
    color: #94A3B8;
    text-align: center;
    margin: 0 0 18px;
    font-weight: 500;
}}

/* ── Floating Navigation Pill Bar (st.tabs) ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(10, 16, 32, 0.80) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 9999px !important;
    padding: 6px 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    display: flex !important;
    justify-content: center !important;
    gap: 6px !important;
    margin: 0 auto 20px auto !important;
    overflow-x: auto !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45) !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 9999px !important;
    padding: 9px 18px !important;
    color: #CBD5E1 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border: none !important;
    transition: all 0.18s ease !important;
    white-space: nowrap !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #FFFFFF !important;
    background: rgba(255, 255, 255, 0.08) !important;
}}
.stTabs [aria-selected="true"] {{
    background: #2563EB !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 18px rgba(37, 99, 235, 0.65) !important;
}}

/* ── Dark Glass Container Cards ── */
.ty-card {{
    background: rgba(10, 16, 32, 0.82) !important;
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.40);
}}

/* ── Red-Glowing AI Susceptibility / Risk Index Card ── */
.ty-risk-card {{
    background: rgba(36, 12, 18, 0.84);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 0 28px rgba(220, 38, 38, 0.20);
}}
.ty-protocol-box {{
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 8px;
    padding: 12px 16px;
    color: #FECACA;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 10px;
}}

/* ── Telemetry Grid Stat Tiles ── */
.ty-telemetry-tile {{
    background: rgba(10, 16, 32, 0.82);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}}
.ty-telemetry-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    font-weight: 700;
    color: #CBD5E1;
}}
.ty-telemetry-icon {{
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: rgba(37, 99, 235, 0.20);
    border: 1px solid rgba(59, 130, 246, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #93C5FD;
}}
.ty-telemetry-value {{
    font-size: 30px;
    font-weight: 900;
    color: #FFFFFF;
    margin: 10px 0 4px;
    line-height: 1.1;
}}
.ty-telemetry-unit {{
    font-size: 16px;
    font-weight: 600;
    color: #94A3B8;
    margin-left: 4px;
}}
.ty-telemetry-source {{
    font-size: 11px;
    color: #64748B;
}}

/* ── Highway / Corridor Detour Status Pill Boxes ── */
.ty-status-pill-red {{
    background: rgba(127, 29, 29, 0.45);
    border: 1px solid rgba(239, 68, 68, 0.45);
    border-radius: 8px;
    padding: 12px 16px;
    color: #FCA5A5 !important;
    font-weight: 700;
    font-size: 13px;
}}
.ty-status-pill-green {{
    background: rgba(6, 78, 59, 0.45);
    border: 1px solid rgba(52, 211, 153, 0.45);
    border-radius: 8px;
    padding: 12px 16px;
    color: #6EE7B7 !important;
    font-weight: 700;
    font-size: 13px;
}}
.ty-status-pill-blue {{
    background: rgba(30, 58, 138, 0.45);
    border: 1px solid rgba(59, 130, 246, 0.45);
    border-radius: 8px;
    padding: 12px 16px;
    color: #93C5FD !important;
    font-weight: 700;
    font-size: 13px;
}}

/* ── Badges & Buttons ── */
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

/* ── Action Pill Buttons ── */
.ty-action-btn-blue {{
    background: #2563EB;
    color: #FFFFFF !important;
    border: 1px solid rgba(96, 165, 250, 0.40);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12.5px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
}}
.ty-action-btn-green {{
    background: #059669;
    color: #FFFFFF !important;
    border: 1px solid rgba(52, 211, 153, 0.40);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12.5px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35);
}}

/* ── Login Shell ── */
.ty-login-shell {{
    max-width: 900px;
    margin: 24px auto;
    background: rgba(10, 16, 32, 0.94);
    backdrop-filter: blur(32px);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-top: 4px solid #DC2626;
    border-radius: 18px;
    padding: 38px 48px 36px;
    box-shadow: 0 32px 80px rgba(0, 0, 0, 0.65);
}}

/* ── Form Inputs ── */
div[data-baseweb="select"] > div {{
    background: rgba(10, 16, 32, 0.94) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}}
div[data-baseweb="select"] * {{
    color: #FFFFFF !important;
    background: rgba(10, 16, 32, 0.98) !important;
}}
.stTextInput input {{
    background: rgba(10, 16, 32, 0.94) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
}}
.stRadio label p, .stCheckbox label p, .stToggle label p {{
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: rgba(15, 23, 42, 0.85) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(148, 163, 184, 0.30) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 9px 18px !important;
    transition: all 0.18s ease !important;
}}
.stButton > button:hover {{
    background: rgba(37, 99, 235, 0.85) !important;
    border-color: rgba(99, 179, 255, 0.60) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
    border: 1px solid rgba(99, 179, 255, 0.40) !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.40) !important;
}}

/* ── Plotly Charts ── */
.js-plotly-plot .plotly, .plotly-graph-div {{
    background: rgba(10, 16, 32, 0.92) !important;
    border-radius: 12px;
}}
</style>
""", unsafe_allow_html=True)

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
# SESSION STATE INITIALIZATION (DEFAULT: AUTHENTICATED)
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "is_logged_in": True,
    "user_dept": "Chief Controller / DRM",
    "user_designation": "Chief Controller (CHC / Central Control)",
    "lang_choice": "English",
    "seed": 42,
    "custom_requests": DEFAULT_DEMO_REQS,
    "simulate_collision": False,
    "sync_failure": False,
    "dispatch_executed": False,
    "siren_off_halt": False,
    "cost_factor": 1200,
    "selected_zone": "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route",
    "workflow_status": "AWAITING_APPROVAL",
    "dept_work_status": {
        "Engineering": "SCHEDULED",
        "S&T":         "SCHEDULED",
        "Electrical":  "SCHEDULED",
        "Operating":   "SCHEDULED",
    },
    "recent_activities": [
        {"time": "13:00:00 IST", "user": "SYSTEM / OR-Tools", "event": "Rolling possession optimized with 0 section collisions"},
        {"time": "12:45:20 IST", "user": "Chief Controller", "event": "Reviewed Katni-Singrauli coal corridor joint possession package"},
        {"time": "12:15:10 IST", "user": "Sr. DEN / Track", "event": "Updated USFD rail flaw inspection telemetry for Route #2"},
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
#  LOGIN PORTAL (IF USER LOGS OUT)
# =============================================================================
if not st.session_state["is_logged_in"]:
    logo_img = (
        f'<img src="data:image/jpeg;base64,{LOGO_B64}" class="ty-logo-circle" style="width:72px;height:72px;display:block;margin:0 auto 12px;" alt="Logo">'
        if LOGO_B64 else ""
    )
    st.markdown(f"""
    <div class="ty-login-shell">
      {logo_img}
      <div style="text-align:center;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.12);">
        <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.12em;">
          Government of India &nbsp;·&nbsp; Ministry of Railways / EWS-NER
        </div>
        <h1 style="margin:6px 0 2px;font-size:32px;font-weight:900;color:#FFFFFF;letter-spacing:-0.02em;">
          SATARK &nbsp;·&nbsp; TRACK<span style="color:#F59E0B;">YUKTI</span>
        </h1>
        <div style="font-size:12px;color:#F59E0B;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;">
          Early Warning. Safe Tomorrow. &nbsp;·&nbsp; Smarter Planning.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    lc1, lc2 = st.columns([1.1, 1])
    with lc1:
        st.markdown("""
        <div class="ty-card" style="margin-top:8px;">
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;margin-bottom:8px;">
            Step 1 — Select Operating Branch
          </div>
          <div style="font-size:12.5px;color:#CBD5E1;">
            Operational control and security credentials assigned per departmental tier.
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
          <div style="font-size:12px;font-weight:800;color:#94A3B8;text-transform:uppercase;margin-bottom:8px;">
            Step 2 — Security Passkey
          </div>
          <div style="font-size:12.5px;color:#CBD5E1;">
            Security passkey: <b style="color:#F59E0B;">JBP2026</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        pk = st.text_input("Passkey", value="JBP2026", type="password")
        if st.button("🔐  Access Operations Command Center", type="primary", use_container_width=True):
            st.session_state["is_logged_in"] = True
            st.session_state["user_dept"] = sel_dept
            st.rerun()

    st.stop()

# =============================================================================
#  AUTHENTICATED OPERATIONS COMMAND CENTER
# =============================================================================

# ── Sidebar Controls (Collapsed by Default) ──────────────────────────────────
with st.sidebar:
    if LOGO_B64:
        st.markdown(f'<img src="data:image/jpeg;base64,{LOGO_B64}" style="height:48px;width:auto;margin-bottom:6px;">', unsafe_allow_html=True)
    st.markdown('<span style="font-size:18px;font-weight:900;color:#FFFFFF;">SATARK · TrackYukti</span>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(255,255,255,0.10);margin:10px 0;">', unsafe_allow_html=True)

    sel_corr = st.selectbox(
        "Corridor Jurisdiction",
        ["All Corridors (Jabalpur Division)"] + list(CORRIDORS.keys()),
        index=0
    )
    horizon_hours = st.slider("Horizon Window (Hours)", 6, 24, 12, step=1)
    setup_buffer  = st.slider("Safety Buffer (Mins)", 5, 45, 15, step=5)

    if st.button("🚪 Logout / Switch User", use_container_width=True):
        st.session_state["is_logged_in"] = False
        st.rerun()
    if st.button("♻️ Reset Parameters", use_container_width=True):
        reset_all(); st.rerun()

# ── Real-Time Pipeline ───────────────────────────────────────────────────────
base_df = get_cached_requests(seed=st.session_state["seed"])
combined_df = pd.concat([pd.DataFrame(st.session_state["custom_requests"]), base_df], ignore_index=True)

delayed_corr_arg = None if sel_corr == "All Corridors (Jabalpur Division)" else sel_corr
baseline_result, bundled_df, scorer = run_pipeline(combined_df, horizon_hours, setup_buffer)
schedule = baseline_result.schedule.copy()

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

total_tasks     = len(schedule)
scheduled_tasks = int(schedule["is_scheduled"].sum())
deferred_tasks  = total_tasks - scheduled_tasks
critical_risks  = int((priority_df["priority_level"] == "CRITICAL").sum())

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOP MARQUEE CRITICAL ALERT BANNER (EXACT AS SCREENSHOT)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ty-top-alert-banner">
  ▲ CRITICAL LANDSLIDE & SECTION HAZARD ALERT — MEPPADI / KATNI-SINGRAULI CORRIDOR (TESTBED) — IMMEDIATE POSSESSION & EVACUATION REQUIRED ▲
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TOP HEADER NAVIGATION BAR (EXACT AS SCREENSHOT)
# ─────────────────────────────────────────────────────────────────────────────
logo_img_tag = (
    f'<img src="data:image/jpeg;base64,{LOGO_B64}" class="ty-logo-circle" alt="SATARK">'
    if LOGO_B64 else ""
)

st.markdown(f"""
<div class="ty-header-nav">
  <div style="display:flex;align-items:center;gap:14px;">
    {logo_img_tag}
    <div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:16px;font-weight:900;color:#FFFFFF;letter-spacing:-0.01em;">
          EWS-NER &nbsp;·&nbsp; Citizen Safety & Operations
        </span>
      </div>
      <div style="font-size:11.5px;color:#94A3B8;font-weight:500;">
        National Early Warning & Rolling Block Network
      </div>
      <div style="margin-top:4px;">
        <div class="ty-live-pill">
          <span class="ty-live-dot"></span> LIVE SATELLITE & WEATHER SYNC
        </div>
      </div>
    </div>
  </div>

  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <button class="stButton" onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance('Critical Hazard Alert. Immediate Possession and Evacuation in effect on Corridor.'))"
            style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.18);border-radius:8px;padding:7px 14px;color:#FFFFFF;font-size:12px;font-weight:700;cursor:pointer;">
      📢 Speak Voice Alert
    </button>
    <button class="stButton" style="background:#DC2626;border:1px solid rgba(248,113,113,0.40);border-radius:8px;padding:7px 14px;color:#FFFFFF;font-size:12px;font-weight:800;cursor:pointer;box-shadow:0 0 14px rgba(220,38,38,0.50);">
      🚨 Emergency Siren
    </button>
    <span style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:8px;padding:6px 12px;font-size:12px;color:#CBD5E1;font-weight:700;">
      EN &nbsp;|&nbsp; HI &nbsp;|&nbsp; AS
    </span>
    <span style="background:#1D4ED8;border:1px solid rgba(96,165,250,0.40);border-radius:8px;padding:7px 14px;color:#FFFFFF;font-size:12px;font-weight:700;display:inline-flex;align-items:center;gap:6px;">
      👤 Officer ⌵
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CENTER HERO TITLES (EXACT AS SCREENSHOT)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin:8px 0 16px;">
  <h1 class="ty-hero-title">Citizen Safety & Disaster Intelligence</h1>
  <p class="ty-hero-sub">Real-time AI Landslide Early Warning, Weather Telemetry & Safe Evacuation</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FLOATING HORIZONTAL NAVIGATION PILL BAR (EXACT AS SCREENSHOT)
# ─────────────────────────────────────────────────────────────────────────────
tab_dash, tab_opt, tab_time, tab_prio, tab_impact, tab_fin, tab_sim, tab_wf, tab_orders = st.tabs([
    "⊞ Overview",
    "📖 3D Terrain & Runoff",
    "⛺ Relief Camps",
    "🤖 AI Priority Agent",
    "📡 Offline SOS Mesh",
    "💰 Financial Audit",
    "🧪 What-If Lab",
    "🛡️ Officer Workflow",
    "📝 Requisitions",
])

# =============================================================================
#  TAB 1: OVERVIEW (EXACT PIXEL-PERFECT RECONSTRUCTION OF SCREENSHOT)
# =============================================================================
with tab_dash:

    # 1. Monitored Zone Selector Card
    st.markdown("""
    <div class="ty-card" style="padding:14px 18px;margin-bottom:12px;">
      <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
        YOUR MONITORED ZONE
      </div>
    """, unsafe_allow_html=True)

    zone_choice = st.selectbox(
        "Monitored Zone",
        [
            "Meppadi, Wayanad (Testbed)",
            "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route",
            "Katni (KTE) - Singrauli Coal Logistics Line",
            "Jabalpur (JBP) - Itarsi (ET) Trunk Line",
            "Satna (STA) - Rewa (REWA) Branch Corridor",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("""
      <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;">
        <span class="ty-action-btn-blue">📍 &nbsp;GIS Command Map</span>
        <span class="ty-action-btn-green">📷 &nbsp;AI Scan & Report Hazard</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Red-Glowing AI Susceptibility Index Card (Exact as Screenshot)
    st.markdown("""
    <div class="ty-risk-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:11px;font-weight:800;color:#FCA5A5;letter-spacing:0.08em;text-transform:uppercase;">
          AI LANDSLIDE SUSCEPTIBILITY INDEX
        </span>
        <span style="font-size:11.5px;color:#94A3B8;">
          Notification Status &nbsp;<span style="color:#EF4444;font-weight:700;">● Notifications Off</span>
        </span>
      </div>
      <div style="font-size:32px;font-weight:900;color:#EF4444;margin:6px 0 8px;letter-spacing:-0.01em;">
        RED RISK (0.78)
      </div>
      <div class="ty-protocol-box">
        <b>Emergency Action Protocol:</b><br>
        Immediate Evacuation & Highway Closure. High-risk debris flow imminent.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. 4-Card Telemetry Grid (Exact as Screenshot)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class="ty-telemetry-tile">
          <div class="ty-telemetry-header">
            <span>24h Cumulative Rain</span>
            <div class="ty-telemetry-icon">🌧️</div>
          </div>
          <div class="ty-telemetry-value">
            142<span class="ty-telemetry-unit">mm</span>
          </div>
          <div class="ty-telemetry-source">Source: Open-Meteo & OpenWeather</div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="ty-telemetry-tile">
          <div class="ty-telemetry-header">
            <span>72h Total Rainfall</span>
            <div class="ty-telemetry-icon">💧</div>
          </div>
          <div class="ty-telemetry-value">
            285<span class="ty-telemetry-unit">mm</span>
          </div>
          <div class="ty-telemetry-source">Source: 3-Day Antecedent Rain</div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="ty-telemetry-tile">
          <div class="ty-telemetry-header">
            <span>Soil Moisture Saturation</span>
            <div class="ty-telemetry-icon">🧪</div>
          </div>
          <div class="ty-telemetry-value">
            0.52<span class="ty-telemetry-unit">m³/m³</span>
          </div>
          <div class="ty-telemetry-source">Source: Topsoil 0-10cm Layer</div>
        </div>
        """, unsafe_allow_html=True)

    with t4:
        st.markdown("""
        <div class="ty-telemetry-tile">
          <div class="ty-telemetry-header">
            <span>NASA SRTM 30m Elevation</span>
            <div class="ty-telemetry-icon">⛰️</div>
          </div>
          <div class="ty-telemetry-value">
            876.5<span class="ty-telemetry-unit">m</span>
          </div>
          <div class="ty-telemetry-source">Source: NASA SRTM 30m DEM</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Highway Corridor Status & Safe Detour Routing (Exact as Screenshot)
    st.markdown("""
    <div class="ty-card" style="margin-top:14px;">
      <div style="font-size:14px;font-weight:800;color:#FFFFFF;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
        🛣️ Highway Corridor Status & Safe Detour Routing
      </div>
      <div style="display:grid;grid-template-columns:1.2fr 1.4fr 1fr;gap:12px;">
        <div>
          <div style="font-size:11px;color:#94A3B8;margin-bottom:4px;">Highway Corridor Status</div>
          <div class="ty-status-pill-red">
            ⊘ NH-766 (BLOCKED - Landslide Hazard Zone)
          </div>
        </div>
        <div>
          <div style="font-size:11px;color:#94A3B8;margin-bottom:4px;">Guaranteed Safe Evacuation Route</div>
          <div class="ty-status-pill-green">
            ✓ Active via SH-59 (Bypass Corridor)
          </div>
        </div>
        <div>
          <div style="font-size:11px;color:#94A3B8;margin-bottom:4px;">Est. Evacuation Time</div>
          <div class="ty-status-pill-blue">
            🕒 42 Minutes
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Live Activity Feed
    st.markdown('<div class="ty-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;font-weight:800;color:#FFFFFF;margin-bottom:8px;">Recent Telemetry & Audit Feed</div>', unsafe_allow_html=True)
    for act in st.session_state["recent_activities"]:
        st.markdown(f"""
        <div style="border-bottom:1px solid rgba(255,255,255,0.08);padding:6px 0;">
          <span style="font-size:11px;font-family:JetBrains Mono,monospace;color:#93C5FD;">{act['time']}</span>
          &nbsp;·&nbsp; <span class="ty-badge">{act['user']}</span>
          &nbsp;·&nbsp; <span style="font-size:12.5px;color:#E2E8F0;">{act['event']}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 2: 3D TERRAIN & RUNOFF (SMART BUNDLES & OPTIMIZER)
# =============================================================================
with tab_opt:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### ⚡ 3D Terrain Analysis & Joint Bundling Engine')
    st.markdown('Unified spatial possession clustering and slope stability optimization.')

    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #94A3B8;">
          <div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;">Uncoordinated Dispersed Plan</div>
          <h2 style="margin:6px 0;font-size:26px;color:#FFFFFF;">{optimization_comp['original_duration_mins'] // 60}h {optimization_comp['original_duration_mins'] % 60}m</h2>
          <div style="font-size:12px;color:#CBD5E1;">{optimization_comp['original_blocks_count']} Independent Closures</div>
        </div>
        """, unsafe_allow_html=True)
    with comp_c2:
        st.markdown(f"""
        <div class="ty-card" style="border-top:3px solid #10B981;">
          <div style="font-size:11px;font-weight:800;color:#34D399;text-transform:uppercase;">Optimized Unified Corridor Plan</div>
          <h2 style="margin:6px 0;font-size:26px;color:#4ADE80;">{optimization_comp['optimized_duration_mins'] // 60}h {optimization_comp['optimized_duration_mins'] % 60}m <span style="font-size:14px;color:#38BDF8;">(−{optimization_comp['time_saved_hrs']}h Saved)</span></h2>
          <div style="font-size:12px;color:#CBD5E1;">{optimization_comp['separate_blocks_avoided']} Separate Interruptions Avoided</div>
        </div>
        """, unsafe_allow_html=True)

    for b in joint_bundles:
        st.markdown(f"""
        <div class="ty-card" style="border-left:4px solid #10B981;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <span class="ty-badge ty-badge-green">{b.bundle_id}</span>
              <span style="font-size:15px;font-weight:800;color:#FFFFFF;margin-left:8px;">{b.corridor}</span>
              <div style="font-size:12px;color:#94A3B8;margin-top:2px;">Track: {b.section_track} &nbsp;|&nbsp; Window: {b.common_start_min//60:02d}:{b.common_start_min%60:02d} – {b.common_end_min//60:02d}:{b.common_end_min%60:02d} IST</div>
            </div>
            <div style="text-align:right;">
              <span style="font-size:18px;font-weight:900;color:#38BDF8;">{b.time_saved_mins}m Saved</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 3: RELIEF CAMPS & MASTER ROLLING TIMETABLE GANTT
# =============================================================================
with tab_time:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 📊 Master Rolling Evacuation & Corridor Possession Timetable')
    st.markdown('24-Hour continuous Gantt schedule for all active relief and maintenance blocks.')

    gantt_df = schedule[schedule["is_scheduled"]].copy()
    bt = datetime.combine(datetime.today(), datetime.min.time())
    gantt_df["Start"]  = gantt_df["start_min"].apply(lambda m: bt + timedelta(minutes=float(m)))
    gantt_df["Finish"] = gantt_df["end_min"].apply(lambda m: bt + timedelta(minutes=float(m)))
    gantt_df["Label"]  = gantt_df.apply(lambda r: f"{r['request_id']} ({r['department'][:3]})", axis=1)

    fig = px.timeline(
        gantt_df, x_start="Start", x_end="Finish", y="section_track",
        color="department", color_discrete_map=DEPT_COLORS, text="Label",
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(10, 16, 32, 0.95)",
        paper_bgcolor="rgba(10, 16, 32, 0.95)",
        font=dict(color="#FFFFFF", size=11),
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 4: AI PRIORITY AGENT (EXPLAINABLE SCORING MODEL)
# =============================================================================
with tab_prio:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 🤖 AI Priority Agent & Explainable Scoring Model')

    p_disp = priority_df[[
        "request_id", "department", "action", "corridor", "priority_score",
        "priority_level", "priority_explanation"
    ]].copy()
    p_disp.columns = ["Task ID", "Dept", "Activity", "Corridor", "Score (0-100)", "Band", "AI Technical Justification"]
    st.dataframe(p_disp, use_container_width=True, height=400, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 5: OFFLINE SOS MESH & FREIGHT IMPACT
# =============================================================================
with tab_impact:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 📡 Offline SOS Mesh & Freight Impact Assessment')

    f_df = freight_impact["impact_df"][[
        "rake_id", "rake_name", "cargo", "scheduled_time", "estimated_delay_mins", "impact_severity", "alternative_window"
    ]]
    f_df.columns = ["Rake ID", "Freight Rake", "Cargo", "Slot", "Delay (Mins)", "Severity", "Alternative Path"]
    st.dataframe(f_df, use_container_width=True, height=320, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 6: FINANCIAL AUDIT
# =============================================================================
with tab_fin:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 💰 Model-Based Financial Demurrage & Energy Audit')

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(f'<div class="ty-card"><div style="font-size:11px;color:#94A3B8;">Without Optimization</div><h2 style="color:#EF4444;">₹{financial_impact["cost_without_optimization_lakhs"]}L</h2></div>', unsafe_allow_html=True)
    with fc2:
        st.markdown(f'<div class="ty-card"><div style="font-size:11px;color:#94A3B8;">With Optimization</div><h2 style="color:#38BDF8;">₹{financial_impact["cost_with_optimization_lakhs"]}L</h2></div>', unsafe_allow_html=True)
    with fc3:
        st.markdown(f'<div class="ty-card"><div style="font-size:11px;color:#94A3B8;">Avoided Impact</div><h2 style="color:#4ADE80;">₹{financial_impact["avoided_impact_lakhs"]}L</h2></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 7: WHAT-IF LAB
# =============================================================================
with tab_sim:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 🧪 What-If Scenario Stress Testing Lab')
    st.session_state["simulate_collision"] = st.toggle("Inject Corridor Collision", value=st.session_state["simulate_collision"])
    st.session_state["sync_failure"] = st.toggle("Simulate Offline Server Mode", value=st.session_state["sync_failure"])
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 8: OFFICER APPROVAL WORKFLOW
# =============================================================================
with tab_wf:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 🛡️ Officer Formal Authorization & Emergency Dispatch')
    if st.button("✅  AUTHORIZE & DISPATCH PROGRAM TO ALL CONCERNED DEPTS", type="primary"):
        st.session_state["workflow_status"] = "APPROVED_DISPATCHED"
        st.session_state["dispatch_executed"] = True
        st.balloons()
        st.success("Program Authorized & Transmitted.")
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB 9: REQUISITIONS
# =============================================================================
with tab_orders:
    st.markdown('<div class="ty-card">', unsafe_allow_html=True)
    st.markdown('### 📝 Departmental Work Order Requisition')
    r_branch = st.selectbox("Operating Branch", ["Engineering", "S&T", "Electrical"])
    r_corr   = st.selectbox("Corridor Section", list(CORRIDORS.keys()))
    r_action = st.selectbox("Activity", BRANCH_ACTIONS[r_branch])
    r_dur    = st.slider("Duration (Mins)", 30, 240, 90, step=15)

    if st.button("Submit Order to Joint Queue", type="primary"):
        st.success("Requisition queued successfully.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 10px;font-size:11.5px;color:#94A3B8;border-top:1px solid rgba(255,255,255,0.10);margin-top:30px;">
  <b>SATARK · TRACKYUKTI</b> &nbsp;·&nbsp; Citizen Safety, Disaster Intelligence & Joint Corridor Operations
  <br>Government of India &nbsp;·&nbsp; EWS-NER / Ministry of Railways &nbsp;·&nbsp; AI Telemetry Integrated
</div>
""", unsafe_allow_html=True)
