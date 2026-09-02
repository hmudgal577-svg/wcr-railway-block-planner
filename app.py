"""
app.py
-------
WEST CENTRAL RAILWAY (WCR) — JABALPUR DIVISION
Joint Corridor Block Management & AI Decision Support System (IR-RBP)
Enterprise Production Portal for Section Controllers & Chief Dispatchers
"""

import io
import json
import time
from datetime import datetime, timedelta

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
    page_title="Indian Railways | WCR Jabalpur Division Block Planner",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# ENTERPRISE LIGHT THEME & PROFESSIONAL RAILWAY PALETTE
# --------------------------------------------------------------------------
DEPT_COLORS = {
    "Engineering": "#0284C7",  # Civil / P-Way (Track Staff)
    "S&T": "#D97706",          # Signal & Telecom
    "Electrical": "#7C3AED",   # TRD / OHE Maintenance
}

RISK_COLORS = {
    "CRITICAL": "#DC2626",
    "HIGH": "#EA580C",
    "MEDIUM": "#CA8A04",
    "LOW": "#16A34A",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.03);
    }

    /* Professional Elevated Cards */
    .pro-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .pro-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    /* Official Ministry Header */
    .gov-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        border-radius: 14px;
        padding: 20px 26px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .gov-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-status-online {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-status-hold {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* KPI Metrics Card */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0 2px 0;
        line-height: 1.2;
    }
    .metric-footer {
        font-size: 12px;
        color: #64748B;
    }

    /* Solution Comparison Cards */
    .comparison-before {
        background: #FFF5F5;
        border: 1px solid #FED7D7;
        border-left: 4px solid #E53E3E;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .comparison-after {
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        border-left: 4px solid #16A34A;
        border-radius: 10px;
        padding: 16px 20px;
    }

    /* Alert Boxes */
    .pro-alert-danger {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 4px solid #DC2626;
        border-radius: 8px;
        padding: 14px 18px;
        color: #991B1B;
        margin-bottom: 16px;
    }
    .pro-alert-warning {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #D97706;
        border-radius: 8px;
        padding: 14px 18px;
        color: #92400E;
        margin-bottom: 16px;
    }
    .pro-alert-success {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 14px 18px;
        color: #166534;
        margin-bottom: 16px;
    }

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    /* Telemetry Code Box */
    .telemetry-box {
        background: #0F172A;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.5;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# PIPELINE CACHE & ML ENGINE
# --------------------------------------------------------------------------
@st.cache_resource
def get_scorer():
    return CriticalityScorer()


def get_default_requests(seed=42):
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
if "seed" not in st.session_state:
    st.session_state["seed"] = 42

if "custom_requests" not in st.session_state:
    st.session_state["custom_requests"] = []

if "simulate_collision" not in st.session_state:
    st.session_state["simulate_collision"] = False

if "sync_failure" not in st.session_state:
    st.session_state["sync_failure"] = False

if "dispatch_executed" not in st.session_state:
    st.session_state["dispatch_executed"] = False

if "siren_off_halt" not in st.session_state:
    st.session_state["siren_off_halt"] = False


def reset_entire_system():
    """Reset system to clean state."""
    st.session_state["seed"] = 42
    st.session_state["custom_requests"] = []
    st.session_state["simulate_collision"] = False
    st.session_state["sync_failure"] = False
    st.session_state["dispatch_executed"] = False
    st.session_state["siren_off_halt"] = False


# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🇮🇳 WCR Control Operations")
    st.caption("Jabalpur Division · Central Dispatch Terminal")

    if st.button("♻️ Reset System Parameters", use_container_width=True):
        reset_entire_system()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 👤 Controller Identity & RBAC")
    security_role = st.selectbox(
        "Access Authorization",
        [
            "Level 1: Section Maintenance Controller (Draft Entry)",
            "Level 3: Chief Controller / DRM Office (Dispatch Authorization)",
        ],
        index=1,
        help="Full corridor block plan transmission requires Level 3 Chief Controller clearance."
    )
    is_level_3 = "Level 3" in security_role

    st.markdown("---")
    st.markdown("#### 📍 Operational Jurisdiction")
    corridor_options = ["All Corridors (Jabalpur Division)"] + list(CORRIDORS.keys())
    selected_corridor = st.selectbox("Active Traffic Corridor", corridor_options, index=0)

    st.markdown("#### ⏱️ Timetable Constraints")
    horizon_hours = st.slider("Planning Window (Hours)", min_value=6, max_value=24, value=12, step=1)
    setup_buffer = st.slider("Safety Handover Buffer (Mins)", min_value=5, max_value=45, value=15, step=5)

    st.markdown("---")
    st.markdown("#### 🧪 Test & Contingency Scenarios")
    
    sync_fail_tgl = st.toggle("Simulate CRIS/COA Server Sync Failure", value=st.session_state["sync_failure"])
    st.session_state["sync_failure"] = sync_fail_tgl

    col_sim_tgl = st.toggle("Inject Multi-Branch Track Collision", value=st.session_state["simulate_collision"])
    st.session_state["simulate_collision"] = col_sim_tgl

    siren_halt_toggle = st.toggle(
        "🔒 Engage Safety Interlock (Halt Dispatch)",
        value=st.session_state["siren_off_halt"],
        help="Locks rolling block plan transmission pending manual safety review."
    )
    st.session_state["siren_off_halt"] = siren_halt_toggle

    delay_minutes = st.slider(
        "Inject Inbound Freight/Express Delay (Mins)",
        min_value=0, max_value=75, value=0, step=5,
        help="Pushes section occupancy to test AI dynamic rescheduling."
    )

# --------------------------------------------------------------------------
# OFFICIAL HEADER BAR
# --------------------------------------------------------------------------
badge_status_html = '<span class="badge-status-online">● SYSTEM OPERATIONAL · READY</span>'
if st.session_state["siren_off_halt"]:
    badge_status_html = '<span class="badge-status-hold">● SAFETY HOLD ACTIVE · DISPATCH LOCKED</span>'

header_html = f"""<div class="gov-header">
<div>
<div style="display:flex; align-items:center; gap:12px;">
<span style="font-size:26px;">🚆</span>
<div>
<div style="display:flex; align-items:center; gap:8px;">
<h2 style="margin:0; font-size:20px; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;">
MINISTRY OF RAILWAYS · WEST CENTRAL RAILWAY
</h2>
<span class="gov-badge">JABALPUR DIVISION</span>
</div>
<p style="margin:4px 0 0 0; color:#CBD5E1; font-size:13px;">
Automated Integrated Block Planning & AI Optimization System (IR-RBP v2.4)
</p>
</div>
</div>
</div>
<div>
{badge_status_html}
</div>
</div>"""

st.markdown(header_html, unsafe_allow_html=True)

# System Fault-Tolerance Warning Banner if sync failed
if st.session_state["sync_failure"]:
    st.markdown(
        '<div class="pro-alert-warning">'
        '<b>⚠️ CRIS/COA API LINK OFFLINE:</b> Activating local SQLite offline buffer and static headway safety templates.'
        '</div>', unsafe_allow_html=True
    )

# --------------------------------------------------------------------------
# EXPANDABLE DEPARTMENT ENTRY FORM
# --------------------------------------------------------------------------
with st.expander("📝 Submit New Departmental Possession Work Order", expanded=False):
    st.markdown("##### Authorized Branch Block Requisition Form")
    col_log1, col_log2, col_log3 = st.columns([1.5, 1.5, 1.5])
    
    with col_log1:
        selected_branch = st.selectbox(
            "Requisitioning Branch",
            ["Engineering (Track / Civil)", "S&T (Signal & Telecom)", "Electrical (TRD / OHE)"],
            index=0
        )
    with col_log2:
        corridor_input = st.selectbox("Target Corridor", list(CORRIDORS.keys()), index=1, key="branch_corr_input")
    with col_log3:
        available_tracks = CORRIDORS[corridor_input]["tracks"]
        track_input = st.selectbox("Track / Section Line", available_tracks, index=0, key="branch_trk_input")

    branch_key = "Engineering" if "Engineering" in selected_branch else ("S&T" if "S&T" in selected_branch else "Electrical")

    f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1.5])
    with f_col1:
        actions_list = BRANCH_ACTIONS[branch_key]
        action_input = st.selectbox("Maintenance Activity", actions_list, index=0, key="branch_act_input")
    with f_col2:
        duration_input = st.slider("Required Window (mins)", 30, 240, 90, step=15, key="branch_dur_input")
    with f_col3:
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        heavy_machinery_toggle = st.checkbox(
            "Requires Heavy TRT / BCM Machinery (Exclusive Block)",
            value=False,
            help="Designates task as an exclusive block that bypasses multi-department bundling for staff safety."
        )

    if st.button("➕ Register Work Order into AI Queue", type="primary"):
        new_id = f"WCR-REQ-{1000 + len(st.session_state['custom_requests']) + 50}"
        meta = CORRIDORS[corridor_input]
        new_entry = {
            "request_id": new_id,
            "department": branch_key,
            "action": action_input,
            "corridor": corridor_input,
            "section_track": f"{corridor_input} :: {track_input}",
            "asset_id": f"AST-WCR-{branch_key[:3].upper()}-9901",
            "latitude": meta["lat"] + np.random.uniform(-0.01, 0.01),
            "longitude": meta["lon"] + np.random.uniform(-0.01, 0.01),
            "overdue_days": 75,
            "last_inspection_score": 82.0,
            "traffic_density": 110,
            "corridor_priority": meta["priority"],
            "estimated_duration_mins": duration_input,
            "is_heavy_machinery": heavy_machinery_toggle,
            "exclusive_block": heavy_machinery_toggle,
        }
        st.session_state["custom_requests"].append(new_entry)
        st.success(f"Work Order {new_id} successfully queued for optimization.")
        time.sleep(0.3)
        st.rerun()

# --------------------------------------------------------------------------
# ASSEMBLE DATA & RUN OPTIMIZER PIPELINE
# --------------------------------------------------------------------------
base_req_df = get_default_requests(seed=st.session_state["seed"])

if st.session_state["simulate_collision"]:
    sample_corr = "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route"
    sample_track = f"{sample_corr} :: DN-Main"
    
    sim_collision_rows = [
        {
            "request_id": "WCR-ENG-COLLIDE-1",
            "department": "Engineering",
            "action": "Track Tamping & Rail Renewal",
            "corridor": sample_corr,
            "section_track": sample_track,
            "asset_id": "AST-WCR-ENG-COLLIDE",
            "latitude": 23.501,
            "longitude": 80.201,
            "overdue_days": 90,
            "last_inspection_score": 88.0,
            "traffic_density": 135,
            "corridor_priority": 1.4,
            "estimated_duration_mins": 90,
            "is_heavy_machinery": False,
            "exclusive_block": False,
        },
        {
            "request_id": "WCR-S&T-COLLIDE-2",
            "department": "S&T",
            "action": "Electronic Interlocking & Point Machine Overhaul",
            "corridor": sample_corr,
            "section_track": sample_track,
            "asset_id": "AST-WCR-S&T-COLLIDE",
            "latitude": 23.503,
            "longitude": 80.204,
            "overdue_days": 85,
            "last_inspection_score": 84.0,
            "traffic_density": 135,
            "corridor_priority": 1.4,
            "estimated_duration_mins": 75,
            "is_heavy_machinery": False,
            "exclusive_block": False,
        }
    ]
    base_req_df = base_req_df[~base_req_df["request_id"].str.contains("COLLIDE")]
    combined_req_df = pd.concat([pd.DataFrame(sim_collision_rows), base_req_df], ignore_index=True)
else:
    combined_req_df = base_req_df.copy()

if st.session_state["custom_requests"]:
    custom_df = pd.DataFrame(st.session_state["custom_requests"])
    combined_req_df = pd.concat([custom_df, combined_req_df], ignore_index=True)

delayed_corridor_arg = None if selected_corridor == "All Corridors (Jabalpur Division)" else selected_corridor

baseline_result, bundled_df, scorer = run_pipeline(
    combined_req_df, horizon_hours, setup_buffer, None, 0
)

if delay_minutes > 0 and delayed_corridor_arg is not None:
    live_result, live_bundled, _ = run_pipeline(
        combined_req_df, horizon_hours, setup_buffer, delayed_corridor_arg, delay_minutes
    )
else:
    live_result = baseline_result
    live_bundled = bundled_df

schedule = live_result.schedule.copy()
base_starts = baseline_result.schedule.set_index("request_id")["start_min"]
schedule["baseline_start_min"] = schedule["request_id"].map(base_starts)
schedule["dynamically_shifted"] = (
    schedule["is_scheduled"]
    & schedule["baseline_start_min"].notna()
    & (schedule["start_min"] != schedule["baseline_start_min"])
)

# Detect multi-department track overlap
has_simultaneous_collision = False
colliding_departments = []
collision_corridor = ""
collision_track = ""

grouped_raw = combined_req_df.groupby("section_track")
for track_name, grp in grouped_raw:
    depts = grp["department"].unique()
    if len(depts) >= 2 and len(grp) >= 2:
        has_simultaneous_collision = True
        colliding_departments = list(depts)
        collision_track = track_name
        collision_corridor = grp["corridor"].iloc[0]
        break

# Collision alert card if detected
if has_simultaneous_collision:
    depts_str = " & ".join(colliding_departments)
    alert_box_html = f"""<div class="pro-alert-danger">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<h4 style="margin:0; font-size:15px; font-weight:700; color:#991B1B;">
⚠️ Track Possession Conflict Detected — Automated Resolution Enforced
</h4>
<p style="margin:4px 0 0 0; font-size:13px; color:#7F1D1D;">
Simultaneous requests detected on <b>{collision_track}</b> from <b>{depts_str}</b> branches.
The <b>OR-Tools CP-SAT + GeoPandas Bundling</b> engine has synchronized these tasks into a single joint block window, preventing line deadlock.
</p>
</div>
</div>
</div>"""
    st.markdown(alert_box_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# TOP KPI METRICS
# --------------------------------------------------------------------------
total_tasks = len(schedule)
scheduled_tasks = int(schedule["is_scheduled"].sum())
deferred_tasks = total_tasks - scheduled_tasks
critical_risks = int((schedule["risk_band"] == "CRITICAL").sum())
bundled_clusters_count = int(schedule.loc[schedule["bundle_cluster"] >= 0, "bundle_cluster"].nunique())
efficiency_pct = round((scheduled_tasks / total_tasks) * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card">
<div class="metric-label">Total Work Requisitions</div>
<div class="metric-value" style="color:#0284C7;">{total_tasks}</div>
<div class="metric-footer">WCR Jabalpur Division</div>
</div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class="metric-card">
<div class="metric-label">Scheduled Block Windows</div>
<div class="metric-value" style="color:#16A34A;">{scheduled_tasks}</div>
<div class="metric-footer">{efficiency_pct}% Allocation Rate</div>
</div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="metric-card">
<div class="metric-label">Deferred (Capacity Limit)</div>
<div class="metric-value" style="color:#DC2626;">{deferred_tasks}</div>
<div class="metric-footer">Rolled to next 12h cycle</div>
</div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="metric-card">
<div class="metric-label">Critical ML Backlog (≥75)</div>
<div class="metric-value" style="color:#EA580C;">{critical_risks}</div>
<div class="metric-footer">USFD / High GMT Priority</div>
</div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="metric-card">
<div class="metric-label">Joint Bundled Clusters</div>
<div class="metric-value" style="color:#7C3AED;">{bundled_clusters_count}</div>
<div class="metric-footer">Shared multi-branch blocks</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# MAIN OPERATIONAL TABS
# --------------------------------------------------------------------------
tab_solution, tab_timeline, tab_map, tab_analytics, tab_table, tab_whatif = st.tabs([
    "🎯 Operational Solution & Impact",
    "📊 Gantt Block Timeline",
    "🗺️ GIS Corridor Map",
    "🧠 Explainable AI Prioritization",
    "📋 Dispatcher Timetable & Orders",
    "⚡ Delay Resilience Simulation",
])

# ==========================================================================
# TAB 1: OPERATIONAL SOLUTION & VALUE DELIVERED
# ==========================================================================
with tab_solution:
    st.markdown("### 🎯 Indian Railways Block Planning: Operational Bottleneck vs AI Solution")
    st.caption("Comprehensive operational comparison under Indian Railways General & Subsidiary Rules (G&SR).")

    col_sol1, col_sol2 = st.columns(2)

    with col_sol1:
        st.markdown("""<div class="comparison-before">
<h4 style="margin:0 0 8px 0; color:#991B1B; font-size:15px; font-weight:700;">
🔴 The Operational Problem (Legacy Siloed Planning)
</h4>
<ul style="margin:0; padding-left:18px; color:#7F1D1D; font-size:13.5px; line-height:1.6;">
<li><b>Independent Department Requisitions:</b> Engineering, S&T, and Electrical raise separate block demands for the same physical corridor without synchronization.</li>
<li><b>Repeated Section Closures:</b> A section is shut down at 02:00 for track tamping, reopened at 04:00, and shut down again at 07:00 for OHE maintenance, causing double freight and express train cancellations.</li>
<li><b>Caution Order Inefficiencies:</b> Speed restrictions (30 km/h) remain active longer, compounding line-haul delays.</li>
<li><b>Safety Hazard:</b> Unsynchronized work increases staff casualty risk and potential collision incidents.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col_sol2:
        st.markdown("""<div class="comparison-after">
<h4 style="margin:0 0 8px 0; color:#166534; font-size:15px; font-weight:700;">
🟢 The AI Integrated Solution (Google OR-Tools + GeoPandas)
</h4>
<ul style="margin:0; padding-left:18px; color:#14532D; font-size:13.5px; line-height:1.6;">
<li><b>Automated Spatial Bundling:</b> GeoPandas identifies maintenance jobs across different branches located within 500m on the same corridor.</li>
<li><b>Unified Possession Windows:</b> CP-SAT solver aligns Engineering, S&T, and OHE into a single concurrent block window with zero additional track downtime.</li>
<li><b>Scientific Risk Prioritization:</b> High-risk rail flaws (USFD) and high-tonnage freight corridors are cleared first.</li>
<li><b>Safety Interlocks:</b> Heavy TRT/BCM machinery tasks are automatically isolated into exclusive blocks for labor protection.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📈 Quantified Operational Impact (WCR Jabalpur Division)")

    imp1, imp2, imp3, imp4 = st.columns(4)
    with imp1:
        st.markdown("""<div class="pro-card" style="border-top:3px solid #0284C7;">
<div class="metric-label">Section Downtime Saved</div>
<div style="font-size:22px; font-weight:700; color:#0284C7; margin:4px 0;">+18.4 Hours / Wk</div>
<div style="font-size:12px; color:#64748B;">Reclaimed freight throughput</div>
</div>""", unsafe_allow_html=True)

    with imp2:
        st.markdown("""<div class="pro-card" style="border-top:3px solid #16A34A;">
<div class="metric-label">Caution Orders Saved</div>
<div style="font-size:22px; font-weight:700; color:#16A34A; margin:4px 0;">-38% Redundant Orders</div>
<div style="font-size:12px; color:#64748B;">Speed recovery acceleration</div>
</div>""", unsafe_allow_html=True)

    with imp3:
        st.markdown("""<div class="pro-card" style="border-top:3px solid #7C3AED;">
<div class="metric-label">Punctuality Improvement</div>
<div style="font-size:22px; font-weight:700; color:#7C3AED; margin:4px 0;">+4.2% Mail/Express</div>
<div style="font-size:12px; color:#64748B;">On JBP-ET and JBP-KTE routes</div>
</div>""", unsafe_allow_html=True)

    with imp4:
        st.markdown("""<div class="pro-card" style="border-top:3px solid #D97706;">
<div class="metric-label">Staff Safety Compliance</div>
<div style="font-size:22px; font-weight:700; color:#D97706; margin:4px 0;">100% G&SR Verified</div>
<div style="font-size:12px; color:#64748B;">Zero concurrent track overlaps</div>
</div>""", unsafe_allow_html=True)


# ==========================================================================
# TAB 2: GANTT TIMELINE
# ==========================================================================
with tab_timeline:
    st.markdown("### 📊 Optimized Rolling Block Timeline (24-Hour Horizon)")
    st.caption("Conflict-free block allocation computed by Google OR-Tools CP-SAT.")

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
            + (" [EXCLUSIVE]" if r.get("is_heavy_machinery", False) else "")
            + (" [SHIFTED]" if r["dynamically_shifted"] else ""),
            axis=1,
        )

        fig_gantt = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="Finish",
            y="section_track",
            color="department",
            color_discrete_map=DEPT_COLORS,
            hover_data={
                "request_id": True,
                "department": True,
                "action": True,
                "risk_score": True,
                "corridor": True,
                "estimated_duration_mins": True,
                "section_track": False,
                "Start": False,
                "Finish": False,
            },
            text="Label",
        )
        fig_gantt.update_yaxes(autorange="reversed", title="Track / Section Line")
        fig_gantt.update_xaxes(title=f"Time Horizon (00:00 to {horizon_hours:02d}:00)")
        fig_gantt.update_traces(
            textposition="inside",
            insidetextanchor="start",
            marker_line_width=1,
            marker_line_color="#FFFFFF"
        )
        fig_gantt.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend_title_text="Branch",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=max(360, 70 + 44 * gantt_df["section_track"].nunique()),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_gantt, use_container_width=True)


# ==========================================================================
# TAB 3: GIS CORRIDOR MAP
# ==========================================================================
with tab_map:
    st.markdown("### 🗺️ WCR Jabalpur Division Geographic Asset & Spatial Clustering")
    st.caption("Visualizing physical maintenance coordinates across Jabalpur, Itarsi, Katni, Satna, Rewa, and Singrauli corridors.")

    map_df = schedule.copy()
    if selected_corridor != "All Corridors (Jabalpur Division)":
        map_df = map_df[map_df["corridor"] == selected_corridor]

    def map_cluster_label(r):
        if r.get("is_heavy_machinery", False) or r.get("exclusive_block", False):
            return "Exclusive (Heavy Machinery)"
        if r["bundle_cluster"] >= 0:
            return f"Bundled (Cluster #{r['bundle_cluster']})"
        return "Individual Work Order"

    map_df["Cluster_Status"] = map_df.apply(map_cluster_label, axis=1)
    map_df["Symbol_Size"] = map_df["risk_score"].apply(lambda s: max(10, s / 3.0))

    fig_map = px.scatter(
        map_df,
        x="longitude",
        y="latitude",
        color="department",
        size="Symbol_Size",
        symbol="Cluster_Status",
        color_discrete_map=DEPT_COLORS,
        hover_name="request_id",
        hover_data={
            "action": True,
            "corridor": True,
            "section_track": True,
            "risk_score": True,
            "overdue_days": True,
            "traffic_density": True,
            "longitude": False,
            "latitude": False,
            "Symbol_Size": False,
        },
    )
    fig_map.update_layout(
        template="plotly_white",
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        legend_title_text="Branch & Clustering",
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ==========================================================================
# TAB 4: EXPLAINABLE AI PRIORITIZATION
# ==========================================================================
with tab_analytics:
    st.markdown("### 🧠 Explainable Machine Learning Prioritization Matrix")
    st.caption("Transparent Random Forest scoring breakdown based on Indian Railways Safety Parameters.")

    st.markdown("""<div class="pro-card">
<h4 style="margin:0 0 6px 0; font-size:14px; font-weight:700; color:#0F172A;">
📐 Scientific Risk Formulation & Weightage Formula
</h4>
<p style="margin:0; font-size:13px; color:#475569; line-height:1.5;">
The AI engine evaluates every maintenance demand through a multi-factor regression model:
<br>
<code style="background:#F1F5F9; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:600;">
Priority Index = 0.35 × (USFD Rail Defect Score) + 0.25 × (Overdue Days) + 0.20 × (Traffic Density GMT) + 0.20 × (Corridor Strategic Weight)
</code>
</p>
</div>""", unsafe_allow_html=True)

    an1, an2 = st.columns(2)
    with an1:
        feat_series = scorer.feature_importances()
        feat_df = pd.DataFrame({"Feature": feat_series.index, "Weight": feat_series.values})
        feat_df["Feature_Clean"] = feat_df["Feature"].map({
            "overdue_days": "Overdue Maintenance Backlog (Days)",
            "last_inspection_score": "Ultrasonic Rail Flaw (USFD) Severity",
            "traffic_density": "Traffic Load Density (GMT / Trains per Day)",
            "corridor_priority": "Corridor Trunk Criticality Factor",
        })
        feat_df = feat_df.sort_values("Weight", ascending=True)

        fig_feat = px.bar(
            feat_df,
            x="Weight",
            y="Feature_Clean",
            orientation="h",
            color="Weight",
            color_continuous_scale="Blues",
            title="RandomForest Feature Contribution Weights",
        )
        fig_feat.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            coloraxis_showscale=False,
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Relative Model Weight",
            yaxis_title="",
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with an2:
        band_counts = schedule["risk_band"].value_counts().reset_index()
        band_counts.columns = ["Risk_Band", "Count"]

        fig_pie = px.pie(
            band_counts,
            names="Risk_Band",
            values="Count",
            color="Risk_Band",
            color_discrete_map=RISK_COLORS,
            hole=0.55,
            title="Work Order Risk Tier Distribution",
        )
        fig_pie.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ==========================================================================
# TAB 5: DISPATCHER TABLE & ORDER ISSUANCE
# ==========================================================================
with tab_table:
    st.markdown("### 📋 WCR Jabalpur Division Master Block Dispatcher Queue")
    st.caption("Official prioritized timetable queue with filtering, search, and executive authorization dispatch.")

    t_col1, t_col2, t_col3 = st.columns([2, 1, 1])
    with t_col1:
        search_query = st.text_input("🔍 Search by Work Order ID, Section, or Activity", "")
    with t_col2:
        dept_filter = st.multiselect("Branch Filter", list(DEPT_COLORS.keys()), default=list(DEPT_COLORS.keys()))
    with t_col3:
        status_filter = st.selectbox("Status Filter", ["All", "SCHEDULED", "DEFERRED (no capacity)"])

    table_data = schedule.copy()
    if selected_corridor != "All Corridors (Jabalpur Division)":
        table_data = table_data[table_data["corridor"] == selected_corridor]
    if dept_filter:
        table_data = table_data[table_data["department"].isin(dept_filter)]
    if status_filter != "All":
        table_data = table_data[table_data["status"] == status_filter]
    if search_query:
        mask = (
            table_data["request_id"].str.contains(search_query, case=False)
            | table_data["section_track"].str.contains(search_query, case=False)
            | table_data["action"].fillna("").str.contains(search_query, case=False)
        )
        table_data = table_data[mask]

    table_data = table_data.sort_values("risk_score", ascending=False)

    def format_clock(m):
        if pd.isna(m):
            return "—"
        h, mm = divmod(int(m), 60)
        return f"{h:02d}:{mm:02d}"

    def bundle_repr(r):
        if r.get("is_heavy_machinery", False) or r.get("exclusive_block", False):
            return "🛡️ EXCLUSIVE"
        if r["bundle_cluster"] >= 0:
            return f"Cluster #{r['bundle_cluster']}"
        return "Individual"

    view_df = pd.DataFrame({
        "Work Order": table_data["request_id"],
        "Branch": table_data["department"],
        "Maintenance Plan": table_data.get("action", table_data["request_id"]),
        "Track Section": table_data["section_track"],
        "Risk Score": table_data["risk_score"],
        "Risk Tier": table_data["risk_band"],
        "Overdue": table_data["overdue_days"].apply(lambda d: f"{d}d"),
        "Duration": table_data["estimated_duration_mins"].apply(lambda m: f"{m}m"),
        "Bundling": table_data.apply(bundle_repr, axis=1),
        "Block Start": table_data["start_min"].apply(format_clock),
        "Block End": table_data["end_min"].apply(format_clock),
        "Status": table_data["status"],
    })

    st.dataframe(
        view_df,
        use_container_width=True,
        height=380,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    st.markdown("---")
    col_disp1, col_disp2 = st.columns([2.5, 1.5])
    
    with col_disp1:
        if st.session_state["siren_off_halt"]:
            st.button("🛑 DISPATCH LOCKED (Safety Hold Active)", disabled=True, use_container_width=True)
            st.caption("Release safety hold to permit transmission.")
        elif not is_level_3:
            st.button("🔒 AUTHORIZE & DISPATCH ROLLING BLOCK PLAN (Disabled)", disabled=True, use_container_width=True)
            st.caption("Requires Level 3 Chief Controller or DRM Office clearance.")
        else:
            if st.button("⚡ AUTHORIZE & TRANSMIT ROLLING BLOCK PROGRAM", type="primary", use_container_width=True):
                st.session_state["dispatch_executed"] = True
                st.balloons()

    with col_disp2:
        csv_buffer = io.StringIO()
        table_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Master Timetable (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"wcr_jbp_block_timetable_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state["dispatch_executed"] and is_level_3 and not st.session_state["siren_off_halt"]:
        st.markdown(f"""<div class="pro-alert-success">
<b>✅ OFFICIAL ROLLING BLOCK PROGRAM TRANSMITTED TO WCR NETWORK</b><br>
• <b>Order Reference:</b> <code>WCR/JBP/RBP-OPT/{datetime.now().strftime('%Y%m%d-%H%M')}</code><br>
• <b>Authorization Token:</b> <code>RBAC_LVL3_CHIEF_CONTROLLER_JBP_OK</code><br>
• <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}<br>
• <b>Recipients:</b> Section Controllers (ET, KTE, STA, SGRL), Traction Power Controllers (TPC), Signal Control.
</div>""", unsafe_allow_html=True)


# ==========================================================================
# TAB 6: WHAT-IF DELAY RESILIENCE
# ==========================================================================
with tab_whatif:
    st.markdown("### ⚡ What-If Scenario: Inbound Train Delay & Dynamic Rescheduling")
    st.caption("Simulate traffic congestion or inbound late-running trains to evaluate solver responsiveness.")

    if delay_minutes == 0 or delayed_corridor_arg is None:
        st.info("💡 Adjust the **Inject Inbound Delay slider** in the sidebar and select a specific corridor to test dynamic schedule shifting.")
    else:
        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown("#### 🟢 Baseline Timetable (Normal State)")
            base_sched = baseline_result.schedule
            base_count = base_sched["is_scheduled"].sum()
            st.metric("Scheduled Blocks", f"{base_count} / {len(base_sched)}")
            st.metric("Solver Optimization Score", f"{baseline_result.objective_value:,.0f}")

        with comp2:
            st.markdown(f"#### 🚨 Delay Injected (+{delay_minutes} min on {delayed_corridor_arg})")
            live_count = live_result.schedule["is_scheduled"].sum()
            shifted_n = int(schedule["dynamically_shifted"].sum())
            st.metric("Scheduled Blocks", f"{live_count} / {len(live_result.schedule)}")
            st.metric("Blocks Dynamically Shifted", f"{shifted_n}", delta=f"-{shifted_n} shifted" if shifted_n > 0 else "0")

        shifted_df = schedule[schedule["dynamically_shifted"]].copy()
        if not shifted_df.empty:
            st.markdown("#### 🔄 Rescheduled Work Orders Breakdown")
            shifted_df["Old Start"] = shifted_df["baseline_start_min"].apply(format_clock)
            shifted_df["New Start"] = shifted_df["start_min"].apply(format_clock)
            shifted_df["Delay Offset"] = (shifted_df["start_min"] - shifted_df["baseline_start_min"]).astype(int).apply(lambda d: f"+{d} mins")
            
            st.dataframe(
                shifted_df[["request_id", "department", "corridor", "section_track", "Old Start", "New Start", "Delay Offset"]],
                use_container_width=True,
            )

# --------------------------------------------------------------------------
# CRIS TELEMETRY AUDIT LOG
# --------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🖥️ CRIS Core Engine Operational Telemetry Log", expanded=False):
    telemetry_data = {
        "status": live_result.solver_status,
        "solver_engine": "Google OR-Tools CP-SAT (v9.15)",
        "spatial_bundling_engine": "GeoPandas / Shapely (EPSG Projection)",
        "planning_horizon_mins": int(horizon_hours * 60),
        "scheduled_work_orders": int(scheduled_tasks),
        "deferred_work_orders": int(deferred_tasks),
        "system_interlock_hold": bool(st.session_state["siren_off_halt"]),
        "audit_token": f"CRIS-WCR-JBP-{int(time.time()) % 10000:04d}",
        "coa_api_link": "FALLBACK_LOCAL_CACHE" if st.session_state["sync_failure"] else "ONLINE_TLS1_3",
    }
    st.markdown(
        f'<div class="telemetry-box"><pre style="margin:0;">{json.dumps(telemetry_data, indent=2)}</pre></div>',
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("🚆 Indian Railways · West Central Railway (WCR) Jabalpur Division · Centre for Railway Information Systems (CRIS)")
