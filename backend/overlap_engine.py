"""
overlap_engine.py
-----------------
Overlap Detection, Task Bundling, and Exclusive Task Analysis Engine for TrackYukti.

Analyzes:
- Corridor, Section, Track, Spatial proximity (lat/lon)
- Temporal overlap (feasible window intersection)
- Departmental compatibility matrix
- Exclusive task safety criteria (BCM, TRT, 25kV OHE isolation)

Classifies into:
- FULL OVERLAP
- PARTIAL OVERLAP
- NO OVERLAP

Generates:
- Joint Work Bundles
- Partial Bundle Opportunities
- Exclusive Task Diagnostics
- Original vs Optimized Plan Comparison
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


# Compatibility matrix between departments:
# (Dept A, Dept B) -> True if they can safely share a joint window under Indian Railways rules
COMPATIBILITY_RULES = {
    ("Engineering", "S&T"): True,        # E.g. P-Way track tamping with point machine & axle counter testing
    ("Engineering", "Electrical"): True, # E.g. Rail de-stressing with OHE contact wire tensioning
    ("S&T", "Electrical"): True,         # E.g. Interlocking overhaul with feeder isolator maintenance
}

# Exclusive activities that cannot be bundled under safety standards
EXCLUSIVE_KEYWORDS = [
    "BCM", "Ballast Cleaning", "TRT", "Track Renewal Train",
    "Substation", "25kV", "Deep Screening", "Continuous Welded Rail",
]


@dataclass
class OverlapPair:
    task_a: str
    task_b: str
    dept_a: str
    dept_b: str
    corridor: str
    track_a: str
    track_b: str
    overlap_type: str  # "FULL OVERLAP", "PARTIAL OVERLAP", "NO OVERLAP"
    overlap_minutes: int
    spatial_dist_m: float
    compatibility: bool
    reason: str


@dataclass
class JointWorkBundle:
    bundle_id: str
    corridor: str
    section_track: str
    participating_departments: List[str]
    tasks: List[Dict[str, Any]]
    common_start_min: int
    common_end_min: int
    optimized_duration_mins: int
    unbundled_total_duration_mins: int
    separate_blocks_avoided: int
    time_saved_mins: int


@dataclass
class PartialBundleOpportunity:
    opportunity_id: str
    corridor: str
    tasks: List[str]
    departments: List[str]
    common_feasible_window: str
    bundled_duration_mins: int
    remaining_work_mins: int
    original_duration_mins: int
    time_saved_mins: int
    recommendation: str


@dataclass
class ExclusiveTask:
    request_id: str
    department: str
    activity: str
    corridor: str
    section_track: str
    reason: str
    min_required_duration_mins: int
    priority_score: float
    priority_level: str


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon coordinates."""
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return float(2.0 * r * np.arcsin(np.clip(np.sqrt(a), 0, 1)))


def is_exclusive_task(row: pd.Series) -> Tuple[bool, str]:
    """Detect if a task must remain EXCLUSIVE and cannot be bundled."""
    if row.get("is_heavy_machinery"):
        return True, "Heavy on-track machinery (TRT / BCM / Tamper) requiring exclusive possession & safety envelope"
    if row.get("exclusive_block"):
        return True, "Safety isolation mandate: Exclusive electrical or mechanical clearance required"

    action = str(row.get("action", ""))
    for kw in EXCLUSIVE_KEYWORDS:
        if kw.lower() in action.lower():
            return True, f"High-risk operation ({kw}): Requires exclusive safety isolation"

    return False, ""


def detect_task_overlaps(schedule_df: pd.DataFrame, max_dist_m: float = 1200.0) -> List[OverlapPair]:
    """
    Analyzes pairs of tasks across corridors and time windows,
    classifying into FULL OVERLAP, PARTIAL OVERLAP, or NO OVERLAP.
    """
    pairs = []
    sched = schedule_df[schedule_df.get("is_scheduled", True)].reset_index(drop=True)
    n = len(sched)

    for i in range(n):
        row_a = sched.iloc[i]
        for j in range(i + 1, n):
            row_b = sched.iloc[j]

            # Spatial distance
            dist = haversine_distance_m(
                row_a.get("latitude", 23.5), row_a.get("longitude", 80.0),
                row_b.get("latitude", 23.5), row_b.get("longitude", 80.0)
            )

            # Temporal overlap
            start_a, end_a = int(row_a.get("start_min", 0)), int(row_a.get("end_min", 90))
            start_b, end_b = int(row_b.get("start_min", 0)), int(row_b.get("end_min", 90))

            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            overlap_min = max(0, overlap_end - overlap_start)

            same_corridor = (row_a.get("corridor") == row_b.get("corridor"))
            same_track = (row_a.get("section_track") == row_b.get("section_track"))
            same_dept = (row_a.get("department") == row_b.get("department"))

            excl_a, _ = is_exclusive_task(row_a)
            excl_b, _ = is_exclusive_task(row_b)

            compat = not (excl_a or excl_b) and (not same_dept)

            if same_corridor and dist <= max_dist_m:
                if overlap_min >= min(end_a - start_a, end_b - start_b) * 0.75 and compat:
                    ov_type = "FULL OVERLAP"
                    reason = f"Same corridor, <= {dist:.0f}m apart, concurrent window ({overlap_min}m shared). Joint possession eligible."
                elif overlap_min > 0:
                    ov_type = "PARTIAL OVERLAP"
                    reason = f"Shared time window of {overlap_min}m with {dist:.0f}m separation. Partial synchronization feasible."
                else:
                    ov_type = "NO OVERLAP"
                    reason = f"Nearby location ({dist:.0f}m) but separated in time schedule."
            else:
                ov_type = "NO OVERLAP"
                reason = "Spatially separated (> 1.2km) or independent corridor alignment."

            pairs.append(OverlapPair(
                task_a=str(row_a.get("request_id")),
                task_b=str(row_b.get("request_id")),
                dept_a=str(row_a.get("department")),
                dept_b=str(row_b.get("department")),
                corridor=str(row_a.get("corridor")),
                track_a=str(row_a.get("section_track")),
                track_b=str(row_b.get("section_track")),
                overlap_type=ov_type,
                overlap_minutes=overlap_min,
                spatial_dist_m=round(dist, 1),
                compatibility=compat,
                reason=reason,
            ))

    return pairs


def build_joint_work_bundles(schedule_df: pd.DataFrame) -> List[JointWorkBundle]:
    """
    Groups scheduled tasks into unified Joint Work Bundles based on bundle_cluster.
    Each bundle acts as ONE coordinated work package.
    """
    bundles = []
    sched = schedule_df[schedule_df.get("is_scheduled", True)].copy()

    # Tasks with bundle_cluster >= 0 are bundled
    clustered = sched[sched.get("bundle_cluster", -1) >= 0]
    for cid, grp in clustered.groupby("bundle_cluster"):
        if len(grp) < 2:
            continue

        tasks_list = grp.to_dict(orient="records")
        depts = sorted(list(grp["department"].unique()))
        corridor = grp["corridor"].iloc[0]
        track = grp["section_track"].iloc[0]

        start_min = int(grp["start_min"].min())
        end_min = int(grp["end_min"].max())
        optimized_dur = end_min - start_min
        unbundled_total = int(grp["estimated_duration_mins"].sum())

        blocks_avoided = len(grp) - 1
        time_saved = max(0, unbundled_total - optimized_dur)

        bundles.append(JointWorkBundle(
            bundle_id=f"JWB-WCR-{cid + 1:03d}",
            corridor=corridor,
            section_track=track,
            participating_departments=depts,
            tasks=tasks_list,
            common_start_min=start_min,
            common_end_min=end_min,
            optimized_duration_mins=optimized_dur,
            unbundled_total_duration_mins=unbundled_total,
            separate_blocks_avoided=blocks_avoided,
            time_saved_mins=time_saved,
        ))

    return bundles


def find_partial_bundle_opportunities(schedule_df: pd.DataFrame) -> List[PartialBundleOpportunity]:
    """
    Identifies tasks on the same corridor that are slightly staggered
    and can be shifted into a partial joint window.
    """
    opps = []
    sched = schedule_df[schedule_df.get("is_scheduled", True)].copy()

    for corridor, grp in sched.groupby("corridor"):
        unbundled = grp[grp.get("bundle_cluster", -1) < 0].reset_index(drop=True)
        if len(unbundled) < 2:
            continue

        for i in range(len(unbundled) - 1):
            r1 = unbundled.iloc[i]
            r2 = unbundled.iloc[i + 1]

            if r1["department"] == r2["department"]:
                continue
            excl1, _ = is_exclusive_task(r1)
            excl2, _ = is_exclusive_task(r2)
            if excl1 or excl2:
                continue

            s1, e1 = int(r1["start_min"]), int(r1["end_min"])
            s2, e2 = int(r2["start_min"]), int(r2["end_min"])

            # Staggered by <= 60 mins
            if abs(s1 - s2) <= 60 and (e1 > s2 or e2 > s1):
                shared_start = max(s1, s2)
                shared_end = min(e1, e2)
                shared_dur = max(0, shared_end - shared_start)
                total_orig = (e1 - s1) + (e2 - s2)
                opt_dur = max(e1, e2) - min(s1, s2)
                saved = total_orig - opt_dur

                opps.append(PartialBundleOpportunity(
                    opportunity_id=f"PBO-{corridor[:3].upper()}-{i+1:02d}",
                    corridor=corridor,
                    tasks=[str(r1["request_id"]), str(r2["request_id"])],
                    departments=[str(r1["department"]), str(r2["department"])],
                    common_feasible_window=f"{shared_start//60:02d}:{shared_start%60:02d} – {shared_end//60:02d}:{shared_end%60:02d}",
                    bundled_duration_mins=opt_dur,
                    remaining_work_mins=max(0, (e1 - s1) - shared_dur),
                    original_duration_mins=total_orig,
                    time_saved_mins=saved,
                    recommendation=f"Advance {r2['request_id']} by {abs(s1 - s2)}m to synchronize with {r1['request_id']}. Saves {saved}m track possession.",
                ))

    return opps


def detect_exclusive_tasks(schedule_df: pd.DataFrame) -> List[ExclusiveTask]:
    """
    Extracts all exclusive tasks that must not be bundled with explicit justification.
    """
    excl_list = []
    for _, row in schedule_df.iterrows():
        is_excl, reason = is_exclusive_task(row)
        if is_excl:
            score = float(row.get("risk_score", row.get("priority_score", 75.0)))
            level = "CRITICAL" if score >= 80 else ("VERY HIGH" if score >= 65 else "HIGH")
            excl_list.append(ExclusiveTask(
                request_id=str(row.get("request_id")),
                department=str(row.get("department")),
                activity=str(row.get("action")),
                corridor=str(row.get("corridor")),
                section_track=str(row.get("section_track")),
                reason=reason,
                min_required_duration_mins=int(row.get("estimated_duration_mins", 90)),
                priority_score=score,
                priority_level=level,
            ))
    return excl_list


def compute_plan_optimization_comparison(schedule_df: pd.DataFrame, bundles: List[JointWorkBundle], exclusive_tasks: List[ExclusiveTask]) -> Dict[str, Any]:
    """
    Computes side-by-side comparison between ORIGINAL (unbundled) PLAN and OPTIMIZED PLAN.
    """
    sched = schedule_df[schedule_df.get("is_scheduled", True)]
    total_scheduled = len(sched)

    # Original uncoordinated plan: every task gets its own separate block
    original_blocks_count = total_scheduled
    original_duration_sum = int(sched["estimated_duration_mins"].sum())

    # In original unbundled plan, each separate block adds 15m safety caution + setup buffer
    original_total_possession_mins = original_duration_sum + (original_blocks_count * 15)

    # Optimized coordinated plan:
    separate_blocks_avoided = sum(b.separate_blocks_avoided for b in bundles)
    optimized_blocks_count = original_blocks_count - separate_blocks_avoided
    bundled_time_saved = sum(b.time_saved_mins for b in bundles)

    optimized_possession_mins = original_total_possession_mins - (bundled_time_saved + (separate_blocks_avoided * 15))

    time_saved_mins = max(0, original_total_possession_mins - optimized_possession_mins)
    time_saved_hrs = round(time_saved_mins / 60.0, 1)

    return {
        "total_tasks_scheduled": total_scheduled,
        "tasks_bundled_count": sum(len(b.tasks) for b in bundles),
        "separate_blocks_avoided": separate_blocks_avoided,
        "original_blocks_count": original_blocks_count,
        "optimized_blocks_count": optimized_blocks_count,
        "original_duration_mins": original_total_possession_mins,
        "optimized_duration_mins": optimized_possession_mins,
        "time_saved_mins": time_saved_mins,
        "time_saved_hrs": time_saved_hrs,
        "exclusive_tasks_count": len(exclusive_tasks),
        "joint_bundles_count": len(bundles),
        "efficiency_gain_pct": round((time_saved_mins / max(1, original_total_possession_mins)) * 100.0, 1),
    }
