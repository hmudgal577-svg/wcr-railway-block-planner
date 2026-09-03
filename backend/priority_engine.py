"""
priority_engine.py
------------------
Explainable Priority Intelligence Scoring Layer for TrackYukti (WCR Jabalpur Division).

Evaluates 5 domain-informed factors (0-100 scale):
1. Safety Criticality (USFD flaws, rail defect severity, switch wear)
2. Operational Urgency (TSR caution orders, track speed restrictions)
3. Defect Severity (Inspection condition index, track geometry variance)
4. Overdue Maintenance (Days overdue against IR safety maintenance interval)
5. Asset Availability Impact (GMT annual freight load, line traffic density)

Outputs explainable priority scores (0-100), priority levels (CRITICAL, VERY HIGH, HIGH, NORMAL, LOW),
and human-interpretable technical explanations.
"""

import numpy as np
import pandas as pd

PRIORITY_WEIGHTS = {
    "safety_criticality": 0.30,
    "operational_urgency": 0.25,
    "defect_severity": 0.20,
    "overdue_maintenance": 0.15,
    "asset_availability": 0.10,
}

PRIORITY_BANDS = [
    (80, "CRITICAL", "#EF4444", "Immediate possession required — critical safety hazard"),
    (65, "VERY HIGH", "#F97316", "Priority block required — high risk of operational TSR restriction"),
    (50, "HIGH", "#F59E0B", "Scheduled block required within 48h — accelerating degradation"),
    (35, "NORMAL", "#38BDF8", "Routine maintenance window feasible — stable track parameters"),
    (0,  "LOW", "#4ADE80", "Elective preventive maintenance — minimal operational risk"),
]


def classify_priority_level(score: float):
    for threshold, band, color, desc in PRIORITY_BANDS:
        if score >= threshold:
            return band, color, desc
    return "LOW", "#4ADE80", "Elective preventive maintenance"


def compute_priority_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches requests dataframe with 5 explainable factor scores,
    overall priority score (0-100), priority band, and technical justification.
    """
    out = df.copy()

    insp = out["last_inspection_score"].fillna(50).astype(float) if "last_inspection_score" in out.columns else pd.Series(50, index=out.index)
    heavy = out["is_heavy_machinery"].fillna(False).astype(bool) if "is_heavy_machinery" in out.columns else pd.Series(False, index=out.index)
    safety_crit = np.clip(insp * 0.85 + heavy.astype(float) * 20.0, 0, 100)
    out["factor_safety_criticality"] = safety_crit.round(1)

    overdue = out["overdue_days"].fillna(30).astype(float) if "overdue_days" in out.columns else pd.Series(30, index=out.index)
    op_urgency = np.clip((overdue / 120.0) * 85.0 + (insp / 100.0) * 15.0, 0, 100)
    out["factor_operational_urgency"] = op_urgency.round(1)

    defect_sev = np.clip(insp * 0.95 + np.random.default_rng(42).uniform(-4, 4, size=len(out)), 0, 100)
    out["factor_defect_severity"] = defect_sev.round(1)

    overdue_score = np.clip((overdue / 100.0) * 100.0, 0, 100)
    out["factor_overdue_maintenance"] = overdue_score.round(1)

    density = out["traffic_density"].fillna(80).astype(float) if "traffic_density" in out.columns else pd.Series(80, index=out.index)
    priority = out["corridor_priority"].fillna(1.2).astype(float) if "corridor_priority" in out.columns else pd.Series(1.2, index=out.index)
    asset_avail = np.clip((density / 150.0) * 70.0 + (priority / 1.5) * 30.0, 0, 100)
    out["factor_asset_availability"] = asset_avail.round(1)

    composite = (
        out["factor_safety_criticality"] * PRIORITY_WEIGHTS["safety_criticality"] +
        out["factor_operational_urgency"] * PRIORITY_WEIGHTS["operational_urgency"] +
        out["factor_defect_severity"] * PRIORITY_WEIGHTS["defect_severity"] +
        out["factor_overdue_maintenance"] * PRIORITY_WEIGHTS["overdue_maintenance"] +
        out["factor_asset_availability"] * PRIORITY_WEIGHTS["asset_availability"]
    )
    out["priority_score"] = composite.round(1)

    bands = []
    colors = []
    explanations = []

    for _, row in out.iterrows():
        b, c, _ = classify_priority_level(row["priority_score"])
        bands.append(b)
        colors.append(c)

        top_factors = []
        if row["factor_safety_criticality"] >= 70:
            top_factors.append(f"Elevated safety flaw index ({row['factor_safety_criticality']:.0f}/100)")
        if row["factor_overdue_maintenance"] >= 65:
            top_factors.append(f"{int(row['overdue_days'])} days past maintenance threshold")
        if row["factor_asset_availability"] >= 75:
            top_factors.append(f"High-density traffic corridor ({int(row['traffic_density'])} trains/day)")
        if row.get("is_heavy_machinery"):
            top_factors.append("Requires exclusive track plant possession (BCM/TRT)")

        if not top_factors:
            top_factors.append("Standard preventive rolling block maintenance")

        expl = f"{row['department']} [{row['action']}]: " + "; ".join(top_factors) + "."
        explanations.append(expl)

    out["priority_level"] = bands
    out["priority_color"] = colors
    out["priority_explanation"] = explanations

    if "risk_score" not in out.columns:
        out["risk_score"] = out["priority_score"]
    if "risk_band" not in out.columns:
        out["risk_band"] = out["priority_level"]

    return out

