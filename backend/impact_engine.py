"""
impact_engine.py
----------------
Passenger & Freight Traffic Impact and Model-Based Financial Estimation Engine
for TrackYukti (WCR Jabalpur Division).

Includes:
1. Passenger Traffic Profile (High, Medium, Low periods & Recommended Windows)
2. Freight Impact Analysis (Simulated Freight Rakes, Delays, Alternatives)
3. Model-Based Financial Demurrage & Punctuality Avoidance Estimation
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


# 24-Hour Passenger Traffic Profile (Typical WCR Jabalpur Trunk Route)
# Hour (0-23) -> (Train count, Traffic Category)
HOURLY_PASSENGER_PROFILE = [
    (0, 2, "LOW", "Night Freight Priority Corridor"),
    (1, 1, "LOW", "Night Freight Priority Corridor"),
    (2, 1, "LOW", "Optimal Heavy Possession Window"),
    (3, 2, "LOW", "Optimal Heavy Possession Window"),
    (4, 3, "MEDIUM", "Early Morning Mail/Express Inbound"),
    (5, 5, "HIGH", "Morning Commuter / Intercity Peak"),
    (6, 8, "HIGH", "Morning Commuter / Intercity Peak"),
    (7, 9, "HIGH", "Morning Superfast / Vande Bharat Corridor"),
    (8, 7, "HIGH", "Morning Express Clearing"),
    (9, 6, "MEDIUM", "Post-Peak Freight Headway"),
    (10, 4, "MEDIUM", "Midday Secondary Passenger"),
    (11, 2, "LOW", "RECOMMENDED MIDDAY ROLLING BLOCK WINDOW"),
    (12, 2, "LOW", "RECOMMENDED MIDDAY ROLLING BLOCK WINDOW"),
    (13, 2, "LOW", "RECOMMENDED MIDDAY ROLLING BLOCK WINDOW"),
    (14, 3, "LOW", "RECOMMENDED MIDDAY ROLLING BLOCK WINDOW"),
    (15, 4, "MEDIUM", "Afternoon Express Convergence"),
    (16, 5, "MEDIUM", "Evening Outbound Staging"),
    (17, 8, "HIGH", "Evening Commuter / Express Peak"),
    (18, 9, "HIGH", "Evening Superfast Corridor"),
    (19, 8, "HIGH", "Evening Mail/Express Peak"),
    (20, 6, "MEDIUM", "Night Long-Distance Dispatch"),
    (21, 5, "MEDIUM", "Night Mail Departures"),
    (22, 3, "LOW", "Night Goods Staging"),
    (23, 2, "LOW", "Night Freight Priority Corridor"),
]


# Simulated Freight Rakes operating in WCR Jabalpur Division
SIMULATED_FREIGHT_RAKES = [
    {"rake_id": "FRT-JBP-701", "name": "Singrauli Coal BoxNHL Rake #1", "corridor": "Katni (KTE) - Singrauli Coal Logistics Line", "cargo": "Thermal Coal", "weight_tonnes": 3800, "scheduled_slot_hr": 2, "priority": "HIGH"},
    {"rake_id": "FRT-JBP-702", "name": "Singrauli Coal BoxNHL Rake #2", "corridor": "Katni (KTE) - Singrauli Coal Logistics Line", "cargo": "Thermal Coal", "weight_tonnes": 3950, "scheduled_slot_hr": 3, "priority": "HIGH"},
    {"rake_id": "FRT-JBP-703", "name": "Katni BCNHL Cement Logistics Rake", "corridor": "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route", "cargo": "Packaged Cement", "weight_tonnes": 2600, "scheduled_slot_hr": 11, "priority": "MEDIUM"},
    {"rake_id": "FRT-JBP-704", "name": "CONCOR Container Rake (Mundra Port Bound)", "corridor": "Jabalpur (JBP) - Itarsi (ET) Trunk Line", "cargo": "Export Containers", "weight_tonnes": 2400, "scheduled_slot_hr": 12, "priority": "HIGH"},
    {"rake_id": "FRT-JBP-705", "name": "Bina POL Tank Wagon Rake (BTPN)", "corridor": "Katni (KTE) - Bina (BINA) Coal Corridor", "cargo": "High Speed Diesel", "weight_tonnes": 3200, "scheduled_slot_hr": 13, "priority": "HIGH"},
    {"rake_id": "FRT-JBP-706", "name": "Rewa Clinker Goods Rake", "corridor": "Satna (STA) - Rewa (REWA) Branch Corridor", "cargo": "Industrial Clinker", "weight_tonnes": 2800, "scheduled_slot_hr": 14, "priority": "MEDIUM"},
    {"rake_id": "FRT-JBP-707", "name": "FCI Foodgrain BCNHL Special", "corridor": "Jabalpur (JBP) - Itarsi (ET) Trunk Line", "cargo": "Wheat / Grains", "weight_tonnes": 2500, "scheduled_slot_hr": 1, "priority": "MEDIUM"},
    {"rake_id": "FRT-JBP-708", "name": "Steel Authority BFR Rake (Bhilai Inbound)", "corridor": "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route", "cargo": "Finished Steel Coils", "weight_tonnes": 3100, "scheduled_slot_hr": 18, "priority": "HIGH"},
]


def get_passenger_traffic_summary() -> Dict[str, Any]:
    """Provides passenger traffic analysis and window recommendations."""
    df = pd.DataFrame(HOURLY_PASSENGER_PROFILE, columns=["hour", "train_count", "category", "notes"])
    low_hours = df[df["category"] == "LOW"]["hour"].tolist()
    high_hours = df[df["category"] == "HIGH"]["hour"].tolist()

    return {
        "profile_df": df,
        "peak_passenger_hours": high_hours,
        "recommended_day_block_window": "11:30 – 15:30 IST (Midday Passenger Valley — Lowest Disruption)",
        "recommended_night_block_window": "00:30 – 04:30 IST (Night Freight Staging — Zero Passenger Cancellations)",
        "total_passenger_trains_24h": int(df["train_count"].sum()),
    }


def compute_freight_impact(schedule_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes which freight rakes interact with scheduled maintenance blocks
    and computes delay estimates, alternative windows, and severity ratings.
    """
    sched = schedule_df[schedule_df.get("is_scheduled", True)].copy()
    impacted_rakes = []
    total_delay_minutes = 0

    for rake in SIMULATED_FREIGHT_RAKES:
        corridor = rake["corridor"]
        slot_hr = rake["scheduled_slot_hr"]
        slot_min_start = slot_hr * 60
        slot_min_end = slot_min_start + 45  # 45-min transit through section

        # Check if any maintenance block occupies this corridor around this time
        matching_blocks = sched[sched["corridor"] == corridor]
        delay = 0
        conflicting_block = None

        for _, blk in matching_blocks.iterrows():
            b_start = int(blk.get("start_min", 0))
            b_end = int(blk.get("end_min", 90))

            if not (slot_min_end < b_start or slot_min_start > b_end):
                # Overlap! Freight must wait or loop
                delay = max(delay, (b_end - slot_min_start) + 10)  # 10m clearing buffer
                conflicting_block = str(blk.get("request_id"))

        if delay > 0:
            severity = "CRITICAL" if delay > 60 else ("MODERATE" if delay > 30 else "MINOR")
            alt_hr = (slot_hr + int(np.ceil(delay / 60.0))) % 24
            alt_window = f"{alt_hr:02d}:30 – {alt_hr+1:02d}:15 IST (Loop Regulated)"
        else:
            delay = 0
            severity = "ZERO IMPACT"
            alt_window = "On-Time Dispatch Path Confirmed"

        total_delay_minutes += delay

        impacted_rakes.append({
            "rake_id": rake["rake_id"],
            "rake_name": rake["name"],
            "corridor": rake["corridor"],
            "cargo": rake["cargo"],
            "scheduled_time": f"{slot_hr:02d}:00 IST",
            "estimated_delay_mins": delay,
            "impact_severity": severity,
            "conflicting_block": conflicting_block if conflicting_block else "None (Clear Line)",
            "alternative_window": alt_window,
        })

    impact_df = pd.DataFrame(impacted_rakes)
    affected_count = int((impact_df["estimated_delay_mins"] > 0).sum())

    return {
        "impact_df": impact_df,
        "affected_freight_trains": affected_count,
        "total_freight_delay_mins": total_delay_minutes,
        "average_delay_mins": round(total_delay_minutes / max(1, affected_count), 1) if affected_count > 0 else 0,
    }


def compute_financial_impact(
    affected_trains: int,
    total_delay_mins: int,
    cost_factor_per_min: float = 1200.0,
    bundling_savings_multiplier: float = 1.45,
) -> Dict[str, Any]:
    """
    Computes model-based financial demurrage & detention estimation.
    Formula:
        Raw Delay Cost = Affected Trains * Delay Minutes * Cost Factor
    Compares Without Optimization (uncoordinated) vs With Optimization (TrackYukti).
    """
    # Optimized (With TrackYukti) cost
    cost_with_opt = float(total_delay_mins * cost_factor_per_min)

    # Without optimization: 45% more delay due to fragmented unbundled blocks and repeated caution orders
    detention_without_opt_mins = int(total_delay_mins * bundling_savings_multiplier) + (affected_trains * 25)
    cost_without_opt = float(detention_without_opt_mins * cost_factor_per_min)

    avoided_impact = max(0.0, cost_without_opt - cost_with_opt)
    avoided_lakhs = round(avoided_impact / 100000.0, 2)
    with_opt_lakhs = round(cost_with_opt / 100000.0, 2)
    without_opt_lakhs = round(cost_without_opt / 100000.0, 2)

    # Environmental / Traction audit estimation
    diesel_hours_saved = round((detention_without_opt_mins - total_delay_mins) / 60.0 * 2.8, 1)
    diesel_litres_saved = int(diesel_hours_saved * 18.5)
    co2_reduction_kg = int(diesel_litres_saved * 2.68)

    return {
        "cost_factor_per_min": cost_factor_per_min,
        "cost_without_optimization_rs": cost_without_opt,
        "cost_without_optimization_lakhs": without_opt_lakhs,
        "cost_with_optimization_rs": cost_with_opt,
        "cost_with_optimization_lakhs": with_opt_lakhs,
        "avoided_impact_rs": avoided_impact,
        "avoided_impact_lakhs": avoided_lakhs,
        "diesel_hours_saved": diesel_hours_saved,
        "diesel_litres_saved": diesel_litres_saved,
        "co2_reduction_kg": co2_reduction_kg,
    }
