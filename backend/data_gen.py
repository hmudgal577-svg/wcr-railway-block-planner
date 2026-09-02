"""
data_gen.py
------------
Generates realistic synthetic maintenance / block requests focused on
West Central Railway (WCR) — Jabalpur Division.

Each row represents ONE maintenance block request raised by a specific branch:
- Engineering (Track Staff)
- S&T (Signal & Telecom)
- Electrical (OHE Maintenance)
"""

import numpy as np
import pandas as pd

DEPARTMENTS = ["Engineering", "S&T", "Electrical"]

# Exact corridors in WCR Jabalpur Division:
# (corridor name -> tracks, anchor coordinates in MP, priority weighting)
CORRIDORS = {
    "Jabalpur (JBP) - Itarsi (ET) Trunk Line": {
        "tracks": ["UP-Main", "DN-Main"],
        "lat": 22.95,
        "lon": 78.85,
        "priority": 1.5,
    },
    "Jabalpur (JBP) - Katni (KTE) Heavy Freight Route": {
        "tracks": ["UP-Main", "DN-Main", "Goods-Loop"],
        "lat": 23.50,
        "lon": 80.20,
        "priority": 1.4,
    },
    "Satna (STA) - Rewa (REWA) Branch Corridor": {
        "tracks": ["Single-Line", "Loop-1"],
        "lat": 24.55,
        "lon": 81.05,
        "priority": 1.1,
    },
    "Katni (KTE) - Singrauli Coal Logistics Line": {
        "tracks": ["Coal-Line-1", "Coal-Line-2", "Single-Line"],
        "lat": 24.05,
        "lon": 81.85,
        "priority": 1.3,
    },
}

BRANCH_ACTIONS = {
    "Engineering": [
        "Track Tamping & Deep Screening",
        "Continuous Welded Rail (CWR) De-Stressing",
        "Turnout Sleeper Renewal (TSR)",
        "Ultrasonic Flaw Detection (USFD) Rail Testing",
    ],
    "S&T": [
        "Electronic Interlocking (EI) Overhaul",
        "Point Machine Motor Overhauling",
        "Digital Axle Counter (DAC) Sensor Calibration",
        "Track Circuit Shunt Resistance Testing",
    ],
    "Electrical": [
        "OHE Catenary Contact Wire Tensioning",
        "Cantilever Assembly Replacement",
        "Section Insulator & Neutral Section Inspection",
        "Traction Power Feeder Isolator Maintenance",
    ],
}


def _jitter_km_to_deg(km: float) -> float:
    """Rough conversion so synthetic points spread realistically near corridor anchor."""
    return km / 111.0


def generate_requests(n_requests: int = 26, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic pool of maintenance/block requests for WCR Jabalpur Division.
    """
    rng = np.random.default_rng(seed)
    corridor_names = list(CORRIDORS.keys())

    rows = []
    for i in range(n_requests):
        corridor = corridor_names[rng.integers(0, len(corridor_names))]
        meta = CORRIDORS[corridor]
        dept = DEPARTMENTS[rng.integers(0, len(DEPARTMENTS))]
        track = meta["tracks"][rng.integers(0, len(meta["tracks"]))]
        action = rng.choice(BRANCH_ACTIONS[dept])

        # scatter assets within ~0-2.5 km of corridor anchor
        lat = meta["lat"] + rng.uniform(-1, 1) * _jitter_km_to_deg(rng.uniform(0, 2.5))
        lon = meta["lon"] + rng.uniform(-1, 1) * _jitter_km_to_deg(rng.uniform(0, 2.5))

        overdue_days = int(rng.integers(0, 120))
        inspection_score = float(np.clip(rng.normal(55, 25), 0, 100))  # higher = worse condition
        traffic_density = int(rng.integers(30, 150))  # trains/day on that line
        duration = int(rng.choice([45, 60, 90, 120, 150, 180]))

        rows.append({
            "request_id": f"WCR-REQ-{1000 + i}",
            "department": dept,
            "action": action,
            "corridor": corridor,
            "section_track": f"{corridor} :: {track}",
            "asset_id": f"AST-WCR-{dept[:3].upper()}-{2000 + i}",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "overdue_days": overdue_days,
            "last_inspection_score": round(inspection_score, 1),
            "traffic_density": traffic_density,
            "corridor_priority": meta["priority"],
            "estimated_duration_mins": duration,
            "is_heavy_machinery": False,
        })

    return pd.DataFrame(rows)
