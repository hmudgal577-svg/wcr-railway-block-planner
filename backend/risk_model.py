"""
risk_model.py
--------------
AI Criticality / Risk Scoring Layer.

Produces a 0-100 risk score for every incoming maintenance request using a
scikit-learn RandomForestRegressor. Since no live historical failure dataset
exists for this prototype, we bootstrap a synthetic-but-plausible training
set from domain-informed feature weightings (overdue days, asset condition
from last inspection, section traffic density, corridor criticality) and fit
the forest on that. This mirrors how the real model would later be re-fit on
actual IPMIS/TMS failure & inspection history without changing the pipeline
architecture -- swap generate_training_data() for a real data-loader and the
rest of the module is unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = [
    "overdue_days",
    "last_inspection_score",
    "traffic_density",
    "corridor_priority",
]


def _synthetic_training_frame(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Bootstrap a plausible historical training set with a known ground-truth
    risk function plus noise, so the forest has real signal to learn from."""
    rng = np.random.default_rng(seed)

    overdue_days = rng.integers(0, 180, size=n).astype(float)
    inspection_score = np.clip(rng.normal(55, 25, size=n), 0, 100)
    traffic_density = rng.integers(10, 160, size=n).astype(float)
    corridor_priority = rng.choice([0.8, 0.9, 1.0, 1.1, 1.4], size=n)

    # Domain-informed ground truth (nonlinear, matches how Indian Railways
    # weighs overdue backlog + asset condition + line criticality/traffic).
    raw = (
        0.32 * (overdue_days / 180 * 100) +
        0.30 * inspection_score +
        0.18 * (traffic_density / 160 * 100) +
        0.20 * (corridor_priority / 1.4 * 100)
    )
    noise = rng.normal(0, 5, size=n)
    risk = np.clip(raw + noise, 0, 100)

    return pd.DataFrame({
        "overdue_days": overdue_days,
        "last_inspection_score": inspection_score,
        "traffic_density": traffic_density,
        "corridor_priority": corridor_priority,
        "risk_score": risk,
    })


class CriticalityScorer:
    """Thin wrapper around a fitted RandomForestRegressor exposing a
    single score_requests(df) -> df entrypoint used by the app layer."""

    def __init__(self, n_estimators: int = 200, random_state: int = 7):
        train_df = _synthetic_training_frame(seed=random_state)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=random_state,
        )
        self.model.fit(train_df[FEATURE_COLUMNS], train_df["risk_score"])

    def score_requests(self, requests_df: pd.DataFrame) -> pd.DataFrame:
        """Attach a `risk_score` (0-100, rounded) and `risk_band` column to
        the incoming requests dataframe. Does not mutate the input."""
        out = requests_df.copy()
        preds = self.model.predict(out[FEATURE_COLUMNS])
        out["risk_score"] = np.clip(preds, 0, 100).round(1)

        def band(score: float) -> str:
            if score >= 75:
                return "CRITICAL"
            if score >= 55:
                return "HIGH"
            if score >= 35:
                return "MEDIUM"
            return "LOW"

        out["risk_band"] = out["risk_score"].apply(band)
        out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)
        return out

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
