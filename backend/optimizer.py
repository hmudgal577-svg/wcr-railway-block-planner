"""
optimizer.py
-------------
Operations Research scheduling layer -- Google OR-Tools CP-SAT.

Given a set of risk-scored, geo-clustered maintenance requests, this module
builds a Constraint Programming model that decides, for the current planning
horizon:
    (a) WHICH requests get a block window (some may not fit -- deferred),
    (b) WHEN each scheduled request's block starts,
subject to two hard rules and one soft (rewarded) preference:

  HARD  1. No two blocks on the SAME physical section/track may overlap
            in time (safety-critical -- you cannot single-line block two
            gangs on the same track at once).
  HARD  2. Every block must fit fully inside the planning horizon.
  SOFT     Requests that were geo-clustered together (different departments,
            <=500m apart, same corridor) are REWARDED for starting at the
            exact same time -- i.e. merged into one combined block window --
            because a single combined possession is operationally cheaper
            (one line-block permission, one safety caution order) than
            several separate ones.

The objective maximizes total scheduled AI-risk-weighted value plus a
bundling bonus, i.e. "get the highest-risk backlog off the books, and merge
whatever can safely be merged."
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from ortools.sat.python import cp_model

BUNDLE_BONUS_WEIGHT = 150  # reward (in objective points) per successful merge


@dataclass
class OptimizerResult:
    schedule: pd.DataFrame       # one row per request with start/end/status
    solver_status: str
    objective_value: float
    horizon_minutes: int


def run_block_optimizer(
    scored_df: pd.DataFrame,
    horizon_hours: int = 12,
    setup_buffer_minutes: int = 15,
    delayed_corridor: Optional[str] = None,
    delay_minutes: int = 0,
    time_limit_s: float = 8.0,
) -> OptimizerResult:
    """
    scored_df must already contain: request_id, department, corridor,
    section_track, risk_score, estimated_duration_mins, bundle_cluster.

    delayed_corridor / delay_minutes implement the live "inject inbound
    train delay" simulation: every request on that corridor becomes
    unavailable (earliest-start pushed back) by delay_minutes, forcing the
    solver to recompute / shift the timeline live.
    """
    df = scored_df.reset_index(drop=True).copy()
    horizon = int(horizon_hours * 60)
    n = len(df)

    model = cp_model.CpModel()

    starts, ends, intervals, scheduled_bools = [], [], [], []

    for i, row in df.iterrows():
        duration = int(row["estimated_duration_mins"]) + int(setup_buffer_minutes)
        earliest_start = 0
        if delayed_corridor and row["corridor"] == delayed_corridor and delay_minutes > 0:
            earliest_start = int(delay_minutes)

        # If a task literally cannot fit in the horizon even starting at the
        # earliest possible moment, it is structurally infeasible this
        # window -- pin is_scheduled to False rather than leaving the solver
        # to discover an infeasible domain.
        latest_start = horizon - duration
        is_scheduled = model.NewBoolVar(f"sched_{i}")
        if latest_start < earliest_start:
            model.Add(is_scheduled == 0)
            latest_start = earliest_start  # dummy valid domain, unused since forced off

        start = model.NewIntVar(earliest_start, max(earliest_start, latest_start), f"start_{i}")
        end = model.NewIntVar(earliest_start, horizon, f"end_{i}")
        interval = model.NewOptionalIntervalVar(start, duration, end, is_scheduled, f"iv_{i}")

        starts.append(start)
        ends.append(end)
        intervals.append(interval)
        scheduled_bools.append(is_scheduled)

    # HARD RULE 1: no-overlap per physical section/track
    for track in df["section_track"].unique():
        idxs = df.index[df["section_track"] == track].tolist()
        if len(idxs) > 1:
            model.AddNoOverlap([intervals[i] for i in idxs])

    # SOFT RULE: multi-department geo-cluster bundling bonus
    bundle_bonus_vars = []
    for cluster_id in sorted(df.loc[df["bundle_cluster"] >= 0, "bundle_cluster"].unique()):
        members = df.index[df["bundle_cluster"] == cluster_id].tolist()
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            bonus = model.NewBoolVar(f"bundle_{cluster_id}_{anchor}_{other}")
            model.Add(starts[anchor] == starts[other]).OnlyEnforceIf(bonus)
            model.Add(scheduled_bools[anchor] == 1).OnlyEnforceIf(bonus)
            model.Add(scheduled_bools[other] == 1).OnlyEnforceIf(bonus)
            bundle_bonus_vars.append(bonus)

    # OBJECTIVE: maximize risk-weighted throughput + bundling bonus
    risk_terms = []
    for i, row in df.iterrows():
        weight = int(round(float(row["risk_score"]) * 10))  # scale to int
        risk_terms.append(weight * scheduled_bools[i])

    model.Maximize(
        sum(risk_terms) + BUNDLE_BONUS_WEIGHT * sum(bundle_bonus_vars)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    rows_out = []
    for i, row in df.iterrows():
        sched = solver.Value(scheduled_bools[i]) == 1
        s = solver.Value(starts[i]) if sched else None
        e = solver.Value(ends[i]) if sched else None
        rows_out.append({
            **row.to_dict(),
            "is_scheduled": sched,
            "status": "SCHEDULED" if sched else "DEFERRED (no capacity)",
            "start_min": s,
            "end_min": e,
        })

    schedule_df = pd.DataFrame(rows_out)
    obj_value = solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0.0

    return OptimizerResult(
        schedule=schedule_df,
        solver_status=status_name,
        objective_value=obj_value,
        horizon_minutes=horizon,
    )
