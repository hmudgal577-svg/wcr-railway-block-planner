# AI-Powered Automatic Block Planning — Indian Railways
### Prototype: maximizing asset (track) availability by intelligently scheduling maintenance blocks

---

## 1. The problem, in plain language

Every day, different departments — **Engineering (track/civil)**, **Signal**,
**OHE (overhead electrification)**, and **Telecom** — need the track taken
out of service ("**blocked**") so their staff can safely inspect or repair
assets. Every block window is track *unavailable* to trains.

Today this is largely manual: each department requests its own block,
often on paper/spreadsheet, often without knowing another department
wants the *same stretch of track* at almost the *same time*. This causes:

- **Overlapping/duplicate blocks** on the same section → wasted line
  capacity, more train delays than necessary.
- **Missed bundling opportunities** — if Engineering and Signal both need
  the same 500m of track, one combined block is operationally cheaper
  (one caution order, one possession, one safety briefing) than two
  separate ones.
- **No objective prioritization** — genuinely overdue/high-risk assets
  don't always get scheduled first; it's often first-come-first-served.
- **No resilience to real-time disruption** — if an inbound train is
  delayed, nobody automatically re-shuffles the block plan.

**Goal of this system:** an AI/optimization layer that ingests all pending
block requests, scores how urgent/risky each one is, automatically finds
requests that can be *physically bundled together*, and then computes a
conflict-free timetable that fits the available planning window — live,
in seconds, and re-computable the moment conditions change (e.g. a delay).

This is fundamentally an **Operations Research scheduling problem** dressed
up with an **ML prioritization layer** and a **GIS proximity layer** — not
"AI" in the generative sense. That framing matters when you explain it to
your team: the intelligence is in *optimization + prioritization + spatial
reasoning*, not a chatbot.

---

## 2. Architecture — 4 layers

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — DASHBOARD (Streamlit)                                 │
│  Dark glassmorphism UI · sidebar controls · live delay injection │
│  Plotly Gantt timeline · AI priority queue table                 │
└───────────────────────────▲───────────────────────────────────────┘
                             │
┌───────────────────────────┴───────────────────────────────────────┐
│  LAYER 3 — OPTIMIZATION ENGINE (Google OR-Tools, CP-SAT solver)   │
│  Decides WHAT gets scheduled and WHEN, subject to:                │
│    HARD:  no two blocks overlap on the same physical track        │
│    HARD:  every block fits inside the planning horizon            │
│    SOFT:  bundled (nearby, cross-department) tasks are rewarded   │
│           for starting simultaneously → merged single possession  │
└───────────────────────────▲───────────────────────────────────────┘
                             │
┌────────────────────┬──────┴───────────────────────────────────────┐
│ LAYER 2a — ML RISK  │ LAYER 2b — GEOSPATIAL BUNDLING               │
│ scikit-learn        │ GeoPandas + Shapely                          │
│ RandomForestRegres- │ Projects lat/lon to true-metre UTM CRS,      │
│ sor → 0-100 risk    │ finds cross-department asset pairs within    │
│ score per request   │ 500m on the same corridor → bundle clusters  │
│ (overdue days,      │ (pure-Python fallback via haversine formula │
│ inspection score,   │  if geopandas isn't installed yet)           │
│ traffic, corridor   │                                              │
│ criticality)        │                                              │
└─────────────────────┴──────────────────────────────────────────────┘
                             │
┌───────────────────────────┴───────────────────────────────────────┐
│  LAYER 1 — DATA                                                    │
│  Synthetic demo generator today; in production, a connector to     │
│  TMS / IPMIS / UFM asset & inspection databases.                   │
└──────────────────────────────────────────────────────────────────┘
```

### File map

| File | Responsibility |
|---|---|
| `backend/data_gen.py` | Synthetic pending-request generator (stand-in for TMS/IPMIS feed) |
| `backend/risk_model.py` | scikit-learn `RandomForestRegressor` → 0–100 criticality score |
| `backend/geo_cluster.py` | GeoPandas/Shapely 500m cross-department proximity bundling |
| `backend/optimizer.py` | OR-Tools CP-SAT scheduling model (the actual "planner") |
| `app.py` | Streamlit dashboard — the only file you run |

Everything is standard-library-adjacent, MIT/Apache/BSD-licensed, **zero
paid dependency** (no PostGIS server, no commercial solver, no Mapbox key).

---

## 3. How the optimizer actually reasons (for your slide/demo talking points)

1. Every pending request becomes an **optional interval variable**
   `[start, start+duration]` in the solver — "optional" because the solver
   is allowed to leave low-priority requests **unscheduled** ("deferred")
   if the horizon is too full, rather than crash.
2. Requests sharing the same **physical section/track** get a
   `NoOverlap` constraint — this is the hard safety rule (you cannot
   authorize two gangs on the same block of track at once).
3. Requests that `geo_cluster.py` found within **500m of each other**
   *and* from **different departments** get an objective bonus for
   sharing an identical start time — the solver will merge them into one
   combined possession whenever it's feasible to do so.
4. The **objective function** = maximize (Σ risk_score × scheduled) +
   (bundling bonus × merged clusters). In plain English: *get as much
   of the highest-risk backlog off the books as possible, and merge
   whatever can safely be merged.*
5. When you move the **"Inject Inbound Train Delay"** slider, every
   request on that corridor gets its earliest-possible-start pushed back
   by that many minutes, and the whole model is **re-solved from
   scratch in real time** — this is what the "DYNAMICALLY SHIFTED"
   badges in the table are showing you.

---

## 4. Running it locally

```bash
# 1. create an isolated environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. install the fully open-source dependency stack
pip install -r requirements.txt

# 3. launch the dashboard
streamlit run app.py
```

It opens at `http://localhost:8501`. No API keys, no database, no internet
connection required — everything (including the "AI") runs locally.

> **Note:** `geopandas`/`shapely` can occasionally need system-level GEOS
> libraries on Windows. If `pip install` fails on those two only, the app
> still runs correctly — `geo_cluster.py` automatically falls back to a
> pure-NumPy haversine-distance implementation so the demo never breaks.

---

## 5. What to say when you present this to your team

- **The problem is a resource-allocation/scheduling problem**, not a
  prediction problem — that's why the core engine is a constraint solver
  (OR-Tools), not a neural network. The ML model's job is narrower: rank
  urgency, not decide the schedule.
- **"AI" here = 3 cooperating techniques**: supervised learning (risk
  ranking), geospatial analysis (bundling detection), and constraint
  optimization (the actual timetable) — worth naming explicitly, since
  judges/reviewers will ask "where's the AI."
- **It's a decision-support tool, not a fully autonomous system** — real
  deployment would still have a human Section Controller reviewing and
  approving the solver's proposed plan before it becomes a live caution
  order.

## 6. Realistic next steps for a production version

- Replace `data_gen.py` with a live connector to Railways' TMS / IPMIS /
  UFM systems (this is the only file that would change).
- Split `optimizer.py` behind a FastAPI microservice endpoint so the solve
  can be triggered by multiple UIs/mobile apps, not just this Streamlit app
  (the prompt's mention of FastAPI/BackgroundTasks maps directly to this).
- Swap GeoPandas' in-memory join for PostGIS `ST_DWithin` once request
  volumes grow past a few thousand rows per planning cycle.
- Add authentication/role-based access (Section Controller vs Department
  Engineer vs read-only Zonal HQ dashboard).
- Persist historical actual-vs-planned data so `risk_model.py` can be
  retrained on real failure/inspection outcomes instead of the synthetic
  bootstrap used in this prototype.
