# System Overview

---

## System Objective

Build a research-grade and production-capable system capable of:

- Processing cenital / nadir aerial video
- Detecting vehicles
- Generating precise per-frame occupancy masks
- Tracking vehicles over time
- Preserving identities after occlusions and re-entry
- Handling multi-level roads (bridges, ramps, overpasses)
- Producing accurate world-coordinate trajectories
- Exporting SSAM-compatible `.trj` files
- Supporting traffic analytics and conflict analysis
- Supporting dense urban traffic scenarios

---

## Core Philosophy

The system should ALWAYS output:

```text
.trj trajectory file
```

from the very first MVP.

The difference between MVPs is:

- trajectory QUALITY
- geometric correctness
- identity persistence
- spatial precision
- physical plausibility

NOT:

- whether trajectories exist

---

## Roadmap: Capability IDs vs Execution Order

**The MVP numbers are capability IDs, not a schedule.** They name a *dependency ladder*
(each capability assumes the ones below it), but the work has deliberately **not** followed
that order: cheap shortcuts were slotted in, two later-MVP foundations were pulled forward,
one milestone was skipped, and a whole supporting layer was built outside the numbering. The
per-MVP files carry a **Status** banner; this is the single reconciliation of plan vs reality.

**The capability ladder (the planned dependency order):**

```
1 → 1.5 → 1.75 → 1.9 → 2 → 3 → 4 → 5 → 6 → 7
```

**Execution status (what's actually been done):**

| MVP | Capability | Status |
| --- | --- | --- |
| 1 | Detection + tracking + image-space SSAM `.trj` | ✅ **Shipped** — with the YOLOv8-VisDrone *emergency* detector, not the planned RT-DETR |
| 1.5 | Fine-tune RT-DETR, remove YOLOv8 | ❌ **Skipped** (leapfrogged by 1.75 + 1.9) |
| 1.75 | Metric sizes/speeds from drone GSD | ✅ **Shipped** |
| 1.9 | ORB ego-motion stabilization | ✅ **Shipped** (optional, off by default) |
| 2 | Single-homography **world projection** | ❌ **Not started** — the load-bearing gap: SSAM positions remain image-space pixels |
| 3 | Multi-homography + Link ID | ❌ Not started |
| 4 | SAM2 segmentation; **dual export begins** | 🟡 **Partially pulled forward** — the dual-export "B-first" architecture is already core; SAM2 not started |
| 5 | FastReID identity persistence | ❌ Not started |
| 6 | Lane graph + Lane ID | ❌ Not started |
| 7 | **Parquet storage** + FiftyOne + Docker | 🟡 **Partially pulled forward** — Parquet is the canonical record; FiftyOne/Docker not started |

**Pulled forward, out of ladder order:** Parquet storage (7) and the dual-export "B-first"
inversion (4) — both landed early because the canonical *track record* needed them now.

**The supporting layer (built outside the MVP numbering):** progress reporting (14), step
timing (15), `.trj` validation (16), time window (17), timestep precision (18), config file
(19), post-hoc render (20), exclusion zones (21), Kalman/RTS smoothing (22), plus the
`final_polish.md` backlog. These are cross-cutting capabilities behind stable ports, not
rungs on the capability ladder.

This section is a **historical reconciliation** — the forward order (2 → 7) is unchanged.

---

## Final Architecture Vision

The final architecture is:

- Multi-Object Tracking

- Segmentation

- ReID

- Multi-Plane Geometry

- Topology-Aware Analytics Platform

---

## Ideal Final System Characteristics

The final system supports:

- Persistent identities
- Multi-level roads
- Precise occupancy masks
- World-space trajectories
- Re-entry recovery
- Dense traffic analytics
- Physically plausible tracking
- SSAM interoperability
- Large-scale processing
- Traffic engineering analytics
- Safety conflict analysis

---

## Most Important Engineering Insight

The hardest problems are NOT:

- detection
- segmentation

Modern models already solve those reasonably well.

The true engineering difficulty is:

- identity persistence
- multi-level geometry
- occlusion recovery
- topology-aware association
- physically meaningful trajectories
- long-term consistency

That is where most production effort goes.
