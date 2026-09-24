# Roadmap: System Overview & Core Philosophy

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

## The Capability Ladder

**The MVP numbers are capability IDs, not a schedule.** They name a *dependency ladder* (each
capability assumes the ones below it):

```
1 → 1.5 → 1.75 → 1.9 → 2 → 3 → 4 → 5 → 6 → 7
```

Execution has never followed this order strictly — cheap shortcuts get slotted in, later-MVP
foundations get pulled forward when a capability is needed early, and a whole supporting layer
(progress reporting, step timing, `.trj` validation, time window, timestep precision, config
file, post-hoc render, exclusion zones, Kalman/RTS smoothing) was built entirely outside the
numbering. That's expected: the ladder names *what depends on what*, not a commitment to build
in that order. Which MVP a given capability belongs to, and its current status, is what each
capability's own design doc and its GitHub Issues describe — see `docs/README.md`'s index and
the org's [Traffic Analysis Pipeline project board](https://github.com/orgs/CentroEstudiosTransporteUCA/projects/1).

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
