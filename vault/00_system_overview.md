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
