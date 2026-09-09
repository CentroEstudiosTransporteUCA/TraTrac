# MVP 4 — PRECISE OCCUPANCY-AWARE TRAJECTORIES

> **Status — 🟡 Partially pulled forward.** The MVP number is a capability ID, not execution
> order — see the roadmap reconciliation in `docs/ROADMAP.md`. This MVP's **dual-export
> "B-first" architecture** ("dual export begins") was pulled forward and is already the core
> design (perception run → Parquet record → post-hoc `.trj`, src/tratrac/application/SMOOTHING.md). The SAM2
> segmentation + mask-based orientation in this milestone are **not started**.

---

## Goal

Replace:

- coarse bounding boxes

With:

- precise segmentation-derived geometry

---

## New Technologies

| Component | Technology |
| --- | --- |
| Segmentation | SAM2 |
| Polygon Extraction | OpenCV Contours |

---

## Pipeline

```text
Video
    ↓
Stabilization
    ↓
RT-DETR
    ↓
SAM2
    ↓
BoT-SORT
    ↓
Mask-Based Orientation
    ↓
Link Assignment (from MVP3)
    ↓
Multi-Homography Projection
    ↓
SSAM + Internal Export
```

---

## Link / Lane IDs

See `docs/roadmap/road_topology.md`.

- **Link ID** — inherited from MVP3 (Strategy A polygons). Mask-based
  centroids feed the same point-in-polygon assigner.
- **Lane ID** — still hardcoded `0`.
