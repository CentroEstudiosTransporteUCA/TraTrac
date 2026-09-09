# MVP 5 — LONG-TERM IDENTITY PERSISTENCE

> **Status — ❌ Not started.** The MVP number is a capability ID, not execution order — see the
> roadmap reconciliation in `docs/ROADMAP.md`.

---

## Goal

Maintain:

- same identity

After:

- occlusion
- disappearance
- re-entry

---

## New Technologies

| Component | Technology |
| --- | --- |
| ReID | FastReID |
| Memory Bank | Embedding Store |

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
FastReID
    ↓
BoT-SORT
    ↓
Identity Memory Matching
    ↓
Link Assignment (from MVP3)
    ↓
Multi-Homography Projection
    ↓
Dual Export
```

---

## Link / Lane IDs

See `docs/roadmap/road_topology.md`.

- **Link ID** — inherited from MVP3 (Strategy A polygons). Re-acquired
  identities carry their link history across occlusions.
- **Lane ID** — still hardcoded `0`.
