# MVP 5 — LONG-TERM IDENTITY PERSISTENCE

> **Status — ❌ Not started, replanned.** The MVP number is a capability ID, not execution order
> — see the roadmap reconciliation in `docs/ROADMAP.md`. **The original ReID plan (FastReID) was
> superseded** after research found nadir drone footage discards most of what vehicle-ReID
> models are trained to discriminate on, and there's no nadir-matched dataset to fine-tune
> FastReID against the way there is for detection. Full comparison and sources:
> `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md`.

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
| ReID appearance signal | **DINOv3** self-supervised embeddings (not FastReID — see `TRACKER_CHOICE.md`) |
| Re-identification gate | Motion-plausibility check against the Kalman state (`application/kalman.py`), not appearance alone |
| Memory Bank | Embedding Store |

Why the change: nadir view loses the visual cues (plates, side profile) vehicle-ReID models rely
on, and no nadir-specific vehicle-ReID dataset exists to fine-tune against — unlike detection,
where UAV-OBB solved that problem cleanly. DINOv3's self-supervised, viewpoint-general training
measurably outperforms supervised ReID checkpoints under large viewpoint shift (90° angular
separation, the closest available proxy for the oblique-to-nadir gap). See `TRACKER_CHOICE.md`
for the full evidence and a real deployment (Songdo, South Korea) that hit this same wall and
had to fuse appearance with a temporal/motion signal to make ReID work at all — which is the
same reasoning behind pairing DINOv3 with motion-plausibility gating here, using infrastructure
(`application/kalman.py`) TraTrac already has instead of building a separate travel-time model.

---

## Pipeline

```text
Video
    ↓
Stabilization
    ↓
YOLO-OBB (MVP1.5, oriented detection)
    ↓
BoT-SORT (OBB-aware tracking)
    ↓
DINOv3 appearance embedding + motion-plausibility gate
    ↓
Identity Memory Matching
    ↓
Link Assignment (from MVP3)
    ↓
Multi-Homography Projection
    ↓
Dual Export
```

Segmentation (previously shown here as a SAM2 step) is MVP4's concern, not MVP5's — see
`docs/roadmap/mvp4.md` for why it's no longer assumed to sit unconditionally in this chain.

---

## Link / Lane IDs

See `docs/roadmap/road_topology.md`.

- **Link ID** — inherited from MVP3 (Strategy A polygons). Re-acquired
  identities carry their link history across occlusions.
- **Lane ID** — still hardcoded `0`.
