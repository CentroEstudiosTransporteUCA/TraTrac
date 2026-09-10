# MVP 4 — PRECISE OCCUPANCY-AWARE TRAJECTORIES

> **Status — 🟡 Partially pulled forward, rescoped, sidecar landed.** The MVP number is a
> capability ID, not execution order — see the roadmap reconciliation in `docs/ROADMAP.md`. This
> MVP's **dual-export "B-first" architecture** ("dual export begins") was pulled forward and is
> already the core design (perception run → Parquet record → post-hoc `.trj` — see
> `src/tratrac/application/SMOOTHING.md`). Its scope has narrowed: **MVP1.5's replanned YOLO-OBB
> detector already reports orientation directly** (see
> `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`), which was this MVP's original
> justification for needing segmentation at all — what segmentation still adds on top is a
> precise occupancy **mask/footprint**, not orientation. **Landed** (Group D1,
> `docs/IMPLEMENTATION_PLAN.md`): the footprint sidecar storage format and
> `tratrac-postprocess --footprint` (`src/tratrac/application/FOOTPRINT.md`) — it reuses Group A's
> OBB dimension slot (`obb_w`/`obb_h`), so `build_state` and world-projection already handle a
> footprint-derived size correctly with no new consumer code. **Not started:** the segmentation
> stage itself (`cli_segment.py`, actually running SAM 3) — needs a GPU + a real model, so nothing
> can populate a footprint sidecar yet; the sidecar format and `--footprint` exist ahead of their
> producer, same as `application/reid_merge.py` landed ahead of DINOv3. Also: **the target model
> is SAM 3, not SAM2** — SAM2 is superseded (Meta shipped SAM 3 in November 2025) and was already
> measured as slower/heavier than current YOLO-seg alternatives for pure instance segmentation.
> See `docs/TECH_STACK.md`'s Segmentation section for sources.

---

## Goal

Replace:

- coarse bounding boxes

With:

- precise segmentation-derived **occupancy/footprint** geometry (orientation is no longer this
  MVP's job — see the status note above)

---

## New Technologies

| Component | Technology |
| --- | --- |
| Segmentation | **SAM 3** (text-promptable — e.g. "segment all cars" — not SAM2) |
| Polygon Extraction | OpenCV Contours |

---

## Pipeline

```text
Video
    ↓
Stabilization
    ↓
YOLO-OBB (MVP1.5, already reports orientation)
    ↓
SAM 3 (occupancy mask only — orientation is no longer sourced here)
    ↓
BoT-SORT (OBB-aware tracking)
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
