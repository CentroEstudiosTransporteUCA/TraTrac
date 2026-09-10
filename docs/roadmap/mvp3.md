# MVP 3 — MULTI-PLANE WORLD TRAJECTORIES

> **Status — 🟡 Partially shipped** (`docs/IMPLEMENTATION_PLAN.md` Groups C1/C5). The MVP number
> is a capability ID, not execution order — see the roadmap reconciliation in `docs/ROADMAP.md`.
> **Landed:** Link ID assignment (Strategy A hand-drawn polygons, `application/ROAD_GRAPH.md`)
> and multi-homography Plane Assignment + Projection (`MultiPlaneTransform`,
> `src/tratrac/application/WORLD_PROJECTION.md`) — `tratrac-postprocess --link-zones`/`--plane-zones`. **Not landed:**
> this milestone's full pipeline still names MVP1.5's YOLO-OBB detector as a stage, which
> doesn't exist as a trained/default detector yet; automatic/assisted plane-and-link authoring
> (Group C4) is a separate, unstarted, repo-boundary-gated task.

---

## Goal

Support:

- bridges
- overpasses
- ramps
- stacked roads

Without:

- false conflicts

---

## New Technologies

| Component | Technology |
| --- | --- |
| Geometry | Multi-Homography |
| Plane Assignment | Polygon Mapping |

---

## Pipeline

```text
Video
    ↓
Stabilization
    ↓
YOLO-OBB (MVP1.5)
    ↓
BoT-SORT
    ↓
Road Plane Assignment
    ↓
Link Assignment
    ↓
Multi-Homography Projection
    ↓
SSAM .trj Export
```

---

## Link / Lane IDs ✅ Landed (both, via Strategy A)

See `docs/roadmap/road_topology.md` and `src/tratrac/application/ROAD_GRAPH.md`.

- **Link ID** — populated via **Strategy A** (hand-drawn link polygons in a
  per-scene JSON, `--link-zones`). `application/road_graph.py`'s `link_id_for_point`
  runs point-in-polygon on each surviving observation's centroid
  (`cli_postprocess.py`'s `_assign_labels`, not a dedicated `RoadGraphAssigner`
  class as originally sketched here), stamping `VehicleState.link_id` per frame.
- **Lane ID** — also landed (Group C2, pulled forward from MVP6 — the capability
  ladder is dependency order, not execution order): the identical mechanism,
  `--lane-zones` / `lane_id_for_point`, stamping `VehicleState.lane_id`.

### Plane assignment vs link assignment ✅ Landed (plane assignment)

These are **different** point-in-polygon passes:

- **Plane assignment** (this MVP, Group C5) sorts points into elevation
  layers — ground, bridge, overpass — to drive multi-homography projection.
  Landed as `MultiPlaneTransform` (`application/coordinate_transforms.py`):
  classification happens **inside** the projector at `to_world()` time (spatial,
  keyed off the point itself) rather than as a separate pre-computed
  per-observation dict the way Link/Lane assignment works — plane membership
  never gets written into `VehicleState`, it only selects which homography
  applies, so there's no output field for a separate pass to populate.
- **Link assignment** sorts vehicles into road-segment identities, written to
  `VehicleState.link_id`.

A bridge plane typically contains many links; a single link can span
multiple planes (an on-ramp). The two assigners share infrastructure
(point-in-polygon over per-scene JSON, `domain/road_graph.py`) but read
different polygon sets, and — unlike the original sketch here — are not
required to run in the same pass: `--plane-zones` only takes effect together
with `--calibration`, independently of whether `--link-zones` is also given.
