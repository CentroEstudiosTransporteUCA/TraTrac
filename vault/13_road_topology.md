# Road Topology Sourcing

SSAM's `Link ID` and `Lane ID` fields (see `04_ssam_format.md`, VEHICLE
record) require knowledge of the road network that does not fall out of
detection or tracking alone. This document defines what those identifiers
mean, where they come from at each MVP, and how the implementation
threads them through `VehicleState`.

---

## Why this matters

Without correct Link / Lane IDs, SSAM:

- misclassifies Lane-Change conflicts as Rear-End (or vice versa)
- aggregates per-link / per-lane analytics into a single bucket
- cannot cross-reference an external `.pth` road network file

Geometric conflict detection (TTC, PET) still works without these IDs —
they are required for *classification* and *grouping*, not for *finding*
conflicts.

---

## Conceptual distinction

### Link

- A **directional** road segment between two decision points
  (intersections, ramps, merges).
- An edge in the road graph. Each direction is its own link.
- Example: "Main St eastbound between Oak Ave and Elm St" is one link;
  the westbound counterpart is a separate link.

### Lane

- A **lateral subdivision** of a link.
- A 3-lane road has lanes 1, 2, 3 *all belonging to the same link*.
- Lane changes happen **within** a link, not between links.

### Important: road_plane ≠ link_id

`07_mvp3.md` introduces "Road Plane Assignment" via polygon mapping. A
*plane* is an elevation surface (ground, bridge, overpass). A *link* is
a road-segment identity. They are orthogonal:

- A bridge plane can carry multiple links (the highway plus its ramps).
- A link can span multiple planes (an on-ramp going from ground to bridge).

So MVP3's plane assignment does **not** automatically yield `link_id`.
Link assignment is a separate (also polygon-based) step that runs
*after* plane assignment.

---

## Sourcing strategies

Ranked by operational complexity.

### Strategy A — Hand-drawn polygons

Per-scene JSON next to the video listing `{ link_id: int, polygon: [(x, y), ...] }`
in image coordinates, and (later) `{ link_id, lane_id, polygon }` for lane strips.
The operator draws the road graph once per camera setup.

**Pros**

- Zero infrastructure.
- Works for fixed-camera installations.
- Scene-specific accuracy.

**Cons**

- Does not scale to ad-hoc scenes.
- Requires manual setup per camera.
- Breaks if the camera moves between recordings.

### Strategy B — Trajectory-clustered geometry

Cluster observed centroid paths to discover lane and link geometry
automatically.

**Pros**

- Self-supervised.
- Scales to new scenes without operator labour.

**Cons**

- Less precise than a real map.
- Fails on low-traffic scenes (no data to cluster).
- Struggles with intersections (paths overlap).

### Strategy C — External map import

Project OpenStreetMap, government GIS, or Mapillary lane geometry into
image coordinates via a georeferenced homography (camera GPS + heading
+ intrinsics).

**Pros**

- Highest accuracy.
- Leverages community-maintained map data.

**Cons**

- Requires camera calibration with real-world coordinates.
- Significant operational complexity.

---

## Adoption per MVP

| MVP | Link ID source | Lane ID source |
| --- | --- | --- |
| 1 | Hardcoded 0 | Hardcoded 0 |
| 1.5 | Hardcoded 0 | Hardcoded 0 |
| 2 | Hardcoded 0 | Hardcoded 0 |
| 3 | Strategy A (hand-drawn link polygons) | Hardcoded 0 |
| 4 | Inherited from MVP3 | Hardcoded 0 |
| 5 | Inherited from MVP3 | Hardcoded 0 |
| 6 | Inherited from MVP3 | Strategy A (hand-drawn lane polygons) |
| 7 | Strategy A → C migration possible | Strategy A → C migration possible |

Strategy B (clustering) is a stretch goal beyond MVP7 and not on the
critical path.

---

## Implementation contract

`tratrac.domain.vehicle.VehicleState` carries:

- `link_id: int = 0` — SSAM Integer field. Non-negative; default 0
  ("unknown / not assigned" sentinel).
- `lane_id: int = 0` — SSAM **Byte** field. Range `[0, 255]` enforced in
  `__post_init__`.

The application layer is responsible for populating these from the
active sourcing strategy. The exporter
(`tratrac.infrastructure.export.ssam_trj.SsamTrjExporter`) reads them
directly from the state — there is no SSAM-specific link / lane logic
inside the exporter.

When MVP3 lands, a new `RoadGraphAssigner` (or similar) belongs in the
application layer, between the tracker and the orientation estimator,
running point-in-polygon assignment per centroid. MVP6 extends the same
component to populate `lane_id`.
