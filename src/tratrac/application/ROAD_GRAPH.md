# Road-topology zones: Link ID, Lane ID, and Plane ID classification

Link ID and Lane ID are sourced via **Strategy A: hand-drawn polygons**, authored once per
camera setup and applied **post-hoc** by `tratrac-postprocess` via `--link-zones`/`--lane-zones`.
Plane ID's *zones and classification helper* also live here, but its *consumer* is
`tratrac-preprocess project`'s `_fit_per_plane` (`cli_preprocess.py`) — plane classification runs
once, at **fit** time, to group calibration correspondences per plane; the resulting homography
row simply embeds that plane's polygon as its `zone` field, so nothing downstream ever
re-classifies by plane again (see `src/tratrac/application/WORLD_PROJECTION.md` and
`src/tratrac/infrastructure/transform/TRANSFORM_SINK.md`'s "One file, one row model" for that
story); this doc covers the shared zone/classification infrastructure the three fields build on.

## Why this matters

SSAM's `Link ID` and `Lane ID` fields (see `infrastructure/export/ssam_trj.py`'s module docstring,
VEHICLE record) require knowledge of the road network that does not fall out of detection or
tracking alone. Without correct values, SSAM misclassifies Lane-Change conflicts as Rear-End (or
vice versa) and aggregates per-link/per-lane analytics into a single bucket. Geometric conflict
detection (TTC, PET) still works without these IDs — they're required for *classification* and
*grouping*, not for *finding* conflicts.

## Link vs. Lane vs. Plane — three orthogonal concepts

- **Link** — a *directional* road segment between two decision points (intersections, ramps,
  merges); an edge in the road graph, each direction its own link. "Main St eastbound between
  Oak Ave and Elm St" is one link, the westbound counterpart a separate one.
- **Lane** — a *lateral subdivision* of a link. A 3-lane road has lanes 1, 2, 3 all belonging to
  the same link; lane changes happen within a link, not between links.
- **Plane** — an *elevation surface* (ground, bridge, overpass), used only to pick which
  homography a point projects through. Orthogonal to both of the above: a bridge plane can carry
  many links (the highway plus its ramps), and a single link can span multiple planes (an
  on-ramp going from ground to bridge). Plane assignment never yields a `link_id` — link
  assignment is a separate, also polygon-based, classification.

## What this adds

A set of **labeled pixel polygons** in `domain/road_graph.py` — `LinkZone(link_id,
reference_frame, polygon)`, `LaneZone(link_id, lane_id, reference_frame, polygon)`, and
`PlaneZone(plane_id, reference_frame, polygon)` — that classify a point into a label. Link/Lane
classify every surviving **observation**, stamped onto `VehicleState.link_id`/`lane_id` (the
SSAM VEHICLE record fields); Plane classifies a **projector query point** at `to_world()` time
instead — it's a purely internal homography-selection key, never written into `VehicleState`
(see "Link vs. Lane vs. Plane" above for why Plane is orthogonal to both: a bridge plane can
carry many links, and a link can span multiple planes).

## Why per-observation, not per-track (unlike exclusion zones)

`application/exclusion.py`'s `excluded_track_ids` does a track-level majority vote — "drop
the whole track" is a binary decision that tolerates a track dipping briefly into a zone.
Link/Lane assignment can't reuse that: a vehicle can **cross links or change lanes mid-track**,
and SSAM's Link ID / Lane ID are per-VEHICLE-RECORD fields (i.e. per frame), not per-track. So
`application/road_graph.py`'s `link_id_for_point`/`lane_id_for_point` classify one point at a
time; `cli_postprocess.py` builds a `{(track_id, frame_index): label}` map from every surviving
observation and stamps it onto the matching smoothed state after the Kalman/RTS pass.

Classification is plain point-in-polygon (`domain/geometry.point_in_polygon`) over the zones
mapped into the global frame — first zone in file order wins on overlap. For Link/Lane, `0` is
the SSAM "unknown" sentinel: a point outside every zone gets `0`, and zone labels themselves
must be positive (`LaneZone.lane_id` is additionally capped at 255 — it's a Byte field in the
SSAM record, same constraint `VehicleState.lane_id.__post_init__` already enforces). Plane is
the one exception: `PlaneZone.plane_id` allows `0` as a legitimate label (e.g. the ground
plane) since it has no SSAM sentinel to reserve — only negative values are rejected. See
`WORLD_PROJECTION.md` for why a projector can't treat "unclassified" as safely defaultable the
way Link/Lane do.

## Where classification runs relative to the other post-hoc stages

Link/Lane classification runs on **image-space** coordinates, at the same stage exclusion
filtering does — after exclusion drops whole tracks, before the already-fitted projection rows
(read from `--transforms`) are applied to project coordinates to world metres. This matches the
composition order documented in `cli_postprocess.py`'s postprocess function docstring:

```
read record
  → filter --exclusion-zones
  → assign --link-zones / --lane-zones
  → apply projection (scale-or-homography rows from --transforms, fitted earlier by
    `tratrac-preprocess project`)
  → smooth (Kalman/RTS)
  → export .trj
```

Zones are authored the same way exclusion zones are (`src/tratrac/application/EXCLUSION_ZONES.md`):
on a reference frame, `0` for a static camera, or any frame of a `tratrac-preprocess` run for a
moving drone (`--transforms transforms.jsonl` maps each zone's `reference_frame` into the global
frame by that frame's pose).

## Components (onion layers)

| Layer | File | Role |
| --- | --- | --- |
| Domain | `domain/road_graph.py` | `LinkZone`/`LinkZones`, `LaneZone`/`LaneZones`, `PlaneZone`/`PlaneZones` — pure value objects, validated label ranges |
| Application | `application/road_graph.py` | `to_global_{link,lane,plane}_polygons` (reference-frame → global, mirrors `application/exclusion.py`), `{link,lane,plane}_id_for_point` (point-in-polygon classification) |
| Infrastructure | `infrastructure/road_graph/json.py` | Sidecar JSON readers (`load_{link,lane,plane}_zones`), mirrors `infrastructure/exclusion/json.py` |
| CLI | `cli_postprocess.py` | `--link-zones`/`--lane-zones` options + `_assign_labels` (shared fitter/classifier plumbing, generic over `LinkZones`/`LaneZones`) and `_apply_link_ids`/`_apply_lane_ids` (stamp the smoothed states) |
| CLI | `cli_preprocess.py` (`project` subcommand) | `--plane-zones` + `_fit_per_plane` (fits one homography row per plane, embedding that plane's polygon as the row's `zone`, instead of stamping a field — see `WORLD_PROJECTION.md`) |

## Sidecar schema

```jsonc
{ "link_zones": [
    { "link_id": 1, "reference_frame": 0, "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
] }
```

```jsonc
{ "lane_zones": [
    { "link_id": 1, "lane_id": 1, "reference_frame": 0, "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
] }
```

```jsonc
{ "plane_zones": [
    { "plane_id": 0, "reference_frame": 0, "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
] }
```

`LaneZone.link_id` is operator documentation / a future cross-checking hook today —
classification itself is plain point-in-polygon over the lane zones, independent of any
separately-computed Link ID for the same point. `plane_zones` has no `link_id`/`lane_id`
fields — it's a standalone label with no cross-referencing metadata.

## Alternative sourcing strategies (not built)

Strategy A doesn't scale to ad-hoc scenes — it requires manual per-camera setup and breaks if
the camera moves between recordings. Two alternatives, ranked by operational complexity, are
tracked as future work (see GitHub Issues): **Strategy B** clusters observed centroid paths to
discover lane/link geometry automatically (self-supervised, but less precise and fails on
low-traffic scenes); **Strategy C** projects OpenStreetMap/GIS/Mapillary lane geometry into
image coordinates via a georeferenced homography (highest accuracy, but needs real-world camera
calibration and is operationally complex). Strategy A remains acceptable for production as long
as scenes are pre-configured.

Automatic/assisted *authoring* of Strategy A's polygons themselves (proposing calibration
correspondences from visible road markings) is a separate, already-shipped concern — see
`scripts/propose_calibration.py`'s module docstring. TraTrac ships only the proposal capability; interactive
confirm/adjust lives in URBAn.
