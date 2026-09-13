# Road-topology zones: Link ID, Lane ID, and Plane ID classification

## Status

Link ID and Lane ID shipped (`docs/IMPLEMENTATION_PLAN.md` Groups C1/C2), Strategy A
(hand-drawn polygons) of `docs/roadmap/road_topology.md`. Optional and off by default, applied
**post-hoc** by `tratrac-postprocess` via `--link-zones`/`--lane-zones`. Plane ID's *zones and
classification helper* also live here (Group C5), but its *consumer* is
`tratrac-preprocess project`'s `_fit_per_plane` (`cli_preprocess.py`) — plane classification runs
once, at **fit** time, to group calibration correspondences per plane; the resulting homography
row simply embeds that plane's polygon as its `zone` field, so nothing downstream ever
re-classifies by plane again (see `src/tratrac/application/WORLD_PROJECTION.md` and
`src/tratrac/infrastructure/transform/TRANSFORM_SINK.md`'s "One file, one row model" for that
story); this doc covers the shared zone/classification infrastructure the three fields build on.

## What this adds

A set of **labeled pixel polygons** in `domain/road_graph.py` — `LinkZone(link_id,
reference_frame, polygon)`, `LaneZone(link_id, lane_id, reference_frame, polygon)`, and
`PlaneZone(plane_id, reference_frame, polygon)` — that classify a point into a label. Link/Lane
classify every surviving **observation**, stamped onto `VehicleState.link_id`/`lane_id` (the
SSAM VEHICLE record fields); Plane classifies a **projector query point** at `to_world()` time
instead — it's a purely internal homography-selection key, never written into `VehicleState`
(`docs/roadmap/road_topology.md` covers what Link/Lane mean conceptually and why Plane is
orthogonal to both: a bridge plane can carry many links, and a link can span multiple planes).

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
composition order in `docs/IMPLEMENTATION_PLAN.md`'s "Composition-root integration order":

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

## Not done by this landing

- **Strategies B/C** (`docs/roadmap/road_topology.md`) — trajectory-clustered or
  externally-sourced (OSM/`.pth`) road geometry — remain unexplored; Strategy A (hand-drawn) is
  what's shipped.
- **Automatic/assisted authoring** (Group C4, `docs/IMPLEMENTATION_PLAN.md`) is a separate,
  already-shipped task, not this doc's concern — see `application/AUTO_CALIBRATION.md`. Its
  repo-boundary question (does TraTrac ship only a correspondence/zone *proposal* capability,
  with interactive confirm/adjust living in URBAn?) is resolved there: yes, proposal-only.
