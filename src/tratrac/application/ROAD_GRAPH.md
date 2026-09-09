# Road-topology zones: Link ID and Lane ID classification

## Status

Shipped (`docs/IMPLEMENTATION_PLAN.md` Groups C1/C2), Strategy A (hand-drawn polygons) of
`docs/roadmap/road_topology.md`. Optional and off by default, applied **post-hoc** by
`tratrac-postprocess` via `--link-zones`/`--lane-zones`.

## What this adds

A set of **labeled pixel polygons** — `LinkZone(link_id, reference_frame, polygon)` and
`LaneZone(link_id, lane_id, reference_frame, polygon)` in `domain/road_graph.py` — that
classify every surviving observation into a Link ID and/or Lane ID, stamped onto
`VehicleState.link_id`/`lane_id` (the SSAM VEHICLE record fields; `docs/roadmap/road_topology.md`
covers what Link/Lane mean conceptually and why they're orthogonal to plane assignment).

## Why per-observation, not per-track (unlike exclusion zones)

`application/exclusion.py`'s `excluded_track_ids` does a track-level majority vote — "drop
the whole track" is a binary decision that tolerates a track dipping briefly into a zone.
Link/Lane assignment can't reuse that: a vehicle can **cross links or change lanes mid-track**,
and SSAM's Link ID / Lane ID are per-VEHICLE-RECORD fields (i.e. per frame), not per-track. So
`application/road_graph.py`'s `link_id_for_point`/`lane_id_for_point` classify one point at a
time; `cli_postprocess.py` builds a `{(track_id, frame_index): label}` map from every surviving
observation and stamps it onto the matching smoothed state after the Kalman/RTS pass.

Classification is plain point-in-polygon (`domain/geometry.point_in_polygon`) over the zones
mapped into the global frame — first zone in file order wins on overlap. `0` is the SSAM
"unknown" sentinel: a point outside every zone gets `0`, and zone labels themselves must be
positive (`LaneZone.lane_id` is additionally capped at 255 — it's a Byte field in the SSAM
record, same constraint `VehicleState.lane_id.__post_init__` already enforces).

## Where classification runs relative to the other post-hoc stages

Link/Lane classification runs on **image-space** coordinates, at the same stage exclusion
filtering does — after exclusion drops whole tracks, before `--calibration` projects
coordinates to world metres. This matches the composition order in
`docs/IMPLEMENTATION_PLAN.md`'s "Composition-root integration order":

```
read record
  → filter --exclusion-zones
  → assign --link-zones / --lane-zones
  → project --calibration
  → smooth (Kalman/RTS)
  → export .trj
```

Zones are authored the same way exclusion zones are (`src/tratrac/application/EXCLUSION_ZONES.md`):
on a reference frame, `0` for a static camera, or one of the run's exported ORB keyframe anchors
for a moving drone (`--anchors manifest.json` maps each zone's `reference_frame` into the global
frame by that anchor's pose).

## Components (onion layers)

| Layer | File | Role |
| --- | --- | --- |
| Domain | `domain/road_graph.py` | `LinkZone`/`LinkZones`, `LaneZone`/`LaneZones` — pure value objects, validated label ranges |
| Application | `application/road_graph.py` | `to_global_link_polygons`/`to_global_lane_polygons` (reference-frame → global, mirrors `application/exclusion.py`), `link_id_for_point`/`lane_id_for_point` (point-in-polygon classification) |
| Infrastructure | `infrastructure/road_graph/json.py` | Sidecar JSON readers (`load_link_zones`/`load_lane_zones`), mirrors `infrastructure/exclusion/json.py` |
| CLI | `cli_postprocess.py` | `--link-zones`/`--lane-zones` options; `_assign_labels` (shared fitter/classifier plumbing, generic over `LinkZones`/`LaneZones`), `_apply_link_ids`/`_apply_lane_ids` (stamp the smoothed states) |

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

`LaneZone.link_id` is operator documentation / a future cross-checking hook today —
classification itself is plain point-in-polygon over the lane zones, independent of any
separately-computed Link ID for the same point.

## Not done by this landing

- **Strategies B/C** (`docs/roadmap/road_topology.md`) — trajectory-clustered or
  externally-sourced (OSM/`.pth`) road geometry — remain unexplored; Strategy A (hand-drawn) is
  what's shipped.
- **Automatic/assisted authoring** (Group C4, `docs/IMPLEMENTATION_PLAN.md`) is a separate,
  gated task with an open repo-boundary question (does TraTrac ship only a correspondence/zone
  *proposal* capability, with interactive confirm/adjust living in URBAn?) — not resolved here.
