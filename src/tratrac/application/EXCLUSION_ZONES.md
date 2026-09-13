# Exclusion zones: post-hoc, track-aware "do-not-analyze" regions

## What this adds

A set of **pixel polygons** marking image regions whose traffic doesn't interest the
analysis — parking lots, buildings, sidewalks, static clutter. A whole **track** is dropped
when the **majority of its observations** fall inside the polygons' union, so the excluded
object never appears in the `.trj`.

Optional and off by default. It is applied **post-hoc** by `tratrac-postprocess`, not by the
run — the spatial analogue of how smoothing and rendering became post-hoc. The perception run
stays pure (detect → track → record); exclusion is an analysis decision on the record.

## Why post-hoc, and why track-aware

Masking was always *post-detection* (you can't make a whole-frame CNN detector skip a region),
so it never needed to be in the run. Moving it out buys two things:

- **One pass.** `tratrac-preprocess` computes ORB once and emits the keyframe anchors before
  `tratrac` ever runs; `tratrac` itself never estimates ego-motion, so there is no separate scout
  pass recomputing ORB, and no replay. (Previously: scout pass + replay run = ORB twice.)
- **Track-aware filtering.** The per-frame in-pipeline mask could only drop individual
  detections. Post-hoc, the whole trajectory is visible, so "this *object* doesn't interest
  me" is expressible: drop a track once a fraction (`--exclusion-min-fraction`, default 0.5)
  of its observations are inside a zone. A car merely *passing through* keeps its track; a car
  *parked* in the zone is dropped.

The test is **centroid-in-polygon** (`domain/geometry.py:point_in_polygon`, even-odd ray
casting, concave-safe) on each observation's recorded centroid — simpler than the old
bbox-raster coverage and closer to what "passing here" means.

## The workflow

```
tratrac-preprocess estimate VIDEO --background-zones bg.json \
        --out transforms.jsonl --anchors-dir anchors/ --meters-per-pixel 0.05
   → anchors/frame_<i>.png     (one per ORB keyframe anchor — the frames to draw on)
   → transforms.jsonl          (dense per-frame ego-motion + scale table, anchor frames included)

[ draw ROI polygons on a frame_<i>.png → zones.json,
  tagging each with reference_frame = that frame's index ]

tratrac --config run.toml   # input.transforms_in = transforms.jsonl

tratrac-postprocess run.parquet --out run.trj \
        --exclusion-zones zones.json --transforms transforms.jsonl
   → tracks mostly inside a zone are dropped; survivors are smoothed into the .trj
```

`tratrac-preprocess estimate` emits anchors via the ORB estimator's existing `anchor_observer`
seam: an `AnchorRecordingEgoMotionEstimator` decorator drains each new anchor and an
`AnchorImageSink` writes just the PNG (no separate manifest — an anchor's pose is already the
transforms file's row at that same frame index, so persisting it twice would only risk the two
going out of sync). `tratrac` itself never estimates ego-motion, resolves scale, fits a
homography, or exports anchors; it only reads `input.transforms_in`.

## Zones live in the global frame; they move with the scene

Each zone is authored on a **reference frame** `R` and carries `reference_frame` in the JSON —
it need not even be one of the exported anchor frames, since the transforms sidecar covers every
frame of the clip. `tratrac-postprocess` reads that sidecar, maps each zone once into the
continuous global stabilization frame via that frame's pose
(`application/exclusion.py:to_global_polygons`, `polygon_global = pose_R.apply(verts)`), then
tests the record's observations — which are already in the global frame — against it
(`excluded_track_ids`). One global space, so a zone tiled from several frames covers the whole
swept scene.

A **static camera** is the degenerate case: omit `--transforms`, every pose is the identity,
global == raw — a fixed screen-space mask authored on any frame.

## Input: sidecar JSON (unchanged shape)

```json
{ "exclusion_zones": [
    { "label": "parking_lot",
      "reference_frame": 0,
      "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
] }
```

`label` optional; `reference_frame` defaults to `0`; `vertices` are pixel coordinates, ≥3 per
polygon. Loaded by `infrastructure/exclusion/json.py` into the pure `ExclusionZones`. A
`reference_frame` outside the transforms sidecar's coverage (i.e. outside the clip) is rejected.

## Files

- `domain/geometry.py` — `Polygon`; `point_in_polygon` (shapely-backed).
- `domain/exclusion.py` — `ExclusionZone` (`reference_frame` + polygon), `ExclusionZones`.
- `application/exclusion.py` — `to_global_polygons` (reference-frame → global) + `excluded_track_ids`
  (track-aware majority filter).
- `infrastructure/exclusion/json.py` — sidecar loader.
- `infrastructure/anchors/` — `AnchorImageSink` (anchor PNGs only, no manifest),
  `AnchorRecordingEgoMotionEstimator` (tees new anchors to the sink).
- `infrastructure/transform/sink.py` — `read_transforms` (the sidecar `_pose_for`/
  `to_global_polygons` resolve a zone's `reference_frame` pose from).
- `infrastructure/video/ego_motion_orb.py` — `anchor_observer` callback (anchor events).
- `cli_preprocess.py` (`tratrac-preprocess`) — `--anchors-dir` exports the anchor PNGs;
  `--out` exports the transforms sidecar every zone's pose is resolved from.
- `cli_postprocess.py` — `--exclusion-zones` / `--transforms` / `--exclusion-min-fraction`;
  filters the record (drop excluded tracks) before smoothing.
