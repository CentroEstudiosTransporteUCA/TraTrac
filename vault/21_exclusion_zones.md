# 21 — Exclusion zones: post-hoc, track-aware "do-not-analyze" regions

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

- **One pass.** The run computes ORB live and **emits its own keyframe anchors**; there is no
  separate scout pass recomputing ORB, and no replay. (Previously: scout pass + replay run =
  ORB twice.)
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
tratrac VIDEO --stabilize --out run.parquet --anchors-dir anchors/
   → anchors/frame_<i>.png   (one per ORB keyframe anchor — the frames to draw on)
   → anchors/manifest.json   (each anchor's frame_index + global pose + image)

[ draw ROI polygons on a frame_<i>.png → zones.json,
  tagging each with reference_frame = that anchor's index ]

tratrac-postprocess run.parquet --out run.trj \
        --exclusion-zones zones.json --anchors anchors/manifest.json
   → tracks mostly inside a zone are dropped; survivors are smoothed into the .trj
```

The run emits anchors via the ORB estimator's existing `anchor_observer` seam: an
`AnchorRecordingEgoMotionEstimator` decorator drains each new anchor and an `AnchorManifestSink`
writes the PNG + manifest (the perception loop is untouched). The **manifest is self-sufficient**
for exclusion — it carries each anchor's pose — so no transform CSV is needed for this workflow.

## Zones live in the global frame; they move with the scene

Each zone is authored on a **reference frame** `R` (an anchor) and carries `reference_frame`
in the JSON. `tratrac-postprocess` reads the anchor manifest, maps each zone once into the
continuous global stabilization frame via that anchor's pose
(`application/exclusion.py:to_global_polygons`, `polygon_global = pose_R.apply(verts)`), then
tests the record's observations — which are already in the global frame — against it
(`excluded_track_ids`). One global space, so a zone tiled from anchors covers the whole swept
scene.

A **static camera** is the degenerate case: omit `--anchors`, every pose is the identity,
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
`reference_frame` that isn't an anchor in the manifest is rejected.

## Files

- `domain/geometry.py` — `Polygon`; `point_in_polygon` (ray casting).
- `domain/exclusion.py` — `ExclusionZone` (`reference_frame` + polygon), `ExclusionZones`.
- `application/exclusion.py` — `to_global_polygons` (reference-frame → global) + `excluded_track_ids`
  (track-aware majority filter).
- `infrastructure/exclusion/json.py` — sidecar loader.
- `infrastructure/anchors/` — `ReferenceFrame`/`write_manifest`/`read_manifest`, `AnchorManifestSink`
  (PNG + manifest), `AnchorRecordingEgoMotionEstimator` (tees new anchors).
- `infrastructure/video/ego_motion_orb.py` — `anchor_observer` callback (anchor events).
- `cli.py` — `--anchors-dir` (export.anchors_dir) wires the anchor sink on a stabilized run.
- `cli_postprocess.py` — `--exclusion-zones` / `--anchors` / `--exclusion-min-fraction`; filters
  the record (drop excluded tracks) before smoothing.
