# 21 — Exclusion zones: image-space "do-not-analyze" regions

## What this adds

A set of **pixel polygons** marking image regions the run must ignore — parking
lots, buildings, sidewalks, static clutter. A detection whose bounding box is
**mostly covered** (more than 50% of its area) by the polygons' **union** is
**dropped before tracking**, so it never enters a trajectory.

Optional and off by default. It is the *spatial* analogue of the `--start`/
`--end` time-window trim (`17_time_window.md`): an image-space input filter that
leaves the trajectory stages untouched. Mechanism-wise it is kin to the hand-drawn
polygons of `13_road_topology.md` (MVP3), but it needs none of MVP3's
homography/plane machinery, so it ships now.

## Coverage by rasterized union (not analytic clipping)

A detection is dropped when the masked **union** of zones covers more than `0.5`
of its bounding box. Coverage is measured by rasterizing: the zones are filled
into a `W×H` binary mask with `cv2.fillPoly`, and a detection's drop test is the
mean of that mask over its (clamped) bbox versus `0.5`. `0.5` lives in
`infrastructure/exclusion/raster.py` as `_MAJORITY`.

Rasterizing — rather than an analytic per-polygon clip — is deliberate: `fillPoly`
fills **concave** polygons correctly and **unions** overlapping/adjacent zones for
free. This dissolves two earlier limitations of an analytic approach (convex-only
area, and per-zone `max` letting a box straddling two zones survive). It also uses
the same primitive the ORB vehicle-mask already uses.

## The seam: a `DetectionMask` port, applied after ego-motion

The drop runs **inside the pipeline**, between ego-motion estimation and
stabilization — *not* at the detector seam. Reason: a moving-drone zone must be
mapped into the frame being processed, which needs that frame's ego-motion pose,
and the pose is produced by `EgoMotionEstimator.estimate()` *after*
`Detector.detect()`. So a detector-seam decorator cannot see the pose.

The per-frame loop (`application/pipeline.py`) is therefore:

```
detections = detect(frame)
observe(detections)                              # ORB vehicle-masking (unchanged)
pose = ego_motion.estimate(frame) or identity    # raw -> global
detections = detection_mask.filter(detections, pose, frame)   # <-- the drop
if ego_motion: detections = [apply_transform(d, pose) ...]    # stabilize
tracker.update(frame, detections)
```

`DetectionMask` (`domain/ports.py`) is a Null-Object port (default
`NullDetectionMask` = pass-through), so a run without zones is byte-identical to
before. The implementation is `RasterExclusionMask`
(`infrastructure/exclusion/raster.py`).

## Zones live in the global frame; they move with the scene

Each zone is authored on a **reference frame** `R` and carries `reference_frame`
in the JSON. At load the polygon is converted **once** into the continuous global
stabilization frame via that frame's pose: `polygon_global = pose_R.apply(verts)`
(`application/exclusion.py:to_global_polygons`). At runtime, for frame `N` with
pose `pose_N`, `RasterExclusionMask` maps the global polygon back into raw-`N`
pixels with `inverse(pose_N)` before rasterizing. Detection (raw `N`) and zone
(mapped to raw `N`) meet in the same frame, so the zone **tracks the scene** as the
drone moves.

A **static camera** is the degenerate case: `reference_frame = 0`, all poses
identity, global == raw — equivalent to a fixed screen-space mask.

## Input: sidecar JSON

A per-scene JSON file referenced from config (`analysis.exclusion_zones`, a
`toggleable_path`: `""` = off), mirroring `calibration.srt`. Loaded by
`infrastructure/exclusion/json.py` into the pure `ExclusionZones` value object.

```json
{ "exclusion_zones": [
    { "label": "parking_lot",
      "reference_frame": 0,
      "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
] }
```

`label` is optional. `reference_frame` defaults to `0`. `vertices` are pixel
coordinates, ≥3 per polygon. Bad path / malformed JSON / too-few vertices fail
fast in the CLI before the costly video open.

## Moving drone: authoring reference frames (the scout — see §Stage 2)

For a moving drone the operator can't predict which frames to draw on, and a zone
drawn on frame 0 isn't visible once the drone pans away. The reference frames are
the **ORB keyframe anchors** (overlap-guaranteed to tile the traversed scene). A
headless **scout pass** (`tratrac-scout`) discovers them: it runs ORB only,
persists the per-frame ego-motion schedule (reusing `CsvTransformSink`), and emits
each anchor frame as a PNG plus a manifest. The operator draws zones over those
PNGs (`reference_frame` = the anchor index). The real run then **replays** the
recorded schedule (`ReplayEgoMotionEstimator`) so its poses match the scout
exactly — a correctness requirement, since zones placed with scout poses must meet
detections placed with the same poses. (Scout + replay land in the second stage of
this work.)

ORB feature-masking of zones is **not** done: in a moving drone the excluded
regions are static ground, exactly the features stabilization needs — masking them
would hurt the fit. Vehicle-masking via `observe()` is unaffected.

## Files

- `domain/geometry.py` — `Polygon` (pure vertex container).
- `domain/exclusion.py` — `ExclusionZone` (`reference_frame` + polygon), `ExclusionZones`.
- `domain/ports.py` — `DetectionMask` port.
- `application/detection_mask.py` — `NullDetectionMask` (Null Object default).
- `application/exclusion.py` — `to_global_polygons` (reference-frame → global).
- `application/pipeline.py` — applies the mask post-estimate, pre-stabilize.
- `infrastructure/exclusion/raster.py` — `RasterExclusionMask` (fillPoly union + coverage).
- `infrastructure/exclusion/json.py` — sidecar loader.
- `application/config.py` — `AnalysisConfig` + `analysis.exclusion_zones`.
- `cli.py` — `--exclusion-zones`, load + existence check, builds `RasterExclusionMask`.
