# 21 — Exclusion zones: image-space "do-not-analyze" regions

## What this adds

A set of **pixel polygons** marking image regions the run must ignore — parking
lots, buildings, sidewalks, static clutter. A detection whose bounding box is
**mostly inside** a zone (more than 50% of its area) is **dropped before
tracking**, so it never enters a trajectory; and the zones are **masked out of
ORB ego-motion feature extraction**, so static objects there cannot bias the
stabilization fit.

Optional and off by default. It is the *spatial* analogue of the `--start`/
`--end` time-window trim (`17_time_window.md`): an image-space input filter that
leaves every downstream stage untouched. The vault did not previously anticipate
it; mechanism-wise it is kin to the hand-drawn polygons of `13_road_topology.md`
(MVP3), but it needs none of MVP3's homography/plane machinery, so it ships now.

## Why the detector seam (not the pipeline)

Pixel polygons only align with detections in **raw pixel space**. In the per-frame
loop the detections are raw only between `detect` and the ego-motion
`apply_transform`; after stabilization they live in the keyframe-anchored global
frame and a raw-pixel polygon no longer matches (see `05_75_mvp1_9.md`).

So the drop happens at the **detector seam**, via a `MaskingDetector` decorator
over the `Detector` port (`infrastructure/detection/masking.py`). Wrapping the
real detector keeps `TrajectoryPipeline` byte-for-byte untouched — the same
decorator idiom as `RecordingEgoMotionEstimator`, the `Timed*` decorators, and
`DecimatingTrajectoryExporter`. It is wrapped *below* `TimedDetector`, so the
detection step timing counts detect + filter together.

The ORB masking is independent: `OrbEgoMotionEstimator` takes the same
`ExclusionZones`, rasterizes them once (raw-pixel, frame size known at first
frame) into a cached 255/0 base mask, and zeros the per-frame vehicle boxes on
top of it. With zones present a mask is now returned even before any detection
arrives.

## The "majority overlap" rule

A detection is excluded when, for **some single zone**, the clipped area of the
zone over the bbox exceeds **0.5**. `0.5` is the *definition* of majority, baked
into `domain/exclusion.py` as `_MAJORITY` — not a config knob.

The area is computed by reusing the stabilizer's existing geometry primitives:
`Polygon.overlap_fraction(bbox)` translates the polygon so the box sits at the
origin (`[0,w] x [0,h]`), runs `_clip_to_rectangle` (Sutherland–Hodgman) and
`_polygon_area` (shoelace), and divides by the box area. No new clipping code.

## Input: sidecar JSON

A per-scene JSON file referenced from config (`analysis.exclusion_zones`, a
`toggleable_path`: `""` = off), mirroring `calibration.srt`. Loaded by
`infrastructure/exclusion/json.py` into the pure `ExclusionZones` value object.
Schema:

```json
{ "exclusion_zones": [
    { "label": "parking_lot", "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
] }
```

`label` is optional (operator documentation). `vertices` are pixel coordinates,
≥3 per polygon. Bad path / malformed JSON / too-few vertices fail fast in the CLI
before the costly video open.

## Caveats (deliberate, documented)

1. **Convexity.** Sutherland–Hodgman is exact only when the *clip window* (the
   bbox) is convex (always true) **and** the subject polygon is convex. A concave
   zone can mis-count its area, so concave regions should be drawn as several
   convex polygons.
2. **Union vs. max.** A box is tested against each zone independently (max), not
   against the union of zones. A vehicle straddling two adjacent zones — say 30%
   in each — survives even though the combined coverage is 60%. Acceptable for
   distinct drawn regions; split-and-redraw if it matters.
3. **Moving drone.** The polygons are **screen-fixed**. With `ego_motion.enabled`
   (a moving drone) a screen-fixed zone masks a *moving* world region, so the
   feature is intended primarily for **static / near-static cameras** — the same
   regime where ORB stabilization itself is advised off on otherwise-static clips.

## Files

- `domain/geometry.py` — `Polygon` value object (`overlap_fraction`).
- `domain/exclusion.py` — `ExclusionZones` (`excludes`, `_MAJORITY`).
- `infrastructure/detection/masking.py` — `MaskingDetector` decorator.
- `infrastructure/video/ego_motion_orb.py` — static-zone feature masking.
- `infrastructure/exclusion/json.py` — sidecar loader.
- `application/config.py` — `AnalysisConfig` + `analysis.exclusion_zones`.
- `cli.py` — `--exclusion-zones`, load + existence check, detector wrap + ORB inject.
