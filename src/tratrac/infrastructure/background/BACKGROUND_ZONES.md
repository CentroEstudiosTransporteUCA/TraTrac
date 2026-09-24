# Background zones: operator-authored ORB masking, no detector needed

## What this adds

A set of **pixel polygons** marking the regions of a frame safe to use for ORB
feature extraction. Used by `tratrac-preprocess`, the detector-free pre-pass that
builds the ego-motion transforms file before the main `tratrac` run — `tratrac`
itself never estimates ego-motion or runs ORB. Mirrors
`src/tratrac/application/EXCLUSION_ZONES.md`'s shape and workflow closely; this
doc only covers what's different.

## Why this exists

Masking vehicles out of ORB's feature extraction is necessary (an unmasked pass
on bare asphalt with dense traffic is a known, literature-confirmed failure mode
— coherently-moving vehicles can dominate RANSAC's inlier consensus), but doing
it from the live detector's own detections would mean either running the
detector twice (once to estimate ego-motion, once for the real perception run)
or entangling ego-motion estimation with detection in a single pass. Background
zones remove the detector from ego-motion estimation entirely: the mask is fully
known ahead of time, from a human, so `tratrac-preprocess` needs no detector at
all, and `tratrac`'s own single detector pass can consume an already-complete
transforms file instead of estimating ego-motion itself.

## The workflow

```
[external tool, out of scope here] operator watches VIDEO, draws the region(s)
  safe for ORB features -> background_zones.json

tratrac-preprocess estimate VIDEO --background-zones background_zones.json \
    --out transforms.jsonl --anchors-dir anchors/ --meters-per-pixel 0.05
   → transforms.jsonl        (the complete per-frame ego-motion + scale table)
   → anchors/frame_<i>.png   (one per ORB keyframe anchor — no separate manifest;
                               an anchor's pose is already transforms.jsonl's row
                               at that frame index)

tratrac --config run.toml   # input.transforms_in = "transforms.jsonl"
   → the detector runs exactly once, against already-known ego-motion
```

The drawing tool itself is out of scope for this repo — same footing as
`calibration.json` and `exclusion_zones.json` today, both externally authored.

## Zones apply per frame range, not per anchor

Unlike exclusion zones (each authored on one anchor, mapped into the global frame
once via that anchor's pose), a background zone is authored directly in **raw**
pixel coordinates and selected by `frame_index` at **estimation** time, before any
global frame exists — masking has to happen before ORB can even discover an
anchor. Each entry's `reference_frame` means "use this polygon from this frame
until a later entry supersedes it": `BackgroundZoneMaskSource.mask_for` resolves
the most recent entry at or before the queried frame (falling back to the earliest
zone for a frame before the first entry). These reference frames do **not** need to
align with ORB's own re-anchor points — the external tool decides how often to
redraw, e.g. every time the drone's framing changes enough that the previous
zone's polygon no longer describes safe background.

## Input: sidecar JSON (same shape as `exclusion_zones.json`)

```json
{ "background_zones": [
    { "reference_frame": 0,   "vertices": [[x1, y1], [x2, y2], [x3, y3]] },
    { "reference_frame": 850, "vertices": [[...]] }
] }
```

`reference_frame` defaults to `0`; `vertices` are pixel coordinates, ≥3 per
polygon. At least one zone is required — unlike exclusion zones, where an empty
list legally means "exclude nothing," a mask source needs *some* keep-region.
Loaded by `infrastructure/background/json.py` into the pure `BackgroundZones`,
sharing `infrastructure/zones.py`'s `{reference_frame, vertices}` parser with
`infrastructure/exclusion/json.py` — same wire shape, different meaning downstream
(mask ORB vs. drop from analytics), so kept as two small domain types rather than
one shape reused by casting.

## Files

- `domain/background.py` — `BackgroundZone` (`reference_frame` + polygon),
  `BackgroundZones` (non-empty).
- `infrastructure/zones.py` — the shared `{reference_frame, vertices}` JSON parser
  (`load_reference_frame_polygons`), used by both this loader and
  `infrastructure/exclusion/json.py`.
- `infrastructure/background/json.py` — `load_background_zones`.
- `infrastructure/video/ego_motion_orb.py` — `MaskSource` Protocol,
  `BackgroundZoneMaskSource` (this doc's consumer, and the only `MaskSource`
  implementation left now that detector-fed masking is gone).
- `cli_preprocess.py` — `tratrac-preprocess estimate`: `--background-zones`, `--out`
  (transforms file), `--anchors-dir`.
- `application/config.py` / `cli.py` — `input.transforms_in`,
  `PrecomputedEgoMotionEstimator` (`infrastructure/transform/sink.py`): the
  `tratrac`-side consumer of a `tratrac-preprocess` run's output; `tratrac`
  never constructs `OrbEgoMotionEstimator` itself.

See `infrastructure/video/ego_motion_orb.py`'s module docstring's "Detector-free ego-motion"
section for the full picture (why this exists, what was tried and rejected first).
