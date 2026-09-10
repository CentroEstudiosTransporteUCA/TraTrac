# Footprint sidecar: mask-derived vehicle dimensions

## Status

Partially shipped (`docs/IMPLEMENTATION_PLAN.md` Group D1, MVP4 remainder). **Landed:** the
sidecar storage format (`infrastructure/tracks/footprint_parquet.py`), the pure geometry it's
built on (`domain/geometry.oriented_extent`), and `tratrac-postprocess --footprint`'s
consumption of it. **Not landed:** the segmentation stage that would actually produce a
footprint sidecar (`cli_segment.py`/`tratrac-segment`, SAM 3) — it needs a GPU and a real SAM 3
model, unavailable in this environment. Like `application/reid_merge.py` before it, this module
is deliberately indifferent to how the polygon was produced — see "Why the storage format
doesn't need SAM 3 to exist" below.

## What this adds

A **footprint** is one polygon per `(track_id, frame_index)` — the vehicle's actual occupancy
outline, as opposed to a bounding box or oriented box's rectangular approximation of it.
`tratrac-postprocess --footprint sidecar.parquet` replaces bbox/OBB-derived `Dimensions` with
footprint-derived ones for every observation the sidecar covers, before smoothing — the real
accuracy payoff segmentation buys over even an oriented box, per `docs/IMPLEMENTATION_PLAN.md`'s
framing of this group.

## Why the storage format doesn't need SAM 3 to exist

"A polygon per track observation" is a general geometric concept, independent of which model
produced it — the same reasoning that let `application/reid_merge.py` land with an opaque
`embedding: tuple[float, ...]` field before DINOv3 existed (`application/REID_MERGE.md`). SAM 3
would eventually produce a per-crop segmentation mask that gets contour-extracted into a
polygon; this module only needs to know "a polygon exists," not the mask format, threshold, or
crop-coordinate convention that produced it. That's what let the storage format, the dimension
math, and the CLI integration land without the segmentation stage that would populate them.

**What's genuinely deferred, not sidestepped:** `cli_segment.py`'s own design — how it prompts
SAM 3 (box- or text-prompted from the OBB, per the plan), what crop size/padding it uses, how it
extracts a contour from a mask — is real work this landing does not attempt, because those
decisions need a real model to validate against, unlike the storage format.

## `domain/geometry.oriented_extent` — deriving `(length, width)` from a polygon

Given a footprint polygon and an optional heading angle, returns the polygon's extent along
that heading and its perpendicular:

- **With a known angle** (from an OBB detection that produced the crop the mask came from):
  projects every vertex onto the heading axis and its perpendicular; the extent along each is
  that axis's `max - min`. Exact when the polygon's true orientation matches the given angle,
  approximate otherwise (a mask's own principal axis need not exactly match a detector's OBB
  angle — no attempt is made to re-derive orientation from the mask itself).
- **Without an angle** (an AABB-only run, no OBB detector): falls back to the polygon's plain
  axis-aligned bounding box extent, not a fitted principal axis (minimum-area rectangle / PCA) —
  simpler, and exactly correct when the vehicle happens to be axis-aligned. Same honesty
  tradeoff `_major_axis_heading` (`application/track_smoothing.py`) already makes for the
  low-speed heading fallback: a documented approximation, not a claim of precision the method
  doesn't have.

## How `--footprint` integrates (reusing Group A's OBB slot, not a new one)

`_apply_footprint` (`cli_postprocess.py`) writes the footprint-derived `(length, width)`
directly into `TrackObservation.obb_w`/`obb_h` — **the same nullable columns Group A's OBB
detector already populates** (`infrastructure/tracks/parquet.py`), not new fields. This means
every downstream consumer treats a footprint-derived size exactly like an OBB-derived one, with
zero new special-casing:

- `build_state` (`application/track_smoothing.py`) already prefers `oriented_size` over the bbox
  when present.
- `_project_observation`'s world-projection path already drops `obb_w`/`obb_h` post-projection
  (image-space quantities, invalidated by a general homography — `application/WORLD_PROJECTION.md`),
  so a footprint-derived size is correctly discarded under `--calibration` the same way an OBB
  angle is, rather than silently misapplied in the wrong coordinate frame.

**Composition-root position:** `--footprint` is applied **first**, immediately after reading the
record, **before** `--reid-merge`. A footprint sidecar is keyed by the *original* `track_id` a
hypothetical segmentation run would have seen (the same ids the raw perception record uses) —
applying it before ReID's track_id remap means the lookup still matches for any track that later
gets merged; applying it after would silently lose footprint coverage for merged fragments,
since their `track_id` would already have changed.

## Sidecar schema (`infrastructure/tracks/footprint_parquet.py`)

Columns: `frame` (int64), `track_id` (int64), `xs`/`ys` (`list<float64>`, parallel arrays of
polygon vertex coordinates — pyarrow has no native point-list type, so this mirrors how other
list-shaped data in this codebase is stored rather than inventing a struct type). Mirrors
`infrastructure/tracks/parquet.py`'s writer/reader shape (buffered row-group writes, a
`...Sink` context manager, a plain read function) rather than a new pattern.

## Not done by this landing

- **The segmentation stage itself** (`cli_segment.py`, SAM 3) — needs a GPU + a real model; see
  above for exactly what's deferred vs. what already landed.
- **No attempt to re-derive orientation from the mask** — `oriented_extent` only *uses* an
  already-known angle (from Group A's OBB detector, when present); it doesn't fit one from the
  polygon itself (e.g. minimum-area rectangle / PCA) when no angle is available.
- **Not validated against a real mask** — `oriented_extent`'s geometry is tested against
  synthetic polygons; the closest thing to a real-footage check any Group D landing got
  (`application/REID_MERGE.md`'s real-footage ReID run) isn't available here, since it needs a
  real segmenter to produce a mask to test against in the first place.
