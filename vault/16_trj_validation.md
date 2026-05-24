# Trajectory Validation

---

## What This Is

A **semantic end-to-end check** on the whole video → `.trj` pipeline:
`scripts/validate_trj.py`. It parses a finished `.trj` and asks whether the
trajectories make *physical sense*, reporting a per-check compliance table
(`compliant instances / total instances`) and, optionally, a CSV of every
non-compliant instance located for inspection in the source video.

It belongs to no MVP — it is QA plumbing, a sibling of the other `scripts/`
diagnostics. Standalone (stdlib only, like `dump_trj.py`): it re-implements the
`.trj` reader rather than importing the package, so it still runs when the
package is broken. The reader mirrors `dump_trj.py`; the format is
`04_ssam_format.md`.

---

## The Load-Bearing Idea: No Ground Truth

There are no hand-labelled trajectories to compare against. That bounds what
"validation" can mean. Without ground truth you can only measure two things:

- **internal consistency** — does the file agree with itself?
- **physical plausibility** — does it violate physics?

You **cannot** measure *accuracy* (is the vehicle really there, is the front
really the front). An early-stage pipeline passes consistency and plausibility
easily, so a high score on those is **not** evidence of quality.

**Continuity is the one exception** and therefore the most valuable check:
"objects do not wink into existence in mid-air" is a physical invariant that
holds independently of the pipeline's own maths — the closest thing to
ground-truth-free truth. It is the check that actually bites on MVP1 output.

---

## The Three Checks

### 1. Continuity (the real quality gate)

Each track's observations are split into **contiguous runs** (segments) by
frame ordinal. Every segment has a start and an end:

- a **start** is compliant only if it is the global first timestep (already in
  view when the clip opens) **or** the centroid is near an image boundary;
- an **end** is compliant only if it is the global last timestep (still in view
  when the clip ends) **or** near a boundary.

This single segment model subsumes births, deaths, **and internal gaps**: a
track that vanishes mid-clip and reappears produces an interior disappearance +
interior appearance, both of which must be boundary-justified or they count
against the score. This is deliberate — gaps are exactly the ID-fragmentation
that MVP1's IoU-only BoT-SORT (no ReID) produces, and surfacing them is the
point. Long-term identity persistence is an explicit MVP5 concern.

Near-boundary uses centroid distance to the nearest edge ≤ `margin` + half the
vehicle's major axis (the body, not just the point, can reach the edge).

### 2. Orientation smoothness

The front/rear axis (heading, recovered as `atan2(front − rear)`) must not jump
between consecutive observations of a track. Compliant if the wrapped angular
change ≤ threshold. This catches **sudden front/rear switches** (the ~180° flip
the MVP1 estimator can produce when a stationary track first moves and the
heading snaps from the bbox major-axis to the velocity direction).

Caveat on what it *cannot* see: heading is EMA-smoothed upstream, so this
largely checks that a smoother produced smooth output. Smoothness is orthogonal
to correctness — a consistently-backwards front/rear scores 100% smooth. It
catches discontinuities, not wrongness.

### 3. Kinematic plausibility (NOT consistency)

The stored Speed and Acceleration are **physical**, in `DIMENSIONS.Units`
(m/s, m/s² or ft/s, ft/s²) per the SSAM spec — *not* pixels, *not* scaled by
`Scale`. Each is checked against a real-world ceiling (unit-aware defaults,
overridable): speed finite and in `[0, max_speed]`; acceleration finite with
`|a| ≤ max_accel`. Defaults reject the impossible, not driving style: ~70 m/s
(252 km/h) and ~12 m/s² (~1.2 g), or the ft equivalents.

**MVP1 caveat — failing is the intended signal.** MVP1 writes
pixel-displacement into these fields while declaring metric units (see
`05_mvp1.md` / `04_ssam_format.md`). So MVP1 output fails these bounds
*wholesale* (speeds of hundreds "m/s", accelerations of thousands "m/s²"). That
is the validator correctly reporting that the kinematics are not yet physical —
the numbers only become meaningful once real metric calibration lands
(MVP1.75 from drone metadata, MVP2 from homography).

#### Why a *consistency* check was rejected

An earlier version also recomputed speed/accel from the stored positions using
the estimator's exact windowed kinematics and compared. It scored ~100% — and
that is precisely why it was removed: it is **tautological**. The exporter
*derives* speed/accel from those positions with that formula, so recomputing
`f(positions)` and checking it equals the stored `f(positions)` can only catch
an export *wiring* bug (byte order, field swap, y-flip/scale error), never a
quality problem. Garbage positions yield *consistent* garbage. Plausibility
against physical units is the check that actually constrains the values.

---

## Output

- **Table** (stdout): header row + grouped checks, each
  `name  compliant / total  pct%`.
- **`--violations-csv PATH`** (opt-in): one row per non-compliant instance —
  `frame_index, timestamp_s, vehicle_id, check, detail, centroid/front/rear
  (image-space px, y-down to match the video), length, width, speed, accel`.
  Positions are converted out of SSAM y-up into image space so a row points
  straight at the frame + spot to inspect. Sorted by frame then vehicle. On
  MVP1 the plausibility checks dominate this file by volume; filter on the
  `check` column for the actionable rows (`appearance`, `disappearance`,
  `heading_switch`).
- **`--fail-under PCT`** (opt-in): exit non-zero if any check is below `PCT`,
  for use as a CI gate.

`frame_index` is the timestep ordinal, which equals the video frame index
because the pipeline emits exactly one TIMESTEP per decoded frame.

---

## Thresholds

All are CLI flags with documented defaults (constants at the top of the
script). They are deliberately conservative sanity bounds, not tuned truth:

- `--boundary-margin` (default 30 grid units, added to half the vehicle major
  axis).
- `--max-heading-step` (default 20°): above this a transition counts as a
  sudden switch. A hard turn at 30 fps is ~1–2°/frame, so 20° passes real
  driving while catching flips.
- `--max-speed` / `--max-accel`: default to the unit-aware physical ceilings;
  override for a specific dataset or once real metric output exists.

---

## Scale Independence

The position-based checks (continuity, orientation) work entirely in the file's
grid units against the `DIMENSIONS` bounds, and the stored grid coordinates are
pixels regardless of `Scale` (the exporter writes `coord / Scale`). So those
checks are unaffected by whether `Scale` is 1.0 (MVP1 pixels-as-metres) or a
real GSD (MVP1.75+). Only kinematic plausibility depends on units, and it reads
them from `DIMENSIONS.Units`.
