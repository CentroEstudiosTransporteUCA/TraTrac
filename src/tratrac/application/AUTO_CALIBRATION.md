# Automatic calibration: correspondence proposal from road geometry

## Status

Shipped, **scoped to proposal only** (GitHub Issues Group C4). The repo-boundary
question the plan flagged as needing an explicit decision before this task could start —
"does TraTrac ship only the correspondence-proposal capability, with interactive confirm/adjust
living in URBAn?" — is resolved: **yes, proposal only.** `scripts/propose_calibration.py`
detects candidate road-marking line segments on a still frame and writes their **image-side**
coordinates in `calibration.json`'s schema, with `world` left `null` for a human (via URBAn, or
hand-editing) to confirm and fill in. TraTrac does not attempt the interactive confirm/adjust
step, and does not attempt to infer real-world coordinates automatically.

## Why proposal-only, grounded in the cited literature

The plan's own text already asserted "manual validation still outperforms automatic, per the
cited source" without naming it. This landing tracked that source down: Popov, Trukhina &
Vashkelis, *Mobile Traffic Camera Calibration from Road Geometry for UAV-Based Traffic
Surveillance* (arXiv:2605.11900, 2026) — a UAV traffic-camera calibration pipeline using the
exact same road-geometry cues (lane markings, road borders, crosswalks) into the exact same
kind of homography TraTrac already fits (`infrastructure/world/calibration.py`). Their own
experiment is the direct evidence for "manual outperforms automatic": they tried "automatic
line-based homography using line detection and vanishing-point heuristics" against
human-in-the-loop calibration, and report "initial fully automatic homography estimation
produced plausible but geometrically imperfect BEV views... This supports the hypothesis that
human-in-the-loop calibration is currently more robust than fully automatic calibration for
real UAV traffic footage." Their own calibration workflow keeps a human selecting points and
assigning metric coordinates "based on lane width and road direction" — i.e. even their
best-performing path is not blind automation, real-world coordinates still come from an
operator. This landing's proposal-only scope isn't a shortcut taken to save effort; it's the
scope the cited literature itself validates as the actually-robust one.

## What it does

`scripts/propose_calibration.py` (stdlib + cv2 + numpy, no `tratrac` package import — a
`calibration.json`-shaped output needs no domain types):

1. **Canny edge detection** + **probabilistic Hough transform** (`cv2.HoughLinesP`) over a
   still frame — finds every fairly-straight edge, most of which are *not* road markings
   (shadows, curbs, vehicle bodies, building edges, sidewalk pavers).
2. **A brightness + local-contrast filter** narrows that down to segments that actually look
   like a painted marking: bright (`--min-brightness`, road paint is white/yellow), **and**
   meaningfully brighter than the surface sampled a few pixels perpendicular to the line on
   either side (`--min-contrast`/`--contrast-offset`). Brightness alone is not enough —
   it also passes uniformly light surfaces (sidewalks, concrete plazas, rooftops) that have no
   real photometric edge; requiring *local* contrast against the surrounding surface is what
   actually distinguishes "a painted line on darker asphalt" from "a bright area."
3. Writes each surviving segment's endpoints as **image-only** candidate correspondences
   (`world: null`) plus an optional annotated overlay PNG for visual review.

**Deliberately not attempted**: classifying *which* marking a line is (lane boundary vs.
crosswalk stripe vs. road border) — Canny + Hough finds edges, not semantics, and a wrong
semantic guess would mislead a human reviewer worse than an unlabeled candidate would.
Real-world coordinate assignment — the literature's own finding above is why this is left to a
human.

## Validated against real footage

Run against a real nadir intersection frame (1920x1080, the same "cruce" clip Group B1 used —
see `src/tratrac/application/REID_MERGE.md`'s real-footage findings): 4,286 raw Hough segments
→ 976 pass a brightness-only filter (too permissive — catches sidewalks, plazas, rooftops) →
**32 pass brightness + local contrast**, visually confirmed (overlay review) to correctly land
on real crosswalk stripes, a lane divider line, and a road border, with imperfect recall (some
crosswalk stripes and one far-side crossing missed) and a couple of false positives (a vehicle
roofline, a shadow edge). This imperfect-but-genuinely-informative result is consistent with —
not worse than — the cited paper's own "plausible but geometrically imperfect" characterization
of automatic line-based calibration; it's exactly why the output is scoped as a proposal a human
reviews, not a calibration TraTrac would use unattended.

## Not done by this landing

- **Parameter tuning is a starting point, not validated across sites.** Defaults were checked
  against one real frame from one intersection; a different lighting condition, marking
  condition, or camera altitude would likely need different `--canny-*`/`--min-brightness`/
  `--min-contrast` values. No sweep or cross-site validation was done.
- **No semantic labeling** of candidates (which marking is which) — see above.
- **No automatic world-coordinate assignment** — by design, per the cited evidence.
- **No integration with `tratrac-postprocess`** — this is a standalone pre-processing step
  whose output (after human review/completion) becomes an ordinary `calibration.json` for the
  already-shipped `--calibration` flag; no new CLI flag was added to consume a "proposal" file
  directly, since it isn't usable until a human has filled in `world` anyway.
- **The interactive confirm/adjust UI** — explicitly out of scope per the resolved repo-boundary
  decision; that's URBAn's job.
