# ReID merge decision: motion-plausibility gating + appearance scoring

## Status

Partially shipped (GitHub Issues Group D2, `docs/roadmap/mvp5.md`). **Landed:**
the "merge decision" and "apply" stages — `application/reid_merge.py`, the
`KinematicKalmanFilter.from_state`/`predict` extension it's built on
(`application/kalman.py`), `infrastructure/reid/json.py`, and `tratrac-postprocess
--reid-merge`. The motion-gate half has also now been **run against real footage**
(`scripts/probe_reid_merge.py` — see "Validated against real footage" below), though with a
placeholder embedding, not DINOv3. **Not landed:** the "embed" stage
(`cli_embed.py`/`tratrac-embed`, DINOv3 per-fragment appearance vectors) — it needs a GPU and a
real DINOv3 model, unavailable in this environment. This module is deliberately indifferent to
how the embedding was produced — see "Why merge decision doesn't need the embed stage to exist"
below.

## What this adds

Long-term identity persistence needs to stitch a tracker's fragments — the same physical
vehicle split into more than one `track_id` across an occlusion — back into one track.
`application/reid_merge.py` decides which fragments are really the same vehicle:

1. **Motion-plausibility gate** — extrapolate the earlier fragment's end kinematic state
   forward to the later fragment's start time, reject the pair if the actual reappearance
   position falls outside a configurable number of standard deviations of the extrapolation.
2. **Appearance score** — among gated (motion-plausible) candidates, cosine similarity
   between each fragment's embedding.
3. **Resolution** — greedy highest-score-first, 1:1 (each fragment absorbs/is absorbed by at
   most one other), with chains collapsing to their earliest fragment's id.

This is the "not appearance alone" design `TRACKER_CHOICE.md`/`mvp5.md` call for: nadir drone
footage discards most of the visual cues vehicle-ReID normally relies on, so a real deployment
facing the same problem (Songdo, South Korea) had to fuse appearance with a temporal/motion
model to make ReID work at all. Rather than build a separate travel-time model, this reuses
`application/kalman.py`, already in the codebase for exactly this kind of prediction.

## Why merge decision doesn't need the embed stage to exist

The three D2 stages split expensive/cacheable from cheap/re-tunable, mirroring
`SMOOTHING.md`'s two-pass design:

1. **Embed** (unbuilt) — the only stage needing video + a GPU + DINOv3. Would write one
   embedding per track fragment to an `embeddings.parquet` sidecar.
2. **Merge decision** (`application/reid_merge.py`) — pure, no video/GPU, fully re-tunable
   offline like `--pos-noise`/`--jerk`. Consumes `TrackFragment`s (whatever produced their
   `embedding` field) and decides merges.
3. **Apply** (`tratrac-postprocess --reid-merge`) — remaps `TrackObservation.track_id`; the
   entire integration point. `_smooth_recording` already groups purely by `track_id`, so
   merged fragments are automatically smoothed as one continuous track with **no new
   smoothing code** — this is why the plan calls "Apply" the entire integration point.

Stage 2's `TrackFragment.embedding` is `tuple[float, ...]`, opaque beyond being comparable by
cosine similarity — nothing in `candidate_pairs`/`resolve_merges` depends on DINOv3
specifically, or on embeddings coming from a real model at all. That's what let this land
without stage 1: the module is exercised with synthetic embedding vectors in
`tests/unit/test_reid_merge.py`, which is a legitimate test of the *decision logic*, but not
a validation that real DINOv3 embeddings on real occlusion footage actually separate
same-vehicle from different-vehicle pairs well enough for the appearance score to be useful —
that validation is still entirely open, and requires stage 1 + a GPU + real footage.

## Composition-root position: before everything else track-lifetime-aware

`--reid-merge` remaps `track_id` **first** in `tratrac-postprocess`, before exclusion
filtering, Link/Lane/Plane assignment, or world projection — all of those are track-lifetime
or per-observation-identity-aware, so occlusion-split fragments must already be one `track_id`
by the time they run (e.g. exclusion's majority vote should judge the vehicle's whole life,
not two separately-judged short fragments):

```
read record
  → apply --reid-merge (D2)         [track_id remap — before anything track-lifetime-aware]
  → filter --exclusion-zones
  → assign --link-zones / --lane-zones / --plane-zones
  → project --calibration
  → smooth (Kalman/RTS)
  → export .trj
```

## `KinematicKalmanFilter.from_state` / `.predict` (Group D2's kalman.py extension)

`application/kalman.py` had a stateful forward filter (`observe()`, built up from raw
measurements) but no way to (a) seed a filter directly from an already-known state — the
whole-track RTS-smoothed end of a fragment is a far better estimate than what a single
`observe()` call would produce, which starts velocity/acceleration at
`_INITIAL_RATE_VARIANCE` (i.e. "unknown") — or (b) extrapolate forward with no new
measurement at all. Both were added:

- `from_state(state, *, pos_noise, jerk, vel_std, accel_std)` — a classmethod seeding
  `_x`/`_p` directly from a `SmoothedSample` plus explicit seeded confidence in each
  component.
- `predict(dt)` — a **pure query** (does not mutate the filter), so the same seeded filter
  can be queried at several candidate reappearance times without recommitting state. Returns
  the predicted state plus its predicted position standard deviation per axis — the gate's
  growing uncertainty.

## Sidecar schema (`infrastructure/reid/json.py`)

```jsonc
{ "merges": { "<old_track_id>": <canonical_track_id> },
  "candidates": [
    { "from_track_id": 1, "to_track_id": 2, "gap_seconds": 1.5, "score": 0.93 }
  ] }
```

`candidates` is written for operator audit (every motion-plausible pair, not just the
resolved merges) but not consumed by `--reid-merge` — only `merges` is. There is no
`tratrac-embed` yet (stage 1+2 wired together with a real embedding) — `scripts/probe_reid_merge.py`
(below) is today's way to produce this file, motion-gate only; a hand-authored file works too
for testing.

## Validated against real footage (motion gate only, no DINOv3)

`scripts/probe_reid_merge.py` runs `candidate_pairs`/`resolve_merges` against a real track
record with every fragment given the **same placeholder embedding** — neutralizing the
appearance-similarity term so only the Kalman motion gate decides merges — and writes the
result for `tratrac-postprocess --reid-merge` to apply. Run against a real ~15-minute
congress-site intersection clip (1920x1080, 30fps, YOLOv8-VisDrone baseline, `conf=0.25`,
default gate parameters):

- **1,545 track fragments → 52 motion-plausible candidate pairs → 43 accepted merges** (36
  canonical tracks absorbed the 43).
- `scripts/validate_trj.py` Continuity compliance **before**: 38.67% appearances / 40.10%
  disappearances compliant (2,873 total continuity events). **After** the 43 merges: 38.74% /
  39.99% — **no measurable improvement**.
- **Why**: continuity events far outnumber tracks (2,873 events across 1,545 fragments before
  merging — an average of ~1.9 per fragment), and separately, track-length distribution on this
  clip shows 47.1% of all fragments last under 1 second (30 frames) and 18.5% under a third of a
  second (10 frames). That's the signature of a low-confidence emergency detector
  (`conf=0.25`) producing many short, flickering, likely-spurious detections — not vehicles
  genuinely disappearing behind an occluder and plausibly reappearing later, which is the
  specific failure mode ReID merging targets. **Conclusion**: on this footage, low continuity
  compliance is dominantly a *detector quality* problem, not an occlusion/identity problem — this
  is real evidence (not a guess) that MVP1.5's YOLO-OBB fine-tune (GitHub Issues
  Group A2, still GPU-blocked) is likely higher-leverage for this metric than finishing ReID's
  embed stage would be, though both remain worth finishing.
- This does **not** mean the motion gate or `resolve_merges` are broken — 52 candidates out of
  ~1.2M possible pairs (1,545²) is exactly what a tight, working gate should produce when most
  fragments genuinely aren't the same reappearing vehicle. It means this particular clip doesn't
  have enough genuine occlusion-driven fragmentation for ReID merging to move the needle much,
  which is itself useful information, not a null result.

## Not done by this landing

- **The embed stage** (`cli_embed.py`, DINOv3) — needs a GPU + real footage; the whole
  appearance-scoring half of this design is unvalidated until it exists. The motion gate alone
  found few candidates on real footage (above) — appearance scoring would only ever *narrow*
  that set further, so this doesn't change the priority conclusion above.
- **`resolve_merges`'s greedy 1:1 heuristic** is not a full assignment solver (Hungarian
  etc.) — on real footage only 52 candidates arose (no contested pairs observed to stress-test
  greedy's suboptimality against); revisit if a future clip shows many contested candidates.
- **Gate parameter defaults** (`max_gap_seconds=2.0`, `seed_vel_std=10.0`,
  `seed_accel_std=5.0`, `max_sigma=3.0`) have now been run once against real footage (above) but
  not swept/tuned — `tratrac-postprocess` still does not expose them as flags; today they're
  `scripts/probe_reid_merge.py` CLI arguments.
