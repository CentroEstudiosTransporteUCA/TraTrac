# ReID merge decision: motion-plausibility gating + appearance scoring

## Status

Partially shipped (`docs/IMPLEMENTATION_PLAN.md` Group D2, `docs/roadmap/mvp5.md`). **Landed:**
the "merge decision" and "apply" stages — `application/reid_merge.py`, the
`KinematicKalmanFilter.from_state`/`predict` extension it's built on
(`application/kalman.py`), `infrastructure/reid/json.py`, and `tratrac-postprocess
--reid-merge`. **Not landed:** the "embed" stage (`cli_embed.py`/`tratrac-embed`, DINOv3
per-fragment appearance vectors) — it needs a GPU and a real DINOv3 model, unavailable in this
environment, so nothing here has been exercised against a real embedding, only synthetic
vectors in tests. This module is deliberately indifferent to how the embedding was produced —
see "Why merge decision doesn't need the embed stage to exist" below.

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
resolved merges) but not consumed by `--reid-merge` — only `merges` is. There is no CLI
command that *produces* this file yet (that's stage 1+2 wired together, i.e. `tratrac-embed`,
unbuilt); today it's produced by calling `application/reid_merge.py` + `save_reid_merge`
directly, or authored by hand for testing.

## Not done by this landing

- **The embed stage** (`cli_embed.py`, DINOv3) — needs a GPU + real footage; the whole
  appearance-scoring half of this design is unvalidated until it exists.
- **`resolve_merges`'s greedy 1:1 heuristic** is not a full assignment solver (Hungarian
  etc.) — judged good enough since fragment merging is a comparatively rare event (an
  occlusion), not a dense every-frame assignment problem; revisit if real footage shows
  contested candidates common enough for greedy's suboptimality to matter.
- **Gate parameter defaults** (`max_gap_seconds`, `seed_vel_std`, `seed_accel_std`,
  `max_sigma`) are not chosen here — no real occlusion footage exists yet to tune them
  against, so `tratrac-postprocess` does not (yet) expose `--reid-merge`'s upstream gate
  parameters as flags; today they're arguments to `candidate_pairs` for a caller (a future
  `tratrac-embed` or a script) to supply.
