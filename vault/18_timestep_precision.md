# Timestep Precision (Output Decimation)

---

## What This Is

An optional `--timestep-precision SECONDS` flag that sets a **minimum interval
between exported TIMESTEP records** in the `.trj`. Off by default (one TIMESTEP
per decoded frame, see the 1:1 frame→TIMESTEP mapping the pipeline produces).

Use it to thin a high-FPS export down to the temporal resolution a downstream
consumer actually needs, without re-encoding the video.

---

## Reading A: It Samples the Output, Not the Processing

There were two readings of "seconds between each timestep":

- **A — export cadence (chosen):** detect, track, and orient run on *every*
  decoded frame; only the *write* to the `.trj` is throttled.
- **B — processing cadence (rejected):** skip frames before the detector.

B was rejected because BoT-SORT's association (IoU + motion model) assumes small
inter-frame motion; decimating its input causes ID switches. The vault chose
BoT-SORT for robustness (`03_tech_stack.md`), and B trades exactly that away for
speed. A keeps tracking quality and only changes the output's temporal density.

Consequence: A gives **no compute saving** — every frame is still detected and
tracked.

---

## Reading B Now Exists Too: `input.process_fps` (Decode-Time Decimation)

The "run faster on long clips" need is now its own, separately-named feature:
`input.process_fps` caps the **processing** cadence (`0.0` = every frame). It is
independent of, and orthogonal to, `export.timestep_precision`.

- **Where:** the **video adapter** (`OpenCvVideoSource`), not the export seam.
  Frames off the target grid are skipped with `cv2.grab()` (advance, no decode);
  only kept frames are `read()`/decoded. So it saves **decode *and* detection +
  tracking** — a real speedup, unlike A.
- **Same grid math:** it reuses the shared `DecimationGrid`
  (`infrastructure/cadence.py`) that backs the exporter — anchored grid, half-frame
  snap, no drift. Extracted so the two features can't diverge.
- **Absolute indices preserved:** skipped frames still advance the absolute index,
  so TIMESTEPs stay on the source clock, `--start`/`--end` composes, and a replay
  transform schedule (keyed by `frame.index`) still lines up.
- **Honest progress:** `total_frames` is "frames this run will process," so the
  adapter reports the grid-accepted count (computed with the same grid that drives
  yielding), not the full window length.
- **The trade-off (the reason it's off by default):** decimating the detector's
  input enlarges inter-frame motion, so BoT-SORT (IoU + constant-velocity model)
  produces **more ID switches**, and ORB ego-motion sees bigger baselines (more
  re-anchoring). This is exactly the quality hit Reading B was rejected for *as the
  timestep feature*; here it is a deliberate, opt-in speed-for-quality knob.
- **Stacking with A:** processing runs first, export decimation thins further. You
  cannot export finer than you process, so the CLI **warns** when
  `timestep_precision` is finer than `1/process_fps` (the export then effectively
  emits at the processing rate).

---

## Where It Lives

In a **`TrajectoryExporter` decorator** — `DecimatingTrajectoryExporter`
(`infrastructure/export/decimating.py`) — not in the pipeline loop.

- The pipeline keeps calling `emit_frame` once per frame; the decorator decides
  whether to forward the call to the real `SsamTrjExporter`.
- This mirrors the `Timed*` decorators (`vault/15_step_timing.md`): cadence
  policy stays out of the orchestrator, and the concrete writer stays dumb.
- It composes *inside* `TimedExporter` (`TimedExporter(Decimating(Ssam))`), so the
  EXPORT step still records once per processed frame and stays aligned with the
  other steps' per-frame ordinals (vault/15). Skipped frames just record the
  cheap "did not write" path.

Contrast with `--start`/`--end` (`vault/17_time_window.md`), which lives in the
video *adapter* because trimming needs a real seek. Decimation needs no seek and
must not change what the tracker sees, so it belongs at the export seam instead.

---

## The Emission Schedule

The first frame is always emitted; its timestamp **anchors** an emission grid at
`anchor + k * interval`. A frame is forwarded once its timestamp reaches the next
grid point, within **half a frame** (`0.5 / fps`) so spacing snaps to the nearest
available frame instead of always rounding up. The grid (not a fixed frame stride)
is the reference, so realized intervals track the requested value with no
cumulative drift, and an interval at or below the frame duration degrades cleanly
to emitting every frame.

Timestamps stay on the source video's absolute clock, so this composes with
`--start`/`--end`: a windowed run anchors the grid at the window's first frame
(e.g. a window starting at 10.0 s emits 10.0, 10.1, … for a 0.1 s interval).

---

## SSAM Coarseness Caveat

`vault/04_ssam_format.md` notes sub-second precision (~1/10 s) is the practical
minimum; once-per-second is too coarse for conflict analysis. A coarse interval
still produces a *syntactically valid* `.trj`, so the CLI **warns** (above 0.5 s)
rather than erroring — the file parses, but its surrogate-safety metrics (TTC,
PET) degrade. Validity is structural; usefulness is the operator's call.

---

## Layering

- `infrastructure/export/decimating.py` — `DecimatingTrajectoryExporter` (the
  decorator + grid math).
- `application/config.py` — validates `export.timestep_precision` (reject < 0;
  `0` = every frame) as part of `RunConfig.resolve` (see `vault/19_config_file.md`).
- `cli.py` — the `--timestep-precision` flag, the coarse-value warning (> 0.5 s),
  and the wrapping wired inside the timing decorator.
