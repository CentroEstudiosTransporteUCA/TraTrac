# Step Timing

---

## What This Is

Opt-in **per-step latency profiling**: how long each per-frame pipeline step takes.
The full chain is **detect → observe → ego_motion → stabilize → track → record** —
every step a port, so every step is timeable. It belongs to no MVP — it is
observability plumbing, a sibling of progress reporting (`src/tratrac/application/PROGRESS_REPORTING.md`).
(The set has churned with the architecture: `orient`/`export` left when kinematics and
the `.trj` moved to the offline `tratrac-postprocess` — src/tratrac/application/SMOOTHING.md; `observe`, `ego_motion`,
`stabilize`, `record` were added so the remaining loop is fully covered. All steps are
wrapped unconditionally, every run — there is no run-level toggle. The three
stabilization-only steps — `observe`, `ego_motion`, `stabilize` — just report near-zero
time on a run whose transforms file has no similarity rows, since `EgoMotionStabilizer`
applying the identity transform is a fast no-op, not an unwired collaborator.)

The difference from progress: progress is always-on and flows *through* the
pipeline (the pipeline emits it). Timing is opt-in and wraps the *ports around*
the pipeline (decorators measure it). The pipeline itself is untouched by
timing — it never holds a timing collaborator.

---

## Homogeneous Ports

Every per-frame step is a uniform port call (`detect`, `observe`, `ego_motion`,
`stabilize`, `track`, `record`), each running **exactly once per frame**, so each is
decoratable the same way and a per-frame timing row assembles cleanly. This is also why
the one previously-inline step, stabilization, was promoted to the `DetectionStabilizer`
port: an `apply_transform` list-comprehension buried in the loop could not be timed by a
decorator without putting a stopwatch *inside* the pipeline; behind a port it joins the
others (and the Null Object cleanly replaces the old `if ego_motion` branch). The other
non-port per-frame work — the timestamp arithmetic and the progress emission — is not
"a step": one is a field access, the other is the reporting channel itself.

---

## The Decorators

`infrastructure/timing/decorators.py` — one decorator per port, each
implementing the port it wraps (GoF Decorator), forwarding the call unchanged,
and timing it:

- `TimedDetector`, `TimedDetectionObserver`, `TimedEgoMotion`, `TimedStabilizer`,
  `TimedTracker`, `TimedTrackSink` — one per timed port. `TimedTrackSink` also delegates
  the context manager (open/flush/close), timing only `record`. `TimedEgoMotion` wraps the
  ORB estimator *innermost* (the transform/anchor recording decorators wrap outside it), so
  its measurement is the ORB work alone, not the recording I/O.
- They delegate the actual measurement to a **`StepStopwatch`** collaborator
  (composition, not a shared base class) that times a callable and reports a
  `StepTiming`.
- **Frame ordinal:** each decorator owns a call counter. Because every step
  runs once per frame in lock-step, the counters stay aligned across decorators
  without any shared state or threaded frame identity. The ordinal equals the
  source `Frame.index` while decoding is contiguous (current behaviour); if a
  future source ever samples frames, the ordinals still stay consistent *with
  each other*.

Why decorators rather than timing inside the pipeline: it keeps the pipeline
free of instrumentation, and timing each port lives with that port. Every timed
step runs once per frame, so uniform decoration stays clean.

---

## The Output Port

`TimingSink` (`domain/ports.py`) — single method `record(StepTiming)`. One
record per step per frame; inherently streaming, which is exactly what a future
telemetry POST wants (no buffering a whole frame before sending).

- `StepTiming` + `PipelineStep` are pure value objects in `domain/timing.py`.
- `CsvTimingSink` (`infrastructure/timing/csv.py`) is the first adapter: a
  **wide** CSV (`frame,detect,observe,ego_motion,stabilize,track,record`, one row per
  frame). It buffers a frame's step durations and flushes the row when the ordinal advances
  (safe because a frame's records arrive consecutively); every column is populated on every
  run — the stabilization-only three just report near-zero time when the transforms file has
  no similarity rows. The last frame, and any partial frame from a mid-step crash, flush on
  close.
- Tomorrow's telemetry/socket sink is just another `TimingSink` — no pipeline,
  domain, or decorator change.

There is **no `Null` sink** (unlike progress): timing does not flow through the
pipeline, so when profiling is off the decorators are simply not inserted and
the run pays zero overhead.

---

## Wiring

Opt-in via the CLI flag `--timing-csv PATH`. When set, the CLI wraps each
collaborator in its `Timed*` decorator and opens a `CsvTimingSink`; when unset,
the bare collaborators run unwrapped.

---

## Layering

- `domain/timing.py` — `PipelineStep`, `StepTiming` (pure value objects).
- `domain/ports.py` — the `TimingSink` port.
- `infrastructure/timing/decorators.py` — `StepStopwatch` + the `Timed*`
  decorators (read the clock = infrastructure).
- `infrastructure/timing/csv.py` — `CsvTimingSink` (file I/O = infrastructure).

---

## Real-footage measurement (CPU baseline)

A real 10-second window of a real intersection clip was run with `--timing-csv` on (CPU, the
only runtime available when this was measured):

| Step | Mean | Share of measured time |
| --- | --- | --- |
| `detect` (YOLOv8-VisDrone) | 114.2 ms/frame | 87.4% |
| `track` (BoT-SORT) | 16.4 ms/frame | 12.6% |
| `record` (Parquet write) | 0.02 ms/frame | ~0% |

The timed steps summed to 98.3% of total wall time — decode + Python loop overhead is only
~1.7% on this run. Inference (`detect`) dominates by nearly an order of magnitude over the next
largest step; decode is not silently serialized behind anything on this codebase's actual
per-frame loop shape. This is CPU-only evidence: swapping the detector onto a GPU would shrink
`detect` by roughly an order of magnitude and could plausibly make decode relatively significant
enough to matter — re-profile with `--timing-csv` once a GPU is available, before deciding
whether a faster decoder or an async runtime is actually worth building.
