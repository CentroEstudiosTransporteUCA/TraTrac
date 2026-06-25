# Step Timing

---

## What This Is

Opt-in **per-step latency profiling**: how long each pipeline step (detect,
track) takes for each frame. It belongs to no MVP — it is observability plumbing,
a sibling of progress reporting (`14_progress_reporting.md`). (Before the export
inversion there were two more steps, `orient` and `export`; both left the run when
kinematics and the `.trj` moved to the offline `tratrac-smooth` — see vault/22.)

The difference from progress: progress is always-on and flows *through* the
pipeline (the pipeline emits it). Timing is opt-in and wraps the *ports around*
the pipeline (decorators measure it). The pipeline itself is untouched by
timing — it never holds a timing collaborator.

---

## Homogeneous Ports

The pipeline's steps are uniform port calls (`detect`, `track`), each running
**exactly once per frame**, so each is decoratable the same way and a per-frame
timing row assembles cleanly. (Historically this section justified promoting
orientation to a batch port for the same uniformity; orientation has since left
the pipeline, but the homogeneity argument is unchanged for the remaining steps.)

---

## The Decorators

`infrastructure/timing/decorators.py` — one decorator per port, each
implementing the port it wraps (GoF Decorator), forwarding the call unchanged,
and timing it:

- `TimedDetector`, `TimedTracker` — one per timed port.
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
  **wide** CSV (`frame,detect,track`, one row per frame). It buffers a frame's
  step durations and flushes the row when the ordinal advances (safe because a
  frame's records arrive consecutively); the last frame, and any partial frame
  from a mid-step crash, flush on close.
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
