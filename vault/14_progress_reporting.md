# Progress Reporting

---

## What This Is

A cross-cutting **output port** for communicating progress while a video is
processed. It belongs to no MVP — it is observability plumbing, symmetric to the
trajectory exporter: the application *emits*, adapters *render*.

---

## The Port

`ProgressReporter` (in `domain/ports.py`) has a single method:

```python
def receive(self, event: ProgressEvent) -> None: ...
```

Single-channel messaging. The pipeline sends a stream of `ProgressEvent`s; each
reporter dispatches on the concrete type and **silently ignores** events it does
not handle. New event types can be added to the family without breaking existing
reporters — the vocabulary is open for extension. (This is why the port is one
`receive(event)` and not a set of lifecycle methods: adding an event must not be
a breaking change.)

---

## The Event Family

A sealed-by-convention hierarchy of value objects in `domain/progress.py`. Each
describes *what happened*, never *how to render it*:

| Event | When | Carries |
| --- | --- | --- |
| `ProcessingStarted` | before the first frame | `VideoMetadata` |
| `FrameProcessed` | after each frame is exported | absolute frame index, frames done (this run), windowed total, timestamp, active tracks; behaviour: `fraction`, `percent` |
| `ProcessingFinished` | after the last frame | frames processed |
| `ProcessingFailed` | a frame raised (error re-raises after) | frame index, error message |

`FrameProcessed.fraction` is `frames_done / total_frames` (not the absolute
`frame_index`): with an analysis window (`vault/17_time_window.md`) the index is
the real frame number, which would mismatch the windowed total. It returns
`0.0` when the total is unknown (`total_frames <= 0`, which OpenCV can report)
and clamps to `1.0` when the reported count under-counts. Rendering and
throttling are the reporter's concern, not the event's.

---

## Wiring

- `TrajectoryPipeline` owns emission — it holds the richest per-frame data
  (frame index *and* vehicle counts). It receives a `ProgressReporter` as an
  injected collaborator.
- Default is `NullProgressReporter` (Null Object, `application/progress.py`) so
  the pipeline always has a reporter and never guards against `None`.
- `ConsoleProgressReporter` (`infrastructure/progress/console.py`) is the first
  real adapter: a throttled, in-place **stderr** line (stdout is reserved for
  the CLI's final summary). The CLI wires it.
- A future UI client (websocket, SSE, GUI callback) is just another adapter
  behind the same port — no pipeline or domain change required.

---

## Layering

- `domain/progress.py` — events (pure value objects).
- `domain/ports.py` — the `ProgressReporter` port.
- `application/progress.py` — `NullProgressReporter` default (no I/O).
- `infrastructure/progress/` — reporters that touch the outside world (stdout,
  sockets, ...). The console adapter lives here because printing is I/O.
