# Time Window (Analysis Trimming)

---

## What This Is

Optional `--start` / `--end` flags that restrict processing to a `[start, end]`
time interval of the input video, instead of the whole file. Off by default
(whole video). Useful for iterating on a clip without re-running the full video
and for excluding takeoff/landing footage.

---

## Where It Lives

In the **`OpenCvVideoSource` adapter**, not in a port-level decorator. The
adapter is the only collaborator that can *seek* (`cv2.CAP_PROP_POS_FRAMES`); a
`VideoSource`-port decorator could only filter `frames()`, which would still
decode-and-discard everything before the start — wasteful on long aerial clips.
Trimming the start needs a real seek, so it belongs with the capture handle.

The pipeline, detector, tracker, orientation, and exporter are **untouched**:
they just see a shorter frame stream.

---

## The Frame-Window Math

`infrastructure/video/window.py` — `FrameWindow`, a pure (no cv2) value object
built by `FrameWindow.from_seconds(fps, total_frames, start_seconds, end_seconds)`.
It owns the policy:

- **Inclusive end:** the frame at `end_seconds` is included.
- **Clamping:** `end` past the last frame is clamped to it.
- **Unknown total:** if the container reports `total_frames <= 0`, the bounds
  that need it (start-past-end check, end clamp, `frame_count`) are skipped and
  seeking is trusted.
- **`frame_count`:** how many frames the window yields, fed to
  `VideoMetadata.total_frames` so progress reporting keeps an accurate
  denominator. `None` when the total is unknown.

Pure so the arithmetic is unit-testable without decoding a real video; the
adapter handles the cv2 glue (read fps/total, seek, stop).

---

## Timestamps Stay Absolute

Frame indices remain **absolute** (real frame numbers), so a window starting at
0:10 emits TIMESTEP 10.0, not 0.0. Rationale: it preserves provenance (where the
clip sits in the source) and is reversible; velocity/acceleration are unaffected
because they depend only on timestamp *deltas*. Re-zeroing to 0 was the
considered alternative and rejected as lossy.

Caveat: keyframe-only codecs may seek to a nearby keyframe rather than the exact
frame. We still *label* frames from the requested start index, so timestamps stay
on the clip clock even if the first decoded pixels are approximate.

---

## CLI Surface

- `--start` / `--end` accept a **timecode**: `HH:MM:SS(.ms)`, `MM:SS(.ms)`, or
  `SS(.ms)` (e.g. `1:30`, `0:01:05.250`, `12.5`). Parsed by `_parse_timecode`
  in `cli.py` into seconds; malformed input raises `typer.BadParameter`.
- Parsing and the `end > start` check run first (fail fast, before the video
  opens). A start past the video duration raises during source open with a
  message naming the duration.

---

## Layering

- `infrastructure/video/window.py` — `FrameWindow` (pure index math).
- `infrastructure/video/opencv.py` — `OpenCvVideoSource` reads fps/total, builds
  the window, seeks, and stops; reports the windowed `total_frames`.
- `cli.py` — `_parse_timecode` + `--start` / `--end`, wired into the source.
