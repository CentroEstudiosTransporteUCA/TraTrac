# Writing a TraTrac run config (`.toml`)

A `tratrac` run is fully described by a **persisted run config**: a single TOML
file naming its input video, output path, and every processing parameter for
the perception pass. There are **no built-in defaults anywhere in the
package** — every key below is mandatory. A missing or invalid value aborts
the run (exit code 2) and lists *every* offending key at once.

This is a deliberate trade of typing convenience for scientific
reproducibility: a run is reconstructable from the config that produced it.
For the design rationale see `src/tratrac/application/CONFIG_DESIGN.md`; for a copyable starting
point see `tratrac.example.toml`.

```bash
uv run tratrac --config run.toml            # the only way to run it
uv run tratrac --config run.toml --check     # validate without running
uv run tratrac --config run.toml --force     # overwrite existing outputs
```

**There are no per-key override flags and no positional `VIDEO` argument** —
`--config` is the only way values reach the run. `--force`/`--no-force` (overwrite
control) and `--check`/`--json` (validate-only) are the only other flags, and
neither is a config key: overwrite policy never affects the trajectories, and
`--check` just short-circuits before the video is ever opened. See
`src/tratrac/CHECK_COMMAND.md`.

**This config covers `tratrac` only** — the perception pass. GSD calibration,
ego-motion stabilization, and world-projection homography are no longer `tratrac`
config keys at all: they're resolved upstream by the separate `tratrac-preprocess`
tool and handed to `tratrac` as one path, `input.transforms_in`. See
"What isn't here anymore" below.

---

## Two rules that govern the whole file

1. **Every key must be present.** Absence is an error. There is no key whose
   omission means "use a sensible default" — sensible defaults do not exist here.
2. **"Disabled" is an explicit value, never a missing key.** A feature you don't
   want is still written, set to its off value:

   | Off value | Meaning |
   | --- | --- |
   | `process_fps = 0.0` | process every frame (no decode-time decimation) |
   | `timing_csv = ""` | profiling off |
   | `start = ""` / `end = ""` | the clip's natural bounds (no trimming) |
   | `force = false` (CLI flag, not a config key) | prompt before overwriting an existing output |

---

## The sections

### `[input]`

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `video` | path string | Must be a non-empty path that resolves to an existing file. |
| `process_fps` | number | Decode-time decimation cap, `>= 0.0`. `0.0` = process every frame. See `src/tratrac/infrastructure/TIMESTEP_PRECISION.md`. |
| `transforms_in` | path string | A `tratrac-preprocess estimate` run's transforms file. Always required — `tratrac` never estimates ego-motion or resolves GSD scale itself, it only reads this file's rows. A static-camera run still needs it, for its scale rows. Must exist on disk (checked before the video opens). See below. |

```toml
[input]
video        = "clips/highway_run3.mp4"
process_fps  = 0.0
transforms_in = "out/highway_run3_transforms.jsonl"
```

### `[detector]`

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `name` | string | `yolov8_visdrone` (current default), `rt_detr` (dormant), or `yolo_obb` (not yet the default — see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`). |
| `checkpoint` | string | HuggingFace repo id, e.g. `Mahadih534/YoloV8-VisDrone`. |
| `conf` | number | Detection confidence threshold, in `[0.0, 1.0]`. |
| `filename` | string | Weights file inside the repo. Consumed only by `yolov8_visdrone`, but **required for every detector** (zero-defaults rule) — still an open question whether to drop or repurpose it for the others, see `DETECTOR_CHOICE.md` "Open questions". |

```toml
[detector]
name       = "yolov8_visdrone"
checkpoint = "Mahadih534/YoloV8-VisDrone"
conf       = 0.25
filename   = "visDrone.pt"
```

### `[runtime]`

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `device` | string | torch device. Must match `cpu`, `mps`, `cuda`, or `cuda:N` (e.g. `cuda:0`). |

```toml
[runtime]
device = "cpu"
```

### `[tracker]`

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `det_thresh` | number | BoT-SORT detection threshold, in `[0.0, 1.0]`. Convention: keep it below `detector.conf`. |

```toml
[tracker]
det_thresh = 0.1
```

### `[export]`

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `out` | path string | Output **track record** path (Parquet) — `tratrac`'s only output. Not a `.trj`: run `tratrac-postprocess` on this file to get one. Parent dirs are created. |

```toml
[export]
out = "out/highway_run3.parquet"
```

### `[window]` — analysis trimming

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `start` | string | `""` = clip start, else a timecode. |
| `end` | string | `""` = clip end, else a timecode. Must be `> 0` and after `start`. |

Timecode formats: `SS(.ms)`, `MM:SS(.ms)`, or `HH:MM:SS(.ms)` — e.g. `12.5`,
`1:30`, `00:01:30.250`.

```toml
[window]
start = ""
end   = ""
```

### `[run]` — run options

| Key | Type | Notes / valid values |
| --- | --- | --- |
| `timing_csv` | string | `""` = profiling off; else a CSV path for per-frame step timings. Must differ from `export.out`. See `src/tratrac/infrastructure/timing/STEP_TIMING.md`. |

`force` is **not** a config key — it's the CLI-only `--force`/`--no-force` flag
(overwrite control is pure I/O, never affects the trajectories, so it's excluded
from the reproducible run spec on purpose). See `src/tratrac/application/CONFIG_DESIGN.md`.

```toml
[run]
timing_csv = ""
```

---

## What isn't here anymore

Earlier revisions of this config had `[calibration]` (GSD scale), `[ego_motion]`
(ORB stabilization toggle + tuning), and `[orientation]` (a live heading-smoothing
window) sections. All three are gone from `tratrac`'s config — not renamed,
removed:

- **GSD calibration and ego-motion** are resolved once, upstream, by
  `tratrac-preprocess estimate` (its own CLI flags — `--meters-per-pixel` or
  `--drone-model`+`--altitude-m`/`--srt` for scale; ORB tuning knobs live only
  here too). The result is written into one shared transforms file, which
  `input.transforms_in` above just names. There is no `[ego_motion]` section or
  `enabled` toggle in `tratrac` to gate on — whether stabilization applies is a
  property of that file's *content* (whether it has similarity rows), not a
  flag. See `src/tratrac/infrastructure/video/EGO_MOTION.md`,
  `src/tratrac/calibration/GSD_CALIBRATION.md`, and
  `src/tratrac/infrastructure/transform/TRANSFORM_SINK.md`.
- **World-projection homography** is fitted by the same tool's `project`
  subcommand (`tratrac-preprocess project --transforms ... --calibration
  calibration.json`), from operator-authored image↔world correspondences, into
  that same transforms file. `tratrac` never fits or applies one — that's
  `tratrac-postprocess`'s job, via its own `--transforms` flag. See
  `src/tratrac/application/WORLD_PROJECTION.md`.
- **Orientation** is no longer computed live. `tratrac` writes only raw tracked
  positions; heading, speed, and acceleration are reconstructed entirely
  offline by `tratrac-postprocess`'s Kalman/RTS smoother, tuned via its own
  `--pos-noise`/`--jerk` flags, not a config section. See
  `src/tratrac/application/SMOOTHING.md`.

See `CLAUDE.md`'s Commands table for every flag on `tratrac-preprocess`,
`tratrac-postprocess`, and `tratrac-render`.

---

## Complete example

```toml
[input]
video         = "clips/highway_run3.mp4"
process_fps   = 0.0
transforms_in = "out/highway_run3_transforms.jsonl"

[detector]
name       = "yolov8_visdrone"
checkpoint = "Mahadih534/YoloV8-VisDrone"
conf       = 0.25
filename   = "visDrone.pt"

[runtime]
device = "cpu"

[tracker]
det_thresh = 0.1

[export]
out = "out/highway_run3.parquet"

[window]
start = ""
end   = ""

[run]
timing_csv = ""
```

---

## What a failed run looks like

A run with missing or invalid keys exits with code 2 and lists everything wrong
in one message — fix them all in one pass:

```
ERROR: invalid run configuration; supply each value via the --config TOML
or its flag:
  - input.video is missing.
  - input.transforms_in does not exist or is not a file -- run tratrac-preprocess first.
  - detector.conf must be in [0.0, 1.0], got 1.5.
  - runtime.device 'gpu' is invalid; expected cpu, mps, or cuda[:N] (e.g. cuda:0).
```
