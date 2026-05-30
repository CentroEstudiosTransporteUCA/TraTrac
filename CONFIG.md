# Writing a TraTrac run config (`.toml`)

A TraTrac run is fully described by a **persisted run config**: a single TOML
file that names its own input video, output path, and every processing
parameter. There are **no built-in defaults anywhere in the package** — every
key below is mandatory. A missing or invalid value aborts the run (exit code 2)
and lists *every* offending key at once.

This is a deliberate trade of typing convenience for scientific
reproducibility: a `.trj` is reconstructable from the config that produced it.
For the design rationale see `vault/19_config_file.md`; for a copyable starting
point see `tratrac.example.toml`.

```bash
uv run tratrac --config run.toml                  # replay; everything from the file
uv run tratrac VIDEO --config run.toml            # positional VIDEO overrides input.video
uv run tratrac VIDEO --config run.toml --conf 0.4 # any flag overrides its config key
```

---

## Two rules that govern the whole file

1. **Every key must be present.** Absence is an error. There is no key whose
   omission means "use a sensible default" — sensible defaults do not exist here.
2. **"Disabled" is an explicit value, never a missing key.** A feature you don't
   want is still written, set to its off value:

   | Off value | Meaning |
   | --- | --- |
   | `timing_csv = ""` | profiling off |
   | `force = false` | prompt before overwriting an existing output |
   | `start = ""` / `end = ""` | the clip's natural bounds (no trimming) |
   | `timestep_precision = 0.0` | emit one TIMESTEP per processed frame |

---

## Resolution & precedence

Each key is resolved independently as: **CLI flag (if passed) → config file
value → error**. A flag overrides only the one key it maps to; everything else
still comes from the file. This lets a stable per-shoot config live on disk while
the command line carries only what changes between runs.

---

## The sections

### `[input]`

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `video` | path string | yes | Must be a non-empty path that resolves to an existing file. | positional `VIDEO` |

```toml
[input]
video = "clips/highway_run3.mp4"
```

### `[detector]`

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `name` | string | yes | `yolov8_visdrone` (MVP1 default) or `rt_detr`. | `--detector` |
| `checkpoint` | string | yes | HuggingFace repo id. e.g. `Mahadih534/YoloV8-VisDrone`, or `PekingU/rtdetr_r18vd` for RT-DETR. | `--checkpoint` |
| `conf` | number | yes | Detection confidence threshold, in `[0.0, 1.0]`. | `--conf` |
| `filename` | string | yes | Weights file inside the repo (e.g. `visDrone.pt`). Consumed only by `yolov8_visdrone`, but **required even for `rt_detr`** (which ignores it) — a deliberate consequence of "every key mandatory". | `--checkpoint-file` |

```toml
[detector]
name       = "yolov8_visdrone"
checkpoint = "Mahadih534/YoloV8-VisDrone"
conf       = 0.25
filename   = "visDrone.pt"
```

### `[runtime]`

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `device` | string | yes | torch device. Must match `cpu`, `mps`, `cuda`, or `cuda:N` (e.g. `cuda:0`). | `--device` |

```toml
[runtime]
device = "cpu"
```

### `[calibration]` — a *one-of*, not "all mandatory"

Calibration methods are mutually exclusive, so exactly **one** must be fully
specified. Specifying both `meters_per_pixel` and `drone_model` is an **error**
(no silent priority). There is **no `.SRT` auto-discovery** — give the path
explicitly.

**Method A — direct GSD:**

| Key | Type | Notes | Override flag |
| --- | --- | --- | --- |
| `meters_per_pixel` | number | Ground sample distance, must be `> 0`. | `--meters-per-pixel` |

```toml
[calibration]
meters_per_pixel = 0.05
```

**Method B — drone geometry** (`drone_model` + one altitude source):

| Key | Type | Notes | Override flag |
| --- | --- | --- | --- |
| `drone_model` | string | One of the registered models (below). Case-insensitive. | `--drone-model` |
| `altitude_m` | number | Flight altitude AGL in metres, `> 0`. **One of** this or `srt`. | `--altitude` |
| `srt` | path string | DJI `.SRT` sidecar with per-frame altitude. **One of** this or `altitude_m`. | `--srt` |

Registered `drone_model` keys: `air_2s`, `mavic_2_pro`, `mavic_3`,
`mini_3_pro`, `mini_4_pro` (add more in `src/tratrac/calibration/drone_specs.py`).

```toml
[calibration]
drone_model = "mavic_3"
altitude_m  = 80.0
# or, instead of altitude_m:
# srt = "clips/highway_run3.SRT"
```

> When using method B, leave `meters_per_pixel` out of the section entirely
> (writing both is the both-methods error).

### `[ego_motion]` — ORB ego-motion (toggle + conditional params)

ORB ego-motion (MVP1.9, see `vault/05_75_mvp1_9.md`). When on, detection and tracking
run on the **raw, full-resolution frame** and the keyframe transform is applied to the
**detections** (coordinates, not pixels), so trajectories are free of drone ego-motion
and nothing is ever cropped to black. The keyframe anchor re-sets once too little of it
stays in view (`min_anchor_overlap`). `enabled` is **always required** (off is explicit);
the other parameters are required **only when `enabled = true`** and may be omitted when
it is `false`.

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `enabled` | boolean | yes | `true` turns stabilization on; `false` leaves coordinates untouched. | `--stabilize` / `--no-stabilize` |
| `n_features` | integer | when enabled | ORB keypoints per frame, `> 0` (e.g. `2000`). | `--orb-features` |
| `match_ratio` | number | when enabled | Lowe ratio test, in `(0, 1)`; lower = stricter matches (e.g. `0.75`). | `--orb-match-ratio` |
| `min_matches` | integer | when enabled | Min good matches to fit a transform, `>= 2`; below it the step is treated as no motion (e.g. `10`). | `--orb-min-matches` |
| `ransac_threshold` | number | when enabled | RANSAC reprojection threshold in pixels, `> 0` (e.g. `3.0`). | `--orb-ransac-threshold` |
| `min_anchor_overlap` | number | when enabled | Re-anchor the keyframe when its shared visible area drops below this fraction, in `(0, 1)` (e.g. `0.6`). | `--min-anchor-overlap` |

```toml
[ego_motion]
enabled            = true
n_features         = 2000
match_ratio        = 0.75
min_matches        = 10
ransac_threshold   = 3.0
min_anchor_overlap = 0.6
```

> To disable, just `enabled = false` — the other keys can be left out entirely.

### `[tracker]`

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `det_thresh` | number | yes | BoT-SORT detection threshold, in `[0.0, 1.0]`. Convention: keep it below `detector.conf`. | `--det-thresh` |

```toml
[tracker]
det_thresh = 0.1
```

### `[orientation]`

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `smoothing_window` | integer | yes | EMA heading window. Must be `>= 2`. | `--smoothing-window` |

```toml
[orientation]
smoothing_window = 5
```

### `[export]`

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `out` | path string | yes | Output `.trj` path (non-empty). Parent dirs are created. | `--out` / `-o` |
| `timestep_precision` | number | yes | Minimum seconds between exported TIMESTEPs. `0.0` = every frame. Values `> 0.5` produce a coarseness warning (still valid, but sparse for SSAM conflict analysis). | `--timestep-precision` |
| `video_out` | string | yes | `""` = no overlay video; else an `.mp4` path. Writes each **raw** frame with bumpers/IDs/trails drawn (trajectories are mapped back onto the raw frame when `ego_motion.enabled`). Only the `.trj` leg is decimated by `timestep_precision`; the video keeps every frame. Must differ from `out` and `run.timing_csv`. | `--video-out` |
| `video_trail` | integer | yes (when `video_out` set) | Trail length in frames for the overlay; `0` = whole path, `N` = rolling window of `N`. Only read when `video_out` is on. | `--video-trail` |

```toml
[export]
out                = "out/highway_run3.trj"
timestep_precision = 0.0
video_out          = ""          # "" = off; else "out/highway_run3_overlay.mp4"
video_trail        = 0           # 0 = whole path (only used when video_out is set)
```

### `[window]` — analysis trimming

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `start` | string | yes | `""` = clip start, else a timecode. | `--start` |
| `end` | string | yes | `""` = clip end, else a timecode. Must be `> 0` and after `start`. | `--end` |

Timecode formats: `SS(.ms)`, `MM:SS(.ms)`, or `HH:MM:SS(.ms)` — e.g. `12.5`,
`1:30`, `00:01:30.250`.

```toml
[window]
start = ""
end   = ""
```

### `[run]` — run options

| Key | Type | Required | Notes / valid values | Override flag |
| --- | --- | --- | --- | --- |
| `force` | boolean | yes | `true` overwrites existing outputs silently; `false` prompts (and errors in a non-TTY run). | `--force` / `--no-force` |
| `timing_csv` | string | yes | `""` = profiling off; else a CSV path for per-frame step timings. Must differ from `export.out`. | `--timing-csv` |

```toml
[run]
force      = false
timing_csv = ""
```

---

## Complete example

```toml
[input]
video = "clips/highway_run3.mp4"

[detector]
name       = "yolov8_visdrone"
checkpoint = "Mahadih534/YoloV8-VisDrone"
conf       = 0.25
filename   = "visDrone.pt"

[runtime]
device = "cpu"

[calibration]
drone_model = "mavic_3"
altitude_m  = 80.0

[ego_motion]
enabled            = true
n_features         = 2000
match_ratio        = 0.75
min_matches        = 10
ransac_threshold   = 3.0
min_anchor_overlap = 0.6

[tracker]
det_thresh = 0.1

[orientation]
smoothing_window = 5

[export]
out                = "out/highway_run3.trj"
timestep_precision = 0.0
video_out          = "out/highway_run3_overlay.mp4"
video_trail        = 0

[window]
start = ""
end   = ""

[run]
force      = false
timing_csv = ""
```

---

## What a failed run looks like

A run with missing or invalid keys exits with code 2 and lists everything wrong
in one message — fix them all in one pass:

```
ERROR: invalid run configuration; supply each value via the --config TOML or its flag:
  - input.video is missing.
  - detector.conf must be in [0.0, 1.0], got 1.5.
  - runtime.device 'gpu' is invalid; expected cpu, mps, or cuda[:N] (e.g. cuda:0).
  - calibration: specify exactly one of meters_per_pixel or drone_model, not both.
  - orientation.smoothing_window must be >= 2.
```
