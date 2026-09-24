# FiftyOne dataset export (`tratrac-fiftyone`)

Builds a [FiftyOne](https://voxel51.com/fiftyone) video dataset from TraTrac's existing outputs
for interactive visual QA — not a new export format, a reader over the track record.

## What it builds

- One FiftyOne video **sample** per source clip.
- **Frame-level detections** from the track record (`infrastructure/tracks/parquet.py`,
  `--record`) and/or the smoothed record (`infrastructure/tracks/smoothed_parquet.py`,
  `--smoothed-record`) — passing both adds two label fields per frame
  (`record_detections`/`smoothed_detections`) so raw and smoothed trajectories can be compared
  directly in the FiftyOne App.
- Each vehicle's FiftyOne `Detection.index` carries its track/vehicle id (FiftyOne's own field
  for video-tracking visualization).

Track-level fields (link/lane id, ReID merge provenance) as label attributes, beyond the
per-frame detections above, are not yet added.

## Deliberately reads `--smoothed-record`, not the SSAM `.trj`

An earlier version read `--trj` directly and normalized its coordinates as raw video pixels —
true only for an uncalibrated `.trj`; a `--calibration`'d one carries real-world metres from a
homography the `.trj` file itself doesn't store, and dividing metres by pixel width/height
silently produced boxes clustered near the origin. This was caught by looking at the rendered
output in the FiftyOne App, not by inspection — every box pinned near the top-left corner
regardless of the vehicle's true position. The fix wasn't a defensive check on `.trj`; it was
building a proper pixel-space output (`tratrac-postprocess --smoothed-record`, see
`application/SMOOTHING.md`'s "Dual-space export" section) from the *same* smoothing pass that
feeds `.trj`, so this tool always has something genuinely pixel-space to read regardless of
calibration.

A straightforward *reader*, not a pipeline change — the same "post-hoc tool over existing
outputs" shape as `scripts/plot_run.py` and `scripts/validate_trj.py`.

## Optional extra, not a core dependency

`fiftyone` (`uv sync --extra fiftyone`) is opt-in. It and its own dependency `voxel51-eta`
hard-require the `opencv-python-headless` distribution, which installs files under the same
path as this project's `opencv-python` (the GUI build, needed for
`scripts/visualize_stabilization.py`'s `cv2.imshow`) — a real, confirmed conflict (installing
both silently corrupted the `cv2` install). `[tool.uv] override-dependencies` excludes
`opencv-python-headless` from resolution entirely, so `cv2` stays resolved to `opencv-python`
even with the extra installed; `fiftyone` still imports fine since Python's `import cv2` doesn't
care which distribution provided it.

`fiftyone` needs a MongoDB to back its dataset store, and its bundled `mongod` doesn't exist on
Linux: `fiftyone-db` (the package that ships it) stopped publishing Linux wheels after version
0.4.5 — every version since ships macOS/Windows binaries only. This is true on any Linux distro.
`fiftyone.core.service.DatabaseService.find_mongod()` never falls back to a system `mongod` on
`PATH` either, so the only way in on Linux is `FIFTYONE_DATABASE_URI` (an env var), which makes
`fiftyone` connect directly to a `mongod` you run yourself instead of looking for the bundled
binary — see `fiftyone.core.odm.database.establish_db_conn`. `run_fiftyone.sh` (repo root
`scripts/`) bootstraps one via MongoDB's official static Linux binary, run through the same
raw-ELF-loader workaround this project's NixOS sessions use for other generic-glibc binaries.
Off Linux, `fiftyone`'s bundled `mongod` should just work with zero setup.

## Verification

The conversion logic (`record_frame_detections`, `smoothed_frame_detections` — pure, no
`fiftyone` import) is unit-tested; `_build_dataset` (the part that calls the `fiftyone` SDK) was
run for real against `cruce.mp4`/`out/cruce.parquet`/`out/cruce_smoothed.parquet`, producing a
persisted FiftyOne dataset with 27,319 frames and both `record_detections`/`smoothed_detections`
label layers populated correctly (spot-checked mid-clip: 16 vehicles in each layer at a
representative frame, correct normalized boxes spread across real vehicle positions, track ids
landing in `Detection.index` as designed).

## Components

- `cli_fiftyone.py` — `tratrac-fiftyone` Typer entry point.
- `scripts/run_fiftyone.sh` — one-command launcher for the bare FiftyOne App: bootstraps MongoDB
  if not already running, then launches the app with no dataset preselected. Idempotent; doesn't
  build a dataset itself.
