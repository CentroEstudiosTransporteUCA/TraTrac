# Persisted Run Config (Zero Hardcoded Defaults)

---

## What This Is

A TraTrac run is fully described by a **`RunConfig`** — input video, detector,
calibration, tracker, orientation, export, analysis window, and run options.
There are **no built-in defaults anywhere in the package**: every value must be
supplied by a TOML config file (`--config PATH`) or a CLI flag. A missing value
fails the run, listing every absent/invalid key at once.

The config is a **persisted, replayable run spec**: it names its own input video
(`input.video`) and output (`export.out`), so a saved config reproduces a run with
no other arguments:

```
uv run tratrac --config run.toml          # replay; everything from the file
uv run tratrac VIDEO --config run.toml     # positional VIDEO overrides input.video
uv run tratrac VIDEO --config run.toml --conf 0.4   # any flag overrides its key
```

---

## Why (the trade)

This extends the calibration philosophy already in MVP1.75 — the CLI refuses to
run uncalibrated rather than emit physically meaningless metric values
(`05_5_mvp1_75.md`) — to *every* parameter. The trade is **typing convenience for
scientific reproducibility**: a `.trj` is reconstructable from the config that
produced it. A silent default (e.g. an unstated `conf=0.25` or `device=cpu`) is
exactly the kind of hidden variable that makes a result hard to reproduce or
defend, so none survive.

---

## Resolution Model

- **Precedence per key:** CLI flag (if passed) > config file value > **error**.
- **`None` is the unset sentinel.** Every value flag defaults to `None` in the
  Typer signature, meaning "not passed on the command line" — a detection
  mechanism, not a behavioural default. Resolution falls through `None` to the
  file value, else records the key as missing.
- **Aggregated failure.** `RunConfig.resolve` collects *all* problems and raises a
  single `ConfigError`, so one run surfaces every missing/invalid key instead of
  one-per-attempt.

---

## "Everything Mandatory" vs. Presence-Toggles

A literally-mandatory `timing_csv` is incoherent — it would force profiling on
every run. The rule that resolves this, while keeping "no hidden default":

> Every key must be **present**, but its *disabled* state is an explicit value the
> operator writes.

- `timing_csv = ""` — profiling off; a path turns it on.
- `video_out = ""` — no overlay video; a path turns it on.
- `transform_csv = ""` — no per-frame transform sidecar; a path turns it on.
- `force = false` — prompt on overwrite; `true` overwrites silently.
- `start = "" / end = ""` — the clip's natural bounds; else a timecode.
- `timestep_precision = 0.0` — emit every frame; else a minimum interval.

Absence of a key is an error; an explicit "off" value is legal. The config is thus
a complete, self-documenting declaration of the run with zero silent behaviour.

A toggleable key can still be **conditionally incoherent**: `export.transform_csv`
(the per-frame ego-motion transform sidecar, `05_75_mvp1_9.md`) only makes sense
when `ego_motion.enabled` is true — with stabilization off every transform is the
identity. Setting it while stabilization is off is therefore an aggregated
`ConfigError`, mirroring the "specify exactly one calibration method" guard: a
present-but-contradictory value is rejected, not silently ignored.

---

## Calibration Is a One-Of (Not "Both Mandatory")

Calibration cannot make every key mandatory because the methods are mutually
exclusive. Exactly one must be fully specified:

- `meters_per_pixel` (direct GSD), **or**
- `drone_model` + one altitude source (`altitude_m > 0` *or* an `srt` path).

Specifying **both** `meters_per_pixel` and `drone_model` is now an **error** — MVP1.75
silently prioritised `meters_per_pixel`; explicitness replaces silent priority.
There is **no `.SRT` sidecar auto-discovery** (it was a hidden default): the SRT
path must be given explicitly.

---

## Where It Lives (layering)

- **`application/config.py`** (pure, no I/O): the `RunConfig` value object and its
  section dataclasses (`InputConfig`, `DetectorConfig`, `CalibrationConfig`, …),
  the `_Resolver` (merge + type-check + collect problems), `ConfigError`, and the
  moved validators (device format, timecode parse, drone-model-known, window
  ordering). `DetectorChoice` moved here (single source of truth). `CalibrationConfig`
  keeps `resolve_scale(metadata)` — behaviour with its data — reusing
  `calibration/gsd.py`, `srt_parser.py`, `drone_specs.py`.
- **`infrastructure/config/toml.py`**: `load_toml(path)` via stdlib `tomllib`
  (Python 3.12, no new dependency) — the lone seam where the dynamically-typed TOML
  document enters; the resolver does the type checking.
- **`cli.py`**: parses flags as `None` sentinels, assembles the dotted-key override
  map, loads the TOML, calls `RunConfig.resolve`, validates the resolved video on
  disk, then builds the adapters from the typed config. Translates `ConfigError`
  into exit code 2.

The CLI is a single-command Typer app, so it is invoked as `tratrac VIDEO …` /
`tratrac --config …` (no `process` subcommand).

---

## Consequence: No Library Pixel Fallback

Removing defaults is **package-wide**, not CLI-only. The adapter constructors
(`EmaOrientationEstimator`, `SsamTrjExporter`, `RtDetrDetector`,
`YoloV8VisDroneDetector`, `BoxmotBotSortTracker`) no longer carry defaults — the
old `scale=1.0` / `meters_per_pixel=1.0` "pixels-as-metres" library escape hatch is
gone. Callers (CLI and tests) pass every value explicitly; tests that want MVP1
pixel behaviour pass `scale=1.0` / `meters_per_pixel=1.0` themselves.

`detector.filename` is required even for `rt_detr` (which ignores it) — a
deliberate consequence of "every key mandatory"; conditional-requiredness was not
worth the special case.
