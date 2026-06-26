# Persisted Run Config (Zero Hardcoded Defaults)

---

## What This Is

A TraTrac run is fully described by a **`RunConfig`** — input video, detector,
calibration, tracker, orientation, export, analysis window, and run options.
There are **no built-in defaults anywhere in the package**: every value must be
supplied by the TOML config file (`--config PATH`). A missing value fails the run,
listing every absent/invalid key at once.

The config is the **single source of truth**: the CLI exposes **no per-key override
flags** — the run is driven entirely by `--config`. The lone operational flag is
`--force` (`run.force`), kept because overwriting outputs is an ad-hoc decision you
shouldn't have to edit the file for. (Earlier revisions mirrored every config key as
a `--flag` override; those were removed — see "Design history" below.)

The config is a **persisted, replayable run spec**: it names its own input video
(`input.video`) and output (`export.out`), so a saved config reproduces a run with
just `--config`:

```
uv run tratrac --config run.toml            # the whole run, from the file
uv run tratrac --config run.toml --force    # same, overwriting existing outputs
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

- **Precedence per key:** config file value > **error**. (The resolver still accepts
  an `overrides` dict and applies it with highest precedence — it is now fed only
  `run.force` from `--force`; the mechanism is retained so a future flag, or a
  programmatic caller, can override a key without reworking resolution.)
- **`None` is the unset sentinel.** An override of `None` means "not supplied" and
  falls through to the file value; `--force`/`--no-force` defaults to `None` in the
  Typer signature so an omitted flag leaves `run.force` to the config.
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
- **`cli.py`**: loads the TOML, assembles a one-entry override map (`run.force` from
  `--force`), calls `RunConfig.resolve`, validates the resolved video on disk, then
  builds the adapters from the typed config. Translates `ConfigError` into exit code 2.

The CLI is a single-command Typer app, invoked as `tratrac --config …` (no `process`
subcommand). Its only options are `--config` and `--force`.

### Design history — why the override flags were removed

The first revision mirrored every config key as a `--flag` (so `--conf 0.4` could tweak
one value without editing the TOML) and allowed a positional `VIDEO` to override
`input.video`. That was removed: the per-key flags duplicated the config surface (every
key needed a flag + a wiring line + help text, and two ways to set one value), and they
weakened the reproducibility argument the config exists to make — a run set partly by
flags is no longer fully captured by its file. Collapsing to **config-only** makes the
file the single, complete, replayable spec; `--force` survives as the one genuinely
ad-hoc operational toggle. The `RunConfig.resolve(file_values, overrides)` signature is
unchanged, so the override path remains available (now exercised only by `--force`).

---

## Consequence: No Library Pixel Fallback

Removing defaults is **package-wide**, not CLI-only. The adapter constructors
(`SsamTrjExporter`, `RtDetrDetector`, `YoloV8VisDroneDetector`,
`BoxmotBotSortTracker`) no longer carry defaults — the
old `scale=1.0` / `meters_per_pixel=1.0` "pixels-as-metres" library escape hatch is
gone. Callers (CLI and tests) pass every value explicitly; tests that want MVP1
pixel behaviour pass `scale=1.0` / `meters_per_pixel=1.0` themselves.

`detector.filename` is required even for `rt_detr` (which ignores it) — a
deliberate consequence of "every key mandatory"; conditional-requiredness was not
worth the special case.
