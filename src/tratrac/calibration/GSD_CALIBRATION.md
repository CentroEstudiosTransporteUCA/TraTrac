# GSD Calibration: Metric Sizes and Speeds from Drone Metadata (MVP1.75)

Delivered before MVP1.5 as an independent shortcut — the MVP number is a capability ID, not
execution order.

---

## Goal

Produce SSAM `.trj` files in which:

- `Length`, `Width`, `Speed`, `Acceleration` are in real metres / m·s⁻¹ / m·s⁻²
- `DIMENSIONS.Scale` carries the metres-per-pixel calibration so any SSAM
  reader recovers real metric coordinates from the stored grid values

…**without** introducing stabilisation, homography, or multi-plane geometry
(those remain MVP2 / MVP3 work).

This is the cheapest possible deliverable that makes the *physical
analysis* of a `.trj` valid. It unblocks downstream traffic-safety
analytics (TTC, PET, real dimensions) for any video where the camera is
a **hovering, nadir-pointing drone** with known specs.

---

## Why this MVP exists

MVP1 emits `Length = 60` "metres" for a 60-pixel bounding box — parseable
SSAM, physically meaningless. MVP2 fixes this as a side effect of
single-homography world projection, but MVP2 is a large piece of work
(SuperPoint + LightGlue stabilisation, homography estimation, multi-frame
calibration). For drone footage the metric conversion is computable from
**metadata alone** — no vision algorithms required.

Splitting this out:

- delivers physically valid sizes and speeds in days, not weeks
- de-risks MVP2 (stabilisation can be debugged in isolation without the
  calibration concern)
- means the first `.trj` files we hand to analysts are scientifically
  meaningful

---

## Technologies

| Component | Technology |
| --- | --- |
| Calibration source | Ground Sample Distance formula |
| Sensor / focal lookup | Per-drone-model config file |
| Altitude source | DJI `.SRT` sidecar parser (and per-drone equivalents) |

No new ML, no new heavy deps.

---

## The formula

```text
GSD (metres/pixel) = (sensor_width_mm × altitude_m) / (focal_length_mm × image_width_pixels)
```

Where:

| Input | Source |
| --- | --- |
| `sensor_width_mm` | Drone spec sheet (e.g. DJI Mavic 3: 17.3 mm 4/3 sensor) |
| `focal_length_mm` | Drone spec sheet — **real focal length, not 35 mm-equivalent** |
| `image_width_pixels` | Read from the video file at runtime |
| `altitude_m` (AGL) | Drone telemetry — DJI ships per-frame altitude in the `.SRT` sidecar; other manufacturers have equivalents |

For hovering drones the altitude is constant per clip; for moving drones
it varies per frame (see Limitations).

---

## Pipeline

GSD calibration is not a step inside the perception run — it's resolved by a separate, earlier
tool, `tratrac-preprocess estimate`, and handed to every downstream step as a row in the shared
transforms file:

```text
Video ─────┐
           ↓
       Drone-Model Config ─→ GSD Calibration ──→ scale row in transforms.jsonl
           ↓                       ↑                         │
       Altitude Source ────────────┘                         │
                                                               ↓
                                          tratrac: YOLOv8-VisDrone → BoT-SORT → track record
                                                               ↓
                                    tratrac-postprocess: Kalman/RTS smoother (unit-aware,
                                    reads the scale row) → SSAM .trj Export (Scale = GSD)
```

The difference from MVP1: the transforms file carries a calibration the offline smoother and
exporter (both in `tratrac-postprocess`) use to produce real units, instead of the `Scale = 1.0`
placeholder MVP1 wrote.

---

## Implementation contract

### New configuration

- Per-drone-model registry — JSON or TOML keyed by drone identifier,
  values are `{ sensor_width_mm, focal_length_mm }`. Initial entries can
  cover whichever drones we actually use; growing the registry is data,
  not code.
- Per-clip altitude — read from sidecar file (default DJI `.SRT`) or
  passed as a CLI flag for clips without telemetry.

### CLI surface

GSD calibration is not part of `tratrac`'s config at all — it's resolved once, upstream, by
`tratrac-preprocess estimate`, and lives only as `scale` rows in the shared transforms file that
`tratrac`'s `input.transforms_in` names. Exactly one calibration method — `--meters-per-pixel`,
or `--drone-model` + an altitude source (`--altitude-m` or `--srt`):

```bash
# Drone metadata via an SRT sidecar (explicit path; no auto-discovery)
uv run tratrac-preprocess estimate VIDEO --out transforms.jsonl --drone-model mavic_3 --srt VIDEO.SRT

# Direct GSD override
uv run tratrac-preprocess estimate VIDEO --out transforms.jsonl --meters-per-pixel 0.05
```

See `application/config.py`'s module docstring, "What isn't here" section, for why this moved out of `tratrac`'s TOML
config, and `src/tratrac/infrastructure/transform/TRANSFORM_SINK.md` for the shared transforms
file this writes into.

### Domain changes

- `VehicleState.dimensions`, `velocity`, `acceleration` are populated in
  **real units** (metres, m/s, m/s²) from this MVP onward. `tratrac-postprocess`'s
  Kalman/RTS smoother (`application/track_smoothing.py`'s `build_state`) multiplies
  pixel-derived values by the transforms file's `scale` row before packing them into the state
  — this now happens entirely offline, not during the perception run.
- `VehicleState.centroid` stays in calibrated grid units (pixels). The
  exporter passes it through unchanged into the SSAM record's X / Y
  fields; SSAM readers recover real metres via `× Scale`.

### Exporter changes

- `SsamTrjExporter` (in `tratrac-postprocess`, the only `.trj` path) takes a required `scale`
  constructor argument; the transforms file's resolved `meters_per_pixel` supplies it. (The old
  `scale=1.0` default was removed by the zero-defaults refactor — see
  `application/config.py`'s module docstring.)
- No struct changes; no on-disk format changes.

### Worked example

DJI Mavic 3, hovering at 50 m AGL, recording at 1920 × 1080:

- `GSD = (17.3 × 50) / (12.29 × 1920) ≈ 0.0367 m/pixel`
- A car with bbox 80 × 35 pixels: `Length ≈ 2.94 m`, `Width ≈ 1.28 m`
- A car moving 250 pixels/sec: `Speed ≈ 9.17 m/s ≈ 33 km/h`
- `DIMENSIONS.Scale = 0.0367` written to the file
- Centroid at pixel `(960, 540)` stored as-is; SSAM reader recovers
  `(35.23 m, 19.82 m)`

All physically meaningful with zero homography work.

---

## What this delivers and what it doesn't

Real metric `Length`, `Width`, `Speed`, `Acceleration` in `.trj`, a valid `DIMENSIONS.Scale` so
SSAM coordinate recovery works, and scientifically valid TTC/PET conflict metrics — but only for
a hovering nadir drone clip. What this shortcut deliberately doesn't cover — stabilization (any
drone or camera motion shows up as fake vehicle velocity), non-nadir perspective correction
(distance-per-pixel varies across the frame when the gimbal isn't straight down), and per-frame
altitude handling for vertically-moving drones — is MVP2's job, via full world projection.

## Limitations

These are the cases where the GSD shortcut breaks and full MVP2 takes
over:

| Scenario | Problem | Fix |
| --- | --- | --- |
| Drone tracks horizontally during clip | Apparent vehicle velocity includes drone motion | MVP2 stabilisation |
| Drone changes altitude during clip | GSD varies per frame; SSAM has one `Scale` value | Either re-scale coordinates per frame so a single Scale stays valid, or split the clip; full solution is MVP2 |
| Gimbal not at −90° (nadir) | Distance-per-pixel varies across the frame | MVP2 single-homography |
| Take-off altitude ≠ ground level under camera | Logged altitude is not AGL | Operator note + manual altitude override |
| Camera is fixed (not a drone) | No GSD applicable | Per-scene config with known reference distance, or MVP2 homography from ground markers |

---

## Verification

- A drone clip with a known `--drone-model` produces a `.trj` whose
  `Length` field matches a hand-measured vehicle to within ~10%.
- `Speed` for a vehicle traveling at a known speed matches to within ~10%.
- `DIMENSIONS.Scale` in the dump-trj output equals the GSD value.
- The domain change is backward-compatible: `meters_per_pixel = 1.0` reproduces MVP1 behaviour.
