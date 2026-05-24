# SSAM Trajectory File Format

Authoritative sources in this folder:

- `SSAM File Format v1.04.pdf` — original Siemens / Gardner Consulting spec (2004).
- `Open Source SSAM File Format v3.0.pdf` — New Global Systems update (2017), backward compatible with 1.04 + adds elevation.

If this document and the PDFs disagree, **the PDFs win** and this file must be updated.

---

## Format Basics

- Binary, record-based.
- Extension: `.trj` or `.TRJ`.
- No header magic; the first byte is the FORMAT record-type byte (0).
- File ends when no more data is available — no explicit EOF record.
- Endianness is declared in the FORMAT record, not fixed.

---

## Type Encodings

| Name | Size | Encoding |
| --- | --- | --- |
| Byte | 1 byte | unsigned |
| Integer | 4 bytes | signed two's complement |
| Float | 4 bytes | signed IEEE 754 single precision |

Multi-byte values (Integer, Float) honor the endianness byte in the FORMAT record. The endianness byte itself is the ASCII character `L` (0x4C) or `B` (0x42).

---

## File Layout

```text
FORMAT
DIMENSIONS
TIMESTEP
    VEHICLE
    VEHICLE
    ...
TIMESTEP
    VEHICLE
    ...
... (more timesteps until EOF)
```

- Exactly one FORMAT record.
- Exactly one DIMENSIONS record.
- Then alternating: one TIMESTEP record followed by a variable number of VEHICLE records.
- The reader uses the leading record-type byte to dispatch.

---

## Record: FORMAT (id = 0)

### v1.04 layout (6 bytes)

| Field | Type | Value |
| --- | --- | --- |
| Record Type | Byte | 0 |
| Endian | Byte | ASCII `L` (little-endian) or `B` (big-endian) |
| Version | Float | 1.04 |

### v3.0 layout (7 bytes)

Adds one byte at the end:

| Field | Type | Value |
| --- | --- | --- |
| Z Value Option | Byte | 0 / blank = no Z in VEHICLE records; non-zero = Z values appended to each VEHICLE record |

---

## Record: DIMENSIONS (id = 1)

22 bytes.

| Field | Type | Notes |
| --- | --- | --- |
| Record Type | Byte | 1 |
| Units | Byte | 0 = English (feet, ft/s, ft/s²); 1 = Metric (m, m/s, m/s²) |
| Scale | Float | Distance per X or Y unit ("per pixel"). Real distance = value × Scale |
| MinX | Integer | Left edge of observation area |
| MinY | Integer | Bottom edge |
| MaxX | Integer | Right edge |
| MaxY | Integer | Top edge |

Constraints:

- Observation area < 10 sq miles.
- Coordinate system is Cartesian with X right, Y up.

---

## Record: TIMESTEP (id = 2)

5 bytes.

| Field | Type | Notes |
| --- | --- | --- |
| Record Type | Byte | 2 |
| Timestep | Float | Seconds since start of simulation/observation |

Sub-second precision (≈ 1/10 s) is the practical minimum; once-per-second is too coarse for conflict analysis.

---

## Record: VEHICLE (id = 3)

### v1.04 layout (42 bytes)

| Field | Type | Notes |
| --- | --- | --- |
| Record Type | Byte | 3 |
| Vehicle ID | Integer | Unique vehicle identifier |
| Link ID | Integer | Road link identifier, where available |
| Lane ID | Byte | Lane identifier, where available — **1 byte, max 255** |
| Front X | Float | Scaled: real X = Front X × DIMENSIONS.Scale |
| Front Y | Float | Scaled |
| Rear X | Float | Scaled |
| Rear Y | Float | Scaled |
| Length | Float | **Unscaled**, in DIMENSIONS.Units |
| Width | Float | **Unscaled**, in DIMENSIONS.Units |
| Speed | Float | **Unscaled**, units/sec |
| Acceleration | Float | **Unscaled**, units/sec² |

### v3.0 supplemental fields (when FORMAT.Z Value Option ≠ 0)

Appended after Acceleration:

| Field | Type | Notes |
| --- | --- | --- |
| Front Z | Float | Same units as DIMENSIONS |
| Rear Z | Float | Same units as DIMENSIONS |

Recommended Z conventions when real elevations are unavailable:

- `-1` = underpass
- `0` = ground level
- `+1` = overpass
- Add ±1 per additional stacked level.

---

## Scaling Semantics — Easy To Get Wrong

The DIMENSIONS.Scale field decouples grid units from physical units:

- X and Y in VEHICLE records are in **grid units** (think: pixel indices).
- Real-world distance = grid value × Scale.
- Length, Width, Speed, Acceleration are already in physical units (DIMENSIONS.Units) and ignore Scale.

So if Scale = 0.25 and Units = Metric, then Front X = 4 means the bumper is at 1.0 m on the real X axis.

---

## Coordinate Orientation vs Image Space

- SSAM: Cartesian, Y grows up.
- Aerial video: image pixels, Y grows down.
- The exporter must flip Y: `y_ssam = image_height − y_image` (after any cropping/stabilization adjustments).

---

## MVP1 Approximation Strategy

MVP1 emits **valid v1.04** (no Z, no multi-plane). Concrete choices:

- Endianness: `L` (dev machine is x86_64).
- Units: Metric (1).
- Scale: 1.0 — one grid unit per "meter". MVP1 has no calibration, so pixels are pretended to be meters. Numbers are syntactically valid but physically meaningless until MVP2.
- DIMENSIONS bounds: `MinX = 0`, `MinY = 0`, `MaxX = image_width`, `MaxY = image_height`.
- Link ID: `0` (no road network).
- Lane ID: `0` (no lane assignment until MVP6).
- Front/Rear X/Y: image pixels after Y-flip.
- Length/Width: bounding-box dimensions in pixels (treated as meters — known wrong).
- Speed/Acceleration: derived from pixel displacement / time (treated as m/s and m/s² — known wrong).
- Timestep: `frame_index / fps`, Float seconds.

MVP2 replaces Scale + coordinates with real metric values from a single homography; MVP3 introduces v3.0 with Z when multi-homography lands.

**CLI policy (MVP1.75+):** the `tratrac` CLI no longer *defaults* to Scale = 1.0. It requires explicit calibration (`--meters-per-pixel`, or `--drone-model` + `--altitude`/`.SRT`) and exits with an error otherwise, so it cannot silently emit physically meaningless metric values. The scale=1.0 pixel mode described above remains available at the **library** level (`EmaOrientationEstimator` / `SsamTrjExporter` default to `scale=1.0`) for callers that knowingly want it.

---

## Record Sizes Cheat Sheet

| Record | v1.04 | v3.0 with Z |
| --- | --- | --- |
| FORMAT | 6 B | 7 B |
| DIMENSIONS | 22 B | 22 B |
| TIMESTEP | 5 B | 5 B |
| VEHICLE | 42 B | 50 B |

A 30 s clip at 30 FPS with ~50 vehicles per frame ≈ `(5 + 50 × 42) × 900 ≈ 1.9 MB` for v1.04.

---

## What Is Out Of Scope For This Document

- The SSAM **Path** file format (`.pth`) is mentioned in v1.04 but not specified.
- The SSAM **conflict output** format (`.csa`) is not part of trajectory input.

Add notes here if those become relevant.
