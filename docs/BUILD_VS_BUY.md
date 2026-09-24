# Build vs. Buy: Hand-Rolled Code With Mature Library Alternatives

An audit of places in the codebase that reimplement something a production-ready external
library already solves, so each can be reviewed and prioritized independently. This is **not**
a to-do list — nothing below is scheduled; it's raised for review. Two items were already found
and fixed in the course of other work (kept here, marked done, for a complete record); the rest
are open for a decision.

Ordered by priority — roughly (value of fixing it) × (confidence it's a real problem, not just
"a library exists") ÷ (blast radius of touching it).

---

## ✅ Done

### Polygon math → `shapely`

`domain/geometry.py` hand-rolled even-odd ray casting (`point_in_polygon`), a Sutherland-Hodgman
polygon clip, and the shoelace area formula (`clipped_overlap_fraction`). Replaced both with
`shapely`-backed implementations behind the same function signatures — no caller changed.
**This wasn't hypothetical risk**: the ray-casting version's boundary-case ambiguity had already
forced a workaround in a plane-zone test (padding polygons a few pixels so correspondence points
didn't land exactly on an edge). `shapely` is GEOS-backed and handles this correctly by
construction. See commit `291a18b`.

### YOLO-OBB training harness → `ultralytics`'s own CLI

GitHub Issues Group A2 had planned a `scripts/train_yolo_obb.py` wrapper script.
`ultralytics` already ships a complete OBB fine-tuning CLI/API (`yolo obb train
data=... model=yolo11n-obb.pt ...` / `YOLO(...).train(...)`), and the target dataset (UAV-OBB)
is already in YOLO-OBB label format — there was nothing left for a wrapper to do beyond
reimplementing that CLI. Dropped the planned file; the plan now points at the real command. See
commit `d5c3597`.

---

## Open — High priority

### Sidecar JSON schema validation → `pydantic` (or `msgspec`)

Three infrastructure adapters hand-roll JSON parsing + validation with `isinstance` chains and
manually constructed `ValueError` messages:

- `infrastructure/exclusion/json.py` (16 `isinstance`/`raise ValueError` sites)
- `infrastructure/road_graph/json.py` (22 sites — link/lane/plane zone loaders)
- `infrastructure/world/calibration.py` (17 sites — correspondence loading)

Same shape every time: check it's a dict, check a required key exists, check field types
(including excluding `bool` from `int` checks, since `isinstance(True, int)` is `True` in
Python), build a path-prefixed error message, repeat per field. ~55 sites of this pattern across
three files.

**Why this is a good fit, not just "a library exists":** `pydantic`'s `ValidationError`
aggregates *every* failing field in one pass, not just the first one hit — which is actually
closer to this project's own stated philosophy (`application/CONFIG_DESIGN.md`'s "list every
missing key, don't fail on the first one") than the current fail-fast code achieves per-file
today. Replacing these three loaders would very likely also be a net *behavior* improvement, not
just less code.

**Risk:** low-moderate. Each loader is a leaf I/O adapter behind a stable function signature
(`load_link_zones(path: Path) -> LinkZones`, etc.) — the domain types (`LinkZone`, `Polygon`, …)
don't need to change, only what's inside these three files. Touches their three test files too.
`pydantic` would be a new runtime dependency (not currently used anywhere in the project).

---

## Open — Medium priority

### `application/config.py`'s `RunConfig` resolution → `pydantic` (or `cattrs`)

560 lines, the same hand-rolled-validation pattern as above but for the entire CLI run spec
(detector, calibration, tracker, orientation, export, analysis window). Deliberately designed
around a "zero hardcoded defaults, aggregate every missing/invalid key into one error" philosophy
(`application/CONFIG_DESIGN.md`) — which `pydantic` could express natively (custom validators
collecting into one `ValidationError`), so the fit is real.

**Why this is lower priority than the sidecar loaders despite being the same pattern:** it's the
single most load-bearing config surface in the whole CLI, it already has a considered design doc
explaining *why* it looks the way it does, and it predates this session — not something written
without knowing the alternative existed. A swap here is an architectural conversation (does
`pydantic`'s validation model actually fit the "resolve from TOML + CLI-flag layering" merge
logic cleanly, or fight it?), not a mechanical find-and-replace like the sidecar loaders.

### Kalman/RTS smoother → `filterpy`

`application/kalman.py` (256 lines) hand-rolls a constant-acceleration Kalman filter and an RTS
backward smoothing pass with raw `numpy` matrix operations. `filterpy` (`KalmanFilter`,
`rts_smoother`) is a mature, widely-used library for exactly this.

**Why this is moderate, not high, priority:** unlike the polygon case, there's no known bug
driving this — the hand-rolled filter is tested and correct. The motion model itself (the CA
`F`/`H`/`Q`/`R` matrices) and this project's specific extensions (`from_state`/`predict` for
ReID motion-gating, added this session) are domain-specific and would need to be written either
way; `filterpy` would mostly remove predict/update-loop and covariance-bookkeeping boilerplate,
not eliminate a class of bug the way `shapely` did for polygons. Real value, but "less code" more
than "fewer bugs."

---

## Open — Low priority

### `Transform2D` affine compose/inverse → `numpy` matrix ops (or `cv2`)

`domain/geometry.py`'s `Transform2D.compose`/`.inverse` hand-derive the 2×3 affine composition
and inversion formulas by hand. Could instead store the transform as a `numpy` 2×3 array and use
`@` for composition, `cv2.invertAffineTransform` for inversion (`cv2` is already a dependency,
used elsewhere for exactly this kind of transform).

**Why this is low priority:** the formulas are short (~25 lines total), already unit-tested, and
have no known bug or edge case that's bitten anything — unlike `point_in_polygon`'s boundary
ambiguity. This is more a style/consistency cleanup than a risk-reduction move.

---

## Considered, not recommended

- **`calibration/srt_parser.py`** (DJI `.SRT` telemetry) — only extracts two regex-matched
  numeric fields from raw lines; never needs SRT's block/timestamp structure at all. A generic
  SRT-parsing library (`srt`, `pysrt`) would add a dependency for structure this code doesn't
  use. Not a good fit.
- **`infrastructure/progress/console.py`** — checked on the assumption it might hand-roll
  throttling that `tqdm` already provides; it already wraps `tqdm` correctly (see its module
  docstring: "tqdm throttles its own redraws, so no manual rate-limiting is needed"). No finding.
- **`scripts/propose_calibration.py`**'s Canny+Hough road-marking detection — already built on
  `cv2` primitives (edge/line detection, not reinvented); the filtering logic on top (brightness/
  contrast heuristics to pick plausible calibration points) is inherently bespoke research code —
  no library solves "propose calibration points from road markings."
- **cv2/boxmot/ultralytics usage elsewhere** (ORB+RANSAC ego-motion, `findHomography`,
  BoT-SORT tracking, YOLO detection/OBB) — already correctly delegating to libraries; nothing
  reinvented.
