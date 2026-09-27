"""Run configuration: the persisted, replayable specification of one analysis.

A TraTrac run is fully described by a ``RunConfig`` — input video and
transforms file, detector, tracker, export, analysis window, and run options.
Every geometric transform (GSD scale, ego-motion, world projection) is
resolved upstream by ``tratrac-preprocess``, not here — ``input.transforms_in``
just names that file. Every value is mandatory: there are **no built-in
defaults anywhere in the
package**. Each parameter must be supplied by a TOML config file or a CLI flag;
if neither supplies it, ``RunConfig.resolve`` fails listing exactly what is
missing. This trades typing convenience for scientific reproducibility — a
``.trj`` is reconstructable from the config that produced it, which names its
own input and output. A silent default (e.g. an unstated ``conf=0.25`` or
``device=cpu``) is exactly the kind of hidden variable that makes a result hard
to reproduce or defend, so none survive — this extends the same calibration
philosophy MVP1.75 already applies to metric output
(``src/tratrac/calibration/GSD_CALIBRATION.md``: refuse to run uncalibrated
rather than emit physically meaningless values) to *every* parameter.

Layering: this module is pure (no I/O, no CLI framework). ``RunConfig`` and its
section dataclasses (``InputConfig``, ``DetectorConfig``, …) are the public,
plain-dataclass shape every caller (``cli.py``, tests, …) has always used —
that shape does not change. Underneath, ``RunConfig.resolve`` merges the TOML
table and CLI overrides into one nested dict (``_merge``) and validates it in
one pass through a tree of internal ``pydantic`` models (``_ConfigModel`` and
its per-section models) that mirror the dataclasses field-for-field.
``pydantic.ValidationError`` already aggregates every problem across every
field in one exception — exactly the "list every missing/invalid key at once"
contract ``ConfigError`` has always made, so adopting it doesn't change that
contract, only what implements it. ``_translate_errors`` turns pydantic's
error list back into this module's own long-standing message wording, so
nothing downstream (including every existing test) needed to change. Custom
business rules (range checks, the device regex, timecode parsing, window
ordering) are pydantic validators, not hand-rolled ``if`` chains, but read the
same either way. There is no ``CalibrationConfig``/``EgoMotionConfig`` here:
GSD-scale resolution (``calibration/gsd.py``, ``srt_parser.py``,
``drone_specs.py``) lives entirely in ``cli_preprocess.py``'s ``estimate``
subcommand now. ``infrastructure/config/toml.py`` is the one seam where the
dynamically-typed TOML document enters (``load_toml``, stdlib ``tomllib``) —
this module does the validating. ``cli.py`` loads the TOML, calls
``RunConfig.resolve`` with an empty override map, validates the resolved video
on disk, then builds the adapters from the typed config; it reads ``--force``
directly (not through resolution) and translates ``ConfigError`` into exit
code 2. See ``src/tratrac/cli.py``'s module docstring for the ``--check``
validation-without-running mode.

Why pydantic, and why not sooner: this was an open question (see the "Build
vs. buy" GitHub Discussion) motivated by two things — pydantic's own
aggregated ``ValidationError`` fitting this module's philosophy more natively
than the hand-rolled resolver it replaced, and schema-drift prevention (a
docstring like this one, or a UI, hand-describing the schema can silently
drift from the code the moment a field changes — this module's own docs did,
twice, before both got folded into this docstring). It wasn't done from the
start because the exact shape of "zero hardcoded defaults, aggregate every
error" was worked out *on this project*, iteratively (see "Design history"
below) — building it by hand first meant not also fighting a library's own
conventions while still discovering what the design needed to be. It wasn't
done immediately once pydantic looked like a fit either, because this is the
single most load-bearing config surface in the CLI: the migration waited for
CI to exist (see ``.github/workflows/ci.yml``) as the safety net a
behavior-preserving swap like this one deserves. ``_ConfigModel`` also now
exposes ``model_json_schema()`` — a machine-readable schema a future doc
generator or URBAn's config-editor UI (see URBAn's ``docs/config_editor_spec.md``)
can read the field list from directly, instead of a second hand-maintained
copy that can drift the same way this docstring's tables already did.

Resolution model: precedence per key is config-file value, then error — the
resolver still accepts an ``overrides`` mapping and applies it with highest
precedence (``None`` means "not supplied," falls through to the file value),
but the CLI now feeds it an empty map; the mechanism is retained so a future
flag or a programmatic caller could override a key without reworking
resolution (see "Design history" below for why the CLI itself doesn't use it
today). ``RunConfig.resolve`` collects *all* problems and raises a single
``ConfigError``, so one run surfaces every missing/invalid key instead of
one-per-attempt.

Design history — why the override flags were removed: the first revision
mirrored every config key as a ``--flag`` (so ``--conf 0.4`` could tweak one
value without editing the TOML) and allowed a positional ``VIDEO`` argument to
override ``input.video``. Both were removed: the per-key flags duplicated the
config surface (every key needed a flag, a wiring line, and help text — two
ways to set one value), and they weakened the reproducibility argument the
config exists to make, since a run set partly by flags is no longer fully
captured by its file. Collapsing to config-only makes the file the single,
complete, replayable spec. ``--force`` survived as the one genuinely ad-hoc
operational toggle, then was taken out of the config too (``run.force``
removed): overwrite policy never affects the trajectories, so it doesn't
belong in the reproducible run spec at all.

No library pixel fallback: removing defaults is package-wide, not CLI-only.
The adapter constructors (``SsamTrjExporter``, ``RtDetrDetector``,
``YoloV8VisDroneDetector``, ``BoxmotBotSortTracker``) carry no defaults either
— the old ``scale=1.0``/``meters_per_pixel=1.0`` "pixels-as-metres" library
escape hatch is gone. Callers (CLI and tests) pass every value explicitly;
tests that want MVP1 pixel behaviour pass ``scale=1.0``/``meters_per_pixel=1.0``
themselves. ``detector.filename`` is required even for ``rt_detr`` (which
ignores it) for the same reason — conditional-requiredness per adapter wasn't
worth the special case.

Usage::

    uv run tratrac --config run.toml            # the only way to run it
    uv run tratrac --config run.toml --check    # validate without running
    uv run tratrac --config run.toml --force    # overwrite existing outputs

There are no per-key override flags and no positional ``VIDEO`` argument —
``--config`` is the only way values reach the run. ``--force``/``--no-force``
(overwrite control) and ``--check``/``--json`` (validate-only) are the only
other CLI flags, and neither is a config key. See ``src/tratrac/cli.py``'s
module docstring for the full ``--check`` validation contract.

This config covers ``tratrac`` only — the perception pass. GSD calibration,
ego-motion stabilization, and world-projection homography are not config keys
here at all: they're resolved upstream by ``tratrac-preprocess`` and handed to
``tratrac`` as one path, ``input.transforms_in`` (see "What isn't here" below).

Two rules govern the whole file:

1. Every key must be **present**. Absence is an error — there is no key whose
   omission means "use a sensible default."
2. "Disabled" is an explicit value, never a missing key: ``process_fps = 0.0``
   (every frame), ``timing_csv = ""`` (profiling off), ``start = "" / end = ""``
   (the clip's natural bounds).

Sections (TOML table -> dataclass -> required keys)::

    [input]                                InputConfig
      video          path, must exist
      process_fps    number >= 0.0; 0.0 = every frame (decode-time decimation,
                     see infrastructure/TIMESTEP_PRECISION.md)
      transforms_in  path to a `tratrac-preprocess estimate` run's transforms
                     file; always required, checked to exist before the video
                     opens — a static-camera run still needs it for its scale
                     rows

    [detector]                             DetectorConfig
      name           "yolov8_visdrone" (current default) | "rt_detr" (dormant)
                     | "yolo_obb" (not yet the default) — see
                     infrastructure/detection/DETECTOR_CHOICE.md
      checkpoint     HuggingFace repo id, e.g. "Mahadih534/YoloV8-VisDrone"
      conf           number in [0.0, 1.0] — detection confidence threshold
      filename       weights file inside the repo; consumed only by
                     yolov8_visdrone but required for every detector (the
                     zero-defaults rule has no per-adapter exception)

    [runtime]                              RuntimeConfig
      device         "cpu" | "mps" | "cuda" | "cuda:N"

    [tracker]                              TrackerConfig
      det_thresh     number in [0.0, 1.0] — BoT-SORT detection threshold,
                     conventionally kept below detector.conf

    [export]                               ExportConfig
      out            output **track record** path (Parquet) — tratrac's only
                     output. Not a .trj: run tratrac-postprocess on this file
                     to get one.

    [window]                               WindowConfig
      start          "" (clip start) | a timecode: SS(.ms) / MM:SS(.ms) /
                     HH:MM:SS(.ms), e.g. "12.5", "1:30", "00:01:30.250"
      end            "" (clip end) | a timecode; must be > 0 and after start

    [run]                                  RunOptionsConfig
      timing_csv     "" (profiling off) | a CSV path for per-frame step
                     timings (see infrastructure/timing/STEP_TIMING.md); must
                     differ from export.out

``force`` is deliberately **not** a config key anywhere — overwrite policy is
pure I/O that never affects the trajectories, so it's excluded from the
reproducible run spec on purpose; it lives only on the CLI's ``--force`` flag.

What isn't here (and why): earlier revisions of this config had
``[calibration]`` (GSD scale), ``[ego_motion]`` (ORB stabilization toggle +
tuning), and ``[orientation]`` (a live heading-smoothing window) sections. All
three are gone, not renamed, removed:

- GSD calibration and ego-motion are resolved once, upstream, by
  ``tratrac-preprocess estimate`` (its own CLI flags — ``--meters-per-pixel``
  or ``--drone-model``/``--altitude-m``/``--srt`` for scale, ORB tuning knobs
  only there too). The result lands in one shared transforms file, which
  ``input.transforms_in`` just names. There is no ``enabled`` toggle to gate
  on here — whether stabilization applies is a property of that file's
  *content* (whether it has similarity rows), not a flag. See
  ``infrastructure/video/ego_motion_orb.py``'s module docstring and
  ``src/tratrac/calibration/GSD_CALIBRATION.md``.
- World-projection homography is fitted by the same tool's ``project``
  subcommand into that same transforms file. ``tratrac`` never fits or
  applies one — that's ``tratrac-postprocess``'s job, via its own
  ``--transforms`` flag. See ``src/tratrac/application/WORLD_PROJECTION.md``.
- Orientation is no longer computed live. ``tratrac`` writes only raw tracked
  positions; heading, speed, and acceleration are reconstructed entirely
  offline by ``tratrac-postprocess``'s Kalman/RTS smoother, tuned via its own
  ``--pos-noise``/``--jerk`` flags, not a config section. See
  ``src/tratrac/application/SMOOTHING.md``.

Complete example (see ``tratrac.example.toml`` at the repo root for a
copyable, fully-commented version)::

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

A run with missing or invalid keys exits with code 2 and lists everything
wrong in one message::

    ERROR: invalid run configuration; supply each value via the --config TOML
    or its flag:
      - input.video is missing.
      - input.transforms_in does not exist or is not a file -- run tratrac-preprocess first.
      - detector.conf must be in [0.0, 1.0], got 1.5.
      - runtime.device 'gpu' is invalid; expected cpu, mps, or cuda[:N] (e.g. cuda:0).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pydantic


class DetectorChoice(StrEnum):
	"""Available detector adapters.

	``yolov8_visdrone`` is the MVP1 emergency detector — community YOLOv8 fine-tuned
	on VisDrone, picked because COCO-pretrained RT-DETR fails on aerial inputs.
	``rt_detr`` stays available as a dormant alternative behind the ``Detector`` port.
	``yolo_obb`` (Group A9, GitHub Issues) wraps ``ultralytics``'s OBB task
	(see ``src/tratrac/infrastructure/detection/yolo_obb.py``) — **not yet the default**: MVP1.5's
	acceptance criterion is that a fine-tuned checkpoint measurably beats the YOLOv8-VisDrone
	baseline (``scripts/probe_detector.py`` + ``scripts/validate_trj.py``) before the default
	changes, and no such checkpoint exists yet (needs a GPU + the UAV-OBB dataset — neither
	available in this environment). See ``src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md``.
	"""

	YOLOV8_VISDRONE = "yolov8_visdrone"
	RT_DETR = "rt_detr"
	YOLO_OBB = "yolo_obb"


class ConfigError(Exception):
	"""Raised when the resolved run configuration is incomplete or invalid.

	Carries every problem found (missing keys, bad types, out-of-range values) so
	the CLI can report them all at once instead of one failure per run.
	"""

	def __init__(self, problems: Sequence[str]) -> None:
		self.problems = list(problems)
		joined = "\n".join(f"  - {problem}" for problem in self.problems)
		super().__init__(
			"invalid run configuration; supply each value via the --config TOML "
			f"or its flag:\n{joined}"
		)


@dataclass(frozen=True, slots=True)
class InputConfig:
	"""The processed video and the transforms file that resolves every geometric
	transform for this run. Per-run, but part of the persisted config so a saved
	config replays without any positional argument."""

	video: Path
	# Cap the processing cadence to this many frames per second (decode-time
	# decimation, see src/tratrac/infrastructure/TIMESTEP_PRECISION.md). ``0.0`` = process every frame.
	process_fps: float
	# A `tratrac-preprocess` run's transforms file (`infrastructure/transform/records.py`).
	# Always required: `tratrac` never estimates ego-motion or resolves the GSD scale
	# itself -- it only ever reads the ego-motion stage's rows out of this file, using
	# the identity when there are none (a static-camera run still needs the file for
	# its per-frame scale rows, which `tratrac-postprocess` reads later). See
	# infrastructure/video/ego_motion_orb.py's module docstring.
	transforms_in: Path


@dataclass(frozen=True, slots=True)
class DetectorConfig:
	name: DetectorChoice
	checkpoint: str
	conf: float
	# Consumed only by the yolov8_visdrone adapter (its HuggingFace Hub filename); rt_detr and
	# yolo_obb ignore it but must still supply a non-empty value (zero-defaults config: still an
	# open question whether to drop or repurpose it, see DETECTOR_CHOICE.md "Open questions").
	filename: str


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
	device: str


@dataclass(frozen=True, slots=True)
class TrackerConfig:
	det_thresh: float


@dataclass(frozen=True, slots=True)
class ExportConfig:
	# The run's primary output: the track record (raw tracked measurements). The
	# offline ``tratrac-postprocess`` pass reads it to produce the SSAM ``.trj`` (src/tratrac/application/SMOOTHING.md).
	# The GSD scale is no longer resolved or written here -- it's one of
	# `input.transforms_in`'s rows, resolved by `tratrac-preprocess estimate`.
	out: Path


@dataclass(frozen=True, slots=True)
class WindowConfig:
	"""Analysis window in seconds. ``None`` means the clip's natural bound."""

	start_seconds: float | None
	end_seconds: float | None


@dataclass(frozen=True, slots=True)
class RunOptionsConfig:
	# ``force`` is intentionally absent: overwrite policy is pure I/O, never affects the
	# trajectories, so it is not part of the reproducible run spec — it lives only on the
	# ``--force`` CLI flag (application/config.py's module docstring), not in the config.
	timing_csv: Path | None  # None = profiling off


@dataclass(frozen=True, slots=True)
class RunConfig:
	"""The complete, validated specification of one analysis run."""

	input: InputConfig
	detector: DetectorConfig
	runtime: RuntimeConfig
	tracker: TrackerConfig
	export: ExportConfig
	window: WindowConfig
	options: RunOptionsConfig

	@classmethod
	def resolve(
		cls,
		file_values: Mapping[str, Any],
		cli_overrides: Mapping[str, Any],
	) -> RunConfig:
		"""Merge a TOML table and CLI overrides into a validated ``RunConfig``.

		Precedence per key: CLI override (non-``None``) > config file > error.
		Collects every problem (via ``_ConfigModel``, see below) and raises a single
		``ConfigError`` if any remain.
		"""
		merged = _merge(file_values, cli_overrides)
		try:
			model = _ConfigModel.model_validate(merged)
		except pydantic.ValidationError as exc:
			raise ConfigError(_translate_errors(exc)) from None

		return cls(
			input=InputConfig(
				video=model.input.video,
				process_fps=model.input.process_fps,
				transforms_in=model.input.transforms_in,
			),
			detector=DetectorConfig(
				name=model.detector.name,
				checkpoint=model.detector.checkpoint,
				conf=model.detector.conf,
				filename=model.detector.filename,
			),
			runtime=RuntimeConfig(device=model.runtime.device),
			tracker=TrackerConfig(det_thresh=model.tracker.det_thresh),
			export=ExportConfig(out=model.export.out),
			window=WindowConfig(
				start_seconds=model.window.start_seconds, end_seconds=model.window.end_seconds
			),
			options=RunOptionsConfig(timing_csv=model.run.timing_csv),
		)


_SECTIONS = ("input", "detector", "runtime", "tracker", "export", "window", "run")


def _merge(file_values: Mapping[str, Any], cli_overrides: Mapping[str, Any]) -> dict[str, Any]:
	"""Fold ``cli_overrides`` (a flat dotted-key map, ``None`` = "not supplied") over
	``file_values`` (the nested TOML table) into one nested dict ready for
	``_ConfigModel.model_validate``. A CLI override always wins when not ``None``;
	a key absent from both is simply missing from the result, which ``_ConfigModel``
	(every field required, no defaults) then reports as such.

	Every section key is always present (``{}`` if the file omitted it entirely) so a
	wholly-missing section still validates *into* its own model -- reporting each of
	its individual leaf fields as missing (``"runtime.device is missing."``) instead of
	the whole section collapsing into one top-level ``"runtime is missing."``.
	"""
	merged: dict[str, dict[str, Any]] = {section: {} for section in _SECTIONS}
	for section, table in file_values.items():
		if isinstance(table, Mapping) and section in merged:
			merged[section].update(table)
	for dotted, value in cli_overrides.items():
		if value is None:
			continue
		section, _, key = dotted.partition(".")
		merged.setdefault(section, {})[key] = value
	return merged


class _StrictModel(pydantic.BaseModel):
	"""Shared config for every section model: no defaults, no unlisted keys."""

	model_config = pydantic.ConfigDict(extra="forbid")


def _require_str(v: Any) -> str:
	if not isinstance(v, str):
		raise ValueError(f"must be a string, got {type(v).__name__}.")
	return v


def _require_number(v: Any) -> float:
	# isinstance(True, int) is True in Python -- a TOML boolean must not silently
	# become 1.0/0.0 for a numeric field.
	if isinstance(v, bool) or not isinstance(v, int | float):
		raise ValueError(f"must be a number, got {type(v).__name__}.")
	return float(v)


def _require_path(v: Any) -> Path:
	if isinstance(v, Path):
		return v
	if isinstance(v, str):
		if not v:
			raise ValueError("must not be empty.")
		return Path(v)
	raise ValueError(f"must be a path string, got {type(v).__name__}.")


def _toggleable_path(v: Any) -> Path | None:
	"""``""``/``None`` -> disabled; else a path string. See ``RunOptionsConfig``."""
	if v == "" or v is None:
		return None
	if isinstance(v, Path):
		return v
	if isinstance(v, str):
		return Path(v)
	raise ValueError('must be a path string or "".')


_Str = pydantic.BeforeValidator(_require_str)
_Number = pydantic.BeforeValidator(_require_number)
_PathField = pydantic.BeforeValidator(_require_path)

_DEVICE_RE = re.compile(r"cpu|mps|cuda(:\d+)?")


def _validated_device(v: str) -> str:
	# Empty already reported as a wrong/missing value upstream -- don't pile a second
	# problem on top of it (mirrors every other "only check format when present" rule
	# in this module).
	if v and _DEVICE_RE.fullmatch(v) is None:
		raise ValueError(f"{v!r} is invalid; expected cpu, mps, or cuda[:N] (e.g. cuda:0).")
	return v


def _parse_timecode(value: str) -> float:
	"""Parse ``SS(.ms)``, ``MM:SS(.ms)``, or ``HH:MM:SS(.ms)`` into seconds."""
	parts = value.strip().split(":")
	if len(parts) > 3:
		raise ValueError(f"timecode has too many ':'-separated parts: {value!r}.")
	try:
		seconds = float(parts[-1])
		minutes = float(parts[-2]) if len(parts) >= 2 else 0.0
		hours = float(parts[-3]) if len(parts) == 3 else 0.0
	except ValueError:
		raise ValueError(
			f"invalid timecode {value!r}; expected SS(.ms), MM:SS, or HH:MM:SS."
		) from None
	if seconds < 0 or minutes < 0 or hours < 0:
		raise ValueError(f"timecode components must be non-negative: {value!r}.")
	return hours * 3600.0 + minutes * 60.0 + seconds


def _window_bound(v: Any) -> float | None:
	"""A required key; ``""`` means the clip's own bound."""
	text = _require_str(v)
	return None if text == "" else _parse_timecode(text)


class _InputModel(_StrictModel):
	video: Annotated[Path, _PathField]
	process_fps: Annotated[float, _Number, pydantic.Field(ge=0.0)]
	transforms_in: Annotated[Path, _PathField]


def _validated_detector_name(v: Any) -> DetectorChoice:
	name = _require_str(v)
	try:
		return DetectorChoice(name)
	except ValueError:
		# Matches the pre-pydantic resolver: an empty string is accepted silently here
		# (required_str never flagged emptiness), not reported as "unknown".
		if not name:
			return DetectorChoice.YOLOV8_VISDRONE
		valid = ", ".join(choice.value for choice in DetectorChoice)
		raise ValueError(f"{name!r} is unknown; valid: {valid}.") from None


class _DetectorModel(_StrictModel):
	name: Annotated[DetectorChoice, pydantic.BeforeValidator(_validated_detector_name)]
	checkpoint: Annotated[str, _Str]
	conf: Annotated[float, _Number, pydantic.Field(ge=0.0, le=1.0)]
	filename: Annotated[str, _Str]


class _RuntimeModel(_StrictModel):
	device: Annotated[str, _Str, pydantic.AfterValidator(_validated_device)]


class _TrackerModel(_StrictModel):
	det_thresh: Annotated[float, _Number, pydantic.Field(ge=0.0, le=1.0)]


class _ExportModel(_StrictModel):
	out: Annotated[Path, _PathField]


class _WindowModel(_StrictModel):
	start_seconds: Annotated[float | None, pydantic.BeforeValidator(_window_bound)]
	end_seconds: Annotated[float | None, pydantic.BeforeValidator(_window_bound)]

	@pydantic.model_validator(mode="before")
	@classmethod
	def _rename_toml_keys(cls, data: Any) -> Any:
		# The TOML table spells these `start`/`end`; the domain object spells them
		# `start_seconds`/`end_seconds`. Rename here so the rest of the model (and the
		# error `loc`, which must read back as `window.start`/`window.end`) stays keyed
		# by the public TOML names -- see `_translate_errors`. Only carry a key over if
		# it was actually present: `.get(..., default=None)` would turn a genuinely
		# absent key into a present-but-None one, losing pydantic's own "missing" report
		# in favor of a wrong "must be a string, got NoneType" one.
		if not isinstance(data, Mapping):
			return data
		renamed: dict[str, Any] = {}
		if "start" in data:
			renamed["start_seconds"] = data["start"]
		if "end" in data:
			renamed["end_seconds"] = data["end"]
		return renamed

	@pydantic.model_validator(mode="after")
	def _check_order(self) -> _WindowModel:
		if self.end_seconds is not None and self.end_seconds <= 0.0:
			raise ValueError("window.end must be greater than zero.")
		if (
			self.start_seconds is not None
			and self.end_seconds is not None
			and self.end_seconds <= self.start_seconds
		):
			raise ValueError("window.end must be after window.start.")
		return self


class _RunOptionsModel(_StrictModel):
	timing_csv: Annotated[Path | None, pydantic.BeforeValidator(_toggleable_path)]


class _ConfigModel(_StrictModel):
	"""The whole merged TOML table, validated in one pass so every problem across
	every section is collected together (``pydantic.ValidationError.errors()``),
	the same aggregate-everything contract ``ConfigError`` has always made.
	"""

	input: _InputModel
	detector: _DetectorModel
	runtime: _RuntimeModel
	tracker: _TrackerModel
	export: _ExportModel
	window: _WindowModel
	run: _RunOptionsModel


# error `type` codes pydantic reports for a field with no default and no value supplied --
# translated to this module's long-standing "{dotted}.{key} is missing." phrasing.
_MISSING_TYPES = frozenset({"missing"})


def _translate_errors(exc: pydantic.ValidationError) -> list[str]:
	"""Turn a ``ValidationError`` into this module's flat ``"section.key: message"``
	problem strings -- the wire format ``ConfigError``/every caller/every test has
	always used, so swapping the validator underneath doesn't ripple outward.
	"""
	problems: list[str] = []
	for error in exc.errors():
		loc = list(error["loc"])
		# _WindowModel's before-validator remaps start_seconds/end_seconds back to the
		# TOML names before pydantic ever sees them, but a *missing* `window` table
		# itself (loc == ("window",)) never reaches that remap -- normalize here too.
		if loc[:2] == ["window", "start_seconds"]:
			loc[1] = "start"
		elif loc[:2] == ["window", "end_seconds"]:
			loc[1] = "end"
		dotted = ".".join(str(part) for part in loc)
		if error["type"] in _MISSING_TYPES:
			if dotted == "run.timing_csv":
				problems.append(f'{dotted} is missing (use "" to disable).')
			else:
				problems.append(f"{dotted} is missing.")
			continue
		if error["type"] == "greater_than_equal" and dotted == "input.process_fps":
			problems.append("input.process_fps must be >= 0 (0 = every frame).")
			continue
		if error["type"] in ("greater_than_equal", "less_than_equal") and dotted in (
			"detector.conf",
			"tracker.det_thresh",
		):
			problems.append(f"{dotted} must be in [0.0, 1.0], got {error['input']!r}.")
			continue
		msg = error["msg"]
		if msg.startswith("Value error, "):
			msg = msg[len("Value error, ") :]
		if dotted == "runtime.device":
			problems.append(f"runtime.device {msg}")
		elif dotted == "detector.name":
			problems.append(f"detector.name {msg}")
		else:
			problems.append(f"{dotted} {msg}")
	return problems
