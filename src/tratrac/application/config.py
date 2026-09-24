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
own input and output. See ``src/tratrac/application/CONFIG_DESIGN.md``.

Layering: this module is pure (no I/O, no CLI framework). The TOML file is read
by ``infrastructure/config/toml.py``; the CLI assembles overrides, validates the
resolved video on disk, and translates ``ConfigError`` into a process exit.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


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
	# src/tratrac/infrastructure/video/EGO_MOTION.md.
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
	# ``--force`` CLI flag (src/tratrac/application/CONFIG_DESIGN.md), not in the config.
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
		Collects every problem and raises a single ``ConfigError`` if any remain.
		"""
		resolver = _Resolver(file_values, cli_overrides)

		video = resolver.required_path("input.video")
		process_fps = resolver.required_float("input.process_fps")
		if resolver.present("input.process_fps") and process_fps < 0.0:
			resolver.problems.append("input.process_fps must be >= 0 (0 = every frame).")
		transforms_in = resolver.required_path("input.transforms_in")

		detector_name = _resolve_detector_name(resolver)
		checkpoint = resolver.required_str("detector.checkpoint")
		conf = resolver.required_float("detector.conf")
		_check_range(conf, 0.0, 1.0, "detector.conf", resolver)
		filename = resolver.required_str("detector.filename")

		device = resolver.required_str("runtime.device")
		_validate_device(device, resolver)

		det_thresh = resolver.required_float("tracker.det_thresh")
		_check_range(det_thresh, 0.0, 1.0, "tracker.det_thresh", resolver)

		out = resolver.required_path("export.out")

		window = WindowConfig(
			start_seconds=_resolve_window_bound(resolver, "window.start"),
			end_seconds=_resolve_window_bound(resolver, "window.end"),
		)
		_validate_window(window, resolver)

		timing_csv = resolver.toggleable_path("run.timing_csv")

		if resolver.problems:
			raise ConfigError(resolver.problems)

		return cls(
			input=InputConfig(video=video, process_fps=process_fps, transforms_in=transforms_in),
			detector=DetectorConfig(
				name=detector_name, checkpoint=checkpoint, conf=conf, filename=filename
			),
			runtime=RuntimeConfig(device=device),
			tracker=TrackerConfig(det_thresh=det_thresh),
			export=ExportConfig(out=out),
			window=window,
			options=RunOptionsConfig(timing_csv=timing_csv),
		)


_MISSING = object()


class _Resolver:
	"""Pulls values from CLI overrides then the TOML table, collecting problems.

	``cli_overrides`` is a flat dotted-key map (e.g. ``"detector.conf"``) whose
	``None`` values mean "not passed on the command line". ``file_values`` is the
	nested TOML table. Missing keys and type errors accumulate in ``problems`` so
	resolution reports them together rather than one per run.
	"""

	def __init__(self, file_values: Mapping[str, Any], cli_overrides: Mapping[str, Any]) -> None:
		# TOML values are dynamically typed; Any is confined to this resolution seam.
		self._file = file_values
		self._cli = cli_overrides
		self.problems: list[str] = []

	def _raw(self, dotted: str) -> Any:
		"""CLI override (if not ``None``), else the file value, else ``_MISSING``."""
		cli_value = self._cli.get(dotted)
		if cli_value is not None:
			return cli_value
		section, _, key = dotted.partition(".")
		table = self._file.get(section)
		if isinstance(table, Mapping) and key in table:
			return table[key]
		return _MISSING

	def present(self, dotted: str) -> bool:
		"""Whether a value was supplied at all (so range checks don't pile a second
		problem on top of an already-recorded "missing")."""
		return self._raw(dotted) is not _MISSING

	def required_str(self, dotted: str) -> str:
		raw = self._raw(dotted)
		if raw is _MISSING:
			self.problems.append(f"{dotted} is missing.")
			return ""
		if not isinstance(raw, str):
			self.problems.append(f"{dotted} must be a string, got {type(raw).__name__}.")
			return ""
		return raw

	def required_float(self, dotted: str) -> float:
		raw = self._raw(dotted)
		if raw is _MISSING:
			self.problems.append(f"{dotted} is missing.")
			return 0.0
		if isinstance(raw, bool) or not isinstance(raw, int | float):
			self.problems.append(f"{dotted} must be a number, got {type(raw).__name__}.")
			return 0.0
		return float(raw)

	def required_path(self, dotted: str) -> Path:
		raw = self._raw(dotted)
		if raw is _MISSING:
			self.problems.append(f"{dotted} is missing.")
			return Path()
		if isinstance(raw, Path):
			return raw
		if isinstance(raw, str):
			if not raw:
				self.problems.append(f"{dotted} must not be empty.")
				return Path()
			return Path(raw)
		self.problems.append(f"{dotted} must be a path string, got {type(raw).__name__}.")
		return Path()

	def toggleable_path(self, dotted: str) -> Path | None:
		"""A required key whose empty value (``""``) means "disabled" -> ``None``."""
		raw = self._raw(dotted)
		if raw is _MISSING:
			self.problems.append(f'{dotted} is missing (use "" to disable).')
			return None
		if raw == "" or raw is None:
			return None
		if isinstance(raw, Path):
			return raw
		if isinstance(raw, str):
			return Path(raw)
		self.problems.append(f'{dotted} must be a path string or "".')
		return None


_DEVICE_RE = re.compile(r"cpu|mps|cuda(:\d+)?")


def _validate_device(device: str, resolver: _Resolver) -> None:
	"""Reject device strings torch won't accept. Heuristic: cpu / mps / cuda[:N]."""
	if device and _DEVICE_RE.fullmatch(device) is None:
		resolver.problems.append(
			f"runtime.device {device!r} is invalid; expected cpu, mps, or cuda[:N] (e.g. cuda:0)."
		)


def _check_range(value: float, low: float, high: float, name: str, resolver: _Resolver) -> None:
	if not low <= value <= high:
		resolver.problems.append(f"{name} must be in [{low}, {high}], got {value}.")


def _resolve_detector_name(resolver: _Resolver) -> DetectorChoice:
	name = resolver.required_str("detector.name")
	try:
		return DetectorChoice(name)
	except ValueError:
		if name:  # empty already reported as missing by required_str
			valid = ", ".join(choice.value for choice in DetectorChoice)
			resolver.problems.append(f"detector.name {name!r} is unknown; valid: {valid}.")
		return DetectorChoice.YOLOV8_VISDRONE


def _resolve_window_bound(resolver: _Resolver, dotted: str) -> float | None:
	"""Resolve a window bound. A required key; ``""`` means the clip's bound."""
	raw = resolver.required_str(dotted)
	if raw == "":
		return None
	try:
		return _parse_timecode(raw)
	except ValueError as exc:
		resolver.problems.append(f"{dotted}: {exc}")
		return None


def _validate_window(window: WindowConfig, resolver: _Resolver) -> None:
	if window.end_seconds is not None and window.end_seconds <= 0.0:
		resolver.problems.append("window.end must be greater than zero.")
	if (
		window.start_seconds is not None
		and window.end_seconds is not None
		and window.end_seconds <= window.start_seconds
	):
		resolver.problems.append("window.end must be after window.start.")


def _parse_timecode(value: str) -> float:
	"""Parse ``SS(.ms)``, ``MM:SS(.ms)``, or ``HH:MM:SS(.ms)`` into seconds.

	Raises ``ValueError`` on malformed input; the caller turns it into a problem.
	"""
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
