"""Run configuration: the persisted, replayable specification of one analysis.

A TraTrac run is fully described by a ``RunConfig`` — input video, detector,
calibration, tracker, orientation, export, analysis window, and run options.
Every value is mandatory: there are **no built-in defaults anywhere in the
package**. Each parameter must be supplied by a TOML config file or a CLI flag;
if neither supplies it, ``RunConfig.resolve`` fails listing exactly what is
missing. This trades typing convenience for scientific reproducibility — a
``.trj`` is reconstructable from the config that produced it, which names its
own input and output. See ``vault/19_config_file.md``.

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

from tratrac.calibration.drone_specs import known_models, lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.calibration.srt_parser import mean_altitude
from tratrac.domain.frame import VideoMetadata


class DetectorChoice(StrEnum):
	"""Available detector adapters.

	``yolov8_visdrone`` is the MVP1 emergency detector — community YOLOv8 fine-tuned
	on VisDrone, picked because COCO-pretrained RT-DETR fails on aerial inputs.
	``rt_detr`` stays available; once a fine-tuned aerial RT-DETR checkpoint exists,
	we make it the default again and drop the YOLO option (see ``vault/05_mvp1.md``).
	"""

	YOLOV8_VISDRONE = "yolov8_visdrone"
	RT_DETR = "rt_detr"


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
	"""The processed video. Per-run, but part of the persisted config so a saved
	config replays without any positional argument."""

	video: Path


@dataclass(frozen=True, slots=True)
class DetectorConfig:
	name: DetectorChoice
	checkpoint: str
	conf: float
	filename: str  # consumed only by the yolov8_visdrone adapter


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
	device: str


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
	"""GSD calibration, a one-of: either a direct ``meters_per_pixel`` or a
	``drone_model`` plus an altitude source (``altitude_m`` or an ``srt`` path).
	``resolve`` guarantees exactly one method is fully specified."""

	meters_per_pixel: float | None
	drone_model: str | None
	altitude_m: float | None
	srt: Path | None

	def resolve_scale(self, metadata: VideoMetadata) -> float:
		"""Resolve the GSD (metres per pixel). Requires video metadata for the
		image width when computing from drone geometry. May raise ``ValueError``
		from the calibration chain (e.g. an SRT with no usable altitudes)."""
		if self.meters_per_pixel is not None:
			return self.meters_per_pixel
		if self.drone_model is None:
			raise ConfigError(["calibration: no method resolved."])
		spec = lookup(self.drone_model)
		if self.altitude_m is not None:
			altitude = self.altitude_m
		elif self.srt is not None:
			altitude = mean_altitude(self.srt)
		else:
			raise ConfigError(["calibration: drone_model needs altitude_m or an srt path."])
		return ground_sample_distance(
			sensor_width_mm=spec.sensor_width_mm,
			focal_length_mm=spec.focal_length_mm,
			altitude_m=altitude,
			image_width_pixels=metadata.width,
		)


@dataclass(frozen=True, slots=True)
class EgoMotionConfig:
	"""ORB video-stabilization settings (MVP1.9, see ``vault/05_75_mvp1_9.md``).

	``enabled`` is the explicit on/off toggle (per the "off is explicit" rule). The
	ORB parameters are only meaningful — and only required by ``resolve`` — when
	``enabled`` is true; when disabled they hold ignored placeholder zeros."""

	enabled: bool
	n_features: int
	match_ratio: float
	min_matches: int
	ransac_threshold: float
	# Minimum fraction of the keyframe anchor still visible before re-anchoring
	# (see vault/05_75_mvp1_9.md). Only meaningful when ``enabled``.
	min_anchor_overlap: float


@dataclass(frozen=True, slots=True)
class TrackerConfig:
	det_thresh: float


@dataclass(frozen=True, slots=True)
class OrientationConfig:
	smoothing_window: int


@dataclass(frozen=True, slots=True)
class ExportConfig:
	out: Path
	timestep_precision: float  # 0.0 = emit every processed frame
	# Optional overlay video output (raw frame + trajectories). ``None`` = off.
	# ``video_trail`` (trail length in frames; 0 = whole path) is only meaningful —
	# and only required by ``resolve`` — when ``video_out`` is set.
	video_out: Path | None
	video_trail: int
	# Optional per-frame ego-motion transform CSV (current frame -> global). ``None``
	# = off. Only meaningful when ego-motion is enabled (``resolve`` enforces this):
	# with stabilization off every transform is the identity, so there is nothing to
	# record. See vault/05_75_mvp1_9.md.
	transform_csv: Path | None


@dataclass(frozen=True, slots=True)
class WindowConfig:
	"""Analysis window in seconds. ``None`` means the clip's natural bound."""

	start_seconds: float | None
	end_seconds: float | None


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
	"""What the run analyzes. ``exclusion_zones`` is a sidecar JSON path of
	image-space polygons whose detections are dropped before tracking (and masked
	out of ORB ego-motion features); ``None`` = no exclusions. See
	vault/21_exclusion_zones.md."""

	exclusion_zones: Path | None


@dataclass(frozen=True, slots=True)
class RunOptionsConfig:
	force: bool
	timing_csv: Path | None  # None = profiling off


@dataclass(frozen=True, slots=True)
class RunConfig:
	"""The complete, validated specification of one analysis run."""

	input: InputConfig
	detector: DetectorConfig
	runtime: RuntimeConfig
	calibration: CalibrationConfig
	ego_motion: EgoMotionConfig
	tracker: TrackerConfig
	orientation: OrientationConfig
	export: ExportConfig
	window: WindowConfig
	analysis: AnalysisConfig
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

		detector_name = _resolve_detector_name(resolver)
		checkpoint = resolver.required_str("detector.checkpoint")
		conf = resolver.required_float("detector.conf")
		_check_range(conf, 0.0, 1.0, "detector.conf", resolver)
		filename = resolver.required_str("detector.filename")

		device = resolver.required_str("runtime.device")
		_validate_device(device, resolver)

		calibration = _resolve_calibration(resolver)
		ego_motion = _resolve_ego_motion(resolver)

		det_thresh = resolver.required_float("tracker.det_thresh")
		_check_range(det_thresh, 0.0, 1.0, "tracker.det_thresh", resolver)

		smoothing_window = resolver.required_int("orientation.smoothing_window")
		if resolver.present("orientation.smoothing_window") and smoothing_window < 2:
			resolver.problems.append("orientation.smoothing_window must be >= 2.")

		out = resolver.required_path("export.out")
		timestep_precision = resolver.required_float("export.timestep_precision")
		if timestep_precision < 0.0:
			resolver.problems.append("export.timestep_precision must be >= 0 (0 = every frame).")
		video_out, video_trail = _resolve_video(resolver)
		transform_csv = resolver.toggleable_path("export.transform_csv")
		if transform_csv is not None and not ego_motion.enabled:
			resolver.problems.append(
				"export.transform_csv requires ego_motion.enabled = true; with stabilization "
				'off every transform is the identity, so there is nothing to record (use "").'
			)

		window = WindowConfig(
			start_seconds=_resolve_window_bound(resolver, "window.start"),
			end_seconds=_resolve_window_bound(resolver, "window.end"),
		)
		_validate_window(window, resolver)

		exclusion_zones = resolver.toggleable_path("analysis.exclusion_zones")

		force = resolver.required_bool("run.force")
		timing_csv = resolver.toggleable_path("run.timing_csv")

		if resolver.problems:
			raise ConfigError(resolver.problems)

		return cls(
			input=InputConfig(video=video),
			detector=DetectorConfig(
				name=detector_name, checkpoint=checkpoint, conf=conf, filename=filename
			),
			runtime=RuntimeConfig(device=device),
			calibration=calibration,
			ego_motion=ego_motion,
			tracker=TrackerConfig(det_thresh=det_thresh),
			orientation=OrientationConfig(smoothing_window=smoothing_window),
			export=ExportConfig(
				out=out,
				timestep_precision=timestep_precision,
				video_out=video_out,
				video_trail=video_trail,
				transform_csv=transform_csv,
			),
			window=window,
			analysis=AnalysisConfig(exclusion_zones=exclusion_zones),
			options=RunOptionsConfig(force=force, timing_csv=timing_csv),
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

	def required_int(self, dotted: str) -> int:
		raw = self._raw(dotted)
		if raw is _MISSING:
			self.problems.append(f"{dotted} is missing.")
			return 0
		if isinstance(raw, bool) or not isinstance(raw, int):
			self.problems.append(f"{dotted} must be an integer, got {type(raw).__name__}.")
			return 0
		return raw

	def required_bool(self, dotted: str) -> bool:
		raw = self._raw(dotted)
		if raw is _MISSING:
			self.problems.append(f"{dotted} is missing.")
			return False
		if not isinstance(raw, bool):
			self.problems.append(f"{dotted} must be true or false, got {type(raw).__name__}.")
			return False
		return raw

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

	def optional_float(self, dotted: str) -> float | None:
		raw = self._raw(dotted)
		if raw is _MISSING:
			return None
		if isinstance(raw, bool) or not isinstance(raw, int | float):
			self.problems.append(f"{dotted} must be a number, got {type(raw).__name__}.")
			return None
		return float(raw)

	def optional_str(self, dotted: str) -> str | None:
		raw = self._raw(dotted)
		if raw is _MISSING:
			return None
		if not isinstance(raw, str):
			self.problems.append(f"{dotted} must be a string, got {type(raw).__name__}.")
			return None
		return raw

	def optional_path(self, dotted: str) -> Path | None:
		raw = self._raw(dotted)
		if raw is _MISSING:
			return None
		if isinstance(raw, Path):
			return raw
		if isinstance(raw, str):
			return Path(raw) if raw else None
		self.problems.append(f"{dotted} must be a path string, got {type(raw).__name__}.")
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


def _resolve_calibration(resolver: _Resolver) -> CalibrationConfig:
	meters_per_pixel = resolver.optional_float("calibration.meters_per_pixel")
	drone_model = resolver.optional_str("calibration.drone_model")
	altitude_m = resolver.optional_float("calibration.altitude_m")
	srt = resolver.optional_path("calibration.srt")

	if meters_per_pixel is not None and drone_model:
		resolver.problems.append(
			"calibration: specify exactly one of meters_per_pixel or drone_model, not both."
		)
		return CalibrationConfig(meters_per_pixel, None, None, None)

	if meters_per_pixel is not None:
		if meters_per_pixel <= 0.0:
			resolver.problems.append("calibration.meters_per_pixel must be positive.")
		return CalibrationConfig(meters_per_pixel, None, None, None)

	if drone_model:
		if drone_model.lower() not in known_models():
			known = ", ".join(known_models())
			resolver.problems.append(
				f"calibration.drone_model {drone_model!r} is unknown; known: {known}."
			)
		altitude_ok = altitude_m is not None and altitude_m > 0.0
		if altitude_m is not None and altitude_m <= 0.0:
			resolver.problems.append("calibration.altitude_m must be positive.")
		if not altitude_ok and srt is None:
			resolver.problems.append(
				"calibration: drone_model needs altitude_m (> 0) or an srt path."
			)
		return CalibrationConfig(
			meters_per_pixel=None,
			drone_model=drone_model,
			altitude_m=altitude_m if altitude_ok else None,
			srt=srt,
		)

	resolver.problems.append(
		"calibration: set meters_per_pixel, or drone_model with altitude_m or an srt path."
	)
	return CalibrationConfig(None, None, None, None)


def _resolve_ego_motion(resolver: _Resolver) -> EgoMotionConfig:
	"""Resolve the ORB stabilization config. ``enabled`` is always required; the ORB
	parameters are required (and range-checked) only when stabilization is on."""
	enabled = resolver.required_bool("ego_motion.enabled")
	if not enabled:
		# Disabled (or the toggle itself was missing — already reported): the ORB
		# parameters are irrelevant. Placeholder zeros; never read downstream.
		return EgoMotionConfig(
			enabled=False,
			n_features=0,
			match_ratio=0.0,
			min_matches=0,
			ransac_threshold=0.0,
			min_anchor_overlap=0.0,
		)

	n_features = resolver.required_int("ego_motion.n_features")
	if resolver.present("ego_motion.n_features") and n_features <= 0:
		resolver.problems.append("ego_motion.n_features must be positive.")
	match_ratio = resolver.required_float("ego_motion.match_ratio")
	if resolver.present("ego_motion.match_ratio") and not 0.0 < match_ratio < 1.0:
		resolver.problems.append("ego_motion.match_ratio must be in (0, 1).")
	min_matches = resolver.required_int("ego_motion.min_matches")
	if resolver.present("ego_motion.min_matches") and min_matches < 2:
		resolver.problems.append("ego_motion.min_matches must be >= 2.")
	ransac_threshold = resolver.required_float("ego_motion.ransac_threshold")
	if resolver.present("ego_motion.ransac_threshold") and ransac_threshold <= 0.0:
		resolver.problems.append("ego_motion.ransac_threshold must be positive.")
	min_anchor_overlap = resolver.required_float("ego_motion.min_anchor_overlap")
	if resolver.present("ego_motion.min_anchor_overlap") and not 0.0 < min_anchor_overlap < 1.0:
		resolver.problems.append("ego_motion.min_anchor_overlap must be in (0, 1).")
	return EgoMotionConfig(
		enabled=True,
		n_features=n_features,
		match_ratio=match_ratio,
		min_matches=min_matches,
		ransac_threshold=ransac_threshold,
		min_anchor_overlap=min_anchor_overlap,
	)


def _resolve_video(resolver: _Resolver) -> tuple[Path | None, int]:
	"""Resolve the optional overlay-video output. ``export.video_out`` is a required
	toggleable key ("" = off, like ``run.timing_csv``); ``export.video_trail`` is
	required (and range-checked) only when the video output is on, mirroring how the
	ORB parameters are required only when ego-motion is enabled."""
	video_out = resolver.toggleable_path("export.video_out")
	if video_out is None:
		# Off (or the key was missing — already reported). The trail length is
		# irrelevant; a placeholder zero that is never read downstream.
		return None, 0
	video_trail = resolver.required_int("export.video_trail")
	if resolver.present("export.video_trail") and video_trail < 0:
		resolver.problems.append("export.video_trail must be >= 0 (0 = whole path).")
	return video_out, video_trail


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
