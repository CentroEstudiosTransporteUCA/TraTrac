"""Scale sidecar: the GSD metres-per-pixel calibration, persisted as its own file.

The "thin layer" between calibration config (a direct ``meters_per_pixel``, or a
drone-model + altitude resolved via the GSD formula, see ``application/config.py``'s
``CalibrationConfig.resolve_scale``) and the generic transform-record schema
(``infrastructure/transform/records.py``): ``cli.py`` resolves the scale once, then
calls ``write_scale`` instead of stamping it into the Parquet track record's schema
metadata. See ``src/tratrac/calibration/GSD_CALIBRATION.md``.
"""

from __future__ import annotations

from pathlib import Path

from tratrac.application.coordinate_transforms import ScaleTransform
from tratrac.infrastructure.transform.records import (
	ScaleRecord,
	publish,
	read_jsonl,
	staging_path,
	write_jsonl,
)


def write_scale(path: Path, scale: float) -> None:
	"""Write the run's GSD metres-per-pixel scale as its own sidecar file.

	Written to a staging path and published in one step: the value is fully known
	before this is called (resolved from config before the frame loop starts), so
	there is no partial-write window to protect against — but every transform
	sidecar uses the same staged-write-then-publish shape for consistency.
	"""
	staging = staging_path(path)
	write_jsonl(staging, [ScaleRecord(scale)])
	publish(staging, path)


def read_scale(path: Path) -> ScaleTransform:
	"""Read the scale sidecar back into a ``ScaleTransform``.

	Raises ``ValueError`` (path-wrapped) on a missing/malformed file or a file that
	doesn't contain exactly one scale record.
	"""
	records = read_jsonl(path)
	if len(records) != 1 or not isinstance(records[0], ScaleRecord):
		raise ValueError(f"{path} must contain exactly one scale record.")
	return ScaleTransform(records[0].scale)
