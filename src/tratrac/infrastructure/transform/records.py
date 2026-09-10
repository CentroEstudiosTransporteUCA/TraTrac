"""The shared on-disk shape for every ``CoordinateTransform`` sidecar.

One tagged union, three record kinds — ``ScaleRecord`` (GSD metric scale),
``SimilarityRecord`` (an ego-motion/anchor-pose ``Transform2D`` anchored to one
frame), ``HomographyRecord`` (one fitted world-projection homography) — written as
JSON Lines. Every sidecar that persists a transform (the ego-motion table, the
anchor manifest's poses, the scale file) is built on this module instead of each
inventing its own JSON/CSV shape, so there is exactly one wire format for "a
transform" in the system. See ``src/tratrac/infrastructure/transform/TRANSFORM_SINK.md``.

Also carries the staged-write-then-``os.replace`` helpers every one-shot-or-streamed
sidecar uses so a reader of the canonical path never observes a half-written file:
either it's not there yet, or it's the complete, finished artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tratrac.domain.geometry import Transform2D


@dataclass(frozen=True, slots=True)
class ScaleRecord:
	"""A constant metres-per-pixel factor (GSD calibration)."""

	scale: float


@dataclass(frozen=True, slots=True)
class SimilarityRecord:
	"""A ``Transform2D`` (ego-motion or anchor pose) anchored to one frame."""

	frame_index: int
	transform: Transform2D


@dataclass(frozen=True, slots=True)
class HomographyRecord:
	"""One fitted world-projection homography (3x3, row-major).

	``frame_index`` (the anchor it was fitted on) and ``plane_id`` (the elevation
	plane it was fitted for) are each optional and mutually exclusive in practice —
	``PerAnchorTransform`` uses the former, ``MultiPlaneTransform`` the latter — but
	both live on one record type rather than two near-identical ones.
	"""

	frame_index: int | None
	plane_id: int | None
	matrix: tuple[float, ...]


TransformRecord = ScaleRecord | SimilarityRecord | HomographyRecord


def to_json(record: TransformRecord) -> dict[str, Any]:
	"""Serialize one record to a JSON-able dict, tagged by ``"type"``."""
	if isinstance(record, ScaleRecord):
		return {"type": "scale", "scale": record.scale}
	if isinstance(record, SimilarityRecord):
		t = record.transform
		return {
			"type": "similarity",
			"frame_index": record.frame_index,
			"a": t.a,
			"b": t.b,
			"c": t.c,
			"d": t.d,
			"tx": t.tx,
			"ty": t.ty,
		}
	return {
		"type": "homography",
		"frame_index": record.frame_index,
		"plane_id": record.plane_id,
		"matrix": list(record.matrix),
	}


def from_json(raw: dict[str, Any]) -> TransformRecord:
	"""Parse one record back. Raises ``ValueError`` on an unknown type or malformed fields."""
	kind = raw.get("type")
	try:
		if kind == "scale":
			return ScaleRecord(scale=float(raw["scale"]))
		if kind == "similarity":
			return SimilarityRecord(
				frame_index=int(raw["frame_index"]),
				transform=Transform2D(
					a=float(raw["a"]),
					b=float(raw["b"]),
					tx=float(raw["tx"]),
					c=float(raw["c"]),
					d=float(raw["d"]),
					ty=float(raw["ty"]),
				),
			)
		if kind == "homography":
			frame_index = raw.get("frame_index")
			plane_id = raw.get("plane_id")
			return HomographyRecord(
				frame_index=int(frame_index) if frame_index is not None else None,
				plane_id=int(plane_id) if plane_id is not None else None,
				matrix=tuple(float(v) for v in raw["matrix"]),
			)
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError(f"malformed {kind!r} transform record: {exc}") from exc
	raise ValueError(f"unknown transform record type: {kind!r}")


def write_jsonl(path: Path, records: Iterable[TransformRecord]) -> None:
	"""Write every record to ``path`` as JSON Lines, one shot (small/known-complete files —
	the scale sidecar, the anchor manifest's poses). Streaming writers append directly via
	``json.dumps``/``to_json`` themselves (see ``infrastructure/transform/sink.py``)."""
	with path.open("w") as handle:
		for record in records:
			handle.write(json.dumps(to_json(record)))
			handle.write("\n")


def read_jsonl(path: Path) -> list[TransformRecord]:
	"""Read a JSON Lines transform-record file back.

	Raises ``ValueError`` (path-wrapped) on a missing file or a malformed *interior*
	line. Tolerates a torn trailing line — a crash mid-write can leave one incomplete
	final line, but every prior line is still a complete, self-contained record — by
	dropping it rather than failing the whole read.
	"""
	try:
		with path.open() as handle:
			lines = handle.readlines()
	except OSError as exc:
		raise ValueError(f"{path} is not a readable transform-record file: {exc}") from exc
	records: list[TransformRecord] = []
	for index, raw_line in enumerate(lines):
		line = raw_line.strip()
		if not line:
			continue
		try:
			records.append(from_json(json.loads(line)))
		except (json.JSONDecodeError, ValueError) as exc:
			if index == len(lines) - 1:
				break
			raise ValueError(
				f"{path}: malformed transform record on line {index + 1}: {exc}"
			) from exc
	return records


def staging_path(path: Path) -> Path:
	"""The path to write to before atomically publishing as ``path``.

	Any reader of ``path`` itself either finds nothing (still being written, or a
	crashed run) or the complete file — never a half-written version under the name
	anything treats as authoritative.
	"""
	return path.with_name(path.name + ".partial")


def publish(staging: Path, path: Path) -> None:
	"""Atomically publish a completed ``staging`` file as ``path`` (``Path.replace``)."""
	staging.replace(path)
