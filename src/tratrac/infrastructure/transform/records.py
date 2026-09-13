"""The shared on-disk shape for the unified transform sidecar.

Every geometric transform in the system — ego-motion, GSD scale, and
world-projection homography — is a **row**: `(frame_index, zone, function)`.
`frame_index` is the primary key (the file is one JSON object per line, in
frame order); `zone` is always a concrete polygon (the full-canvas rectangle
when nothing narrower applies, never a null/optional field); `function` is the
type-specific payload. Multi-frame applicability (a constant scale, a
frame-invariant per-plane homography) is materialized as repeated rows written
once at build time, not a matching rule resolved at read time — see
`src/tratrac/infrastructure/video/EGO_MOTION.md`.

Two objects per transformation kind, single-responsibility each:

* A `<Kind>Function` — the math only (`apply`/`reverse`), never touches JSON.
* A `<Kind>Codec` — serialization only (`encode`/`decode`), never touches a point.

A registry maps each kind's `type` tag to its codec, so decoding a line is a
dict lookup, never a hand-written branch. There is only one `homography` kind:
what used to be a "per-anchor" (whole-scene) vs. "per-plane" (grade-separated)
distinction is nothing more than what `zone` a given row carries.

Also carries the staged-write-then-`Path.replace` helpers every sidecar uses
so a reader of the canonical path never observes a half-written file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from tratrac.domain.geometry import Point2D, Transform2D


class TransformFunction(Protocol):
	"""The math a transform row applies: maps one point to another. No JSON here."""

	def apply(self, point: Point2D) -> Point2D: ...

	@property
	def preserves_shape(self) -> bool:
		"""Whether this function preserves angles and length ratios (any similarity
		transform: identity, uniform scale, translate+rotate+scale) as opposed to a
		general homography, whose perspective distortion can shear/rotate differently
		across the image.

		A capability the function declares about *itself* — the same pattern
		``InvertibleTransformFunction`` already uses — so a caller deciding whether an
		oriented (OBB) size/heading survives being carried through the transform never
		needs to know or check which concrete kind it's holding.
		"""
		...


@runtime_checkable
class InvertibleTransformFunction(TransformFunction, Protocol):
	"""A ``TransformFunction`` that can also map a point back the other way."""

	def reverse(self, point: Point2D) -> Point2D: ...


@dataclass(frozen=True, slots=True)
class IdentityFunction:
	"""The no-op: pass the point through unchanged."""

	def apply(self, point: Point2D) -> Point2D:
		return point

	def reverse(self, point: Point2D) -> Point2D:
		return point

	@property
	def preserves_shape(self) -> bool:
		return True


@dataclass(frozen=True, slots=True)
class ScaleFunction:
	"""A constant isotropic scale (GSD calibration)."""

	factor: float

	def __post_init__(self) -> None:
		if self.factor <= 0.0:
			raise ValueError(f"ScaleFunction factor must be positive, got {self.factor}.")

	def apply(self, point: Point2D) -> Point2D:
		return Point2D(point.x * self.factor, point.y * self.factor)

	def reverse(self, point: Point2D) -> Point2D:
		return Point2D(point.x / self.factor, point.y / self.factor)

	@property
	def preserves_shape(self) -> bool:
		return True


@dataclass(frozen=True, slots=True)
class SimilarityFunction:
	"""A 4-DOF similarity (ego-motion): translate + rotate + uniform scale."""

	transform: Transform2D

	def apply(self, point: Point2D) -> Point2D:
		return self.transform.apply(point)

	def reverse(self, point: Point2D) -> Point2D:
		return self.transform.inverse().apply(point)

	@property
	def preserves_shape(self) -> bool:
		return True


def _homography_apply(matrix: NDArray[np.float64], point: Point2D) -> Point2D:
	projected = matrix @ np.array([point.x, point.y, 1.0])
	w = float(projected[2])
	if w == 0.0:
		raise ValueError("homography mapped a point to infinity (w = 0).")
	return Point2D(float(projected[0]) / w, float(projected[1]) / w)


@dataclass(frozen=True, slots=True)
class HomographyFunction:
	"""A 3x3 projective transform (world projection), row-major, 9 entries.

	One kind covers both the whole-scene case (a row whose ``zone`` is the full
	canvas) and a grade-separated scene's per-plane case (a row whose ``zone`` is
	that plane's polygon) — see the module docstring.
	"""

	matrix: tuple[float, ...]

	def __post_init__(self) -> None:
		if len(self.matrix) != 9:
			raise ValueError(f"HomographyFunction matrix needs 9 entries, got {len(self.matrix)}.")

	def _as_array(self) -> NDArray[np.float64]:
		return np.array(self.matrix, dtype=np.float64).reshape(3, 3)

	def apply(self, point: Point2D) -> Point2D:
		return _homography_apply(self._as_array(), point)

	def reverse(self, point: Point2D) -> Point2D:
		inverse = np.linalg.inv(self._as_array()).astype(np.float64)
		return _homography_apply(inverse, point)

	@property
	def preserves_shape(self) -> bool:
		return False


class TransformCodec(Protocol):
	"""Serialization only for one transformation kind. Never touches a point."""

	type_tag: str

	def encode(self, function: Any) -> dict[str, Any]: ...

	def decode(self, raw: dict[str, Any]) -> TransformFunction: ...


class IdentityCodec:
	type_tag = "identity"

	def encode(self, function: IdentityFunction) -> dict[str, Any]:
		del function
		return {}

	def decode(self, raw: dict[str, Any]) -> IdentityFunction:
		del raw
		return IdentityFunction()


class ScaleCodec:
	type_tag = "scale"

	def encode(self, function: ScaleFunction) -> dict[str, Any]:
		return {"factor": function.factor}

	def decode(self, raw: dict[str, Any]) -> ScaleFunction:
		try:
			return ScaleFunction(factor=float(raw["factor"]))
		except (KeyError, TypeError, ValueError) as exc:
			raise ValueError(f"malformed scale function: {exc}") from exc


class SimilarityCodec:
	type_tag = "ego_motion"

	def encode(self, function: SimilarityFunction) -> dict[str, Any]:
		t = function.transform
		return {"a": t.a, "b": t.b, "c": t.c, "d": t.d, "tx": t.tx, "ty": t.ty}

	def decode(self, raw: dict[str, Any]) -> SimilarityFunction:
		try:
			return SimilarityFunction(
				Transform2D(
					a=float(raw["a"]),
					b=float(raw["b"]),
					c=float(raw["c"]),
					d=float(raw["d"]),
					tx=float(raw["tx"]),
					ty=float(raw["ty"]),
				)
			)
		except (KeyError, TypeError, ValueError) as exc:
			raise ValueError(f"malformed ego_motion function: {exc}") from exc


class HomographyCodec:
	type_tag = "homography"

	def encode(self, function: HomographyFunction) -> dict[str, Any]:
		return {"matrix": list(function.matrix)}

	def decode(self, raw: dict[str, Any]) -> HomographyFunction:
		try:
			matrix = tuple(float(v) for v in raw["matrix"])
		except (KeyError, TypeError, ValueError) as exc:
			raise ValueError(f"malformed homography function: {exc}") from exc
		return HomographyFunction(matrix=matrix)


# Registered by concrete Function type (for encoding, where the caller already has an
# instance) and by wire tag (for decoding, where the caller only has a "type" string).
# Adding a fifth transformation kind means adding one Function + one Codec + one entry
# here -- nothing else in this module branches on kind.
_CODECS_BY_TYPE: dict[type, TransformCodec] = {
	IdentityFunction: IdentityCodec(),
	ScaleFunction: ScaleCodec(),
	SimilarityFunction: SimilarityCodec(),
	HomographyFunction: HomographyCodec(),
}
_CODECS_BY_TAG: dict[str, TransformCodec] = {
	codec.type_tag: codec for codec in _CODECS_BY_TYPE.values()
}


@dataclass(frozen=True, slots=True)
class TransformRow:
	"""One row: where/when a ``function`` applies. No math, no serialization of its own.

	``zone`` is always a concrete polygon (never ``None``) -- the full-canvas
	rectangle when the function applies everywhere spatially.
	"""

	frame_index: int
	zone: tuple[Point2D, ...]
	function: TransformFunction


def encode_row(row: TransformRow) -> dict[str, Any]:
	"""Serialize one row to a JSON-able dict, tagged by ``"type"``."""
	codec = _CODECS_BY_TYPE.get(type(row.function))
	if codec is None:
		raise ValueError(f"no codec registered for {type(row.function).__name__!r}.")
	return {
		"frame_index": row.frame_index,
		"type": codec.type_tag,
		"zone": [[p.x, p.y] for p in row.zone],
		"function": codec.encode(row.function),
	}


def decode_row(raw: dict[str, Any]) -> TransformRow:
	"""Parse one row back. Raises ``ValueError`` on an unknown type or malformed fields."""
	try:
		frame_index = int(raw["frame_index"])
		type_tag = raw["type"]
		zone_raw = raw["zone"]
		function_raw = raw["function"]
	except (KeyError, TypeError) as exc:
		raise ValueError(f"malformed transform row: {exc}") from exc
	codec = _CODECS_BY_TAG.get(type_tag)
	if codec is None:
		raise ValueError(f"unknown transform type: {type_tag!r}")
	if not isinstance(zone_raw, list) or len(zone_raw) < 3:
		raise ValueError(f"zone must be an array of at least 3 [x, y] pairs, got {zone_raw!r}.")
	try:
		zone = tuple(Point2D(float(v[0]), float(v[1])) for v in zone_raw)
	except (TypeError, IndexError, ValueError) as exc:
		raise ValueError(f"zone must be an array of [x, y] pairs: {exc}") from exc
	if not isinstance(function_raw, dict):
		raise ValueError(f'"function" must be an object, got {function_raw!r}.')
	function = codec.decode(function_raw)
	return TransformRow(frame_index=frame_index, zone=zone, function=function)


def write_jsonl(path: Path, rows: Iterable[TransformRow]) -> None:
	"""Write every row to ``path`` as JSON Lines, one shot (small/known-complete files).
	Streaming writers append directly via ``json.dumps``/``encode_row`` themselves (see
	``infrastructure/transform/sink.py``)."""
	with path.open("w") as handle:
		for row in rows:
			handle.write(json.dumps(encode_row(row)))
			handle.write("\n")


def read_jsonl(path: Path) -> list[TransformRow]:
	"""Read a JSON Lines transform-row file back.

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
	rows: list[TransformRow] = []
	for index, raw_line in enumerate(lines):
		line = raw_line.strip()
		if not line:
			continue
		try:
			rows.append(decode_row(json.loads(line)))
		except (json.JSONDecodeError, ValueError) as exc:
			if index == len(lines) - 1:
				break
			raise ValueError(f"{path}: malformed transform row on line {index + 1}: {exc}") from exc
	return rows


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
