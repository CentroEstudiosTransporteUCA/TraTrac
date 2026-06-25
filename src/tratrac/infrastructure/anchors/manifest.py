"""Keyframe-anchor manifest: the operator-facing index of frames to draw zones on.

The run emits each ORB keyframe **anchor** (see vault/21_exclusion_zones.md) as a PNG plus
a manifest row carrying the frame index, the global ego-motion pose, and the image name.
The manifest is self-sufficient for exclusion: the post-process pass reads each anchor's
pose from here to map zones authored on that anchor into the global frame — no separate
transform CSV needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tratrac.domain.geometry import Transform2D


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
	"""One keyframe anchor: its index, global pose, and exported image filename."""

	frame_index: int
	pose: Transform2D
	image_name: str


def write_manifest(path: Path, references: list[ReferenceFrame], *, video: str) -> None:
	"""Write the anchor manifest as JSON."""
	document = {
		"video": video,
		"reference_frames": [
			{
				"frame_index": ref.frame_index,
				"image": ref.image_name,
				"pose": _pose_dict(ref.pose),
			}
			for ref in references
		],
	}
	with path.open("w") as handle:
		json.dump(document, handle, indent=2)


def read_manifest(path: Path) -> list[ReferenceFrame]:
	"""Read an anchor manifest back into ``ReferenceFrame``s.

	Raises ``FileNotFoundError`` if absent and ``ValueError`` on malformed content
	(re-wrapped with the file path).
	"""
	try:
		with path.open() as handle:
			document: Any = json.load(handle)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{path} is not valid JSON: {exc}") from exc
	if not isinstance(document, dict) or "reference_frames" not in document:
		raise ValueError(f'{path} must be an anchor manifest with a "reference_frames" array.')
	try:
		return [_parse_reference(raw) for raw in document["reference_frames"]]
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError(f"{path} is not a valid anchor manifest: {exc}") from exc


def _parse_reference(raw: Any) -> ReferenceFrame:
	pose = raw["pose"]
	return ReferenceFrame(
		frame_index=int(raw["frame_index"]),
		pose=Transform2D(
			a=float(pose["a"]),
			b=float(pose["b"]),
			tx=float(pose["tx"]),
			c=float(pose["c"]),
			d=float(pose["d"]),
			ty=float(pose["ty"]),
		),
		image_name=str(raw["image"]),
	)


def _pose_dict(pose: Transform2D) -> dict[str, float]:
	return {"a": pose.a, "b": pose.b, "tx": pose.tx, "c": pose.c, "d": pose.d, "ty": pose.ty}
