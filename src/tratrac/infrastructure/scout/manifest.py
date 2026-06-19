"""Reference-frame manifest emitted by the scout pass.

The scout discovers the ORB keyframe anchors of a clip — the frames an operator
draws exclusion zones on (see vault/21_exclusion_zones.md). For each it records the
frame index, the global ego-motion pose, and the PNG written for drawing. The
manifest is operator-facing JSON; the real run reads poses from the transform CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tratrac.domain.geometry import Transform2D


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
	"""One keyframe anchor: its index, global pose, and exported image filename."""

	frame_index: int
	pose: Transform2D
	image_name: str


def write_manifest(
	path: Path,
	references: list[ReferenceFrame],
	*,
	video: str,
	transforms_name: str,
) -> None:
	"""Write the reference-frame manifest as JSON."""
	document = {
		"video": video,
		"transforms": transforms_name,
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


def _pose_dict(pose: Transform2D) -> dict[str, float]:
	return {
		"a": pose.a,
		"b": pose.b,
		"tx": pose.tx,
		"c": pose.c,
		"d": pose.d,
		"ty": pose.ty,
	}
