"""Tests for the scout pass: the manifest writer and the runner over a fake source.

cv2/numpy only — no real video file, no model downloads. See
vault/21_exclusion_zones.md.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import Transform2D
from tratrac.infrastructure.scout.manifest import ReferenceFrame, write_manifest
from tratrac.infrastructure.scout.runner import run_scout


def _textured(seed: int) -> NDArray[np.uint8]:
	rng = np.random.default_rng(seed)
	return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


def _shifted(image: NDArray[np.uint8], tx: float) -> NDArray[np.uint8]:
	import cv2

	matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, 0.0]], dtype=np.float64)
	h, w = image.shape[:2]
	out: NDArray[np.uint8] = cv2.warpAffine(image, matrix, (w, h))
	return out


class _FakeVideoSource:
	"""A VideoSource yielding a fixed list of frames; context-manageable."""

	def __init__(self, frames: list[Frame]) -> None:
		self._frames = frames

	@property
	def metadata(self) -> VideoMetadata:
		return VideoMetadata(width=200, height=200, fps=30.0, total_frames=len(self._frames))

	def frames(self) -> Iterator[Frame]:
		yield from self._frames

	def __enter__(self) -> _FakeVideoSource:
		return self

	def __exit__(self, *exc: object) -> None:
		return None


class TestWriteManifest:
	def test_writes_reference_frames(self, tmp_path: Path) -> None:
		refs = [ReferenceFrame(0, Transform2D.identity(), "frame_0.png")]
		path = tmp_path / "refs_manifest.json"
		write_manifest(path, refs, video="clip.mp4", transforms_name="transforms.csv")
		doc = json.loads(path.read_text())
		assert doc["video"] == "clip.mp4"
		assert doc["transforms"] == "transforms.csv"
		assert doc["reference_frames"][0]["frame_index"] == 0
		assert doc["reference_frames"][0]["image"] == "frame_0.png"
		assert doc["reference_frames"][0]["pose"]["a"] == 1.0


class TestRunScout:
	def test_writes_transforms_manifest_and_pngs(self, tmp_path: Path) -> None:
		base = _textured(0)
		# Frame 0 anchors (identity); a large shift under a high overlap threshold
		# forces a re-anchor on frame 1.
		frames = [
			Frame(index=0, pixels=base),
			Frame(index=1, pixels=_shifted(base, 40.0)),
		]
		out_dir = tmp_path / "refs"
		refs = run_scout(
			_FakeVideoSource(frames),
			out_dir,
			n_features=2000,
			match_ratio=0.75,
			min_matches=10,
			ransac_threshold=3.0,
			min_anchor_overlap=0.99,
			video_label="fake.mp4",
		)

		# At least the frame-0 anchor; a PNG and manifest entry per reference frame.
		assert refs[0].frame_index == 0
		assert (out_dir / "frame_0.png").is_file()
		manifest = json.loads((out_dir / "refs_manifest.json").read_text())
		assert [r["frame_index"] for r in manifest["reference_frames"]] == [
			r.frame_index for r in refs
		]

		# transforms.csv has one row per processed frame.
		with (out_dir / "transforms.csv").open() as handle:
			rows = list(csv.DictReader(handle))
		assert [int(row["frame"]) for row in rows] == [0, 1]

	def test_injected_image_writer_is_used(self, tmp_path: Path) -> None:
		written: list[Path] = []
		run_scout(
			_FakeVideoSource([Frame(index=0, pixels=_textured(1))]),
			tmp_path / "refs",
			n_features=2000,
			match_ratio=0.75,
			min_matches=10,
			ransac_threshold=3.0,
			min_anchor_overlap=0.6,
			video_label="fake.mp4",
			image_writer=lambda path, _pixels: written.append(path),
		)
		assert written == [tmp_path / "refs" / "frame_0.png"]
