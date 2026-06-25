"""Tests for keyframe-anchor emission: manifest round-trip, the PNG sink, and the
recording decorator. cv2 is replaced by an injected image writer. See vault/21."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.infrastructure.anchors.manifest import ReferenceFrame, read_manifest, write_manifest
from tratrac.infrastructure.anchors.recording import AnchorRecordingEgoMotionEstimator
from tratrac.infrastructure.anchors.sink import AnchorManifestSink

_POSE = Transform2D(a=1.0, b=0.0, tx=30.0, c=0.0, d=1.0, ty=-12.0)


def _frame(index: int) -> Frame:
	return Frame(index=index, pixels=np.zeros((4, 4, 3), dtype=np.uint8))


class TestManifest:
	def test_round_trips(self, tmp_path: Path) -> None:
		path = tmp_path / "manifest.json"
		refs = [
			ReferenceFrame(5, _POSE, "frame_5.png"),
			ReferenceFrame(12, Transform2D.identity(), "frame_12.png"),
		]
		write_manifest(path, refs, video="clip.mp4")
		back = read_manifest(path)
		assert back == refs

	def test_rejects_non_manifest_json(self, tmp_path: Path) -> None:
		path = tmp_path / "manifest.json"
		path.write_text('{"not": "a manifest"}')
		with pytest.raises(ValueError, match="reference_frames"):
			read_manifest(path)


class TestAnchorManifestSink:
	def test_writes_one_png_per_anchor_and_a_manifest(self, tmp_path: Path) -> None:
		written: list[tuple[str, tuple[int, ...]]] = []

		def fake_writer(path: Path, pixels: NDArray[np.uint8]) -> None:
			written.append((path.name, pixels.shape))

		out_dir = tmp_path / "anchors"
		with AnchorManifestSink(out_dir, video_label="clip.mp4", image_writer=fake_writer) as sink:
			sink.record(_frame(5), _POSE)
			sink.record(_frame(12), Transform2D.identity())

		assert [name for name, _ in written] == ["frame_5.png", "frame_12.png"]
		refs = read_manifest(out_dir / "manifest.json")
		assert [(r.frame_index, r.image_name) for r in refs] == [
			(5, "frame_5.png"),
			(12, "frame_12.png"),
		]
		assert refs[0].pose == _POSE


class _FakeOrb:
	"""Stand-in estimator that fires the anchor queue on selected frames (like the ORB observer)."""

	def __init__(self, pending: list[Transform2D], anchor_at: set[int]) -> None:
		self._pending = pending
		self._anchor_at = anchor_at

	def estimate(self, frame: Frame) -> Transform2D:
		if frame.index in self._anchor_at:
			self._pending.append(_POSE)
		return Transform2D.identity()


class _RecordingSink:
	def __init__(self) -> None:
		self.calls: list[tuple[int, Transform2D]] = []

	def record(self, frame: Frame, pose: Transform2D) -> None:
		self.calls.append((frame.index, pose))

	def __enter__(self) -> _RecordingSink:
		return self

	def __exit__(self, *args: object) -> None:
		return None


class TestAnchorRecording:
	def test_tees_only_the_frames_that_reanchor(self) -> None:
		pending: list[Transform2D] = []
		sink = _RecordingSink()
		estimator = AnchorRecordingEgoMotionEstimator(_FakeOrb(pending, {0, 3}), pending, sink)
		for i in range(5):
			estimator.estimate(_frame(i))
		assert [index for index, _ in sink.calls] == [0, 3]
		assert sink.calls[0][1] == _POSE
