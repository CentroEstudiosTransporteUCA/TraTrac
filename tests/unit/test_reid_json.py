"""Tests for the reid_merge.json reader/writer. Pure stdlib — no cv2/model downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tratrac.application.reid_merge import MergeCandidate
from tratrac.infrastructure.reid.json import load_reid_merge, save_reid_merge


class TestLoadReidMerge:
	def test_reads_merges_keyed_by_int(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		path.write_text(json.dumps({"merges": {"2": 1, "3": 1}}))
		assert load_reid_merge(path) == {2: 1, 3: 1}

	def test_missing_candidates_is_fine(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		path.write_text(json.dumps({"merges": {}}))
		assert load_reid_merge(path) == {}

	def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		path.write_text(json.dumps({"other": {}}))
		with pytest.raises(ValueError, match="merges"):
			load_reid_merge(path)

	def test_malformed_json_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		path.write_text("{not json")
		with pytest.raises(ValueError, match="not valid JSON"):
			load_reid_merge(path)

	def test_non_integer_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		path.write_text(json.dumps({"merges": {"abc": 1}}))
		with pytest.raises(ValueError, match="integer"):
			load_reid_merge(path)

	def test_non_integer_value_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		path.write_text(json.dumps({"merges": {"2": "one"}}))
		with pytest.raises(ValueError, match="integer"):
			load_reid_merge(path)


class TestSaveReidMerge:
	def test_round_trips(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		save_reid_merge(
			path,
			{2: 1, 3: 1},
			candidates=[MergeCandidate(from_track_id=1, to_track_id=2, gap_seconds=1.5, score=0.9)],
		)
		assert load_reid_merge(path) == {2: 1, 3: 1}
		document = json.loads(path.read_text())
		assert document["candidates"][0]["from_track_id"] == 1

	def test_empty_merges_round_trips(self, tmp_path: Path) -> None:
		path = tmp_path / "merge.json"
		save_reid_merge(path, {})
		assert load_reid_merge(path) == {}
