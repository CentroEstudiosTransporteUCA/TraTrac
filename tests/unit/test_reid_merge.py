"""Tests for the ReID merge-decision stage: motion-plausibility gating, appearance scoring,
and greedy 1:1 resolution. Pure — no video/GPU/embeddings model, matching
docs/IMPLEMENTATION_PLAN.md's Group D2 test-file map entry."""

from __future__ import annotations

import pytest

from tratrac.application.kalman import SmoothedSample
from tratrac.application.reid_merge import (
	MergeCandidate,
	TrackFragment,
	candidate_pairs,
	resolve_merges,
)
from tratrac.domain.geometry import Point2D

_STATIONARY = SmoothedSample(px=0.0, py=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0)


def _fragment(
	track_id: int,
	*,
	start_time: float,
	end_time: float,
	start_position: Point2D = Point2D(0.0, 0.0),
	end_state: SmoothedSample = _STATIONARY,
	embedding: tuple[float, ...] = (1.0, 0.0),
) -> TrackFragment:
	return TrackFragment(
		track_id=track_id,
		start_time=start_time,
		end_time=end_time,
		start_position=start_position,
		end_state=end_state,
		embedding=embedding,
	)


_GATE_KWARGS = {
	"max_gap_seconds": 5.0,
	"pos_noise": 2.0,
	"jerk": 1.0,
	"seed_vel_std": 1.0,
	"seed_accel_std": 1.0,
	"max_sigma": 3.0,
}


class TestTrackFragment:
	def test_end_before_start_raises(self) -> None:
		with pytest.raises(ValueError, match="end_time"):
			_fragment(1, start_time=5.0, end_time=1.0)

	def test_empty_embedding_raises(self) -> None:
		with pytest.raises(ValueError, match="embedding"):
			_fragment(1, start_time=0.0, end_time=1.0, embedding=())


class TestCandidatePairs:
	def test_plausible_nearby_reappearance_is_a_candidate(self) -> None:
		earlier = _fragment(1, start_time=0.0, end_time=1.0, embedding=(1.0, 0.0))
		later = _fragment(
			2, start_time=2.0, end_time=3.0, start_position=Point2D(0.5, 0.0), embedding=(1.0, 0.0)
		)
		[candidate] = candidate_pairs([earlier, later], **_GATE_KWARGS)
		assert candidate.from_track_id == 1
		assert candidate.to_track_id == 2
		assert candidate.gap_seconds == pytest.approx(1.0)
		assert candidate.score == pytest.approx(1.0)  # identical embeddings

	def test_gap_beyond_max_is_excluded(self) -> None:
		earlier = _fragment(1, start_time=0.0, end_time=1.0)
		later = _fragment(2, start_time=100.0, end_time=101.0)
		assert candidate_pairs([earlier, later], **_GATE_KWARGS) == []

	def test_later_starting_before_earlier_ends_is_excluded(self) -> None:
		earlier = _fragment(1, start_time=0.0, end_time=5.0)
		later = _fragment(2, start_time=1.0, end_time=6.0)  # overlaps, doesn't reappear after
		assert candidate_pairs([earlier, later], **_GATE_KWARGS) == []

	def test_same_track_id_is_never_paired_with_itself(self) -> None:
		fragment = _fragment(1, start_time=0.0, end_time=1.0)
		assert candidate_pairs([fragment], **_GATE_KWARGS) == []

	def test_motion_implausible_reappearance_is_gated_out(self) -> None:
		# Stationary at the origin; "reappears" 500px away a moment later -> not plausible,
		# even with an identical embedding.
		earlier = _fragment(1, start_time=0.0, end_time=1.0, embedding=(1.0, 0.0))
		later = _fragment(
			2,
			start_time=1.5,
			end_time=2.5,
			start_position=Point2D(500.0, 500.0),
			embedding=(1.0, 0.0),
		)
		assert candidate_pairs([earlier, later], **_GATE_KWARGS) == []

	def test_moving_fragment_extrapolates_before_gating(self) -> None:
		moving = SmoothedSample(px=0.0, py=0.0, vx=10.0, vy=0.0, ax=0.0, ay=0.0)
		earlier = _fragment(1, start_time=0.0, end_time=1.0, end_state=moving)
		# 2s later at 10px/s -> expected near x=20; a reappearance there is plausible even
		# though it's far from the fragment's *own* start position.
		later = _fragment(2, start_time=3.0, end_time=4.0, start_position=Point2D(20.0, 0.0))
		[candidate] = candidate_pairs([earlier, later], **_GATE_KWARGS)
		assert candidate.to_track_id == 2

	def test_mismatched_embedding_length_raises(self) -> None:
		earlier = _fragment(1, start_time=0.0, end_time=1.0, embedding=(1.0, 0.0))
		later = _fragment(
			2,
			start_time=1.5,
			end_time=2.5,
			start_position=Point2D(0.1, 0.0),
			embedding=(1.0, 0.0, 0.0),
		)
		with pytest.raises(ValueError, match="equal length"):
			candidate_pairs([earlier, later], **_GATE_KWARGS)


class TestResolveMerges:
	def test_single_candidate_resolves(self) -> None:
		merges = resolve_merges([MergeCandidate(1, 2, gap_seconds=1.0, score=0.9)])
		assert merges == {2: 1}

	def test_no_candidates_resolves_to_nothing(self) -> None:
		assert resolve_merges([]) == {}

	def test_higher_score_wins_a_contested_later_fragment(self) -> None:
		merges = resolve_merges(
			[
				MergeCandidate(1, 3, gap_seconds=1.0, score=0.5),
				MergeCandidate(2, 3, gap_seconds=1.0, score=0.9),
			]
		)
		assert merges == {3: 2}

	def test_higher_score_wins_a_contested_earlier_fragment(self) -> None:
		merges = resolve_merges(
			[
				MergeCandidate(1, 2, gap_seconds=1.0, score=0.9),
				MergeCandidate(1, 3, gap_seconds=1.0, score=0.5),
			]
		)
		assert merges == {2: 1}
		assert 3 not in merges

	def test_chain_resolves_to_the_earliest_root(self) -> None:
		merges = resolve_merges(
			[
				MergeCandidate(1, 2, gap_seconds=1.0, score=0.9),
				MergeCandidate(2, 3, gap_seconds=1.0, score=0.9),
			]
		)
		assert merges == {2: 1, 3: 1}
