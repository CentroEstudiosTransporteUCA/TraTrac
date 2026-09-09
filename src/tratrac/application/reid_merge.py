"""Offline ReID fragment-merge decision: motion-plausibility gating + appearance scoring.

Pure application logic for Group D2's "merge decision" stage (``docs/IMPLEMENTATION_PLAN.md``,
``docs/roadmap/mvp5.md``): no video, no GPU, fully re-tunable offline like
``--pos-noise``/``--jerk``. Consumes track fragments — a tracker splits the same physical
vehicle into more than one ``track_id`` across an occlusion — each carrying an appearance
embedding (Stage 1's output, not built here; this module doesn't care how the embedding was
produced, only that fragments' embeddings are comparable by cosine similarity), and decides
which fragments are really the same vehicle reappearing.

Two-part decision per candidate pair (an earlier fragment ending, a later fragment beginning):

1. **Motion-plausibility gate** — extrapolate the earlier fragment's end kinematic state
   forward to the later fragment's start time (``application.kalman.KinematicKalmanFilter``'s
   ``from_state``/``predict``), and reject the pair if the later fragment's actual start
   position falls outside a configurable number of standard deviations of the extrapolated
   position. An implausible reappearance is rejected regardless of how similar the crops look
   — this is the "not appearance alone" gate ``TRACKER_CHOICE.md``/``mvp5.md`` describe, mirroring
   the Songdo deployment's appearance+temporal fusion using infrastructure already in this repo.
2. **Appearance score** — among gated candidates, cosine similarity between embeddings.

Resolution is greedy highest-score-first, 1:1 (each fragment absorbs/is absorbed by at most
one other) — not a full assignment solver (Hungarian etc.); fragment merging is a comparatively
rare event (an occlusion), not a dense every-frame assignment problem, so the simpler heuristic
is judged good enough. Chains (fragment A's end merges into B's start, B's end merges into C's
start) resolve transitively to a single canonical id per chain — the earliest fragment's id.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tratrac.application.kalman import KinematicKalmanFilter, SmoothedSample
from tratrac.domain.geometry import Point2D


@dataclass(frozen=True, slots=True)
class TrackFragment:
	"""One track's observed lifespan, endpoint kinematics, and appearance — the unit merge
	candidates are built from.

	``end_state`` is the track's smoothed kinematic state (``application/kalman.py``) at
	``end_time`` — the seed for extrapolating this fragment's motion forward into a later
	fragment's start. ``embedding`` is Stage 1's per-fragment appearance vector (e.g. DINOv3),
	opaque to this module beyond being comparable by cosine similarity.
	"""

	track_id: int
	start_time: float
	end_time: float
	start_position: Point2D
	end_state: SmoothedSample
	embedding: tuple[float, ...]

	def __post_init__(self) -> None:
		if self.end_time < self.start_time:
			raise ValueError(
				f"end_time ({self.end_time}) must be >= start_time ({self.start_time})."
			)
		if not self.embedding:
			raise ValueError("embedding must not be empty.")


@dataclass(frozen=True, slots=True)
class MergeCandidate:
	"""One motion-plausible, appearance-scored fragment pair: ``from_track_id`` (earlier)
	reappearing as ``to_track_id`` (later)."""

	from_track_id: int
	to_track_id: int
	gap_seconds: float
	score: float


def candidate_pairs(
	fragments: Sequence[TrackFragment],
	*,
	max_gap_seconds: float,
	pos_noise: float,
	jerk: float,
	seed_vel_std: float,
	seed_accel_std: float,
	max_sigma: float,
) -> list[MergeCandidate]:
	"""Every motion-plausible ``(earlier ends, later begins)`` pair, scored by appearance.

	``O(n^2)`` over the fragment count — fine for a clip's track count. ``max_gap_seconds``
	bounds which pairs are even considered (SSAM-scale occlusions, not a whole-video re-entry
	window); ``pos_noise``/``jerk`` match the smoother's own units (pixels); ``seed_vel_std``/
	``seed_accel_std`` set the seeded confidence in the extrapolation (see
	``KinematicKalmanFilter.from_state``); ``max_sigma`` is the gate threshold in standard
	deviations of the extrapolated position.
	"""
	candidates: list[MergeCandidate] = []
	for earlier in fragments:
		filt = KinematicKalmanFilter.from_state(
			earlier.end_state,
			pos_noise=pos_noise,
			jerk=jerk,
			vel_std=seed_vel_std,
			accel_std=seed_accel_std,
		)
		for later in fragments:
			if later.track_id == earlier.track_id:
				continue
			gap = later.start_time - earlier.end_time
			if not (0.0 < gap <= max_gap_seconds):
				continue
			predicted, std_x, std_y = filt.predict(gap)
			dx = (predicted.px - later.start_position.x) / std_x
			dy = (predicted.py - later.start_position.y) / std_y
			if math.hypot(dx, dy) > max_sigma:
				continue
			score = _cosine_similarity(earlier.embedding, later.embedding)
			candidates.append(MergeCandidate(earlier.track_id, later.track_id, gap, score))
	return candidates


def resolve_merges(candidates: Sequence[MergeCandidate]) -> dict[int, int]:
	"""Greedy highest-score-first 1:1 resolution into ``{old_track_id: canonical_track_id}``.

	Each track id absorbs/is absorbed by at most one other. Chains resolve transitively: if
	fragment A's end merges into B's start, and B's end merges into C's start, both B and C
	map to A's id (the chain's *root*, its earliest fragment) — the vehicle's original
	identity carries forward rather than jumping to its latest fragment's id. Track ids with
	no accepted merge are simply absent from the result (the caller's default is identity,
	matching how Link/Lane assignment treats an unclassified point as ``0``/unchanged).
	"""
	claimed_later: set[int] = set()
	claimed_earlier: set[int] = set()
	direct_predecessor: dict[int, int] = {}  # later_track_id -> its immediate earlier fragment
	for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
		if candidate.to_track_id in claimed_later or candidate.from_track_id in claimed_earlier:
			continue
		claimed_later.add(candidate.to_track_id)
		claimed_earlier.add(candidate.from_track_id)
		direct_predecessor[candidate.to_track_id] = candidate.from_track_id

	def root(track_id: int) -> int:
		seen = {track_id}
		while track_id in direct_predecessor:
			track_id = direct_predecessor[track_id]
			if track_id in seen:
				raise ValueError(f"cycle detected in ReID merge chain at track {track_id}.")
			seen.add(track_id)
		return track_id

	return {later: root(later) for later in direct_predecessor}


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
	if len(a) != len(b):
		raise ValueError(f"embeddings must have equal length, got {len(a)} and {len(b)}.")
	dot = sum(x * y for x, y in zip(a, b, strict=True))
	norm_a = math.sqrt(sum(x * x for x in a))
	norm_b = math.sqrt(sum(x * x for x in b))
	if norm_a == 0.0 or norm_b == 0.0:
		raise ValueError("cannot compute cosine similarity against a zero embedding vector.")
	return dot / (norm_a * norm_b)
