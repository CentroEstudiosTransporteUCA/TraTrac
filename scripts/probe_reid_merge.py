#!/usr/bin/env python3
"""Probe ``application/reid_merge.py``'s motion-plausibility gate against a real track record,
without a real appearance embedding.

Stage 1 (DINOv3 embedding) is unbuilt — needs a GPU, see
``src/tratrac/application/REID_MERGE.md``. Every fragment here gets the **same constant
embedding**, so ``resolve_merges``'s appearance-similarity term is neutralized (every candidate
scores identically) and only the Kalman motion gate decides which fragments merge. This is a
legitimate first empirical check of whether motion gating *alone* measurably improves track
continuity (``scripts/validate_trj.py``'s Continuity checks) on real footage, ahead of DINOv3
existing to add the appearance half — and the concrete way to tune the gate parameters
(``--max-gap-seconds``, ``--seed-vel-std``, ``--seed-accel-std``, ``--max-sigma``) against real
data, which ``REID_MERGE.md`` flagged as an open question with "no real occlusion footage
exists yet to tune them against."

Writes a ``reid_merge.json`` (see ``infrastructure/reid/json.py``) for
``tratrac-postprocess --reid-merge``.

Imports the ``tratrac`` package directly (unlike most ``scripts/`` tools) to exercise the real
``reid_merge``/``kalman`` code paths — the same lone-exception discipline
``scripts/visualize_stabilization.py`` already uses for exercising the real stabilizer.

Usage:
	uv run python scripts/probe_reid_merge.py RECORD.parquet --out merge.json
		[--max-gap-seconds S] [--pos-noise PX] [--jerk Q]
		[--seed-vel-std V] [--seed-accel-std A] [--max-sigma N]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from tratrac.application.kalman import SmoothedSample, smooth_track
from tratrac.application.reid_merge import TrackFragment, candidate_pairs, resolve_merges
from tratrac.domain.geometry import Point2D
from tratrac.infrastructure.reid.json import save_reid_merge
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording, read_tracks

# Placeholder appearance vector: identical for every fragment, so cosine similarity is always
# 1.0 and candidate_pairs' scoring is neutral — only the motion gate filters candidates.
_PLACEHOLDER_EMBEDDING = (1.0,)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument("record", type=Path, help="Track record (export.out) to probe.")
	parser.add_argument("--out", type=Path, required=True, help="Output reid_merge.json.")
	parser.add_argument(
		"--max-gap-seconds",
		type=float,
		default=2.0,
		help="Max occlusion gap to consider a candidate pair, seconds.",
	)
	parser.add_argument(
		"--pos-noise", type=float, default=2.0, help="Measurement-noise std, px (fragment fit)."
	)
	parser.add_argument("--jerk", type=float, default=20.0, help="Process jerk spectral density.")
	parser.add_argument(
		"--seed-vel-std",
		type=float,
		default=10.0,
		help="Seeded velocity-uncertainty std for extrapolation, px/s.",
	)
	parser.add_argument(
		"--seed-accel-std",
		type=float,
		default=5.0,
		help="Seeded acceleration-uncertainty std for extrapolation, px/s^2.",
	)
	parser.add_argument(
		"--max-sigma",
		type=float,
		default=3.0,
		help="Gate threshold, standard deviations of the extrapolated position.",
	)
	args = parser.parse_args()

	if not args.record.exists():
		print(f"Record not found: {args.record}", file=sys.stderr)
		return 1

	recording = read_tracks(args.record)
	print(f"{args.record}: {len(recording.observations)} observations")

	fragments = _fragments(recording)
	print(f"{len(fragments)} track fragments")

	candidates = candidate_pairs(
		fragments,
		max_gap_seconds=args.max_gap_seconds,
		pos_noise=args.pos_noise,
		jerk=args.jerk,
		seed_vel_std=args.seed_vel_std,
		seed_accel_std=args.seed_accel_std,
		max_sigma=args.max_sigma,
	)
	print(f"{len(candidates)} motion-plausible candidate pairs")

	merges = resolve_merges(candidates)
	canonical_count = len(set(merges.values())) if merges else 0
	print(f"{len(merges)} fragments merged into {canonical_count} canonical tracks")

	args.out.parent.mkdir(parents=True, exist_ok=True)
	save_reid_merge(args.out, merges, candidates)
	print(f"Wrote {args.out}")
	return 0


def _fragments(recording: TrackRecording) -> list[TrackFragment]:
	fps = recording.metadata.fps
	by_track: dict[int, list[TrackObservation]] = defaultdict(list)
	for observation in recording.observations:
		by_track[observation.track_id].append(observation)

	fragments: list[TrackFragment] = []
	for track_id, observations in by_track.items():
		observations.sort(key=lambda o: o.frame_index)
		xs = [o.cx for o in observations]
		ys = [o.cy for o in observations]
		timestamps = [o.frame_index / fps for o in observations]
		smoothed: list[SmoothedSample] = smooth_track(xs, ys, timestamps, pos_noise=2.0, jerk=20.0)
		fragments.append(
			TrackFragment(
				track_id=track_id,
				start_time=timestamps[0],
				end_time=timestamps[-1],
				start_position=Point2D(xs[0], ys[0]),
				end_state=smoothed[-1],
				embedding=_PLACEHOLDER_EMBEDDING,
			)
		)
	return fragments


if __name__ == "__main__":
	sys.exit(main())
