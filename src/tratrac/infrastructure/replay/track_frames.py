"""Windowed video replay keyed to a track record.

Shared infra for Group D (GitHub Issues): segmentation (D1) and ReID-embed
(D2) both need to re-open the source video and pair each frame with the track record's
observations at that frame, so each can crop per-track image patches. Extracted from the
windowed-reopen + ``round(timestamp * fps)``-bucketing pattern ``cli_render.py`` already proved
for ``.trj``/``VehicleState`` playback, here doing the same thing one level earlier in the
pipeline — over the raw ``TrackRecording``/``TrackObservation`` the perception run wrote,
before smoothing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from tratrac.domain.frame import Frame
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

# Seconds of clip kept past the last observed frame so it's included in the window — mirrors
# cli_render.py's _TAIL_BUFFER_SECONDS (the record carries frame indices, converted to seconds
# for OpenCvVideoSource's window, which is expressed in seconds).
_DEFAULT_PADDING_SECONDS = 0.5


def iter_track_frames(
	video_path: Path,
	recording: TrackRecording,
	*,
	padding_seconds: float = _DEFAULT_PADDING_SECONDS,
) -> Iterator[tuple[Frame, list[TrackObservation]]]:
	"""Yield each frame the recording covers, paired with that frame's observations.

	Re-opens ``video_path`` windowed to ``[first_observed_frame, last_observed_frame]``
	(+ ``padding_seconds`` tail) rather than the whole clip, so a short analysis window on a
	long source doesn't decode video outside the recording's span — the same reasoning
	``cli_render.py`` windows to a ``.trj``'s span. A frame with no recorded observations
	still yields with an empty list (e.g. a frame between two tracks' lifespans within the
	window); an empty recording yields nothing.

	Frame indices stay absolute (matching ``TrackObservation.frame_index``), so callers don't
	need to remap between windowed and source-video coordinates.
	"""
	if not recording.observations:
		return
	fps = recording.metadata.fps
	by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
	for observation in recording.observations:
		by_frame[observation.frame_index].append(observation)

	start_seconds = min(by_frame) / fps
	end_seconds = max(by_frame) / fps + padding_seconds
	with OpenCvVideoSource(
		video_path, start_seconds=start_seconds, end_seconds=end_seconds
	) as source:
		for frame in source.frames():
			yield frame, by_frame.get(frame.index, [])
