"""Frame-window arithmetic: turn a start/end time interval into a frame range.

Pure (no cv2) so the index math — seconds→frames, inclusive end, clamping to the
clip length — is testable in isolation. ``OpenCvVideoSource`` uses it to seek to
the start frame and to know when to stop. See vault/17_time_window.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameWindow:
	"""The inclusive range of absolute frame indices to decode.

	``end_frame`` is ``None`` when the window runs to end-of-stream. ``frame_count``
	is how many frames the window yields, or ``None`` when the container did not
	report a reliable total (so progress has no denominator). Indices are absolute
	(real frame numbers), keeping exported TIMESTEPs on the source video's clock.
	"""

	start_frame: int
	end_frame: int | None
	frame_count: int | None

	@classmethod
	def from_seconds(
		cls,
		*,
		fps: float,
		total_frames: int,
		start_seconds: float | None,
		end_seconds: float | None,
	) -> FrameWindow:
		"""Build the window from a time interval; ``end`` is inclusive of its frame.

		``total_frames <= 0`` means the container did not report a count, so the
		bounds that depend on it (start-past-end, end clamping, ``frame_count``)
		are skipped and seeking is trusted.
		"""
		start_frame = round(start_seconds * fps) if start_seconds is not None else 0
		if start_frame < 0:
			raise ValueError(f"start resolves to a negative frame ({start_frame}).")
		end_frame = int(end_seconds * fps) if end_seconds is not None else None

		known_total = total_frames if total_frames > 0 else None
		if known_total is not None and start_frame >= known_total:
			raise ValueError(
				f"start is at or beyond the video duration (~{known_total / fps:.3f}s)."
			)
		if end_frame is not None and known_total is not None:
			end_frame = min(end_frame, known_total - 1)
		if end_frame is not None and end_frame < start_frame:
			raise ValueError(f"end frame ({end_frame}) precedes start frame ({start_frame}).")

		if end_frame is not None:
			frame_count: int | None = end_frame - start_frame + 1
		elif known_total is not None:
			frame_count = known_total - start_frame
		else:
			frame_count = None
		return cls(start_frame=start_frame, end_frame=end_frame, frame_count=frame_count)

	def includes(self, index: int) -> bool:
		"""Whether an absolute frame index still falls within the window."""
		return self.end_frame is None or index <= self.end_frame
