"""DJI .SRT telemetry parser.

DJI drones write a per-frame subtitle (.srt) file alongside each video clip,
containing fields like ``[rel_alt: 50.000 abs_alt: 100.500]``. We only need
the altitude for MVP1.75; the other fields are ignored.

``rel_alt`` is altitude *above the take-off point*. That equals altitude
above ground level (AGL) only when take-off was on the ground directly
below the camera. For terrain with elevation changes, the operator must
override altitude manually.

Per-frame altitude is summarised as a single mean for MVP1.75; per-frame
Scale is deferred (SSAM v1.04 has only one Scale value in DIMENSIONS).
"""

from __future__ import annotations

import re
from pathlib import Path

_REL_ALT_RE = re.compile(r"rel_alt\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_ABS_ALT_RE = re.compile(r"abs_alt\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_altitudes(srt_path: Path) -> list[float]:
	"""Return one altitude (m, AGL) per SRT entry that carried a usable value.

	Prefers ``rel_alt`` (relative to take-off). Falls back to ``abs_alt``
	when ``rel_alt`` is missing. Entries with neither are skipped.

	Raises ``FileNotFoundError`` if ``srt_path`` does not exist.
	"""
	text = srt_path.read_text(encoding="utf-8", errors="replace")
	altitudes: list[float] = []
	for line in text.splitlines():
		rel = _REL_ALT_RE.search(line)
		if rel is not None:
			altitudes.append(float(rel.group(1)))
			continue
		abs_match = _ABS_ALT_RE.search(line)
		if abs_match is not None:
			altitudes.append(float(abs_match.group(1)))
	return altitudes


def mean_altitude(srt_path: Path) -> float:
	"""Mean altitude across a DJI .SRT file. Raises if no readable altitudes."""
	altitudes = extract_altitudes(srt_path)
	if not altitudes:
		raise ValueError(f"No rel_alt / abs_alt values found in {srt_path}.")
	return sum(altitudes) / len(altitudes)
