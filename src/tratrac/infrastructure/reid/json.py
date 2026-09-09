"""Sidecar JSON reader/writer for the ReID merge decision (Group D2, ``docs/IMPLEMENTATION_PLAN.md``).

``application/reid_merge.py``'s ``resolve_merges`` output — plus the full candidate list, kept
for operator audit — round-trips through this file so the "merge decision" and "apply" stages
(``cli_postprocess.py --reid-merge``) can run as separate, independently re-run steps: re-tune
the gate/scoring, re-run just the merge decision, no re-embedding.

Schema::

    { "merges": { "<old_track_id>": <canonical_track_id>, ... },
      "candidates": [
        { "from_track_id": .., "to_track_id": .., "gap_seconds": .., "score": .. }
      ] }

``candidates`` is optional on read (informational; the apply stage only consumes ``merges``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tratrac.application.reid_merge import MergeCandidate


def load_reid_merge(path: Path) -> dict[int, int]:
	"""Parse a ``reid_merge.json`` into ``{old_track_id: canonical_track_id}``.

	Raises ``FileNotFoundError`` if ``path`` is absent and ``ValueError`` on any malformed
	content, re-wrapped with the path so the CLI can report it cleanly.
	"""
	try:
		with path.open("rb") as handle:
			document: Any = json.load(handle)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{path} is not valid JSON: {exc}") from exc

	if not isinstance(document, dict) or "merges" not in document:
		raise ValueError(f'{path} must be a JSON object with a "merges" object.')
	raw_merges = document["merges"]
	if not isinstance(raw_merges, dict):
		raise ValueError(f'{path}: "merges" must be an object.')

	merges: dict[int, int] = {}
	for old_id_str, canonical_id in raw_merges.items():
		try:
			old_id = int(old_id_str)
		except ValueError as exc:
			raise ValueError(f"{path}: merges key {old_id_str!r} is not an integer.") from exc
		if isinstance(canonical_id, bool) or not isinstance(canonical_id, int):
			raise ValueError(f"{path}: merges[{old_id_str!r}] must be an integer.")
		merges[old_id] = canonical_id
	return merges


def save_reid_merge(
	path: Path, merges: dict[int, int], candidates: list[MergeCandidate] | None = None
) -> None:
	"""Write ``resolve_merges``'s output (plus the optional full candidate list) to ``path``."""
	document = {
		"merges": {str(old_id): canonical_id for old_id, canonical_id in merges.items()},
		"candidates": [
			{
				"from_track_id": c.from_track_id,
				"to_track_id": c.to_track_id,
				"gap_seconds": c.gap_seconds,
				"score": c.score,
			}
			for c in (candidates or [])
		],
	}
	path.write_text(json.dumps(document, indent=2))
