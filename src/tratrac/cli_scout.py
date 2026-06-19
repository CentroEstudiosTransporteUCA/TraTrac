"""Typer entry point for the scout pass (``tratrac-scout``).

Discovers a clip's ORB keyframe anchors — the reference frames an operator draws
exclusion zones on — and persists the per-frame ego-motion schedule the real run
replays. See vault/21_exclusion_zones.md.

Unlike ``tratrac`` (a fully reproducible run with zero defaults), the scout is a
discovery tool, so its ORB parameters carry sensible defaults. The schedule it
emits is what the real run replays, so these values need not match any later run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tratrac.infrastructure.scout.runner import run_scout
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

app = typer.Typer(
	name="tratrac-scout",
	help="Discover ORB keyframe anchors and persist the ego-motion schedule.",
	no_args_is_help=True,
)


@app.command()
def scout(
	video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video.")],
	out_dir: Annotated[
		Path,
		typer.Option("--out-dir", "-o", file_okay=False, help="Directory for PNGs/manifest/CSV."),
	],
	orb_features: Annotated[
		int, typer.Option("--orb-features", help="ORB keypoints per frame.")
	] = 2000,
	orb_match_ratio: Annotated[
		float, typer.Option("--orb-match-ratio", help="Lowe ratio test, (0, 1).")
	] = 0.75,
	orb_min_matches: Annotated[
		int, typer.Option("--orb-min-matches", help="Min good matches to fit a transform.")
	] = 10,
	orb_ransac_threshold: Annotated[
		float, typer.Option("--orb-ransac-threshold", help="RANSAC reprojection threshold (px).")
	] = 3.0,
	min_anchor_overlap: Annotated[
		float,
		typer.Option("--min-anchor-overlap", help="Re-anchor below this shared area, (0, 1)."),
	] = 0.6,
) -> None:
	"""Scout VIDEO into OUT_DIR: transforms.csv, refs_manifest.json, one PNG per anchor."""
	try:
		with OpenCvVideoSource(video) as source:
			references = run_scout(
				source,
				out_dir,
				n_features=orb_features,
				match_ratio=orb_match_ratio,
				min_matches=orb_min_matches,
				ransac_threshold=orb_ransac_threshold,
				min_anchor_overlap=min_anchor_overlap,
				video_label=str(video),
			)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc
	typer.echo(
		f"Scouted {video} -> {len(references)} reference frames in {out_dir} "
		f"(draw zones on the PNGs; point the run's ego_motion.transforms at transforms.csv)."
	)


if __name__ == "__main__":
	app()
