"""TOML config-file reader.

Thin I/O adapter: reads a persisted run config into the nested table that
``tratrac.application.config.RunConfig.resolve`` consumes. Parsing uses the
stdlib ``tomllib`` (Python 3.12+), so no third-party dependency is needed.

This is the only seam where the dynamically-typed TOML document enters the
package; the application resolver does the type checking and validation.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
	"""Parse a TOML file into a nested dict.

	Raises ``FileNotFoundError`` if ``path`` does not exist and ``ValueError`` on
	malformed TOML (re-wrapping ``tomllib.TOMLDecodeError`` with the file path so
	the CLI can report it cleanly).
	"""
	try:
		with path.open("rb") as handle:
			return tomllib.load(handle)
	except tomllib.TOMLDecodeError as exc:
		raise ValueError(f"{path} is not valid TOML: {exc}") from exc
