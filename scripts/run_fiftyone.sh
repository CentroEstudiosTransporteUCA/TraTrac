#!/usr/bin/env bash
# Bootstrap MongoDB (fiftyone-db has no Linux wheel -- see CLAUDE.md Dependency Notes) and
# launch the FiftyOne App against an already-built dataset.
#
# Usage: scripts/run_fiftyone.sh [DATASET_NAME]
#   DATASET_NAME defaults to "cruce". The dataset must already exist (built via
#   `tratrac-fiftyone ... --dataset-name NAME`) -- this script only runs the app, it doesn't
#   build a dataset.
#
# Idempotent: safe to re-run. Reuses an already-running mongod on the expected port; downloads
# the mongod binary only once (cached under .fiftyone-mongo/, gitignored).
#
# Ctrl+C stops the FiftyOne App; mongod is left running in the background (--fork) so
# re-running this script is fast next time. Kill it yourself when done with it for good:
#   pkill -f 'mongod.*fiftyone-mongo'

set -euo pipefail

DATASET_NAME="${1:-cruce}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONGO_DIR="$REPO_ROOT/.fiftyone-mongo"
BIN_DIR="$MONGO_DIR/bin"
DATA_DIR="$MONGO_DIR/data"
MONGOD="$BIN_DIR/mongod"
PORT=27117
DB_URI="mongodb://127.0.0.1:${PORT}"

mkdir -p "$BIN_DIR" "$DATA_DIR"

# --- 1. Get a mongod binary (download once, cached) ------------------------------------
if [ ! -x "$MONGOD" ]; then
	echo "==> Downloading MongoDB (first run only)..."
	MONGO_TGZ="$MONGO_DIR/mongo.tgz"
	curl -fL -o "$MONGO_TGZ" \
		"https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.14.tgz"
	tar xzf "$MONGO_TGZ" -C "$MONGO_DIR"
	rm "$MONGO_TGZ"
	find "$MONGO_DIR" -maxdepth 1 -name "mongodb-linux-*" -type d | while read -r d; do
		cp "$d/bin/mongod" "$MONGOD"
	done
	chmod +x "$MONGOD"
fi

# --- 2. Start mongod if it isn't already running ----------------------------------------
mongod_running() {
	(exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && exec 3>&- 3<&-
}

if mongod_running; then
	echo "==> mongod already running on port ${PORT}."
else
	echo "==> Starting mongod on port ${PORT}..."
	MONGOD_ARGS=(--dbpath "$DATA_DIR" --port "$PORT" --fork --logpath "$DATA_DIR/mongod.log" --bind_ip 127.0.0.1)

	# Try running it directly first (works on non-NixOS Linux); only fall back to the raw
	# ELF-loader trick if that fails, so this script isn't NixOS-only.
	if ! "$MONGOD" "${MONGOD_ARGS[@]}" 2>/dev/null; then
		if [ ! -e /etc/NIXOS ] && [ -z "${IN_NIX_SHELL:-}" ]; then
			echo "Direct exec failed and this doesn't look like NixOS -- see mongod's own" \
				"error above." >&2
			exit 1
		fi
		echo "==> Direct exec failed (expected on NixOS) -- retrying via the raw ELF loader."
		CURL_LIB="$(nix build 'nixpkgs#curl^out' --no-link --print-out-paths)/lib"
		OPENSSL_LIB="$(nix build 'nixpkgs#openssl^out' --no-link --print-out-paths)/lib"
		LOADER="$(ldd "$(command -v python3)" | grep -oE '/nix/store/[^ ]*/ld-linux[^ ]*\.so\.[0-9]+' | head -1)"
		if [ -z "$LOADER" ]; then
			echo "Could not find the Nix-provided ELF loader via python3 -- inspect" \
				"'ldd \$(command -v python3)' yourself." >&2
			exit 1
		fi
		nix develop "$REPO_ROOT" --command bash -c "
			export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$CURL_LIB:$OPENSSL_LIB\"
			'$LOADER' --library-path \"\$LD_LIBRARY_PATH\" '$MONGOD' ${MONGOD_ARGS[*]}
		"
	fi

	for _ in $(seq 1 20); do
		mongod_running && break
		sleep 0.5
	done
	mongod_running || { echo "mongod did not come up -- check $DATA_DIR/mongod.log" >&2; exit 1; }
fi

# --- 3. Launch the FiftyOne App -----------------------------------------------------------
echo "==> Launching the FiftyOne App for dataset \"${DATASET_NAME}\"..."
echo "    (Ctrl+C stops the app; mongod keeps running in the background.)"
cd "$REPO_ROOT"
FIFTYONE_DATABASE_URI="$DB_URI" uv run --extra fiftyone python -c "
import sys
import fiftyone as fo

name = '${DATASET_NAME}'
if not fo.dataset_exists(name):
	existing = fo.list_datasets()
	print(f'No dataset named {name!r} in this database.', file=sys.stderr)
	if existing:
		print(f'Datasets that do exist here: {existing}', file=sys.stderr)
	else:
		print(
			'No datasets exist here yet -- build one first, e.g.:\n'
			f'  uv run --extra fiftyone tratrac-fiftyone VIDEO --dataset-name {name} '
			'--record RECORD.parquet --trj RUN.trj',
			file=sys.stderr,
		)
	sys.exit(1)

session = fo.launch_app(fo.load_dataset(name))
session.wait()
"
