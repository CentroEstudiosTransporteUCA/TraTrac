# Scope — `tratrac --check` (✅ landed)

A validation-only mode so the Tauri client (and CI) can verify a config **without
running** the pipeline. Goal: make `application/config.py` the single source of truth
for validation, instead of the UI reimplementing range/coherence rules in JS/Rust
(see `docs/tauri_ui_spec.md`, principio 2).

---

## Command surface

```
tratrac --config run.toml --check [--json]
```

- `--check` — validate and exit. **Never** opens the video, **never** writes outputs,
  **never** prompts. `--force` is irrelevant under `--check` (no writes) and is ignored.
- `--json` — emit a machine-readable report to **stdout** (only meaningful with
  `--check`). Without it, problems print human-readable to stderr.
- Lives as a **flag on the existing `process` command**, not a new subcommand — keeps
  the single-command, config-only design intact (vault/19).

### Exit codes (contract for the UI)
| Code | Meaning |
| --- | --- |
| `0` | config valid |
| `2` | config invalid (same code `ConfigError` already uses) |

### JSON shape (`--json`)
```json
{
  "ok": false,
  "problems": [
    "detector.conf must be in [0.0, 1.0], got 1.4.",
    "calibration: specify exactly one of meters_per_pixel or drone_model, not both.",
    "export.out must be a file path, not a directory: out/."
  ]
}
```
`problems` is a flat list of human strings (the same messages `ConfigError` and the CLI
guards already produce — no new message taxonomy). The UI shows them verbatim and keys
its enable/disable state off exit code. `ok == (problems == [])`.

---

## What `--check` validates (and what it deliberately does not)

Layered, cheapest first; each layer runs only if the previous parsed:

- **L1 — TOML parse** (`load_toml`). A syntax error is a single fatal problem
  (`"config: <msg>"`); resolution can't proceed.
- **L2 — `RunConfig.resolve`** (the bulk; already aggregated). Missing keys, type
  errors, ranges, the calibration one-of, the `transform_csv`/`anchors_dir` ⇒
  `ego_motion.enabled` coherence guard. **This is exactly what a UI form needs.**
- **L3 — static run guards** (only if L1+L2 yield a `run`). The post-resolve checks
  already in `process`: video file exists, output path-types (file vs dir),
  path collisions (`out` ≠ `timing_csv` ≠ `transform_csv`). Cheap, no video decode.

**Out of scope (documented, not silently skipped):**

- **L4 — anything that opens the video or the network**: `resolve_scale` for
  `drone_model` calibration (needs `metadata.width`), `.SRT` altitude parsing, detector
  checkpoint download/availability, device reachability. These are *run-time* concerns,
  not config-shape concerns. A future `--check-deep` could add them; v1 stays fast and
  offline so the UI can call it on every form edit.
- **Warnings** (e.g. `tracker.det_thresh >= detector.conf` — legal but unusual). v1 is
  errors-only; a `"warnings": [...]` field is a forward-compatible extension point.

---

## Refactor required (small, and an improvement)

Today the L3 guards in `process` raise `typer.BadParameter` **one at a time** (first
failure wins). To report them aggregated like L2, extract a shared helper:

```python
def static_run_problems(run: RunConfig) -> list[str]:
        """Filesystem/path problems detectable without opening the video."""
```

Both paths consume it:
- `--check` merges `ConfigError.problems` (L2) with `static_run_problems` (L3) into one
  report.
- `process` calls it, and if non-empty raises one aggregated error instead of the
  current first-fail sequence.

**Behaviour change to flag:** `process` will now surface *all* path problems at once
rather than one per re-run. This is consistent with how L2 already behaves and is
strictly more informative — but it is a (minor) change to existing run behaviour, so it
needs your sign-off.

The helper stays in `cli.py` (it does filesystem `is_file`/`is_dir` probes — an
infrastructure concern, not pure `application`). The pure path-collision subset *could*
move to `application`, but I'd keep it together unless you want it unit-tested in
isolation.

---

## Layering / files touched

- `cli.py` — add `--check`/`--json` options to `process`; extract `static_run_problems`;
  branch to a `_emit_check_report(...)` that prints JSON or human lines and exits.
  No new module; no change to `application/config.py` (its `ConfigError.problems` is
  already the contract).
- `tests/unit` — CLI-level tests: a valid config → exit 0, `{"ok": true, "problems": []}`;
  an invalid one → exit 2 with the expected problem strings; assert `--check` opens no
  video (no checkpoint download — keep it out of the `slow` mark).

## Non-goals (explicit)

- No change to the run itself, the config schema, or any message text.
- No new dependency (stdlib `json`).
- No deep/online validation in v1 (see L4 above).
