# Running TraTrac on Google Colab (GPU)

Copy-paste recipe to run the current 3-tool pipeline (`tratrac` → `tratrac-postprocess` →
`tratrac-render`) on a Colab GPU (T4 is enough) and compare a stabilized vs. non-stabilized
run with `validate_trj`. Each fenced block is one Colab **cell**.

> This guide previously targeted an older CLI generation (a `--video-out` flag on `tratrac`
> itself, a `tratrac-smooth` command, a standalone `render_violations.py` script, and an
> `orientation.method` config key comparing EMA vs. inline-Kalman vs. offline-RTS smoothing).
> None of that exists anymore: the live pipeline computes no kinematics at all now (the
> export inversion, see `CLAUDE.md`), so the only smoother left is the offline Kalman/RTS
> pass in `tratrac-postprocess`, and rendering/violation-marking are both handled by
> `tratrac-render` in one pass. This rewrite targets the current 3-tool pipeline; the
> remaining meaningful ablation axis is **stabilized vs. non-stabilized** (`ego_motion.enabled`),
> not smoothing strategy.

**Drive layout this assumes** — create it and upload your files:

```
/content/drive/MyDrive/URBAn/TraTrac/
├── resources/        # cruce_simple.mp4, rotonda.mp4
├── configs/          # cruce.toml, rotonda_nostab.toml, rotonda_stab.toml
└── outputs/          # created automatically; one folder per config
```

The three `.toml`s are copies of `tratrac.example.toml` with `input.video`, `export.out`, and
(for the `rotonda_*` pair) `[ego_motion]` edited per config; they're gitignored, so create and
upload them yourself. `cruce.toml` needs no stabilization (a static/near-static shot);
`rotonda_nostab.toml` / `rotonda_stab.toml` are the same moving-drone clip with
`ego_motion.enabled = false` / `true`.

The repo pins torch to the **CPU** index; on a GPU we swap to the **cu126** index (the only
one carrying torch 2.12 at time of writing — check `pyproject.toml` for the current pin).

---

## 0. Runtime

`Runtime → Change runtime type → T4 GPU`, then run the cells in order.

## 1. Confirm the GPU

```python
!nvidia-smi -L
```

## 2. Install uv

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
# Colab exports MPLBACKEND=module://matplotlib_inline...; ultralytics imports matplotlib at
# import time, and that inline backend isn't in the uv venv, so force a headless one.
os.environ["MPLBACKEND"] = "Agg"
!uv --version
```

## 3. Clone the repo (private → needs a token)

Add a GitHub PAT (scope `repo`) in Colab's **🔑 Secrets** panel as `GH_TOKEN`, then:

```python
from google.colab import userdata
token = userdata.get("GH_TOKEN")
!git clone https://{token}@github.com/CentroEstudiosTransporteUCA/TraTrac.git
%cd TraTrac
!git log --oneline -1
```

## 4. Swap torch CPU → CUDA, then sync (~2.5 GB download, a few minutes)

Check `pyproject.toml`'s `[tool.uv.sources]` for the current CUDA tag before running this —
torch's published cu-tags change over time. The `sed` normalizes whatever the index currently
is, so it's re-runnable.

```python
!sed -i -E 's#download.pytorch.org/whl/(cpu|cu[0-9]+)#download.pytorch.org/whl/cu126#g' pyproject.toml
!uv lock && uv sync
!uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

`cuda` must print `True`. If it doesn't, the runtime's NVIDIA driver is too old for the cu-tag
above — recycle the runtime / pick a standard GPU, or lower the cu-tag to match.

## 5. Mount Drive and set the paths

```python
from google.colab import drive
drive.mount("/content/drive")
VIDEO_DIR  = "/content/drive/MyDrive/URBAn/TraTrac/resources"
CONFIG_DIR = "/content/drive/MyDrive/URBAn/TraTrac/configs"
OUTPUT_DIR = "/content/drive/MyDrive/URBAn/TraTrac/outputs"
!ls -lh "$VIDEO_DIR" "$CONFIG_DIR"
```

The `ls` must show your two `.mp4`s and the three `.toml`s. Configs use absolute Drive paths
for `input.video` and `export.out`, so the runs below need nothing else. (Outputs are written
straight to Drive, so they survive the runtime shutting down.)

## 6. Pass 1 — perception (writes the track record)

First run downloads the YOLOv8-VisDrone weights. Each config writes its track record
(`export.out`, a `.parquet` file) — no flags, everything comes from the TOML.

```python
!uv run tratrac --config "$CONFIG_DIR/cruce.toml"
!uv run tratrac --config "$CONFIG_DIR/rotonda_nostab.toml"
!uv run tratrac --config "$CONFIG_DIR/rotonda_stab.toml"
```

## 7. Pass 2 — offline smoothing (writes the `.trj`)

`tratrac-postprocess` runs the Kalman/RTS smoother and writes the SSAM `.trj`. For the
stabilized run, keep the record's projected coordinates as-is (no `--calibration` needed
beyond what's already baked into MVP1.75 GSD calibration in the config).

```python
!uv run tratrac-postprocess "$OUTPUT_DIR/cruce/cruce.parquet" \
    --out "$OUTPUT_DIR/cruce/cruce.trj"
!uv run tratrac-postprocess "$OUTPUT_DIR/rotonda_nostab/rotonda_nostab.parquet" \
    --out "$OUTPUT_DIR/rotonda_nostab/rotonda_nostab.trj"
!uv run tratrac-postprocess "$OUTPUT_DIR/rotonda_stab/rotonda_stab.parquet" \
    --out "$OUTPUT_DIR/rotonda_stab/rotonda_stab.trj"
```

Re-tune the smoother for free (no re-detection) by re-running any of these lines with
`--pos-noise 1.5 --jerk 5` and re-validating below.

## 8. Validate, then render each result video

Validate all three variants, dumping each one's violations to a CSV:

```python
for folder, stem in [("cruce", "cruce"), ("rotonda_nostab", "rotonda_nostab"), ("rotonda_stab", "rotonda_stab")]:
    print(f"================ {stem} ================")
    !uv run python scripts/validate_trj.py "{OUTPUT_DIR}/{folder}/{stem}.trj" \
        --violations-csv "{OUTPUT_DIR}/{folder}/{stem}_violations.csv"
```

Render one result video per variant — trajectories drawn, violations marked in the same
pass. The `rotonda_stab` run needs `--transforms` (the ego-motion sidecar CSV, only produced
when `ego_motion.enabled` and `export.transform_csv` are both set in the config) so the
marks map from the stabilized coordinate frame back onto the raw video:

```python
!uv run tratrac-render "$VIDEO_DIR/cruce_simple.mp4" \
    --trj "$OUTPUT_DIR/cruce/cruce.trj" \
    --violations "$OUTPUT_DIR/cruce/cruce_violations.csv" \
    --out "$OUTPUT_DIR/cruce/cruce_result.mp4" --force
!uv run tratrac-render "$VIDEO_DIR/rotonda.mp4" \
    --trj "$OUTPUT_DIR/rotonda_nostab/rotonda_nostab.trj" \
    --violations "$OUTPUT_DIR/rotonda_nostab/rotonda_nostab_violations.csv" \
    --out "$OUTPUT_DIR/rotonda_nostab/rotonda_nostab_result.mp4" --force
!uv run tratrac-render "$VIDEO_DIR/rotonda.mp4" \
    --trj "$OUTPUT_DIR/rotonda_stab/rotonda_stab.trj" \
    --transforms "$OUTPUT_DIR/rotonda_stab/rotonda_stab_transforms.csv" \
    --violations "$OUTPUT_DIR/rotonda_stab/rotonda_stab_violations.csv" \
    --out "$OUTPUT_DIR/rotonda_stab/rotonda_stab_result.mp4" --force
```

### What to read

- **orientation smoothness % / speed-accel plausibility %** (from step 8's `validate_trj`
  output) — stabilization should raise both on the moving clip: compare
  `rotonda_nostab` → `rotonda_stab`.
- **Three `<variant>_result.mp4`** — the visual checks: each variant's trajectories + red
  violation marks. Add `--checks appearance,...` to `tratrac-render` to mark only some rule
  types; default marks all.

## Notes

- **Full clips:** raise the configs' `[window] end` (or set it `""`) once a short run looks
  right — minutes on a T4.
- **Skip the overlay videos:** the render step (8) is the slow part per config; skip it for a
  faster metrics-only pass over steps 1–8's validation output.
