# Running the TraTrac ablation on Google Colab (GPU)

Copy-paste recipe to run the four ablation configs + the RTS smoother on a Colab GPU
(T4 is enough) and compare the innovations with `validate_trj`. Each fenced block is one
Colab **cell**. The configs are self-contained (absolute Drive paths, `device = "cuda"`,
outputs grouped per config), so the runs take **no flags**.

**Drive layout this assumes** — create it and upload your files:
```
/content/drive/MyDrive/URBAn/TraTrac/
├── resources/        # cruce_simple.mp4, rotonda.mp4
├── configs/          # cruce_ema.toml, cruce_kalman.toml, rotonda_nostab.toml, rotonda_stab.toml  (+ ABLATION.md)
└── outputs/          # created automatically; one folder per config
```
(the `.toml`s + `ABLATION.md` are the repo's `.configs/colab/` files; they're gitignored, so
upload them to Drive yourself.)

The repo pins torch to the **CPU** index; on a GPU we swap to the **cu126** index (the only
one carrying torch 2.12).

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
!git log --oneline -1   # should show a2b9166 (the Kalman commit) or later
```

## 4. Swap torch CPU → CUDA, then sync (~2.5 GB download, a few minutes)
torch 2.12 publishes **only** for cu126 (cu124 stops at 2.6.0, cu128 at 2.11.0). The `sed`
normalizes whatever the index currently is, so it's re-runnable.
```python
!sed -i -E 's#download.pytorch.org/whl/(cpu|cu[0-9]+)#download.pytorch.org/whl/cu126#g' pyproject.toml
!uv lock && uv sync
!uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
The last line must print `torch 2.12.0+cu126 cuda True`.

## 5. Mount Drive and set the paths
```python
from google.colab import drive
drive.mount("/content/drive")
VIDEO_DIR  = "/content/drive/MyDrive/URBAn/TraTrac/resources"
CONFIG_DIR = "/content/drive/MyDrive/URBAn/TraTrac/configs"
OUTPUT_DIR = "/content/drive/MyDrive/URBAn/TraTrac/outputs"
!ls -lh "$VIDEO_DIR" "$CONFIG_DIR"
```
The `ls` must show your two `.mp4`s and the four `.toml`s. The configs already point at
these absolute paths, so the runs below need nothing else. (Outputs are written straight to
Drive, so they survive the runtime shutting down.)

## 6. Run the four ablation configs on the GPU
First run downloads the YOLOv8-VisDrone weights; each config writes into
`outputs/<config-name>/`.
```python
!uv run tratrac --config "$CONFIG_DIR/cruce_ema.toml"
!uv run tratrac --config "$CONFIG_DIR/cruce_kalman.toml"
!uv run tratrac --config "$CONFIG_DIR/rotonda_nostab.toml"
!uv run tratrac --config "$CONFIG_DIR/rotonda_stab.toml"
```

## 7. Offline forward+RTS smoothing of the two track sidecars
```python
!uv run tratrac-smooth "$OUTPUT_DIR/cruce_ema/cruce_ema_tracks.csv" \
    --out "$OUTPUT_DIR/cruce_ema/cruce_ema_smooth.trj"
!uv run tratrac-smooth "$OUTPUT_DIR/rotonda_stab/rotonda_stab_tracks.csv" \
    --out "$OUTPUT_DIR/rotonda_stab/rotonda_stab_smooth.trj"
```

## 8. Compare — `validate_trj` across every output
```python
print("================ cruce_ema  (EMA baseline) ================")
!uv run python scripts/validate_trj.py "$OUTPUT_DIR/cruce_ema/cruce_ema.trj"
print("================ cruce_kalman  (inline forward Kalman) ================")
!uv run python scripts/validate_trj.py "$OUTPUT_DIR/cruce_kalman/cruce_kalman.trj"
print("================ cruce_ema_smooth  (offline RTS) ================")
!uv run python scripts/validate_trj.py "$OUTPUT_DIR/cruce_ema/cruce_ema_smooth.trj"
print("================ rotonda_nostab  (no stabilization) ================")
!uv run python scripts/validate_trj.py "$OUTPUT_DIR/rotonda_nostab/rotonda_nostab.trj"
print("================ rotonda_stab  (ORB stabilization) ================")
!uv run python scripts/validate_trj.py "$OUTPUT_DIR/rotonda_stab/rotonda_stab.trj"
print("================ rotonda_stab_smooth  (stab + RTS) ================")
!uv run python scripts/validate_trj.py "$OUTPUT_DIR/rotonda_stab/rotonda_stab_smooth.trj"
```

### What to read
- **speed / acceleration plausibility %** — the smoother should raise it sharply.
  `cruce_ema → cruce_ema_smooth` is the headline number for the smoothing decision;
  `cruce_kalman` shows what the *live* forward filter alone buys (with some lag).
- **orientation smoothness %** — stabilization should raise it on the moving clip:
  `rotonda_nostab → rotonda_stab`.
- Each config's `outputs/<name>/` folder also has a `*_timing.csv` and, where applicable,
  the `*_tracks.csv` / `*_transforms.csv`.

## Notes
- **Re-tune the smoother for free** (no re-detection): re-run a cell 7 line with
  `--jerk 5 --pos-noise 1.5` and re-validate.
- **Inline Kalman on the moving clip:** set `orientation.method = "kalman"` in
  `rotonda_stab.toml` to combine stabilization + live filtering.
- **Full clips:** raise the configs' `[window] end` (or set it `""`) once a 30 s run looks
  right — minutes on a T4.
- **cu-tag:** torch 2.12 is published only for **cu126** (cu124 stops at 2.6.0, cu128 at
  2.11.0). If `cuda True` never prints, the runtime's NVIDIA driver is too old — recycle the
  runtime / pick a standard GPU.
- **Eyeball it:** set `export.video_out` in a config to also write an overlay `.mp4`
  (bumpers/IDs/trails) before trusting the numbers.
