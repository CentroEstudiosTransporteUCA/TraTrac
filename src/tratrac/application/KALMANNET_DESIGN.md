# KalmanNet-family adaptive noise: design spike (Group E)

## Status

**Research/design spike only — no code.** This is the concrete next step
GitHub Issues Group E calls for ("a research/design spike producing
`application/KALMANNET_DESIGN.md` before any code"), and what `SMOOTHING.md`'s "Frontier
upgrade" section flagged as "a genuine design-pass item, not a drop-in swap." This doc goes
one level deeper than that flag — architecture mechanics, a reconsidered recommendation on
which KalmanNet variant to target, an integration surface, and a training-data plan — but
stops short of writing any Python. No dataset plan existed before this doc; there still isn't
a *committed* one, but there is now a concrete, checkable proposal (below) instead of an open
question.

## What this would replace

`application/kalman.py`'s `smooth_track`/`KinematicKalmanFilter` use **fixed, hand-tuned**
`pos_noise`/`jerk` hyperparameters (measurement-noise std, process spectral density) — one pair
per `tratrac-postprocess` invocation, applied uniformly to every track in the run. In reality,
detector jitter and how "responsive" a track's true motion is both vary — by detector
confidence, by vehicle size, by whether the vehicle is turning vs. cruising. A learned,
per-timestep-adaptive noise/gain would let the filter trust measurements more or less
depending on local conditions, instead of one global compromise value.

## Architecture: KalmanNet learns the **gain**, not the dynamics

The foundational KalmanNet paper (Revach et al., 2022) embeds an RNN **inside** the Kalman
filter's recursion rather than replacing the filter: at each step, the RNN outputs the **Kalman
gain matrix directly** (subsuming what the analytic `P`-covariance recursion computes in a
classical filter), fed features derived from the innovation (`z − h(x̂)`) and consecutive
state-estimate differences. Critically for TraTrac:

- **It requires a known state-transition/observation model** — the prediction step is still
  `x̂ₖ|ₖ₋₁ = f(x̂ₖ₋₁|ₖ₋₁)`. It does **not** learn vehicle dynamics from scratch.
- It does **not** discard the CA structure `application/kalman.py` already has. `_transition`
  (the constant-acceleration `f`) and `_H` (the position-only observation model `h`) are
  exactly the "structural SS model" KalmanNet is designed to wrap — they would be **reused
  unchanged**; only the gain computation (today: `_update`'s analytic `gain = (p @ H.T) / s`)
  would be replaced by a learned one.

This is a materially smaller integration than it might sound: TraTrac's CA model is already
correct per the literature (`RESEARCH_NOTES.md`); this upgrade targets specifically the one
piece that's currently a hand-tuned constant.

## Reconsidered recommendation: plain **unsupervised** KalmanNet, not MAML-KalmanNet first

`SMOOTHING.md`/`TECH_STACK.md` named **MAML-KalmanNet** as the more "relevant" descendant,
reasoning that "TraTrac won't have a large labeled-trajectory dataset to train on." This spike's
research changes that recommendation:

- **Plain KalmanNet has a genuinely unsupervised training mode** (Revach et al., *Unsupervised
  Learned Kalman Filtering*, 2021) that needs **no ground-truth states at all** — the loss is
  the squared innovation magnitude `‖Δy_t‖²`, computable from the observed sequence alone,
  because KalmanNet's hybrid architecture already predicts the next *observation* internally as
  part of running the filter. Unsupervised KalmanNet is reported to reach performance close to
  the supervised variant when noise statistics are unknown.
- **TraTrac already produces exactly the input this needs, as a side effect of normal use**:
  every `tratrac` run's Parquet track record (`export.out`) is a set of noisy pixel-centroid
  sequences — no manual labeling step, no separate data-collection effort. This directly
  answers the blocker that made MAML look necessary.
- **MAML-KalmanNet solves a different problem than "no labels"**: meta-learning is for fast
  adaptation across many distinct *tasks* (e.g. wildly different noise/motion regimes) from very
  few examples per task. TraTrac's actual problem — one long-running deployment accumulating
  plenty of unlabeled trajectories over time — is a better match for plain unsupervised training
  than for meta-learning across few-shot tasks. MAML-KalmanNet remains a plausible **second**
  step (e.g. if TraTrac needs a filter that adapts quickly to a brand-new camera altitude/site
  with only a handful of recorded clips), not the first one to build.

**Revised recommendation: target plain KalmanNet, trained unsupervised on TraTrac's own
recorded track data.** Recursive KalmanNet (positive-definite/unbiased covariance propagation)
remains worth a second look once a first KalmanNet integration exists and its failure modes
(if any — e.g. ill-conditioned or overconfident gains) are actually observed, rather than
designed against speculatively.

## Training data plan (the concrete part this spike adds)

- **Source**: any `tratrac` run's Parquet track record already recorded for this project (no
  new collection process). Each track (one `track_id`'s sequence of `(cx, cy, frame_index)`) is
  one training sequence.
- **Sequence length**: a civil-engineering deployment of unsupervised KalmanNet (Wang et al.,
  2024, structural response reconstruction) trained on segments as short as **10 seconds** and
  generalized to reconstructing **39-second** sequences — evidence that short segments, cut from
  longer recordings, are enough for the network to learn gain computation rather than
  overfitting to specific trajectories. TraTrac's own tracks (bounded by a vehicle's visible
  dwell time in frame) are naturally in a comparable range, so no special long-sequence
  collection effort looks necessary — an assumption to verify empirically, not taken on faith.
- **Diversity, not volume, is the open question.** The literature evidence above speaks to
  *sequence length* being forgiving; it says nothing about how many *distinct* tracks/sites/
  vehicle-motion-regimes are needed before the learned gain generalizes rather than
  overfitting to one camera setup. This project has no real-footage corpus yet
  (no full real-footage validation pass has cleared `validate_trj` yet — see the B1/B2 issues
  in GitHub Issues) — so there is currently not even one deployment's worth of diverse recorded
  tracks to train against, let alone several. **This is the actual blocker**, not "no labels."

## Proposed integration surface (design-only; not implemented)

- New module, e.g. `application/kalmannet.py`, implementing the **same per-axis CA structure**
  as `application/kalman.py` (`_transition`, `_H` reused or duplicated) but with `_update`'s
  gain computation swapped for a small RNN's output.
- Selected via an explicit opt-in (mirroring how `ego_motion.enabled` and `DetectorChoice` work
  elsewhere in this repo — "off is explicit," no silent default change): a new
  `tratrac-postprocess` flag, not a replacement for `--pos-noise`/`--jerk`, so the classical
  filter stays available and remains the default until this is validated.
- Training happens **offline**, outside the `tratrac`/`tratrac-postprocess` runtime — a
  standalone training script (not designed here) producing a checkpoint the new module loads,
  the same shape as the detector/tracker checkpoints already loaded elsewhere in this repo.
- **Evaluation**: `scripts/validate_trj.py` compliance comparison against the classical
  fixed-parameter filter, the same acceptance-gate pattern already used for every other
  quality upgrade in this repo (detector swap, ego-motion, world projection) — not a new
  validation philosophy.

## Open questions (deliberately not resolved here)

1. **RNN input feature design** — the architecture summary found in this research explicitly
   calls this "domain and problem dependent"; TraTrac's specific feature set (what exactly to
   feed the gain network beyond the raw innovation) needs its own design pass once real
   training data exists to iterate against.
2. **Training compute** — KalmanNet's per-timestep network is small (nothing like a detection
   or segmentation foundation model), so CPU training may be *practical* here even though this
   project's other CPU-blocked upgrades (YOLO-OBB fine-tuning, DINOv3, SAM 3) are not — but this
   is an untested assumption, not a verified claim, and should be checked before being relied on.
3. **Diversity requirement** (above) is the real open blocker — resolving it needs either real
   multi-site footage or a deliberate synthetic-diversity strategy (varied synthetic
   noise/motion regimes), neither designed here.
4. **Framework/dependency choice** — no PyTorch training-loop dependency exists in this project
   yet (today's `torch` pin is inference-only, CPU wheels); adding a training loop is a real,
   if small, new dependency surface, not assumed away by this doc.

## Sources

- [KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics (Revach et al., foundational paper)](https://arxiv.org/abs/2107.10043)
- [KalmanNet architecture summary — gain-network mechanism, inputs, model requirements](https://www.emergentmind.com/topics/neural-network-aided-kalman-filtering)
- [Unsupervised Learned Kalman Filtering (Revach et al., 2021) — innovation-magnitude loss, no ground truth needed](https://arxiv.org/abs/2110.09005)
- [Structural Dynamic Response Reconstruction via Unsupervised RNN-aided Kalman Filter (Wang et al., 2024) — short-segment training, long-sequence generalization](https://onlinelibrary.wiley.com/doi/full/10.1155/2024/7481513)
- [MAML-KalmanNet: model-agnostic meta-learning variant (IEEE TSP 2025)](https://dl.acm.org/doi/abs/10.1109/TSP.2025.3540018)
- [Recursive KalmanNet — positive-definite/unbiased covariance propagation](https://onlinelibrary.wiley.com/doi/10.1002/acs.3982)
- [Latent-KalmanNet — learned Kalman filtering from high-dimensional signals](https://arxiv.org/pdf/2304.07827)
