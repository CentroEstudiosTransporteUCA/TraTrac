# syntax=docker/dockerfile:1

# Multi-stage PyTorch+CUDA build (Group F, docs/roadmap/mvp7.md's "Exploration pass" design):
# a `-devel` base (compilers/headers) builds the venv, a matching `-runtime` base (no build
# tools, smaller) ships it — cited ~60% image-size reduction from this pattern alone.
#
# The devel/runtime tag pair below was verified against Docker Hub's registry (as of
# 2026-09-10) to exist and match this project's exact `torch>=2.12.0` pin, using the cu130
# CUDA index Group A1 (docs/IMPLEMENTATION_PLAN.md) already verified resolves for that pin
# (cu128 only publishes wheels up to torch 2.11.0). Not build-tested — no `docker` available
# in the environment this was written in. Re-verify both tags still exist before relying on
# this if building much later; PyTorch's supported CUDA versions shift with each release.
#
# Not decided here: which CUDA/cuDNN version to pin for a *specific* deployment GPU (this
# picks the version already validated for dev/training, not a deployment target).

ARG PYTORCH_DEVEL_IMAGE=pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel
ARG PYTORCH_RUNTIME_IMAGE=pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

FROM ${PYTORCH_DEVEL_IMAGE} AS build

# The base image already ships the Python this project targets (3.12); don't let uv
# download its own.
ENV UV_PYTHON_DOWNLOADS=never
# uv's documented Docker pattern (astral-sh/uv's own install docs). Pin to the version this
# project develops against (`.python-version`-adjacent — check `uv --version` locally if this
# drifts) rather than `:latest`, so a rebuild months later doesn't silently pick up a
# different uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# The one manual edit Group A1 (docs/IMPLEMENTATION_PLAN.md) decided on: point
# [tool.uv.sources]'s torch/torchvision index at the CUDA wheel index instead of the CPU one
# this repo defaults to. Deliberately explicit here, not a silent default baked into
# pyproject.toml itself — A1 rejected a uv-extras approach that let bare `uv sync`/`uv run`
# silently resolve to CUDA in a normally-CPU dev environment. A CUDA base image's build stage
# is unambiguously GPU-bound already, so the same rewrite is safe and expected in this
# Dockerfile specifically.
RUN sed -i 's#https://download.pytorch.org/whl/cpu#https://download.pytorch.org/whl/cu130#' \
	pyproject.toml uv.lock

# Re-lock against the rewritten index first (the checked-in uv.lock pins CPU wheel hashes/
# URLs, which won't satisfy a CUDA index) in its own layer so dependency installs are cached
# separately from copying the source tree below.
RUN uv lock && uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

FROM ${PYTORCH_RUNTIME_IMAGE} AS runtime

WORKDIR /app
# uv installs the project in editable mode by default, so the venv's .pth reference needs
# the source tree present at the same path — not just the venv itself. This is still the
# multi-stage build's real win: no compiler toolchain, no build-only headers, no uv, none of
# the devel base's several-GB of extra layers ship in the final image.
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["tratrac"]
