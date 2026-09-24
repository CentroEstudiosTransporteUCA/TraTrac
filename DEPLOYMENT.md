# Docker deployment

A multi-stage build: a `-devel` base (compilers/headers) for the build stage, a matching
`-runtime` base (no build tools, smaller) for the final stage, copying over only the built
virtualenv/artifacts — this pattern is cited as cutting image size by roughly 60% versus a
single-stage devel image.

- **Build stage**: `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel` (matches this project's
  `torch>=2.12.0` pin and the `cu130` CUDA index that resolves for it — see `pyproject.toml`),
  `uv sync` with `[tool.uv.sources]` rewritten at build time (via `sed`, visible in the
  `Dockerfile`, not baked into the checked-in `pyproject.toml`) to point at the CUDA
  torch/torchvision index instead of the CPU one this repo defaults to for local dev — the same
  manual-edit decision this project always uses for the CPU→CUDA swap (see `CLAUDE.md`
  Dependency Notes), reused for deployment rather than dev, and deliberately not a `uv` extras
  approach (bare `uv sync`/`uv run` silently resolving to CUDA would be exactly as unsafe in a
  Dockerfile's own dev-facing commands as it was found to be locally — the CUDA base image
  itself, not an extra flag, is what makes this build unambiguously GPU-bound).
- **Runtime stage**: the matching `-runtime` base, `COPY --from=build` the built `.venv` +
  `src/` (the venv's editable-install `.pth` needs the source tree present, not just the venv),
  no compiler toolchain or `uv` binary shipped.

Not designed here: which CUDA/cuDNN version to pin for the actual deployment GPU (this picks the
version already validated for local dev, not a specific deployment target — re-pin if they
differ).

## Licensing

`boxmot`/`ultralytics` are AGPL-3.0; TraTrac is GPL-3.0, and GPLv3 §13 explicitly permits the
combination (see `CLAUDE.md` Dependency Notes). A deployed image does carry AGPL's own
network-interaction clause if the image is ever offered as a hosted/network service, which a
Docker build is a plausible step toward.

## Not build-tested

The `pytorch/pytorch` devel/runtime tag pair and the `ghcr.io/astral-sh/uv` version pin were
both confirmed to exist via their registries' own APIs, but an actual `docker build` has not
been run in this environment (no `docker` available where the `Dockerfile` was written) — run
one before trusting this in a real deployment.
