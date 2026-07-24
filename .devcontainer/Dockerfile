# Cobaya development container.
# Base image ships a non-root `vscode` user (uid 1000) with passwordless sudo,
# which Claude Code requires (it refuses to run as root).
ARG VARIANT=3.14-bookworm
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

# System build dependencies:
#   gfortran / build-essential / cmake : compile CAMB, CLASS, polychord, ...
#   libopenmpi-dev / openmpi-bin       : OpenMPI + mpicc for mpi4py and MPI tests
#   ripgrep                            : used by Claude Code's search tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        gfortran build-essential cmake \
        libopenmpi-dev openmpi-bin \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Let git trust the bind-mounted workspace (host uid differs from container uid).
RUN git config --system --add safe.directory '*'

# uv: used to create/populate the devcontainer-local venv below.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# The repo's working copy on the host has its own `.venv/` (created by the
# host OS, e.g. Windows: `.venv/Scripts/python.exe`); that folder is
# bind-mounted into the container too. If VS Code's Python/Ruff environment
# auto-detection is left to find "the workspace venv" it discovers that
# host-layout folder and tries to run it inside Linux, which fails (wrong
# layout entirely). Keeping the container's venv under $HOME instead of
# ${workspaceFolder}/.venv avoids that clash; devcontainer.json points
ENV COBAYA_VENV=/home/vscode/.local/share/cobaya-devcontainer/.venv \
    UV_CACHE_DIR=/home/vscode/.cache/uv \
    PRE_COMMIT_HOME=/home/vscode/.cache/pre-commit \
    RUFF_CACHE_DIR=/home/vscode/.cache/ruff \
    CODEX_HOME=/home/vscode/.codex

# Pre-install everything that doesn't depend on the (not-yet-mounted) repo
# source, so it's baked into the image layer instead of reinstalled on every
# container create. `cobaya` itself is installed editable in
# postCreateCommand, once the workspace is actually mounted. Package list
# mirrors cobaya's own [devel]/[speed] extras (pyproject.toml) plus the extra
# test deps .github/workflows/tests.yml installs (pytest-xdist, matplotlib,
# coverage, numba for the "Latest Python" row).
RUN mkdir -p "$(dirname "$COBAYA_VENV")" "$PRE_COMMIT_HOME" "$UV_CACHE_DIR" "$RUFF_CACHE_DIR" "$CODEX_HOME" \
    && chown -R vscode:vscode /home/vscode/.local /home/vscode/.cache "$CODEX_HOME" \
    && su vscode -c "uv venv '$COBAYA_VENV' --python /usr/local/bin/python3" \
    && su vscode -c "uv pip install --python '$COBAYA_VENV/bin/python3' --upgrade \
        pip setuptools wheel \
        numpy scipy pandas PyYAML requests py-bobyqa GetDist rapidfuzz \
        packaging tqdm portalocker dill typing_extensions \
        ruff pre-commit pyupgrade \
        pytest pytest-xdist pytest-cov flaky coverage \
        mpi4py iminuit numba matplotlib"
