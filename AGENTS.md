# Running in the devcontainer

The commands below assume the provided
**dev container** (`.devcontainer/`), which reproduces the CI workflow
(`.github/workflows/tests.yml`) locally.

## Python path / interpreter
- Interpreter: a **devcontainer-local venv** at
  `/home/vscode/.local/share/cobaya-devcontainer/.venv`

## Packages path (downloaded data/code)

- External packages (cosmology codes, likelihood data) live at
  `COBAYA_PACKAGES_PATH=/home/vscode/cobaya/packages`, backed by the persistent
  `cobaya-packages` volume (subdirs `code/` and `data/`).
- `cobaya-install` also records this path in `~/.config/cobaya/config.yaml`, so
  `cobaya run ...` finds it even without the env var (the env var takes precedence).
- Check the resolved path: `python -m cobaya.install --show-packages-path`.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `COBAYA_PACKAGES_PATH` | Where external packages/data are installed. |
| `COBAYA_INSTALL_SKIP` | Comma-separated keywords of components to skip on install (defaults skip the multi-GB Planck likelihoods and polychord). |
| `COBAYA_TEST_SKIP` | Comma-separated keywords of tests to skip. |
| `COBAYA_DEVCONTAINER_INSTALL_DATA` | `1` (default) install cosmo test data on create; `0` to skip. |

## Running tests

These mirror the CI job steps (`--skip-not-installed` xfails components whose
dependencies aren't installed instead of erroring):

```bash
# Lint (ruff) — CI runs this on cobaya/
ruff check cobaya/

# Fast / non-cosmology tests (parallel)
pytest tests/ -n auto -k "not cosmo" --skip-not-installed

# Cosmology tests (needs the cosmo data install; run with fewer workers)
pytest tests/ -vv -s -k "cosmo" -n 2 --skip-not-installed

# MPI tests (2 ranks; --oversubscribe for machines with few cores)
mpiexec -np 2 --oversubscribe python -m pytest -m mpi tests/

# A single test
pytest tests/test_cosmo_planck_2018.py::test_planck_2018_p_camb -s
```

To (re)install components on demand:

```bash
python -m cobaya.install cosmo-tests --no-progress-bars   # all tested cosmo packages
python -m cobaya.install <component_or_yaml>              # a specific one
python -m cobaya.install polychord                        # skipped by default (see COBAYA_INSTALL_SKIP)
```

## Code style

Formatting and linting use **ruff** (config in `pyproject.toml`, line length 90,
double quotes). Format-on-save and import organization are enabled via
`.vscode/settings.json`. Run `ruff check cobaya/` and `ruff format` before committing.

`pre-commit` hooks (`.pre-commit-config.yaml`: trailing-whitespace, pyupgrade,
ruff-check, ruff-format) are installed automatically by `post-create.sh`
(`pre-commit install --install-hooks`), so they run on `git commit`. Run them
manually with `pre-commit run --all-files`.
