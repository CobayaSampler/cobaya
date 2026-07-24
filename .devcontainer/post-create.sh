#!/usr/bin/env bash
# Provision the cobaya dev container. Heavy/pure dependencies are already
# baked into the image (see Dockerfile); this script only does what needs
# the mounted workspace: install cobaya itself (editable) into the
# devcontainer-local venv, wire up pre-commit, and download the cosmo test
# data into the persistent packages volume.
# Idempotent: safe to re-run (uv/cobaya-install detect what's already there).
set -euo pipefail

PYTHON="${COBAYA_VENV}/bin/python3"

echo ">>> Fixing ownership of packages volume (${COBAYA_PACKAGES_PATH})"
# Named volumes mount root-owned; hand the packages dir to the dev user.
sudo chown -R vscode:vscode "${COBAYA_PACKAGES_PATH}"

echo ">>> Installing cobaya (editable) into ${COBAYA_VENV}"
uv pip install --python "${PYTHON}" -e .

echo ">>> Wiring up pre-commit hooks"
# .git/hooks lives on the bind-mounted workspace. On Docker Desktop's Windows
# bind mounts (9p/drvfs), chmod'ing a freshly written hook script there is
# unreliable (intermittent EPERM/EACCES) and, worse, can leave a
# container-only interpreter path baked into a hook file inside the *host's*
# real .git directory (breaking `git commit` on the host). Route around it
# entirely: point core.hooksPath at a container-local directory and write
# the hook script there ourselves, instead of running `pre-commit install`
# (which always targets .git/hooks).
HOOKS_DIR="${HOME}/.cache/pre-commit-hooks"
mkdir -p "${HOOKS_DIR}"
git config core.hooksPath "${HOOKS_DIR}"
cat > "${HOOKS_DIR}/pre-commit" <<HOOK
#!/usr/bin/env bash
exec "${PYTHON}" -m pre_commit hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit --hook-dir "${HOOKS_DIR}" -- "\$@"
HOOK
chmod +x "${HOOKS_DIR}/pre-commit"
"${COBAYA_VENV}/bin/pre-commit" install-hooks

if [ "${COBAYA_DEVCONTAINER_INSTALL_DATA:-1}" = "1" ]; then
    echo ">>> Installing cosmology test packages/data into ${COBAYA_PACKAGES_PATH}"
    echo "    (one-time per volume; large Planck likelihoods skipped via COBAYA_INSTALL_SKIP)"
    "${PYTHON}" -m cobaya.install cosmo-tests --no-progress-bars --skip-global
else
    echo ">>> Skipping cobaya data install (COBAYA_DEVCONTAINER_INSTALL_DATA=0)"
fi

echo ">>> Devcontainer ready. See AGENTS.md for how to run tests."
