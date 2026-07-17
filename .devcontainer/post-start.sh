#!/usr/bin/env bash
set -euo pipefail

# VS Code's Dev Containers extension copies the host's ~/.gitconfig into the
# container so credentials/identity carry over. On a Windows host that file
# can contain a `safe.directory` entry with a Windows-style path (e.g.
# `c:/Work`, added by Git for Windows / VS Code on the host side). Git
# inside the (Linux) container rejects that as "not absolute" and prints a
# warning on every single git invocation. The system config already trusts
# everything (`git config --system --add safe.directory '*'` in the
# Dockerfile), so these entries are redundant anyway: strip any that aren't
# a real Linux absolute path or the `*` wildcard.
if [ -f "${HOME}/.gitconfig" ]; then
    mapfile -t _valid_safe_dirs < <(git config --global --get-all safe.directory 2>/dev/null | grep -E '^(/|\*$)' || true)
    if git config --global --get-all safe.directory >/dev/null 2>&1; then
        git config --global --unset-all safe.directory
        for _dir in "${_valid_safe_dirs[@]}"; do
            git config --global --add safe.directory "${_dir}"
        done
    fi
    unset _valid_safe_dirs _dir
fi

CODEX_DIR="${HOME}/.codex"
CONFIG_FILE="${CODEX_DIR}/config.toml"
TMP_FILE="$(mktemp)"

trap 'rm -f "${TMP_FILE}"' EXIT

mkdir -p "${CODEX_DIR}"

if [ -f "${CONFIG_FILE}" ]; then
    awk '
        /^[[:space:]]*approval_policy[[:space:]]*=/ { next }
        /^[[:space:]]*sandbox_mode[[:space:]]*=/ { next }
        { print }
    ' "${CONFIG_FILE}" > "${TMP_FILE}"
else
    : > "${TMP_FILE}"
fi

cat >> "${TMP_FILE}" <<'EOF'
approval_policy = "never"
sandbox_mode = "danger-full-access"
EOF

mv "${TMP_FILE}" "${CONFIG_FILE}"
trap - EXIT