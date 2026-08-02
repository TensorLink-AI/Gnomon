#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY="${GNOMON_REPOSITORY:-TensorLink-AI/Gnomon}"
VERSION="${GNOMON_VERSION:-main}"
# --local installs the checkout this script lives in, rather than fetching
# from GitHub. Without it, `cd Gnomon && bash install.sh` silently installs
# the remote default branch — not the source the reader just changed.
LOCAL_SOURCE="${GNOMON_LOCAL:-}"
INSTALL_ROOT="${GNOMON_INSTALL_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/gnomon}"
BIN_DIR="${GNOMON_BIN_DIR:-${XDG_BIN_HOME:-${HOME}/.local/bin}}"

usage() {
  printf '%s\n' \
    "Install Gnomon in an isolated Python environment." \
    "" \
    "Usage: bash install.sh [options]" \
    "" \
    "Options:" \
    "  --local             Install this checkout instead of fetching from GitHub" \
    "  --version REF       Git tag, branch, or commit (default: main)" \
    "  --repository OWNER/REPO  Source repository" \
    "  --install-root DIR  Environment storage directory" \
    "  --bin-dir DIR       Command directory" \
    "  -h, --help          Show this help"
}

while (($#)); do
  case "$1" in
    --local)
      LOCAL_SOURCE=1
      shift
      ;;
    --version)
      VERSION="${2:?--version requires a value}"
      shift 2
      ;;
    --repository)
      REPOSITORY="${2:?--repository requires a value}"
      shift 2
      ;;
    --install-root)
      INSTALL_ROOT="${2:?--install-root requires a value}"
      shift 2
      ;;
    --bin-dir)
      BIN_DIR="${2:?--bin-dir requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "$LOCAL_SOURCE" ]]; then
  if [[ ! -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    printf 'No pyproject.toml beside install.sh; --local needs a checkout.\n' >&2
    exit 2
  fi
elif [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid repository: %s\n' "$REPOSITORY" >&2
  exit 2
fi
if [[ -z "$LOCAL_SOURCE" ]] &&
   { [[ ! "$VERSION" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$VERSION" == *..* ]]; }; then
  printf 'Invalid version/ref: %s\n' "$VERSION" >&2
  exit 2
fi

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  printf 'Gnomon requires Python 3.11 or newer.\n' >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT/releases" "$BIN_DIR"
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RELEASE_DIR="$INSTALL_ROOT/releases/$RELEASE_ID"
LINK_TMP="$BIN_DIR/.gnomon-$RELEASE_ID"

cleanup_failed_install() {
  rm -f "$LINK_TMP"
  if [[ ! -x "$RELEASE_DIR/bin/gnomon" ]]; then
    rm -rf "$RELEASE_DIR"
  fi
}
trap cleanup_failed_install EXIT

SOURCE_URL="https://github.com/$REPOSITORY/archive/$VERSION.tar.gz"
SOURCE_ARCHIVE="$RELEASE_DIR/gnomon-source.tar.gz"
if [[ -n "$LOCAL_SOURCE" ]]; then
  printf 'Installing Gnomon from the local checkout at %s using %s...\n' "$SCRIPT_DIR" "$PYTHON_BIN"
else
  printf 'Installing Gnomon from %s at %s using %s...\n' "$REPOSITORY" "$VERSION" "$PYTHON_BIN"
fi
"$PYTHON_BIN" -m venv "$RELEASE_DIR"
"$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check --upgrade pip
if [[ -n "$LOCAL_SOURCE" ]]; then
  "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SCRIPT_DIR"
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh api "repos/$REPOSITORY/tarball/$VERSION" > "$SOURCE_ARCHIVE"
  "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SOURCE_ARCHIVE"
else
  "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SOURCE_URL"
fi
rm -f "$SOURCE_ARCHIVE"
"$RELEASE_DIR/bin/gnomon" capabilities >/dev/null

ln -s "$RELEASE_DIR/bin/gnomon" "$LINK_TMP"
mv -f "$LINK_TMP" "$BIN_DIR/gnomon"
trap - EXIT

printf 'Gnomon installed successfully: %s\n' "$BIN_DIR/gnomon"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  printf 'Add %s to PATH, then run: gnomon capabilities\n' "$BIN_DIR"
else
  printf 'Run: gnomon capabilities\n'
fi
