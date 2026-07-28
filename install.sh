#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY="${HEADWATER_REPOSITORY:-TensorLink-AI/headwater}"
VERSION="${HEADWATER_VERSION:-main}"
INSTALL_ROOT="${HEADWATER_INSTALL_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/headwater}"
BIN_DIR="${HEADWATER_BIN_DIR:-${XDG_BIN_HOME:-${HOME}/.local/bin}}"

usage() {
  printf '%s\n' \
    "Install Headwater in an isolated Python environment." \
    "" \
    "Usage: bash install.sh [options]" \
    "" \
    "Options:" \
    "  --version REF       Git tag, branch, or commit (default: main)" \
    "  --repository OWNER/REPO  Source repository" \
    "  --install-root DIR  Environment storage directory" \
    "  --bin-dir DIR       Command directory" \
    "  -h, --help          Show this help"
}

while (($#)); do
  case "$1" in
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

if [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid repository: %s\n' "$REPOSITORY" >&2
  exit 2
fi
if [[ ! "$VERSION" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$VERSION" == *..* ]]; then
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
  printf 'Headwater requires Python 3.11 or newer.\n' >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT/releases" "$BIN_DIR"
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RELEASE_DIR="$INSTALL_ROOT/releases/$RELEASE_ID"
LINK_TMP="$BIN_DIR/.headwater-$RELEASE_ID"

cleanup_failed_install() {
  rm -f "$LINK_TMP"
  if [[ ! -x "$RELEASE_DIR/bin/headwater" ]]; then
    rm -rf "$RELEASE_DIR"
  fi
}
trap cleanup_failed_install EXIT

SOURCE_URL="https://github.com/$REPOSITORY/archive/$VERSION.tar.gz"
SOURCE_ARCHIVE="$RELEASE_DIR/headwater-source.tar.gz"
printf 'Installing Headwater from %s at %s using %s...\n' "$REPOSITORY" "$VERSION" "$PYTHON_BIN"
"$PYTHON_BIN" -m venv "$RELEASE_DIR"
"$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check --upgrade pip
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh api "repos/$REPOSITORY/tarball/$VERSION" > "$SOURCE_ARCHIVE"
  "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SOURCE_ARCHIVE"
else
  "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SOURCE_URL"
fi
rm -f "$SOURCE_ARCHIVE"
"$RELEASE_DIR/bin/headwater" capabilities >/dev/null

ln -s "$RELEASE_DIR/bin/headwater" "$LINK_TMP"
mv -f "$LINK_TMP" "$BIN_DIR/headwater"
trap - EXIT

printf 'Headwater installed successfully: %s\n' "$BIN_DIR/headwater"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  printf 'Add %s to PATH, then run: headwater capabilities\n' "$BIN_DIR"
else
  printf 'Run: headwater capabilities\n'
fi
