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
INSTALL_ROOT="$(cd -- "$INSTALL_ROOT" && pwd)"
BIN_DIR="$(cd -- "$BIN_DIR" && pwd)"
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RELEASE_DIR="$INSTALL_ROOT/releases/$RELEASE_ID"
INSTALL_SUCCEEDED=0

cleanup_failed_install() {
  if [[ "$INSTALL_SUCCEEDED" != 1 ]] &&
     [[ "$(readlink "$BIN_DIR/gnomon" 2>/dev/null || true)" != "$RELEASE_DIR/bin/gnomon" ]]; then
    rm -rf "$RELEASE_DIR"
  fi
}
trap cleanup_failed_install EXIT
mkdir -p "$RELEASE_DIR"
touch "$RELEASE_DIR/.installing"

SOURCE_COMMIT=""
SOURCE_ARCHIVE="$RELEASE_DIR/gnomon-source.tar.gz"
if [[ -n "$LOCAL_SOURCE" ]]; then
  printf 'Installing Gnomon from the local checkout at %s using %s...\n' "$SCRIPT_DIR" "$PYTHON_BIN"
else
  printf 'Installing Gnomon from %s at %s using %s...\n' "$REPOSITORY" "$VERSION" "$PYTHON_BIN"
fi
"$PYTHON_BIN" -m venv "$RELEASE_DIR"
"$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check --upgrade pip
if [[ -n "$LOCAL_SOURCE" ]]; then
  if command -v git >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true)"
  fi
  "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SCRIPT_DIR"
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  SOURCE_COMMIT="$(gh api "repos/$REPOSITORY/commits/$VERSION" --jq .sha)"
  [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf 'Could not resolve source commit.\n' >&2; exit 1; }
  gh api "repos/$REPOSITORY/tarball/$SOURCE_COMMIT" > "$SOURCE_ARCHIVE"
  GNOMON_BUILD_COMMIT="$SOURCE_COMMIT" "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SOURCE_ARCHIVE"
else
  SOURCE_COMMIT="$("$PYTHON_BIN" - "$REPOSITORY" "$VERSION" <<'PY'
import json, sys, urllib.parse, urllib.request
repository, ref = sys.argv[1:]
url = f"https://api.github.com/repos/{repository}/commits/{urllib.parse.quote(ref, safe='')}"
request = urllib.request.Request(url, headers={"User-Agent": "gnomon-installer"})
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.load(response)["sha"])
PY
)"
  [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf 'Could not resolve source commit.\n' >&2; exit 1; }
  SOURCE_URL="https://github.com/$REPOSITORY/archive/$SOURCE_COMMIT.tar.gz"
  GNOMON_BUILD_COMMIT="$SOURCE_COMMIT" "$RELEASE_DIR/bin/python" -m pip install --disable-pip-version-check "$SOURCE_URL"
fi
rm -f "$SOURCE_ARCHIVE"
"$RELEASE_DIR/bin/gnomon" capabilities >/dev/null
cp -- "${BASH_SOURCE[0]}" "$RELEASE_DIR/install.sh"
# Use only the standard library here so this installer can also activate older
# tagged packages that predate the release-management module.
"$RELEASE_DIR/bin/python" - "$INSTALL_ROOT" "$BIN_DIR/gnomon" "$REPOSITORY" "$VERSION" "$LOCAL_SOURCE" "$SOURCE_COMMIT" <<'PY'
from datetime import datetime, timezone
import fcntl, hashlib, json, os
from pathlib import Path
import sys
from uuid import uuid4
import gnomon

root, command, repository, ref, local, commit = sys.argv[1:]
prefix = Path(sys.prefix).resolve()
package = Path(gnomon.__file__).parent
try:
    from gnomon.build_info import build_info
    build = build_info()
except ImportError:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    build = {"package_version": gnomon.__version__, "commit": commit or None,
             "source_sha256": digest.hexdigest(), "provenance": "installer_receipt"}
receipt = {"schema_version": 1, "release_id": prefix.name, "install_root": root, "command": command,
           "repository": repository, "requested_ref": "local" if local else ref,
           "source": "local" if local else "github", "installed_at": datetime.now(timezone.utc).isoformat(), "build": build}
(prefix / "gnomon-install.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
link = Path(command)
temporary = link.parent / (".gnomon-" + uuid4().hex)
with (Path(root) / ".install.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        (prefix / ".installing").unlink()
        temporary.symlink_to(prefix / "bin" / "gnomon")
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)
print("Build: " + build.get("build_id", str(build)))
PY
INSTALL_SUCCEEDED=1
trap - EXIT

printf 'Gnomon installed successfully: %s\n' "$BIN_DIR/gnomon"
printf 'Python API: %s -c "import gnomon"\n' "$RELEASE_DIR/bin/python"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  printf 'Add %s to PATH, then run: gnomon capabilities\n' "$BIN_DIR"
else
  printf 'Run: gnomon capabilities\n'
fi
