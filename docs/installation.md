# Installation options

## One-command Bash installer

From a cloned checkout:

```bash
bash install.sh
```

Directly from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/TensorLink-AI/Gnomon/main/install.sh | bash
```

The direct `curl` form requires the repository to be public. While it is
private, clone it with an authenticated GitHub account and run `bash install.sh`.
The installer detects an authenticated `gh` CLI and uses it to download private
source archives without placing the token in a URL or subprocess argument.

The installer requires Python 3.11 or newer. It creates an isolated virtual
environment under `~/.local/share/gnomon/releases/` and links the executable
to `~/.local/bin/gnomon`. It does not use `sudo` or modify system Python.

If `~/.local/bin` is not on `PATH`, add it in your shell configuration:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Pin an installer and release

Executing a mutable `main` script is convenient but less suitable for controlled
environments. After release tags exist, pin both the downloaded installer and
the installed source:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/TensorLink-AI/Gnomon/v0.1.0/install.sh \
  | bash -s -- --version v0.1.0
```

For higher assurance, download the script first, inspect it, and verify a known
checksum before running it.

Supported installer overrides:

```bash
bash install.sh --help
bash install.sh --version main
bash install.sh --install-root /opt/gnomon --bin-dir /usr/local/bin
```

Writing to system paths may require appropriate permissions. Re-running the
installer creates a new isolated release and atomically repoints the command;
older release environments are retained for manual rollback or removal.

## Install with uv

From a checkout:

```bash
uv tool install .
```

From GitHub:

```bash
uv tool install 'git+https://github.com/TensorLink-AI/Gnomon.git@main'
```

## Install from PyPI

The published distribution is `gnomon-forecast`; it installs the `gnomon`
command:

```bash
pipx install gnomon-forecast
# or
uv tool install gnomon-forecast
```

Pin `gnomon-forecast==<version>` in controlled environments. Use the checkout
or GitHub methods above when testing unreleased changes.

## Install Parquet support

The default Bash installer installs CSV support only. With uv or pip, request:

```bash
uv tool install 'gnomon-forecast[parquet]'
```
