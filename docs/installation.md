# Installation options

## One-command Bash installer

From a cloned checkout:

```bash
bash install.sh
```

Directly from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/TensorLink-AI/headwater/main/install.sh | bash
```

The direct `curl` form requires the repository to be public. While it is
private, clone it with an authenticated GitHub account and run `bash install.sh`.
The installer detects an authenticated `gh` CLI and uses it to download private
source archives without placing the token in a URL or subprocess argument.

The installer requires Python 3.11 or newer. It creates an isolated virtual
environment under `~/.local/share/headwater/releases/` and links the executable
to `~/.local/bin/headwater`. It does not use `sudo` or modify system Python.

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
  https://raw.githubusercontent.com/TensorLink-AI/headwater/v0.1.0/install.sh \
  | bash -s -- --version v0.1.0
```

For higher assurance, download the script first, inspect it, and verify a known
checksum before running it.

Supported installer overrides:

```bash
bash install.sh --help
bash install.sh --version main
bash install.sh --install-root /opt/headwater --bin-dir /usr/local/bin
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
uv tool install 'git+https://github.com/TensorLink-AI/headwater.git@main'
```

## Install from PyPI

After the first tagged release has been published:

```bash
pipx install headwater-forecast
# or
uv tool install headwater-forecast
```

Until that release exists, use the checkout, GitHub, or Bash installer methods.

## Install Parquet support

The default Bash installer installs CSV support only. With uv or pip, request:

```bash
uv tool install 'headwater-forecast[parquet]'
```
