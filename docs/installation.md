# Installation

Python 3.11–3.13. The core has no required third-party dependencies.

The distribution name used by pip is **gnomon-forecast**; the Python import name
is **gnomon**. A `gnomon_forecast-*.dist-info` directory is package metadata, not
an importable module.

Install Gnomon:

```bash
python -m pip install 'gnomon-forecast==1.1.2'
gnomon --version
gnomon infer --provider last_value --request '{"history":[1,2,3],"horizon":2}'
```

For a local custom provider, install Gnomon in the **same Python environment**
as the provider and all its dependencies, then use that environment's `gnomon`
or `python -m gnomon`. An isolated Gnomon environment cannot import PyTorch or
another model library installed in a different environment.

The repository also includes `install.sh` for a standalone CLI environment:

```bash
bash install.sh --version v1.1.2
```

The management commands below require **Gnomon 1.1.0 or newer**. Before the
version tag/package is published, install main with `bash install.sh --version main`,
or use `bash install.sh --local` for your checkout.

The standalone installer creates a private environment under
`~/.local/share/gnomon/releases/` and places a `gnomon` symlink in `~/.local/bin`.
It does not install into your normal Python. To run the API in its environment:

```bash
gnomon environment
gnomon python -c 'from gnomon import GnomonSession; print(GnomonSession.from_config().capabilities()["runtime_version"])'
gnomon python your_script.py
```

`gnomon environment` reports the exact interpreter and package paths. If using
an older install without these commands, the installer prints the interpreter
path; run that interpreter directly. To use Gnomon in a notebook or an existing
application, install it into that application's Python using
`python -m pip install gnomon-forecast`, then `from gnomon import GnomonSession`.

Manage standalone installs explicitly:

```bash
gnomon releases
gnomon update --version main
gnomon rollback RELEASE_ID
gnomon releases --prune
gnomon releases --prune --keep 0 --apply
```

The release list shows which environment is active, its requested ref, version,
commit and source fingerprint. Legacy installs with unknown commits are marked
as such and fingerprinted without executing their code. Updates resolve a ref to
a commit before downloading and activate the new environment only after validation.
When that clean commit is already active and the source fingerprint still matches,
`gnomon update` returns `changed: false` and `reason: already_up_to_date` without
creating an environment. Direct `install.sh` remains an explicit reinstall.

Updates export the active environment's installed dependency versions and restore
them in the replacement, including provider packages and optional file-format
dependencies. Local/editable package sources must remain available. `pip check`
must pass before activation; failed exports, unavailable dependencies or incompatible
versions preserve the active environment. Gnomon itself is installed from the
selected Git commit, and pip is managed by the installer.
Failed installs are cleaned up. Rollback atomically selects an existing release.
If an update or rollback selects a release that predates these commands, use the returned `management_command`
path to manage installs from the newer environment.

Pruning previews candidates unless `--apply` is supplied. It keeps one inactive
release by default and always protects the active, running and in-progress
environments. New standalone environments hold a shared lock for the lifetime of
each Python process, including direct API use and MCP servers. Linux process
inspection also detects older environments without these locks. If an older
environment cannot be inspected reliably, the list reports `usage: unknown` and
pruning preserves it. Processes detected after a preview are rechecked before
removal and reported in `skipped_in_use` when necessary.
These commands manage `install.sh` environments; pip installs are
managed with that interpreter's pip.

`gnomon --version` reports a qualified build ID, such as
`1.1.0+g0123456789ab.s0123456789ab.dirty`. The Git commit, source SHA-256,
package version and dirty state are also exposed by `gnomon environment` and
capabilities, and embedded in wheels and source distributions. The source hash
covers Gnomon's Python source files; it is distinct from the wheel's byte hash.
Source archives without commit information report a source hash and an unknown
commit. The package version used by pip remains the release/development version.

That isolation suits the built-in baselines and remote providers. For a local
custom provider, use the shared model environment described above, or explicitly
install the provider and its dependencies into the environment created by the
script.

From a checkout use `python -m pip install .`.
For development use `python -m pip install -e '.[dev]'`.
Optional file readers are `.[parquet]` and `.[excel]`.
Register the model as a callable or factory; Gnomon has no per-library installer
or built-in TSFM catalogue.

See [offline installation](offline-installation.md) and
[provider configuration](production/INFERENCE.md).
