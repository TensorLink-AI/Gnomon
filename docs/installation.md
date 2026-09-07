# Installation

Python 3.11–3.13. The core has no required third-party dependencies.

Install Gnomon:

```bash
python -m pip install 'gnomon-forecast==1.0.1'
gnomon --version
gnomon infer --provider last_value --request '{"history":[1,2,3],"horizon":2}'
```

For a local custom provider, install Gnomon in the **same Python environment**
as the provider and all its dependencies, then use that environment's `gnomon`
or `python -m gnomon`. An isolated Gnomon environment cannot import PyTorch or
another model library installed in a different environment.

The repository also includes `install.sh` for a standalone CLI environment:

```bash
bash install.sh --version v1.0.1
```

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
