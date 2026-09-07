# Installation

Python 3.11–3.13. The core has no required third-party dependencies.

Install Gnomon:

```bash
python -m pip install 'gnomon-forecast==1.0.0'
gnomon --version
gnomon infer --provider last_value --request '{"history":[1,2,3],"horizon":2}'
```

From a checkout use `python -m pip install .`.
For development use `python -m pip install -e '.[dev]'`.
Optional file readers are `.[parquet]` and `.[excel]`.
Install your model software in your own environment and register a callable or
factory; Gnomon has no per-library installer or built-in TSFM catalogue.

See [offline installation](offline-installation.md) and
[provider configuration](production/INFERENCE.md).
