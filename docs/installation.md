# Installation

Python 3.11–3.13. The core has no required third-party dependencies.

This checkout is the unreleased 0.9 development line. Install it from source:

```bash
python -m pip install .
gnomon --version
gnomon infer --provider last_value --request '{"history":[1,2,3],"horizon":2}'
```

For development use `python -m pip install -e '.[dev]'`.
Optional file readers are `.[parquet]` and `.[excel]`.
Install your model software in your own environment and register a callable or
factory; Gnomon has no per-library installer or built-in TSFM catalogue.

The published `gnomon-forecast==0.8.0rc3` is the previous candidate and still
contains the retired legacy workflows. It is not the code documented here.
No 0.9 package has been published by this cleanup.

See [offline installation](offline-installation.md) and
[provider configuration](production/INFERENCE.md).
