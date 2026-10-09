"""Strategy files: the versioned, diffable definition of what the agent trades on.

A strategy file holds `[strategy]` metadata plus `[forecast]` and `[risk]` keys that
override the agent config. Its revision is `<name>@<hash>` over the EFFECTIVE
forecast and risk parameters, so the ledger ties every decision to the exact
parameters that produced it, and two files with the same parameters share a revision.
"""
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from pathlib import Path
import tomllib

from . import config as config_mod

POLICY_SECTIONS = ("forecast", "risk")
# Shadow providers are observed, never traded on: they do not change the policy.
_NOT_POLICY = {"shadow_vol_models", "shadow_direction_providers"}


def policy_params(config) -> dict:
    out = {}
    for name in POLICY_SECTIONS:
        out[name] = {k: v for k, v in asdict(getattr(config, name)).items() if k not in _NOT_POLICY}
    return out


def revision(config) -> str:
    digest = sha256(json.dumps(policy_params(config), sort_keys=True, default=list).encode()).hexdigest()
    return f"{config.strategy_name or config.agent.policy_revision}@{digest[:12]}"


def _override(section, raw, name):
    known = {f.name for f in fields(section)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"strategy [{name}] unknown keys: {sorted(unknown)}")
    return replace(section, **{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def apply_strategy(config, path):
    raw = tomllib.loads(Path(path).read_text())
    unknown = set(raw) - {"strategy", *POLICY_SECTIONS}
    if unknown:
        raise ValueError(f"strategy file {path}: unknown sections {sorted(unknown)}")
    meta = raw.get("strategy", {})
    if set(meta) - {"name", "description", "parent"}:
        raise ValueError(f"strategy file {path}: unknown [strategy] keys")
    updated = {name: _override(getattr(config, name), raw.get(name, {}), name) for name in POLICY_SECTIONS}
    return config_mod.validate(replace(config, **updated, strategy_name=meta.get("name", ""),
                                       strategy_parent=meta.get("parent", "")))


def with_overrides(config, overrides: dict, *, name=None):
    """A child strategy: `config`'s policy with dotted overrides like {"risk.z_full": 2.0}."""
    updated = {}
    for key, value in overrides.items():
        section, _, field_name = key.partition(".")
        if section not in POLICY_SECTIONS or not field_name:
            raise ValueError(f"{key}: overrides must name forecast.<key> or risk.<key>")
        updated.setdefault(section, {})[field_name] = value
    child = replace(config, **{s: _override(getattr(config, s), v, s) for s, v in updated.items()},
                    strategy_name=name or config.strategy_name or config.agent.policy_revision,
                    strategy_parent=config.revision)
    return config_mod.validate(child)


def parse_value(text):
    """CLI value: TOML scalar or array (`2`, `0.5`, `true`, `"ephemeris"`, `[0.1, 0.9]`), else a bare string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _toml(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")


def dump(config, path, *, description=""):
    """Write `config`'s full policy as a self-contained strategy file."""
    lines = ["[strategy]", f"name = {_toml(config.strategy_name or config.agent.policy_revision)}"]
    if description:
        lines.append(f"description = {_toml(description)}")
    if config.strategy_parent:
        lines.append(f"parent = {_toml(config.strategy_parent)}")
    lines.append(f"# revision = {config.revision}")
    for name, params in policy_params(config).items():
        lines += ["", f"[{name}]"] + [f"{k} = {_toml(v)}" for k, v in params.items()]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def diff(parent, child) -> dict:
    a, b = policy_params(parent), policy_params(child)
    return {f"{s}.{k}": [a[s][k], b[s][k]] for s in POLICY_SECTIONS for k in a[s] if a[s][k] != b[s][k]}
