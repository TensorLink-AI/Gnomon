"""Agent configuration: one TOML file, validated at startup. Unknown keys fail."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib

MODES = ("backtest", "paper", "live")


@dataclass(frozen=True)
class AgentSection:
    mode: str = "paper"
    account: str = "vanta-miner"
    state_dir: str = "state"
    policy_revision: str = "volsize-dir-v1"
    # Live only: a user-approved promotion record and the paper ledger it cites
    # (see skills/trade-with-gnomon/references/promotion.md).
    promotion_record: str = ""
    paper_ledger: str = ""


@dataclass(frozen=True)
class UniverseSection:
    # Vanta trade_pair ids. Crypto pairs are Hyperliquid USDC perps.
    pairs: tuple[str, ...] = ("BTCUSDC", "ETHUSDC", "SOLUSDC")


@dataclass(frozen=True)
class MarketDataSection:
    source: str = "hyperliquid"  # hyperliquid | csv
    url: str = "https://api.hyperliquid.xyz/info"
    csv_dir: str = ""
    timeout_seconds: float = 20.0
    stale_after_minutes: int = 20


@dataclass(frozen=True)
class ForecastSection:
    # Gnomon providers TOML (e.g. with an Ephemeris provider). Empty = local models only.
    providers_config: str = ""
    # Price-direction provider. "rw" is the random-walk (zero-return) baseline,
    # "local/momentum" a strawman candidate; use the Ephemeris provider name from
    # providers_config (e.g. "ephemeris") for TSFM forecasts.
    direction_provider: str = "rw"
    direction_baseline: str = "rw"
    vol_model: str = "vol/har"  # vol/last | vol/ewma | vol/har
    vol_baseline: str = "vol/last"
    horizon_hours: int = 4
    history_hours: int = 336
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class RiskSection:
    # Risk budget per position: hourly stdev of equity, in bps.
    target_vol_bps_per_hour: float = 8.0
    # Agent caps. Vanta tier-2 crypto caps are 1.0x per position and 2.0x portfolio.
    max_position_leverage: float = 0.5
    max_portfolio_leverage: float = 1.5
    allow_short: bool = True
    # Direction needs |z| >= z_full for full size; below the cost hurdle it is zero.
    z_full: float = 1.0
    min_rebalance_leverage: float = 0.02
    # Halts well inside Vanta's elimination lines (5% intraday, 8% EOD).
    intraday_drawdown_halt: float = 0.03
    eod_drawdown_halt: float = 0.05
    initial_equity: float = 150_000.0


@dataclass(frozen=True)
class CostSection:
    fee_bps: float = 3.0  # Vanta crypto spread fee: 0.03% of order value, per order
    slippage_bps: float = 2.0
    # Hyperliquid funding is charged on HL-sourced pairs; 10.95%/yr ~ 0.125 bps/h.
    funding_bps_per_hour: float = 0.125


@dataclass(frozen=True)
class VantaSection:
    url: str = "http://127.0.0.1:8088"
    api_key_env: str = "VANTA_API_KEY"
    timeout_seconds: float = 120.0
    # Vanta enforces a 5s cooldown between orders on the same pair.
    same_pair_cooldown_seconds: float = 6.0


@dataclass(frozen=True)
class AgentConfig:
    agent: AgentSection = field(default_factory=AgentSection)
    universe: UniverseSection = field(default_factory=UniverseSection)
    market_data: MarketDataSection = field(default_factory=MarketDataSection)
    forecast: ForecastSection = field(default_factory=ForecastSection)
    risk: RiskSection = field(default_factory=RiskSection)
    costs: CostSection = field(default_factory=CostSection)
    vanta: VantaSection = field(default_factory=VantaSection)
    base_dir: Path = Path(".")

    def path(self, value: str) -> Path:
        """Resolve a configured path relative to the config file."""
        p = Path(value)
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def _section(cls, raw, name):
    raw = dict(raw or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"[{name}] unknown keys: {sorted(unknown)}")
    for key, value in raw.items():
        if isinstance(value, list):
            raw[key] = tuple(value)
    return cls(**raw)


def validate(config: AgentConfig) -> AgentConfig:
    a, f, r = config.agent, config.forecast, config.risk
    if a.mode not in MODES:
        raise ValueError(f"agent.mode must be one of {MODES}")
    if a.mode == "live" and not (a.promotion_record and a.paper_ledger):
        raise ValueError("live mode requires agent.promotion_record and agent.paper_ledger")
    if not config.universe.pairs:
        raise ValueError("universe.pairs is empty")
    if config.market_data.source not in ("hyperliquid", "csv"):
        raise ValueError("market_data.source must be hyperliquid or csv")
    if not 1 <= f.horizon_hours <= 48:
        raise ValueError("forecast.horizon_hours must be within 1..48")
    if f.history_hours < 96:
        raise ValueError("forecast.history_hours must be at least 96")
    if f.history_hours * 12 > 5000 and config.market_data.source == "hyperliquid":
        raise ValueError("Hyperliquid serves at most 5000 candles: history_hours <= 416")
    if not (0 < r.max_position_leverage <= 1.0 and 0 < r.max_portfolio_leverage <= 2.0):
        raise ValueError("risk leverage caps exceed Vanta tier-2 crypto limits (1.0x / 2.0x)")
    if not (0 < r.intraday_drawdown_halt < 0.05 and 0 < r.eod_drawdown_halt < 0.08):
        raise ValueError("drawdown halts must sit inside Vanta's 5% intraday / 8% EOD eliminations")
    return config


def load(path: str | Path) -> AgentConfig:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    sections = {f.name: f.type for f in fields(AgentConfig) if f.name != "base_dir"}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    built = {name: _section(cls, raw.get(name), name) for name, cls in sections.items()}
    return validate(AgentConfig(**built, base_dir=path.parent))
