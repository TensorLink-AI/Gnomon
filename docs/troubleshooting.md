# Troubleshooting

- Unknown provider: check `gnomon capabilities --providers-config providers.toml`.
  Install/load the model in your own software and register its callable or factory.
- Unsupported request: check provider capabilities, exact shapes, frequency and
  cutoff fields. Unsupported covariates/quantiles are errors, not silently dropped.
- Invalid input: select columns explicitly and inspect repairs before forecasting.
  Mixed naive/aware timestamps and ambiguous revisions need explicit semantics.
- Expired reference: inspect again, or retrieve durable evidence from the ledger.
- Result retention limit: execution may already have happened. Check the receipt
  before retrying paid work.
- Retired command/profile: follow [migration](../COMPATIBILITY.md); it is not an
  alias for the new contract.
- No routing winner: retain the baseline fallback; missing comparable evidence is
  not proof that a model is worse.

[Provider and evaluation limits](production/INFERENCE.md).
