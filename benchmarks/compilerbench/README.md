# CompilerBench

CompilerBench measures the language-to-`TemporalQuestion` boundary separately
from numerical estimation. Its 80 deterministic held-out cases cover all
properties, exact series, pair, each, aggregate, material ambiguity and
tempting invented targets. It contains no forecast labels or answer options.

The runner distinguishes semantic refusals from provider/infrastructure
failures. A transport failure can never earn refusal credit.
Malformed structured output is a completed incorrect compilation, not an
honest refusal. Use a different `--seed` for a held-out phrasing set; whole
cases receive bounded `--infrastructure-retries`, recorded in the summary.

```bash
python3 -m benchmarks.compilerbench.run_compilerbench \
  --model deepseek-v4-flash-0731 \
  --base-url https://api.engy.ai/v1 \
  --api-key-env ENGY_API_KEY \
  --count 80 --workers 8 --output-dir results/compilerbench
```

Release gates are at least 95% overall accuracy, 98% exact-target accuracy,
90% ambiguity-refusal accuracy, zero invented targets accepted, and complete
execution without infrastructure cases hidden from the denominator.
