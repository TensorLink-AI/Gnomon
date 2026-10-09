"""CLI: python -m vanta_agent <command> --config agent.toml

Trading:  run [--once] | status | flatten [--yes] | resume --eod-halt X | backtest
Research: research snapshot|report|compare|sweep|promote|trials
Evidence: guide   (rank forecasts on realised outcomes in the paper/live ledger)
"""
import argparse
import json
import sys

from . import config as config_mod


def _grid(items):
    from .strategy import parse_value
    grid = {}
    for item in items or ():
        key, _, values = item.partition("=")
        if not values:
            raise SystemExit(f"--vary expects key=v1,v2,... (got {item!r})")
        grid[key] = [parse_value(v) for v in _split(values)]
    return grid


def _split(text):
    """Split on commas outside brackets/quotes, so `[0.1, 0.9],[0.05, 0.95]` works."""
    parts, depth, quote, buf = [], 0, False, ""
    for ch in text:
        if ch == '"':
            quote = not quote
        elif not quote and ch == "[":
            depth += 1
        elif not quote and ch == "]":
            depth -= 1
        if ch == "," and depth == 0 and not quote:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    return parts + ([buf.strip()] if buf.strip() else [])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vanta_agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "backtest", "status", "flatten", "resume", "research", "guide"])
    parser.add_argument("action", nargs="?", choices=["snapshot", "report", "compare", "sweep", "promote", "trials"],
                        help="research action")
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true", help="run: one cycle, then exit")
    parser.add_argument("--hours", type=int, default=168, help="backtest: hours to simulate; snapshot: hours to fetch")
    parser.add_argument("--max-remote-calls", type=int, default=0, help="budget for billable (Ephemeris) forecasts")
    parser.add_argument("--eod-halt", type=float, help="resume: new EOD drawdown halt line (< 0.08)")
    parser.add_argument("--yes", action="store_true", help="flatten: confirm in live mode")
    parser.add_argument("--candidate", help="research compare/promote: candidate strategy file")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="research compare: candidate = champion with this change (repeatable)")
    parser.add_argument("--vary", action="append", metavar="KEY=V1,V2",
                        help="research sweep: values to try for one key, one at a time (repeatable)")
    parser.add_argument("--window", choices=["tune"], default="tune")
    parser.add_argument("--ledger", help="guide: ledger path (default: state/<mode>.db)")
    parser.add_argument("--min-origins", type=int, default=48, help="guide: matched origins needed to suggest")
    args = parser.parse_args(argv)
    cfg = config_mod.load(args.config)
    show = lambda obj: print(json.dumps(obj, indent=2, default=str))

    if args.command == "backtest":
        from .backtest import run
        return show(run(cfg, test_hours=args.hours, max_remote_calls=args.max_remote_calls))
    if args.command == "research":
        return show(_research(parser, args, cfg))
    if args.command == "guide":
        from .guide import guide
        return show(guide(cfg, ledger_path=args.ledger, min_origins=args.min_origins))
    if args.command == "status":
        from .state import Book
        book = Book.load(cfg.path(cfg.agent.state_dir) / f"book-{cfg.agent.mode}.json", cfg.risk.initial_equity)
        return show(dict(mode=cfg.agent.mode, strategy=cfg.revision, equity_estimate=book.equity,
                         positions=book.positions, intraday_drawdown=book.intraday_drawdown,
                         eod_drawdown=book.eod_drawdown, halted_until=book.halted_until or None,
                         halt_reason=book.halt_reason or None, needs_reconcile=book.needs_reconcile))
    if args.command == "resume":
        from .agent import resume
        if args.eod_halt is None:
            parser.error("resume requires --eod-halt")
        return show(resume(cfg, eod_halt=args.eod_halt))

    from .agent import Agent
    agent = Agent(cfg)
    if args.command == "flatten":
        if cfg.agent.mode == "live" and not args.yes:
            parser.error("flatten in live mode sends FLAT orders to Vanta; pass --yes to confirm")
        return show(agent.flatten_all())
    if args.once:
        return show(agent.cycle())
    agent.run_forever()


def _research(parser, args, cfg):
    from . import research, strategy
    if args.action is None:
        parser.error("research needs an action: snapshot, report, compare, sweep, promote or trials")
    if args.action == "snapshot":
        return research.snapshot(cfg, hours=args.hours)
    if args.action == "trials":
        path = cfg.path(cfg.research.work_dir) / "trials.jsonl"
        return research.summarize_trials(research.read_trials(path), dataset=research.dataset_id(cfg))
    lab = research.Lab(cfg, max_remote_calls=args.max_remote_calls)
    try:
        if args.action == "report":
            result, _ = lab.evaluate(cfg)
            return dict(dataset=lab.dataset, window="tune", hours=len(lab.windows["tune"]),
                        holdout_hours=len(lab.windows["holdout"]), **result, remote_calls=lab.remote_calls)
        if args.action == "sweep":
            grid = _grid(args.vary)
            if not grid:
                parser.error("sweep needs at least one --vary key=v1,v2")
            return lab.sweep(cfg, grid)
        if args.candidate:
            candidate = strategy.apply_strategy(cfg, args.candidate)
        elif args.set:
            candidate = strategy.with_overrides(cfg, {k: strategy.parse_value(v) for k, v in
                                                      (s.split("=", 1) for s in args.set)})
            strategy.dump(candidate, lab.work / "candidates" / f"{candidate.revision.replace('@', '-')}.toml",
                          description=f"--set from {cfg.revision}")
        else:
            parser.error(f"research {args.action} needs --candidate FILE or --set KEY=VALUE")
        if args.action == "compare":
            record = lab.compare(cfg, candidate)
            return {k: v for k, v in record.items() if k not in ("champion_result", "challenger_result")} | dict(
                champion_strategy=record["champion_result"]["strategy"],
                challenger_strategy=record["challenger_result"]["strategy"],
                challenger_forecast_quality=record["challenger_result"]["forecast_quality"],
                remote_calls=lab.remote_calls)
        if not cfg.agent.strategy:
            parser.error("promote needs [agent] strategy = <champion file> in the agent config")
        return lab.promote(cfg, candidate, cfg.path(cfg.agent.strategy))
    finally:
        lab.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:  # user-facing refusals: print the reason, not a traceback
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        sys.exit(2)
