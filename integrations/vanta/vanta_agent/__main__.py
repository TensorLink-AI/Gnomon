"""CLI: python -m vanta_agent {run,backtest,status,flatten,resume} --config agent.toml"""
import argparse
import json
import sys

from . import config as config_mod


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vanta_agent", description=__doc__)
    parser.add_argument("command", choices=["run", "backtest", "status", "flatten", "resume"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true", help="run: one cycle, then exit")
    parser.add_argument("--hours", type=int, default=168, help="backtest: hours to simulate")
    parser.add_argument("--max-remote-calls", type=int, default=0,
                        help="backtest: budget for billable (Ephemeris) forecasts")
    parser.add_argument("--eod-halt", type=float, help="resume: new EOD drawdown halt line (< 0.08)")
    parser.add_argument("--yes", action="store_true", help="flatten: confirm in live mode")
    args = parser.parse_args(argv)
    cfg = config_mod.load(args.config)
    show = lambda obj: print(json.dumps(obj, indent=2, default=str))

    if args.command == "backtest":
        from .backtest import run
        return show(run(cfg, test_hours=args.hours, max_remote_calls=args.max_remote_calls))
    if args.command == "status":
        from .state import Book
        book = Book.load(cfg.path(cfg.agent.state_dir) / f"book-{cfg.agent.mode}.json", cfg.risk.initial_equity)
        return show(dict(mode=cfg.agent.mode, equity_estimate=book.equity, positions=book.positions,
                         intraday_drawdown=book.intraday_drawdown, eod_drawdown=book.eod_drawdown,
                         halted_until=book.halted_until or None, halt_reason=book.halt_reason or None,
                         needs_reconcile=book.needs_reconcile))
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


if __name__ == "__main__":
    sys.exit(main())
