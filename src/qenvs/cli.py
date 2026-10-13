"""``qenvs`` command line interface (no extra dependencies; ``--json`` on every command).

Designed to be read by humans and by coding agents alike::

    qenvs list --json
    qenvs info qubit/StateTransfer-v0 --json
    qenvs run qubit/StateTransfer-v0 --policy pi-pulse --json
    qenvs validate --json
    qenvs doctor --json

Errors are reported as ``{"error": {"code": ..., "message": ...}}`` with exit code 1
when ``--json`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from qenvs import toolkit


def _emit(obj: object, as_json: bool, text: str) -> None:
    if as_json:
        json.dump(obj, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(text + "\n")


def _parse_kv(items: list[str] | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            msg = f"expected key=value, got {item!r}"
            raise argparse.ArgumentTypeError(msg)
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_list(args: argparse.Namespace) -> int:
    """List registered tasks."""
    tasks = toolkit.list_tasks()
    text = "\n".join(f"{t['id']:32s} {t['model']:24s} {t['description']}" for t in tasks)
    _emit(tasks, args.json, text)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Describe one task."""
    info = toolkit.task_info(args.task)
    lines = [f"{info['id']}  ({info['model']})", "", info["description"], "", "kwargs:"]
    lines += [f"  {k}: {v}" for k, v in info["kwargs"].items()]
    lines += ["", "defaults: " + json.dumps(info["defaults"]), "", "references:"]
    lines += [f"  - {r}" for r in info["references"]]
    _emit(info, args.json, "\n".join(lines))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Open-loop rollout with a reference policy."""
    out = toolkit.run_rollout(args.task, policy=args.policy, seed=args.seed, **_parse_kv(args.set))
    text = (
        f"{out['task']} policy={out['policy']} horizon={out['horizon']} dt={out['dt']}\n"
        f"return={out['return']:.6f}  final={json.dumps(out['final'])}"
    )
    _emit(out, args.json, text)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run physics checks and print errors vs tolerances."""
    out = toolkit.validate()
    rows = [f"{'check':30s} {'category':16s} {'error':>10s} {'tol':>8s}  status"]
    for c in out["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        rows.append(
            f"{c['name']:30s} {c['category']:16s} "
            f"{c['error']:10.2e} {c['tolerance']:8.0e}  {status}"
        )
    rows.append("")
    rows += [f"{k}: {v}" for k, v in out["environment"].items()]
    _emit(out, args.json, "\n".join(rows))
    return 0 if out["all_passed"] else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report installation health."""
    out = toolkit.doctor()
    lines = [f"{k}: {v}" for k, v in out["environment"].items()]
    lines += ["", "optional:"] + [f"  {k}: {v}" for k, v in out["optional"].items()]
    lines += ["", f"smoke: {out['smoke']}"]
    _emit(out, args.json, "\n".join(lines))
    return 0 if out["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    from qenvs import __version__

    p = argparse.ArgumentParser(prog="qenvs", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"qenvs {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list registered tasks")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("info", help="describe a task (physics, kwargs, references)")
    s.add_argument("task")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("run", help="open-loop rollout with a reference policy")
    s.add_argument("task")
    s.add_argument("--policy", choices=toolkit.POLICIES, default="pi-pulse")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="task kwargs, e.g. --set gamma1=0.1"
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("validate", help="run physics validation checks and report errors")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("doctor", help="check installation, backend and optional extras")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)
    return p


ERROR_CODES: dict[type[Exception], str] = {
    KeyError: "unknown_task",
    ValueError: "invalid_argument",
    argparse.ArgumentTypeError: "invalid_argument",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError, argparse.ArgumentTypeError) as e:
        code = next(c for t, c in ERROR_CODES.items() if isinstance(e, t))
        msg = e.args[0] if e.args else str(e)
        if getattr(args, "json", False):
            json.dump({"error": {"code": code, "message": msg}}, sys.stdout)
            sys.stdout.write("\n")
        else:
            sys.stderr.write(f"error [{code}]: {msg}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
