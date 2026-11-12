#!/usr/bin/env python3
"""Record tutorial replays on a local GPU: each script with ``--record recordings/<id>.json``.

usage: python tutorials/tools/record_all.py [--only ID ...] [--device cuda:0] [--timing]

Runs every tutorial in ``tutorials/index.toml`` (or only the given ids) from the tutorials folder,
in the active Python environment, which needs the Newton build the docs describe. ``--timing`` also
runs ``tools/timing.py`` for tutorials with a ``[timing]`` table and stores the cost rows in their
recordings; without it, cost rows already in a recording are kept.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

TUTORIALS = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="tutorial ids to record (default: all)")
    ap.add_argument("--device", help="Warp device passed to each script, for example cuda:0")
    ap.add_argument(
        "--timing", action="store_true", help="also measure the cost tables with tools/timing.py"
    )
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per script")
    args = ap.parse_args()

    tutorials = load_tutorials()
    unknown = set(args.only or []) - {t["id"] for t in tutorials}
    if unknown:
        raise SystemExit(f"unknown tutorial ids: {', '.join(sorted(unknown))}")
    selected = [t for t in tutorials if not args.only or t["id"] in args.only]
    env = dict(os.environ, NEWTON_COMMIT=os.environ.get("NEWTON_COMMIT") or newton_commit())
    (TUTORIALS / "recordings").mkdir(exist_ok=True)

    timing = measure_timing(selected, env) if args.timing else {}
    failed = []
    for t in selected:
        out = TUTORIALS / "recordings" / f"{t['id']}.json"
        previous = json.loads(out.read_text()) if out.exists() else {}
        cmd = [sys.executable, t["file"], "--record", str(out)]
        if args.device:
            cmd += ["--device", args.device]
        print(f"=== {t['id']} ({t['file']})", flush=True)
        try:
            subprocess.run(cmd, cwd=TUTORIALS, env=env, check=True, timeout=args.timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"FAILED {t['id']}: {e}", flush=True)
            failed.append(t["id"])
            continue
        recording = json.loads(out.read_text())
        rows = timing.get(t["id"]) or previous.get("timing")
        if rows:
            recording["timing"] = rows
            out.write_text(json.dumps(recording, separators=(",", ":"), ensure_ascii=False))
    print(f"recorded {len(selected) - len(failed)} of {len(selected)}", flush=True)
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def load_tutorials():
    index = tomllib.loads((TUTORIALS / "index.toml").read_text())
    by_id = {}
    for path in TUTORIALS.glob("*.toml"):
        if path.name in ("index.toml", "ruff.toml"):
            continue
        meta = tomllib.loads(path.read_text())
        by_id[meta["id"]] = dict(meta, file=f"{path.stem}.py")
    return [by_id[i] for group in index["group"] for i in group["tutorials"]]


def measure_timing(selected, env):
    """Cost rows per tutorial id, from one tools/timing.py run over the groups they name."""
    groups = {t["timing"]["group"]: t["id"] for t in selected if "timing" in t}
    if not groups:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "timing.json"
        subprocess.run(
            [sys.executable, str(TUTORIALS / "tools" / "timing.py"), str(out), *groups],
            env=env,
            check=True,
        )
        data = json.loads(out.read_text())
    return {
        tid: {"gpu": data["gpu"], "rows": [r for r in data["rows"] if r["group"] == group]}
        for group, tid in groups.items()
    }


def newton_commit():
    """The installed Newton's git commit when it comes from a clone; recordings link to it."""
    try:
        import newton
    except ImportError:
        return ""
    root = Path(newton.__file__).resolve().parent.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


if __name__ == "__main__":
    main()
