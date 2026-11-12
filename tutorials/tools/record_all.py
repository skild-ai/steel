#!/usr/bin/env python3
"""Record tutorial replays on a local GPU: each script with ``--record recordings/<id>.json``.

usage: python tutorials/tools/record_all.py [--only ID ...] [--device cuda:0] [--timing]

Runs every tutorial in ``tutorials/index.toml`` (or only the given ids) from the tutorials folder,
in the active Python environment, which needs the Newton build the docs describe. ``--timing`` also
runs the timing runner (``tools/timing*.py``) of every ``[timing]`` group and stores the cost rows,
with the Newton commit and runner hash they were measured with, in the recordings. Without it, a
recording keeps its old cost rows only if neither its script nor the Newton commit changed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib
from timing_groups import runners

TUTORIALS = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="tutorial ids to record (default: all)")
    ap.add_argument("--device", help="Warp device passed to each script, for example cuda:0")
    ap.add_argument(
        "--timing", action="store_true", help="also measure the cost tables (tools/timing*.py)"
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

    timing = measure_timing(selected, env, args.device, args.timeout) if args.timing else {}
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
        rows = timing.get(t["id"])
        unchanged = (
            previous.get("script_sha256") == recording["script_sha256"]
            and previous.get("newton_commit") == recording["newton_commit"]
        )
        if rows is None and "timing" in previous:
            if unchanged:
                rows = previous["timing"]
            else:
                print(f"{t['id']}: dropped its old cost table; record with --timing", flush=True)
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


def measure_timing(selected, env, device, timeout):
    """Cost rows per tutorial id, one runner call per tools/timing*.py over the groups it owns."""
    owner = runners()
    by_group = {}
    for t in selected:
        if "timing" in t:
            by_group.setdefault(t["timing"]["group"], []).append(t["id"])
    missing = sorted(g for g in by_group if g not in owner)
    if missing:
        raise SystemExit(f"no timing runner defines: {', '.join(missing)}")
    if device and device.startswith("cuda:"):
        env = dict(env, CUDA_VISIBLE_DEVICES=device.split(":", 1)[1])
    harness = (TUTORIALS / "tools" / "timing.py").read_bytes()
    result = {}
    for runner in sorted({owner[g] for g in by_group}):
        groups = [g for g in by_group if owner[g] == runner]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "timing.json"
            cmd = [sys.executable, str(runner), str(out), *groups]
            subprocess.run(cmd, env=env, check=True, timeout=timeout * len(groups))
            data = json.loads(out.read_text())
        provenance = {
            "gpu": data["gpu"],
            "newton_commit": env.get("NEWTON_COMMIT") or None,
            "runner": runner.name,
            "runner_sha256": hashlib.sha256(runner.read_bytes() + harness).hexdigest(),
        }
        for group in groups:
            rows = [r for r in data["rows"] if r["group"] == group]
            for tid in by_group[group]:
                result[tid] = dict(provenance, rows=rows)
    return result


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
