# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""``python -m shelf``: list, run and check examples, and build the preset matrix."""

from __future__ import annotations

import argparse
import datetime
import functools
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import looks, modes, presets, thumbs

ROOT = Path(__file__).resolve().parent.parent
SHELF = ROOT / "shelf"
MATRIX = ROOT / "PRESETS.md"
MARK = {"works": "✅", "unstable": "⚠️", "unsupported": "—", "untested": "·"}


def examples() -> dict[str, Path]:
    """Example name -> its ``example.py``, in shelf order."""
    return {p.parent.name: p for p in sorted(SHELF.glob("*/*/example.py"))}


def find(name: str) -> Path:
    found = examples()
    if name in found:
        return found[name]
    close = [n for n in found if name in n]
    hint = f"; did you mean {', '.join(close[:5])}?" if close else ""
    sys.exit(f"no example named {name!r}{hint}")


def section_of(path: Path) -> str:
    return path.parent.parent.name


# --- list ------------------------------------------------------------------------------------------


def cmd_list(args) -> None:
    for name, path in examples().items():
        if args.section and not section_of(path).startswith(args.section):
            continue
        data = presets.read(path)
        cells = [f"{k}:{data[k].get('status', 'untested')}" for k in presets.names(path)]
        print(f"{section_of(path):30s} {name:45s} default={data.get('default', '-'):14s} {' '.join(cells)}")


# --- run -------------------------------------------------------------------------------------------


def cmd_mode(args, rest: list[str]) -> None:
    """play / bench / tune: select the preset, then run the example in that mode."""
    if args.cmd in ("play", "run", "tune") and args.example is None:
        path = _browse(args)  # the picked example; args.solver is the chip picked, if one
        if path is None:
            return
        args.example = path.parent.name
    else:
        path = find(args.example)
    overrides = dict(presets.parse_override(s) for s in args.set)
    presets.select(args.solver, overrides, file=args.presets)
    # Child processes (and examples that respawn) see the same choice.
    if args.solver:
        os.environ["SHELF_SOLVER"] = args.solver
    if args.presets:
        os.environ["SHELF_PRESETS"] = args.presets
    if overrides:
        os.environ["SHELF_SET"] = json.dumps(overrides)
    preset = presets.load(path)
    if not getattr(args, "quiet_preset", False):
        source = f" from {preset.path.name}" if preset.path.name != "presets.toml" else ""
        print(
            f"[shelf] {args.example}: {args.cmd} mode, preset [{preset.name}]{source} ({preset.status})",
            file=sys.stderr,
        )

    steel_solvers = _track_steel_solvers() if args.check_capacity else []
    loop = None
    if args.cmd == "bench":
        if any(a == "--viewer" or a.startswith("--viewer=") for a in rest):
            sys.exit("bench runs headless; drop --viewer")
        if args.world_count is not None:
            rest = ["--world-count", str(args.world_count), *rest]
        # The frame count only bounds Newton's own loop, which bench replaces.
        loop = modes.Throughput(args.example, args.frames, args.warmup, args.json)
        modes.run_example(path, ["--viewer", "null", "--quiet", *rest], loop)
    else:
        # play and tune keep the window: picking another example in the bar ends this one's loop and
        # runs the next in the same viewer, at its default preset and arguments
        while True:
            if args.cmd == "tune":
                loop = host = modes.Tuner(path, overrides, args.presets)
                modes.run_example(path, rest, loop, look=args.look)
            else:
                host = modes.PlayHost(path)
                modes.run_example(path, rest, look=args.look, host=host)
            if host.next_example is None or (loop is not None and not loop.ran):
                break
            path, rest, overrides, args.presets = find(host.next_example), [], {}, None
            solver = getattr(host, "next_solver", None)  # a chip in the browser; None: the default
            for var in ("SHELF_SOLVER", "SHELF_PRESETS", "SHELF_SET"):
                os.environ.pop(var, None)
            presets.select(solver)
            if solver:
                os.environ["SHELF_SOLVER"] = solver
            preset = presets.load(path)
            print(
                f"[shelf] {path.parent.name}: {args.cmd} mode, preset [{preset.name}] ({preset.status})",
                file=sys.stderr,
            )
    if loop is not None and not loop.ran:
        sys.exit(
            f"{path.parent.name} runs its own loop instead of newton.examples.run(), "
            f"so {args.cmd} cannot time or tune it; only play works for this example"
        )
    for solver in steel_solvers:
        try:
            solver.check_constraint_capacity()
        except RuntimeError as error:
            sys.exit(str(error).split(". ")[0])


def _track_steel_solvers() -> list:
    """Every SolverSteel built from here on, for checking after the run that none dropped rows."""
    import newton.solvers

    cls = newton.solvers.SolverSteel
    built = []
    init = cls.__init__

    @functools.wraps(init)  # presets.make_solver reads the signature
    def tracked_init(self, *a, **k):
        init(self, *a, **k)
        built.append(self)

    cls.__init__ = tracked_init
    return built


def _browse(args) -> Path | None:
    """``shelf play`` or ``shelf tune`` with no example: the browser in an empty window; the picked example's path, or None.

    Previews the look lacks are rendered in the background meanwhile (see common/thumbs.py); nothing is
    rendered when an example is named, so `shelf play <example>` never waits for them.
    """
    if args.solver or args.set or args.presets:
        sys.exit("name an example to use --solver, --set or --presets")
    looks.load(args.look)  # a wrong name fails here, before any preview build starts
    building, total = thumbs.start_background(args.look)
    if building:
        first = building >= total
        what = (
            f"the previews for the example browser in the background: one frame of each of the {total} examples, "
            f"about {thumbs.FIRST_BUILD_MINUTES} min on an RTX 3090 the first time"
            if first
            else f"{building} new preview{'s' if building != 1 else ''} for the example browser in the background"
        )
        print(f"[shelf] rendering {what}. They are cached under {thumbs.folder(args.look)}; the browser fills in as "
              "they finish.", file=sys.stderr)  # fmt: skip
    host = modes.run_gallery(args.look, building, total)
    if host.next_example is None:
        return None
    args.solver = host.next_solver
    return find(host.next_example)


# --- check -----------------------------------------------------------------------------------------


def newton_commit() -> str:
    try:
        info = json.loads(importlib.metadata.distribution("newton").read_text("direct_url.json") or "{}")
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            return commit[:9]
    except importlib.metadata.PackageNotFoundError:
        pass
    import newton

    # Editable or source installs: ask git in the source tree, if there is one.
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(newton.__file__).parent), "rev-parse", "--short=9", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"v{newton.__version__}"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return "cpu"


def check_args(data: dict, name: str) -> list[str]:
    block = data.get(name, {})
    args = block.get("check", {}).get("args")
    if args is None:
        args = data.get("check", {}).get("args", [])
    return [str(a) for a in args]


def run_check(example: str, path: Path, name: str, timeout: int, file: str | None) -> tuple[bool, str, float]:
    data = presets.read(path, file)
    cmd = [
        sys.executable, "-m", "shelf", "run", example, "--solver", name, "--quiet-preset", "--check-capacity",
        *(["--presets", file] if file else []),
        "--viewer", "null", "--test", "--quiet", *check_args(data, name),
    ]  # fmt: skip
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} s", time.perf_counter() - start
    elapsed = time.perf_counter() - start
    if proc.returncode == 0:
        return True, "", elapsed
    lines = [ln for ln in (proc.stderr or proc.stdout).strip().splitlines() if ln.strip()]
    return False, (lines[-1] if lines else f"exit code {proc.returncode}")[:300], elapsed


def write_result(path: Path, name: str, ok: bool, note: str, stamp: str, file: str | None) -> None:
    import tomlkit

    toml_path = presets.presets_path(path, file)
    doc = tomlkit.parse(toml_path.read_text())
    block = doc[name]
    block["status"] = "works" if ok else "unstable"
    block["checked"] = stamp
    if ok and "note" in block and str(block["note"]).startswith("check failed:"):
        del block["note"]
    if not ok:
        block["note"] = f"check failed: {note}"
    toml_path.write_text(tomlkit.dumps(doc))


def cmd_check(args) -> None:
    targets = examples() if args.all else {n: find(n) for n in args.examples}
    if not targets:
        sys.exit("name examples to check, or pass --all")
    stamp = f"{datetime.date.today().isoformat()}, newton {newton_commit()}, {gpu_name()}"
    failures = 0
    for example, path in targets.items():
        data = presets.read(path, args.presets)
        if args.solver:
            wanted = [args.solver]
        elif args.every:
            # Placeholders (no settings) have no code path in the example yet; nothing to check.
            wanted = presets.ready(path, args.presets)
        else:
            wanted = [n for n in presets.names(path, args.presets) if data[n].get("status") in ("works", "unstable")]
        for name in wanted:
            if name not in data:
                print(f"{example:45s} {name:14s} no such preset")
                failures += 1
                continue
            if data[name].get("status") == "unsupported":
                print(f"{example:45s} {name:14s} unsupported, skipped")
                continue
            ok, note, elapsed = run_check(example, path, name, args.timeout, args.presets)
            failures += not ok
            print(f"{example:45s} {name:14s} {'works' if ok else 'FAILED':8s} {elapsed:6.1f} s  {note}", flush=True)
            if not args.dry_run:
                write_result(path, name, ok, note, stamp, args.presets)
    # PRESETS.md describes presets.toml only, not files saved by `shelf tune`.
    if not args.dry_run and not args.no_matrix and not args.presets:
        write_matrix()
    sys.exit(1 if failures else 0)


# --- thumbs ----------------------------------------------------------------------------------------


def cmd_thumbs(args) -> None:
    targets = examples() if args.all or not args.examples else {n: find(n) for n in args.examples}
    if args.missing:
        targets = thumbs.missing(targets, args.look)
    if not targets:
        print("nothing to render: every preview is cached (--force renders again)")
        return
    if not args.quiet:
        print(f"rendering {len(targets)} previews in the {args.look} look, {args.jobs} at a time, to {thumbs.folder(args.look)}")
    start = time.perf_counter()
    failures = thumbs.build(targets, args.look, args.jobs, args.timeout, args.force, args.quiet)
    if not args.quiet:
        print(f"{len(targets) - failures} rendered, {failures} failed, {time.perf_counter() - start:.0f} s")
    sys.exit(1 if failures else 0)


def cmd_thumb(args) -> None:
    """One preview in this process (the `thumbs` command runs one of these per example)."""
    thumbs.render(find(args.example), Path(args.out), args.frame, args.look)


# --- matrix ----------------------------------------------------------------------------------------


def write_matrix() -> None:
    cols = list(presets.SOLVERS)
    lines = [
        "# Preset matrix",
        "",
        "Generated by `python -m shelf matrix`; do not edit. Each cell is that solver's preset status;",
        "click an example for its `presets.toml`. *other* lists coupled presets (several solvers in one",
        "scene) and `none` (examples that build no physics solver).",
        "",
        f"{MARK['works']} works · {MARK['unstable']} unstable · {MARK['unsupported']} unsupported · "
        f"{MARK['untested']} untested. **Bold** is the example's default.",
        "",
    ]
    current = None
    for name, path in examples().items():
        section = section_of(path)
        if section != current:
            current = section
            lines += ["", f"## {section}", "", "| Example | " + " | ".join(cols) + " | other |",
                      "| --- |" + " :-: |" * (len(cols) + 1)]  # fmt: skip
        data = presets.read(path)
        rel = path.parent.relative_to(ROOT).as_posix()
        cells = []
        for col in cols:
            mark = MARK[data.get(col, {}).get("status", "untested")]
            cells.append(f"**{mark}**" if data.get("default") == col else mark)
        coupled = [
            f"{'**' if data.get('default') == n else ''}{n} {MARK[data[n].get('status', 'untested')]}"
            f"{'**' if data.get('default') == n else ''}"
            for n in presets.names(path)
            if n not in presets.SOLVERS
        ]
        lines.append(f"| [{name}]({rel}/presets.toml) | " + " | ".join(cells) + f" | {', '.join(coupled)} |")
    totals = dict.fromkeys(presets.STATUSES, 0)
    for path in examples().values():
        data = presets.read(path)
        for name in presets.names(path):
            totals[data[name].get("status", "untested")] += 1
    summary = ", ".join(f"{n} {s}" for s, n in totals.items())
    lines.insert(7, f"{len(examples())} examples, {sum(totals.values())} presets: {summary}.")
    MATRIX.write_text("\n".join(lines) + "\n")
    print(f"wrote {MATRIX.relative_to(ROOT)}")


# --- entry point -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m shelf", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list examples and their preset statuses")
    p.add_argument("section", nargs="?", help="only folders starting with this, e.g. 02")

    def mode_parser(name, help_text, aliases=()):
        p = sub.add_parser(name, help=help_text, aliases=list(aliases))
        if name in ("play", "tune"):
            p.add_argument("example", nargs="?", help="an example from `list`; with none, the example browser opens")
        else:
            p.add_argument("example")
        p.add_argument("--solver", help="preset to use (default: the file's default)")
        p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a setting")
        p.add_argument("--presets", metavar="FILE", help="preset file in the example's folder (default: presets.toml)")
        p.add_argument(
            "--look", default="steel", metavar="NAME",
            help="palette and lighting from common/looks.toml (default: steel; 'newton' is Newton's own look)",
        )  # fmt: skip
        p.add_argument("--quiet-preset", action="store_true", help=argparse.SUPPRESS)
        p.add_argument("--check-capacity", action="store_true", help=argparse.SUPPRESS)
        return p

    mode_parser(
        "play", "interactive: the viewer, mouse picking, pause and reset; other arguments go to the example",
        aliases=("run",),
    )  # fmt: skip
    p = mode_parser("bench", "throughput: headless step time, throughput and GPU memory at a chosen world count")
    p.add_argument("--world-count", type=int, help="worlds to simulate (examples that support --world-count)")
    p.add_argument("--frames", type=int, default=300, help="frames to time")
    p.add_argument("--warmup", type=int, default=30, help="frames to run before timing")
    p.add_argument("--json", metavar="FILE", help="append the result as a JSON line")
    mode_parser("tune", "live tuning: the viewer plus a panel to edit, apply and save the preset's settings")

    p = sub.add_parser("check", help="run examples headless in test mode and record the result in presets.toml")
    p.add_argument("examples", nargs="*")
    p.add_argument("--all", action="store_true", help="every example")
    p.add_argument("--solver", help="only this preset")
    p.add_argument("--every", action="store_true", help="every preset not marked unsupported, including untested")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--dry-run", action="store_true", help="do not write results")
    p.add_argument("--presets", metavar="FILE", help="check this preset file instead of presets.toml")
    p.add_argument("--no-matrix", action="store_true", help=f"record results but leave {MATRIX.name} alone")

    sub.add_parser("matrix", help=f"write {MATRIX.name}")

    p = sub.add_parser("thumbs", help="render the example browser's previews into the cache, one frame per example")
    p.add_argument("examples", nargs="*")
    p.add_argument("--all", action="store_true", help="every example (the default when none is named)")
    p.add_argument("--missing", action="store_true", help="only examples with no preview and no recorded failure")
    p.add_argument("--look", default="steel", metavar="NAME", help="the look to render in (default: steel)")
    p.add_argument("--jobs", type=int, default=thumbs.DEFAULT_JOBS, help=f"examples rendering at once (default {thumbs.DEFAULT_JOBS})")
    p.add_argument("--timeout", type=int, default=thumbs.DEFAULT_TIMEOUT, help="seconds per example")
    p.add_argument("--force", action="store_true", help="render again, including examples whose last render failed")
    p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("_thumb", help=argparse.SUPPRESS)  # one example in this process; `thumbs` runs these
    p.add_argument("example")
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=presets.DEFAULT_THUMB_FRAME)
    p.add_argument("--look", default="steel")

    args, rest = parser.parse_known_args(argv)
    if args.cmd in ("play", "run", "bench", "tune"):
        cmd_mode(args, rest)
        return
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    {"list": cmd_list, "check": cmd_check, "matrix": lambda a: write_matrix(), "thumbs": cmd_thumbs, "_thumb": cmd_thumb}[
        args.cmd
    ](args)
