# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Write plan.json for an examples tree: which scenes render with which Steel preset, and which are skipped.

usage: python3 plan.py <examples dir> <out plan.json> [scene ...]

A scene renders with its [steel] table when that table has settings and is not unsupported; otherwise
with a coupled preset whose name contains "steel" (steel_vbd, kamino_steel, ...), preferring the
file's default. Variants with kind = "steel" (e.g. steel_off) are compare toggles and never chosen.
A scene with neither is excluded (outside Steel's scope) or skipped with a label, per steel_scope.py.
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from steel_scope import classify

SETTING_KEYS = ("sim", "collision", "solver", "model", "entries")
DEFAULT_SECONDS = 10.0
MAX_SECONDS = 20.0


def has_settings(table: dict) -> bool:
    return any(k in table for k in SETTING_KEYS)


def check_args(data: dict, name: str) -> list[str]:
    args = data.get(name, {}).get("check", {}).get("args")
    return [str(a) for a in (data.get("check", {}).get("args", []) if args is None else args)]


def num_frames(args: list[str]) -> int | None:
    for i, a in enumerate(args):
        if a == "--num-frames" and i + 1 < len(args):
            return int(args[i + 1])
        if a.startswith("--num-frames="):
            return int(a.split("=", 1)[1])
    return None


def choose(data: dict) -> tuple[str | None, str, str]:
    """(preset, kind, why) for one presets.toml."""
    if data.get("default") == "none":
        return None, "skipped", "the example builds no physics solver"
    steel = data.get("steel")
    if steel is not None and steel.get("status") != "unsupported" and has_settings(steel):
        return "steel", "steel", ""
    coupled = [
        k for k, v in data.items()
        if isinstance(v, dict) and k != "steel" and "steel" in k.split("_") and v.get("kind") != "steel"
        and v.get("status") != "unsupported" and has_settings(v)
    ]  # fmt: skip
    if coupled:
        pick = data["default"] if data.get("default") in coupled else coupled[0]
        return pick, "steel-coupled", ""
    if steel is None:
        return None, "skipped", "no [steel] table"
    if steel.get("status") == "unsupported":
        return None, "skipped", f"steel unsupported: {steel.get('reason', '')}".strip()
    return None, "skipped", "steel table has no settings (no Steel code path yet)"


def main() -> None:
    tree, out, only = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), set(sys.argv[3:])
    sha = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no", "."],
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    pyproject = (tree / "pyproject.toml").read_text()
    newton_pin = next((ln.split("@")[-1].strip().strip('",') for ln in pyproject.splitlines() if "newton" in ln and "@" in ln), "")
    render, skipped, excluded = [], [], []
    for toml in sorted(tree.glob("shelf/*/*/presets.toml")):
        data = tomllib.loads(toml.read_text())
        scene, category = toml.parent.name, toml.parent.parent.name
        if only and scene not in only:
            continue
        preset, kind, why = choose(data)
        gallery = data.get("gallery", {})
        base = {"scene": scene, "category": category, "path": str(toml.parent.relative_to(tree)),
                "short": gallery.get("short", ""), "title": gallery.get("title", scene)}  # fmt: skip
        if preset is None:
            kind, text = classify(scene) or ("unlisted", "")
            row = {**base, "reason": why, "default": data.get("default")}
            if kind == "excluded":
                excluded.append({**row, "reason": text, "shelf_reason": why})
            else:
                label = {"preset": "no Steel preset yet", "feature": "needs Steel feature"}.get(kind, "no Steel preset")
                skipped.append({**row, "label": label, "detail": text})
            continue
        table = data[preset]
        fps = float(table.get("sim", {}).get("fps", 60) or 60)
        args = check_args(data, preset)
        nf = num_frames(args)
        seconds = DEFAULT_SECONDS
        if nf and nf / fps > DEFAULT_SECONDS:
            seconds = min(nf / fps, MAX_SECONDS)
        speed = 2.0 if seconds > DEFAULT_SECONDS else 1.0
        frame_hint = gallery.get("frame")
        poster = min(0.9, max(0.1, frame_hint / fps / seconds)) if frame_hint else 0.5
        render.append({
            **base, "preset": preset, "kind": kind, "status": table.get("status", "untested"),
            "checked": table.get("checked", ""), "note": table.get("note", ""),
            "settings": {k: table[k] for k in SETTING_KEYS if k in table},
            "check_args": args, "sim_seconds": round(seconds, 3), "speed": speed, "poster_frac": round(poster, 3),
        })  # fmt: skip
    plan = {"commit": sha, "dirty": bool(dirty), "newton_pin": newton_pin, "render": render, "skipped": skipped,
            "excluded": excluded}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1))
    if dirty:
        print("warning: the examples tree has uncommitted changes; the audit does not describe its commit")
    print(f"{len(render)} to render ({sum(e['kind'] == 'steel-coupled' for e in render)} coupled), {len(skipped)} skipped, {len(excluded)} excluded")


if __name__ == "__main__":
    main()
