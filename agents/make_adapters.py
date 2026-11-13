# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Write the optional per-tool pointers to the workflows in agents/workflows/ (steps get none).

usage: python agents/make_adapters.py [--check]

Each pointer holds front matter built from the workflow's summary lines and one link to its README; the README is
the only copy of the procedure. --check exits 1 when a pointer is missing, stale or orphaned.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "agents" / "workflows"
# Tools that discover skill folders, and where their pointers go.
TARGETS = {"claude": ROOT / ".claude" / "skills"}
PREFIX = "steel-"


def main() -> None:
    check = "--check" in sys.argv[1:]
    wanted = {}
    for readme in sorted(WORKFLOWS.glob("*/README.md")):
        name = readme.parent.name
        text = readme.read_text()
        goal = summary(text, "Goal", readme)
        use = summary(text, "Use when", readme)
        for target in TARGETS.values():
            wanted[target / f"{PREFIX}{name}" / "SKILL.md"] = (
                f"---\nname: {PREFIX}{name}\ndescription: {json.dumps(f'Goal: {goal} Use when {use}')}\n---\n\n"
                f"Follow [agents/workflows/{name}/README.md](../../../agents/workflows/{name}/README.md).\n"
            )
    existing = {p for target in TARGETS.values() for p in target.glob(f"{PREFIX}*/SKILL.md")}
    stale = sorted(str(p.relative_to(ROOT)) for p, text in wanted.items() if not p.exists() or p.read_text() != text)
    orphans = sorted(str(p.relative_to(ROOT)) for p in existing - set(wanted))
    if check:
        if stale or orphans:
            sys.exit(f"adapters out of date: {stale + orphans}; run python agents/make_adapters.py")
        print(f"{len(wanted)} adapters up to date")
        return
    for path, text in wanted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for orphan in orphans:
        (ROOT / orphan).unlink()
        (ROOT / orphan).parent.rmdir()
    print(f"wrote {len(wanted)} adapters, removed {len(orphans)}")


def summary(text: str, key: str, path: Path) -> str:
    """The value of a ``> **Key:** value`` summary line."""
    m = re.search(rf"^> \*\*{re.escape(key)}:\*\* (.+)$", text, re.M)
    if not m:
        sys.exit(f"{path} has no '> **{key}:**' summary line")
    return m.group(1).strip()


if __name__ == "__main__":
    main()
