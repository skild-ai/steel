# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Which timing runner in tools/ measures each cost-table group, found without importing Newton."""

import ast
import hashlib
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))

from source_hash import source_bytes, source_sha256  # noqa: E402

__all__ = ["runners", "source_sha256", "workload_sha256"]


def runners():
    """{group name: runner path} from GROUPS in timing.py and GROUP_NAMES in timing_*.py."""
    out = {}
    for path in sorted(TOOLS.glob("timing*.py")):
        if path.name == "timing_groups.py":
            continue
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name == "GROUPS" and isinstance(node.value, ast.Dict):
                keys = [k.value for k in node.value.keys]
            elif name == "GROUP_NAMES":
                keys = ast.literal_eval(node.value)
            else:
                continue
            for key in keys:
                if key in out:
                    raise SystemExit(
                        f"timing group {key} is defined in {out[key].name} and {path.name}"
                    )
                out[key] = path
    return out


def workload_sha256(runner):
    """Hash of everything that builds a runner's timed workload: the runner, timing.py and the
    tutorial scripts it loads scenes from."""
    text = runner.read_text()
    scripts = sorted(set(re.findall(r"[\"'](\d\d_[a-z0-9_]+)[\"']", text)))
    parts = [runner, TOOLS / "timing.py"] + [TOOLS.parent / f"{s}.py" for s in scripts]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise SystemExit(f"{runner.name} loads missing scripts: {', '.join(missing)}")
    h = hashlib.sha256()
    for part in parts:
        h.update(part.name.encode() + b"\0" + source_bytes(part))
    return h.hexdigest()
