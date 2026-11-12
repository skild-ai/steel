"""Which timing runner in tools/ measures each cost-table group, found without importing Newton."""

import ast
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


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
