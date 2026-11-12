# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Per-step cost for the row-order, fused-limit and pre-elimination tutorials, timed like timing.py.

usage: python tutorials/tools/timing_rows.py OUT.json [group ...]
"""

import importlib.util
import json
import os
import sys
import warnings

import newton
import timing
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
TUTORIALS = os.path.dirname(HERE)
GROUP_NAMES = ["row_schedule", "fuse", "preelim_mimics"]


def tutorial(name):
    sys.argv = sys.argv[:1]
    sys.path.insert(0, TUTORIALS)
    spec = importlib.util.spec_from_file_location(name, os.path.join(TUTORIALS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unfinalized(build):
    """Return a scene() that gives the tutorial's builder before finalize, for replicate()."""

    def scene():
        captured = {}
        original = newton.ModelBuilder.finalize

        def keep(self, *args, **kwargs):
            captured["builder"] = self
            return original(self, *args, **kwargs)

        newton.ModelBuilder.finalize = keep
        try:
            build()
        finally:
            newton.ModelBuilder.finalize = original
        return captured["builder"]

    return scene


def main():
    out, names = sys.argv[1], sys.argv[2:] or GROUP_NAMES
    rs = tutorial("27_row_schedule")
    dr = tutorial("39_drive_rows_velocity")
    pm = tutorial("45_preelimination_mimics")
    o = timing.o
    groups = {
        "row_schedule": (
            unfinalized(rs.build_stop),
            [
                o(
                    f'row_schedule="{s}"',
                    {"enable_joint_limits": True, "iterations": 4, "row_schedule": s},
                )
                for s in ("interleaved", "contact_then_internal", "phased")
            ],
        ),
        "fuse": (
            unfinalized(dr.build_flywheel),
            [
                o(
                    "fuse_joint_velocity_limits=False",
                    {
                        "drive_mode": "rows",
                        "enable_joint_velocity_limits": True,
                        "fuse_joint_velocity_limits": False,
                    },
                ),
                o(
                    "fuse_joint_velocity_limits=True (default)",
                    {"drive_mode": "rows", "enable_joint_velocity_limits": True},
                ),
            ],
        ),
        "preelim_mimics": (
            unfinalized(lambda: pm.build()),
            [
                o("no pre-elimination, iterations=2", {"iterations": 2}),
                o("no pre-elimination, iterations=12", {"iterations": 12}),
                o(
                    "include_mimics=True, iterations=2 (disabled: 9 rows)",
                    {"iterations": 2, "enable_bilateral_preelimination": True},
                ),
                o(
                    "include_mimics=False, iterations=2",
                    {
                        "iterations": 2,
                        "enable_bilateral_preelimination": True,
                        "bilateral_preelimination_include_mimics": False,
                    },
                ),
            ],
        ),
    }
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            scene, options = groups[name]
            rows += timing.group(name, scene, options)
    json.dump({"gpu": wp.get_device().name, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
