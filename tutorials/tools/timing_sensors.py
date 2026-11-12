# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Per-step cost of sleeping options for the contact-sensor tutorial, with timing.py's method.

usage: python tutorials/tools/timing_sensors.py OUT.json [sensors_sleep]
"""

import json
import sys

import newton
import timing
import warp as wp

GROUP_NAMES = ["sensors_sleep"]


def towers_with_sensor_bases():
    """Three 4-box towers, the contact-sensor tutorial's scene."""
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.7)
    for k in range(3):
        for i in range(4):
            box = b.add_body(
                xform=wp.transform((0.3 * k - 0.3, 0.0, 0.05 + 0.1005 * i), wp.quat_identity())
            )
            b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return b


def main():
    o = timing.o
    options = [
        o("no sleeping"),
        o("enable_sleeping=True", {"enable_sleeping": True}),
        o(
            "sleeping, sleep_skip_constraints=False",
            {"enable_sleeping": True, "sleep_skip_constraints": False},
        ),
    ]
    # 120 warm-up frames (4 s) so the towers are asleep; 64+ replicated worlds stay awake
    rows = timing.group(
        "sensors_sleep", towers_with_sensor_bases, options, world_counts=(1, 8), warm_frames=120
    )
    json.dump({"gpu": wp.get_device().name, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
