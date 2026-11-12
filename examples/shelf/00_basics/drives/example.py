# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Joint motors
#
# Drive rows (drive_mode="rows"): arms stepping to 1.2 rad with effort limits 3, 10 and 60 N m, capped at 2 rad/s.
#
# Ported from steel-webgpu's tools/scenes.py (drives); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run drives
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from common.steel_scenes import BLUE, GOLD, GREY, RED, TEAL, SceneExample, cfg, tf


class Example(SceneExample):
    file = __file__
    camera = {'eye': (0.6, -1.6, 0.9), 'target': (0.2, 0.0, 0.5)}

    def build(self):
        """Drive rows (drive_mode="rows"): arms stepping to 1.2 rad with effort limits 3, 10 and 60 N m, capped at 2 rad/s."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        joints = []
        for i, effort in enumerate((3.0, 10.0, 60.0)):
            y = (i - 1) * 0.3
            base = b.add_link(xform=tf((0.0, y, 0.5)))
            b.add_shape_box(base, xform=tf((0.2, 0, 0)), hx=0.2, hy=0.02, hz=0.02, cfg=cfg(density=500), color=[BLUE, TEAL, GOLD][i])
            b.add_shape_sphere(base, xform=tf((0.4, 0, 0)), radius=0.05, cfg=cfg(density=3000), color=RED)
            j = b.add_joint_revolute(-1, base, parent_xform=tf((0.0, y, 0.5)), axis=(0, -1, 0), target_pos=1.2, target_ke=2000.0,
                                     target_kd=60.0, effort_limit=effort, velocity_limit=2.0)
            b.add_articulation([j])
            joints.append(j)
        return b, {}, "Drive rows: arms driven to 1.2 rad with effort limits 3, 10 and 60 N m and a 2 rad/s velocity limit (drive_mode rows)."

    def check(self):
        """The 60 N m arm holds the 1.2 rad target; the 3 N m arm cannot lift its load and hangs below horizontal."""
        q = self.state_0.joint_q.numpy()
        assert abs(q[2] - 1.2) < 0.02, f"the 60 N m arm is at {q[2]:.3f} rad, not at its 1.2 rad target"
        assert q[0] < 0.0, f"the 3 N m arm is at {q[0]:.3f} rad: its effort limit should not lift the load"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
