# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Parallel worlds
#
# Batched worlds: 16 copies of a pendulum-and-tower scene stepped in one Sim, one world per cell.
#
# Ported from steel-webgpu's tools/scenes.py (worlds); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run worlds
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
    camera = {'eye': (1.6, -2.6, 2.0), 'target': (0.0, 0.0, 0.1)}

    def build(self):
        """Batched worlds: 16 copies of a pendulum-and-tower scene stepped in one Sim, one world per cell."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        for w in range(16):
            cell = newton.ModelBuilder()
            swing = math.radians(30 + 4 * w)  # each world releases its pendulum from a different angle
            arm = cell.add_link(xform=tf((0.0, 0.0, 0.5)))
            cell.add_shape_capsule(arm, xform=tf(q=wp.quat_rpy(0.0, math.pi / 2, 0.0)), radius=0.015, half_height=0.12, cfg=cfg(density=400), color=GREY)
            cell.add_shape_sphere(arm, xform=tf((0.16, 0, 0)), radius=0.04, cfg=cfg(density=4000), color=RED)
            j = cell.add_joint_revolute(-1, arm, parent_xform=tf((-0.2, 0, 0.5), wp.quat_rpy(0.0, -swing, 0.0)), axis=(0, 1, 0))
            cell.add_articulation([j])
            for k in range(3):
                box = cell.add_body(xform=tf((0.0, 0.0, 0.03 + 0.06 * k)))
                cell.add_shape_box(box, hx=0.03, hy=0.05, hz=0.03, cfg=cfg(density=300), color=[BLUE, TEAL, GOLD][k])
            b.add_world(cell, xform=tf(((w % 4 - 1.5) * 0.6, (w // 4 - 1.5) * 0.6, 0.0)))
        return b, {}, "Batched worlds: 16 worlds, each a pendulum released from a different angle into a tower, in one Sim."

    def check(self):
        """Every world's tower still stands where it was built (the pendulums swing beside them)."""
        z = self.state_0.body_q.numpy()[:, 2].reshape(16, 4)[:, 1:]
        assert np.allclose(z, (0.03, 0.09, 0.15), atol=0.003), f"a tower box moved: heights {z.min(0)}..{z.max(0)}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
