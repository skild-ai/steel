# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Joint speed limits
#
# Joint velocity limits: two double pendulums released horizontal; the gold one has |qd| <= 1.5 rad/s.
#
# Ported from steel-webgpu's tools/scenes.py (vel_limit); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run vel_limit
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
    camera = {'eye': (0.3, -2.0, 1.1), 'target': (0.3, 0.0, 0.55)}

    def build(self):
        """Joint velocity limits: two double pendulums released horizontal; the gold one has |qd| <= 1.5 rad/s."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        for y, vmax, color in ((0.25, 1e6, GREY), (-0.25, 1.5, GOLD)):
            prev, joints = -1, []
            for k in range(2):
                link = b.add_link(xform=tf((0.15 + 0.3 * k, y, 0.8)))
                b.add_shape_capsule(link, xform=tf(q=wp.quat_rpy(0.0, math.pi / 2, 0.0)), radius=0.03, half_height=0.12,
                                    cfg=cfg(density=800), color=color)
                parent_xform = tf((0, y, 0.8)) if prev < 0 else tf((0.15, 0, 0))
                joints.append(b.add_joint_revolute(prev, link, parent_xform=parent_xform, child_xform=tf((-0.15, 0, 0)),
                                                   axis=(0, 1, 0), velocity_limit=vmax))
                prev = link
            b.add_articulation(joints)
        return b, {}, "Velocity limits: the grey double pendulum swings freely; the gold one is held to 1.5 rad/s per joint.", {}

    def check(self):
        """The gold joints stay at or below their 1.5 rad/s limit."""
        qd = np.abs(self.state_0.joint_qd.numpy())
        assert qd[2:].max() <= 1.5 * 1.05, f"a gold joint moves at {qd[2:].max():.3f} rad/s, over its 1.5 rad/s limit"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
