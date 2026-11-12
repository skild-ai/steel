# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Six-axis joints
#
# D6 joints: a driven 2-translation + yaw gantry carrying a pendulum, a universal joint and a gimbal.
#
# Ported from steel-webgpu's tools/scenes.py (d6); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run d6
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
    camera = {'eye': (0.9, -2.0, 1.3), 'target': (-0.05, 0.0, 0.6)}

    def build(self):
        """D6 joints: a driven 2-translation + yaw gantry carrying a pendulum, a universal joint and a gimbal."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        D = newton.ModelBuilder.JointDofConfig
        link_cfg = cfg(density=500)
        # Gantry: x/y slides limited to +-0.3 m and driven to (0.25, -0.2); yaw driven to 1 rad.
        head = b.add_link(xform=tf((-0.4, 0, 0.9)))
        b.add_shape_box(head, hx=0.08, hy=0.08, hz=0.03, cfg=link_cfg, color=GOLD)
        slide = dict(limit_lower=-0.3, limit_upper=0.3, target_ke=400.0, target_kd=40.0, armature=0.05)
        j0 = b.add_joint_d6(
            -1, head, parent_xform=tf((-0.4, 0, 0.9)),
            linear_axes=[D(axis=(1, 0, 0), target_pos=0.25, **slide), D(axis=(0, 1, 0), target_pos=-0.2, **slide)],
            angular_axes=[D(axis=(0, 0, 1), target_pos=1.0, target_ke=40.0, target_kd=4.0, armature=0.01)],
        )
        bob = b.add_link(xform=tf((-0.4, 0, 0.55)))
        b.add_shape_capsule(bob, radius=0.025, half_height=0.15, cfg=link_cfg, color=BLUE)
        j1 = b.add_joint_ball(head, bob, parent_xform=tf((0, 0, -0.03)), child_xform=tf((0, 0, 0.17)))
        b.add_articulation([j0, j1])
        # Universal joint released from a tilt; its first axis stops at -0.15 rad on the return swing.
        u = b.add_link(xform=tf((0.3, 0.25, 0.6)))
        b.add_shape_box(u, hx=0.02, hy=0.02, hz=0.18, cfg=link_cfg, color=TEAL)
        lim = dict(limit_lower=-0.5, limit_upper=0.5)
        ju = b.add_joint_d6(-1, u, parent_xform=tf((0.3, 0.25, 0.8)), child_xform=tf((0, 0, 0.2)),
                            angular_axes=[D(axis=(1, 0, 0), limit_lower=-0.15, limit_upper=0.5), D(axis=(0, 1, 0), **lim)])
        b.add_articulation([ju])
        # Gimbal: three rotations, each limited.
        g = b.add_link(xform=tf((0.3, -0.25, 0.6)))
        b.add_shape_box(g, hx=0.12, hy=0.02, hz=0.02, cfg=link_cfg, color=RED)
        b.add_shape_box(g, xform=tf((0, 0, 0.1)), hx=0.015, hy=0.015, hz=0.1, cfg=link_cfg, color=RED)
        jg = b.add_joint_d6(-1, g, parent_xform=tf((0.3, -0.25, 0.8)), child_xform=tf((0, 0, 0.2)),
                            angular_axes=[D(axis=(1, 0, 0), **lim), D(axis=(0, 1, 0), **lim), D(axis=(0, 0, 1), limit_lower=-1.0, limit_upper=1.0)])
        b.add_articulation([jg])
        model_q = {"universal": (ju, (0.45, 0.3)), "gimbal": (jg, (0.4, -0.35, 0.0))}
        return b, {}, "D6 joints: a driven gantry (two slides and yaw) carrying a pendulum, a universal joint that stops on a limit, and a gimbal.", model_q

    def check(self):
        """The gantry reaches its driven pose and every limited coordinate stays within its limits."""
        q = self.state_0.joint_q.numpy()
        assert np.allclose(q[:3], (0.25, -0.2, 1.0), atol=0.05), f"the gantry is at {q[:3]}, not at its targets (0.25, -0.2, 1.0)"
        lower, upper = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        # the ball joint's quaternion (coordinates 3..6) has no limits; the D6 coordinates map one to one onto DOFs
        dof = [0, 1, 2, 6, 7, 8, 9, 10]
        coord = [0, 1, 2, 7, 8, 9, 10, 11]
        over = np.maximum(q[coord] - upper[dof], lower[dof] - q[coord])
        assert over.max() < 0.02, f"a D6 coordinate is {over.max():.3f} past its limit"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
