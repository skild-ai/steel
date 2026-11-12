# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Ragdoll
#
# Floating base: a jointed figure with a free-floating torso tumbling down stairs.
#
# Ported from steel-webgpu's tools/scenes.py (ragdoll); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run ragdoll
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
    camera = {'eye': (0.6, -2.6, 1.4), 'target': (0.6, 0.0, 0.35)}

    def build(self):
        """Floating base: a jointed figure with a free-floating torso tumbling down stairs."""
        b = newton.ModelBuilder()
        self.preset.apply_model(b)  # the preset's builder settings (the contact gap), before any shape
        b.add_ground_plane()
        for k in range(6):
            h = 0.08 * (5 - k) + 0.04
            b.add_shape_box(-1, xform=tf((0.22 * k, 0, h / 2)), hx=0.11, hy=0.4, hz=h / 2, color=GREY)
        body_cfg = cfg(mu=0.8, density=1000)
        start = np.array((-0.05, 0.0, 0.95))
        torso = b.add_link(xform=tf(start, wp.quat_rpy(0.4, 0.5, 0.0)))
        b.add_shape_capsule(torso, radius=0.07, half_height=0.12, cfg=body_cfg, color=BLUE)
        joints = [b.add_joint_free(torso)]
        head = b.add_link(xform=tf(start + (0, 0, 0.29)))
        b.add_shape_sphere(head, radius=0.08, cfg=body_cfg, color=GOLD)
        joints.append(b.add_joint_ball(torso, head, parent_xform=tf((0, 0, 0.2)), child_xform=tf((0, 0, -0.09))))
        limbs = [((0.0, 0.16, 0.17), (1, 0, 0)), ((0.0, -0.16, 0.17), (1, 0, 0)), ((0.0, 0.06, -0.2), (0, 1, 0)), ((0.0, -0.06, -0.2), (0, 1, 0))]
        for anchor, axis in limbs:
            down = -0.12
            upper = b.add_link(xform=tf(start + anchor + (0, 0, down)))
            b.add_shape_capsule(upper, radius=0.035, half_height=0.08, cfg=body_cfg, color=TEAL)
            joints.append(b.add_joint_ball(torso, upper, parent_xform=tf(anchor), child_xform=tf((0, 0, 0.12))))
            lower = b.add_link(xform=tf(start + anchor + (0, 0, 2 * down)))
            b.add_shape_capsule(lower, radius=0.03, half_height=0.08, cfg=body_cfg, color=TEAL)
            joints.append(
                b.add_joint_revolute(upper, lower, parent_xform=tf((0, 0, -0.12)), child_xform=tf((0, 0, 0.12)), axis=axis,
                                     limit_lower=-2.2, limit_upper=0.0)
            )
        b.add_articulation(joints)
        return b, {torso: (2.0, 0, 0.3, 0, 4.0, 0)}, "Floating base: a 10-body ragdoll with a free-floating torso is thrown onto a staircase."

    def check(self):
        """The ragdoll tumbles to the foot of the stairs and lies still (every body under 2 cm/s, torso 0.9-1.25 m
        out)."""
        qd = self.state_0.body_qd.numpy()
        q = self.state_0.body_q.numpy()
        speed = np.linalg.norm(qd[:, :3], axis=1)
        assert speed.max() < 0.02, f"body {speed.argmax()} still moves at {speed.max():.3f} m/s"
        assert 0.9 < q[0, 0] < 1.25, f"the torso ended at x={q[0, 0]:.2f} m, not at the foot of the stairs"
        assert q[:, 2].min() > 0.0, "a body is below the floor"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
