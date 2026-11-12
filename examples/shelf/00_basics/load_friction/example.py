# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Load-dependent friction
#
# Load-dependent joint friction: two loaded pendulums, the gold hinge with joint_friction_load_coefficient 0.4.
#
# Ported from steel-webgpu's tools/scenes.py (load_friction); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run load_friction
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
    camera = {'eye': (0.4, -1.3, 0.8), 'target': (0.1, 0.0, 0.5)}

    def build(self):
        """Load-dependent joint friction: two loaded pendulums, the gold hinge with joint_friction_load_coefficient 0.4."""
        b = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(b)
        b.add_ground_plane()
        joints = []
        for y, coef, color in ((0.2, 0.0, GREY), (-0.2, 0.4, GOLD)):
            arm = b.add_link(xform=tf((0.0, y, 0.6)))
            b.add_shape_capsule(arm, xform=tf((0.15, 0, 0), wp.quat_rpy(0.0, math.pi / 2, 0.0)), radius=0.012, half_height=0.15, cfg=cfg(density=300), color=GREY)
            b.add_shape_sphere(arm, xform=tf((0.3, 0, 0)), radius=0.05, cfg=cfg(density=4000), color=color)
            j = b.add_joint_revolute(-1, arm, parent_xform=tf((0.0, y, 0.6)), axis=(0, 1, 0),
                                     custom_attributes={"steel:joint_friction_load_coefficient": coef})
            b.add_articulation([j])
            joints.append(j)
        start = {"grey": (joints[0], (-0.8,)), "gold": (joints[1], (-0.8,))}
        return b, {}, "Load-dependent joint friction: two loaded pendulums released at 45 deg; the gold hinge's friction grows with the load it carries and holds it.", start

    def step(self):
        super().step()
        if self.test_mode:  # the grey pendulum's widest swing so far, for check()
            self.grey_swing = max(getattr(self, "grey_swing", 0.0), abs(float(self.state_0.joint_q.numpy()[0]) + 0.8))

    def check(self):
        """The gold hinge's load-dependent friction holds its pendulum at the release angle; the grey one swings."""
        q = self.state_0.joint_q.numpy()
        assert abs(q[1] + 0.8) < 0.01, f"the gold pendulum moved to {q[1]:.3f} rad from its -0.8 rad release"
        assert self.grey_swing > 1.0, f"the grey pendulum swung only {self.grey_swing:.3f} rad: it should swing freely"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
