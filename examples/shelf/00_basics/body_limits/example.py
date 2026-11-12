# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Speed caps
#
# Steel's per-body attributes: max linear velocity, max angular velocity and angular damping.
#
# Ported from steel-webgpu's tools/scenes.py (body_limits); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run body_limits
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
    camera = {'eye': (0.0, -2.0, 1.0), 'target': (0.0, 0.0, 0.3)}

    def build(self):
        """Steel's per-body attributes: max linear velocity, max angular velocity and angular damping."""
        b = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(b)
        b.add_ground_plane(cfg=cfg(mu=0.0))
        twists = {}
        rows = (
            ("max linear velocity 1 m/s", {"steel:rigid_body_max_linear_velocity": 1.0}, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.6),
            ("max angular velocity 5 rad/s", {"steel:rigid_body_max_angular_velocity": 5.0}, (0.0, 0.0, 0.0, 0.0, 0.0, 30.0), 0.3),
            ("angular damping 2 /s", {"steel:rigid_body_angular_damping": 2.0}, (0.0, 0.0, 0.0, 0.0, 0.0, 30.0), 0.3),
        )
        for i, (_, custom, tw, z) in enumerate(rows):
            for k, limited in enumerate((False, True)):
                body = b.add_body(xform=tf((-0.4 + 0.4 * i, -0.15 + 0.3 * k, z)), custom_attributes=custom if limited else None)
                b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg(density=500, mu=0.0), color=[GREY, GOLD][k])
                twists[body] = tw
        return b, twists, "Per-body limits (gold vs grey): a 1 m/s fall cap, a 5 rad/s spin cap and 2 /s angular damping on frictionless spinners."

    def check(self):
        """At the end of the test run (100 frames): the twists reached the spinners and each gold limit acts."""
        w = np.abs(self.state_0.body_qd.numpy()[:, 5])
        # bodies: (grey, gold) per row; rows: fall cap, spin cap, angular damping
        assert w[2] > 25.0 and w[4] > 25.0, f"the grey spinners lost their 30 rad/s twist: {w[[2, 4]]}"
        assert 4.0 < w[3] <= 5.05, f"the spin-capped box spins at {w[3]:.3f} rad/s, not at its 5 rad/s cap"
        assert w[5] < 0.1 * w[4], f"the damped spinner still spins at {w[5]:.3f} rad/s"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
