# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Kinematic paddle
#
# Kinematic bodies: a kinematic paddle sweeps boxes at a prescribed spin; a gravity-free box floats.
#
# Ported from steel-webgpu's tools/scenes.py (kinematic); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run kinematic
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
    camera = {'eye': (0.6, -1.2, 0.8), 'target': (0.0, 0.0, 0.1)}

    def build(self):
        """Kinematic bodies: a kinematic paddle sweeps boxes at a prescribed spin; a gravity-free box floats."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        paddle = b.add_body(xform=tf((0.0, 0.0, 0.05)), is_kinematic=True)
        b.add_shape_box(paddle, hx=0.42, hy=0.015, hz=0.04, cfg=cfg(density=500), color=RED)
        twists = {paddle: (0.0, 0.0, 0.0, 0.0, 0.0, 1.2)}
        for k in range(10):
            # half a spacing off steel-webgpu's ring (2 pi k / 10), whose boxes at 0 and pi start 45 mm inside the
            # paddle and are shot out by the depenetration (at 3.5 m at 16 substeps) before the paddle reaches them
            a = 2 * math.pi * (k + 0.5) / 10
            body = b.add_body(xform=tf((0.32 * math.cos(a), 0.32 * math.sin(a), 0.03)))
            b.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03, cfg=cfg(density=300), color=[BLUE, TEAL, GOLD][k % 3])
        floater = b.add_body(xform=tf((0.0, 0.0, 0.4)), disable_gravity=True)
        b.add_shape_box(floater, hx=0.04, hy=0.04, hz=0.04, cfg=cfg(density=300), color=GOLD)
        twists[floater] = (0.0, 0.0, 0.0, 0.3, 0.5, 0.0)
        return b, twists, "Kinematic bodies: a kinematic paddle (prescribed 1.2 rad/s spin, infinite mass) sweeps boxes; the gold box above has gravity disabled."



if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
