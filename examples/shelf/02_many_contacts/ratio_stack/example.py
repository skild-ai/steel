# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Mass-ratio stack
#
# Mass-ratio stack: six cubes, each about 3x the mass of the one below (about 270:1 overall).
#
# Ported from steel-webgpu's tools/scenes.py (ratio_stack); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run ratio_stack
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
    camera = {'eye': (12.0, -20.0, 9.0), 'target': (0.0, 0.0, 4.5)}

    def build(self):
        """Mass-ratio stack: six cubes, each about 3x the mass of the one below (about 270:1 overall)."""
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=cfg(mu=0.8))
        side, z = 0.5, 0.0
        for k in range(6):
            z += side / 2
            body = b.add_body(xform=tf((0, 0, z)))
            b.add_shape_box(body, hx=side * 0.49, hy=side * 0.49, hz=side * 0.49, cfg=cfg(density=1000, mu=0.8), color=[BLUE, TEAL, GOLD, RED][k % 4])
            z += side / 2
            side *= 1.45
        return b, {}, "Mass-ratio stack: six cubes, each about 3x heavier than the one below. Momentum keeps the tower standing; switch it off and it topples within seconds."

    def check(self):
        """The tower stands: every cube ends within 5 cm of its height in a stack of touching cubes."""
        body = self.model.shape_body.numpy()
        half = self.model.shape_scale.numpy()[:, 2]
        z = self.state_0.body_q.numpy()[:, 2]
        top = 0.0
        for k in range(self.model.body_count):
            hz = half[body == k][0]
            assert abs(z[k] - (top + hz)) < 0.05, f"cube {k} at z={z[k]:.3f}, touching stack puts it at {top + hz:.3f}"
            top += 2.0 * hz


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
