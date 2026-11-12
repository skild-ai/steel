# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Overlapping boxes
#
# Velocity pass: boxes spawned 15 mm into each other and the floor.
#
# Ported from steel-webgpu's tools/scenes.py (overlap); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run overlap
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
    camera = {'eye': (0.5, -1.3, 0.6), 'target': (0.0, 0.0, 0.12)}

    def build(self):
        """Velocity pass: boxes spawned 15 mm into each other and the floor."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        for i in range(5):
            for k in range(4):
                body = b.add_body(xform=tf(((i - 2) * 0.13, 0.0, 0.035 + 0.055 * k), wp.quat_rpy(0.0, 0.0, 0.3 * k)))
                b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.035, cfg=cfg(density=400), color=[BLUE, TEAL, GOLD, RED][k])
        return b, {}, "Velocity pass: 20 boxes start 15 mm deep in each other; with velocity_iterations the push-out does not launch them."

    def check(self):
        """Every column stands: each box within 2 cm of its column's axis and 5 mm of its stacked height
        (0.035 + 0.07 k). Not for [steel_off]: without the velocity pass the push-out topples a column, which is what
        that variant shows."""
        if self.preset.name == "steel_off":
            return
        q = self.state_0.body_q.numpy()[:, :3].reshape(5, 4, 3)
        x0 = (np.arange(5) - 2) * 0.13
        off = np.hypot(q[..., 0] - x0[:, None], q[..., 1])
        assert off.max() < 0.02, f"a box is {1e3 * off.max():.0f} mm off its column"
        dz = np.abs(q[..., 2] - (0.035 + 0.07 * np.arange(4)))
        assert dz.max() < 0.005, f"a box sits {1e3 * dz.max():.1f} mm off its stacked height"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
