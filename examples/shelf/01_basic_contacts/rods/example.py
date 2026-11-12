# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Sliding blocks
#
# Coupled friction (friction_mode): tall blocks launched sliding on a high-friction floor.
#
# Ported from steel-webgpu's tools/scenes.py (rods); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run rods
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
    camera = {'eye': (0.2, -1.6, 0.6), 'target': (0.0, 0.0, 0.1)}

    def build(self):
        """Coupled friction (friction_mode): tall blocks launched sliding on a high-friction floor."""
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=cfg(mu=0.9))
        twists = {}
        for i, v in enumerate((1.0, 1.5, 2.0, 2.5, 3.0)):
            body = b.add_body(xform=tf((-0.6, (i - 2) * 0.18, 0.12)))
            b.add_shape_box(body, hx=0.03, hy=0.05, hz=0.12, cfg=cfg(mu=0.9, density=600), color=[BLUE, TEAL, GOLD, RED, GREY][i])
            twists[body] = (v, 0, 0, 0, 0, 0)
        return b, twists, "Coupled friction: tall blocks launched at 1 to 3 m/s on a mu 0.9 floor slide and tip; with friction_mode bisection_desaxce (normal target mu |v_t|) sliding blocks hop."

    def test_post_step(self):
        self._zmax = np.maximum(getattr(self, "_zmax", 0.0), self.state_0.body_q.numpy()[:, 2])

    def check(self):
        """Blocks launched at 1-3 m/s slide farther the faster they start, 0.12-0.65 m (16 x 50: 0.17-0.59), stay
        in their lane and never hop (coupled friction without de Saxce's lift)."""
        q = self.state_0.body_q.numpy()
        slide = q[:, 0] + 0.6
        assert np.all(np.diff(slide) > 0), f"slides not ordered by launch speed: {np.round(slide, 3)}"
        assert slide.min() > 0.12 and slide.max() < 0.65, f"slides {np.round(slide, 3)} m outside 0.12-0.65"
        lane = np.abs(q[:, 1] - (np.arange(5) - 2) * 0.18)
        assert lane.max() < 0.01, f"block {lane.argmax()} wandered {1e3 * lane.max():.0f} mm out of its lane"
        assert self._zmax.max() < 0.13, f"a block hopped to z={self._zmax.max():.3f} m"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
