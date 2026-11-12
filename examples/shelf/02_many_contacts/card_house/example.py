# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Card house
#
# Card house: four levels of leaning cards and flat separators held up by friction alone.
#
# Ported from steel-webgpu's tools/scenes.py (card_house); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run card_house
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
    camera = {'eye': (3.5, -6.0, 3.6), 'target': (0.0, 0.0, 1.8)}

    def build(self):
        """Card house: four levels of leaning cards and flat separators held up by friction alone."""
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=cfg(mu=0.9))
        w, t, depth, theta = 1.2, 0.05, 1.0, 1.22
        foot, h = w * math.cos(theta), w * math.sin(theta)
        pitch = 2 * foot + 0.15
        card = cfg(density=200, mu=0.9)
        levels = 4
        for level in range(levels):
            n = levels - level
            z0 = level * (h + t + t * math.cos(theta))
            x0 = -(n - 1) * pitch / 2
            for i in range(n):
                cx = x0 + i * pitch
                for side in (-1, 1):
                    # Faces (not mid-planes) meet at the apex, and the lower corners rest on the floor.
                    x = cx + side * (foot / 2 + t / (2 * math.sin(theta)))
                    body = b.add_body(xform=tf((x, 0, z0 + h / 2 + t / 2 * math.cos(theta)), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), side * theta)))
                    b.add_shape_box(body, hx=w / 2, hy=depth / 2, hz=t / 2, cfg=card, color=[BLUE, TEAL, GOLD, RED][(level + i) % 4])
            if level < levels - 1:
                for i in range(n - 1):
                    body = b.add_body(xform=tf((x0 + (i + 0.5) * pitch, 0, z0 + h + t * math.cos(theta) + t / 2)))
                    # 0.99 pitch, not steel-webgpu's 1.05: at 1.05 neighbouring separators overlap 49 mm end to end and
                    # start the scene by pushing each other apart; at 0.99 they leave a 1 cm gap over each apex and
                    # still rest on both cards' outer corners (5 cm from the apex)
                    b.add_shape_box(body, hx=pitch * 0.99 / 2, hy=depth / 2, hz=t / 2, cfg=card, color=GREY)
        return b, {}, "Card house: four levels of leaning cards and flat separators stand on friction alone."

    def check(self):
        """The house stands: every card and separator stays within 25 mm of where it was built (it settles ~11 mm)."""
        moved = np.linalg.norm(self.state_0.body_q.numpy()[:, :3] - self.model.body_q.numpy()[:, :3], axis=1)
        assert moved.max() < 0.025, f"body {moved.argmax()} moved {1e3 * moved.max():.0f} mm: the house is coming down"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
