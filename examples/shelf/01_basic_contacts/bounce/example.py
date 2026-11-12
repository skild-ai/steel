# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Bounce
#
# Restitution: five balls, each on a pad of the same restitution (0 ... 0.9), dropped from 0.8 m.
#
# Ported from steel-webgpu's tools/scenes.py (bounce); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run bounce
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
    camera = {'eye': (1.25, -1.6, 1.25), 'target': (0.1, 0.05, 0.15)}

    def build(self):
        """Restitution: five balls, each on a pad of the same restitution (0 ... 0.9), dropped from 0.8 m."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        twists = {}
        for i, e in enumerate((0.0, 0.25, 0.5, 0.75, 0.9)):
            x = (i - 2) * 0.25
            b.add_shape_box(-1, xform=tf((x, 0, 0.02)), hx=0.1, hy=0.1, hz=0.02, cfg=cfg(restitution=e), color=GREY)
            body = b.add_body(xform=tf((x, 0, 0.8)))
            b.add_shape_sphere(body, radius=0.04, cfg=cfg(restitution=e, density=1000), color=[BLUE, TEAL, GOLD, RED, BLUE][i])
        return b, twists, "Restitution: balls on pads of restitution 0, 0.25, 0.5, 0.75 and 0.9, left to right."

    def test_post_step(self):
        self._z = getattr(self, "_z", []) + [self.state_0.body_q.numpy()[:, 2].copy()]

    def check(self):
        """Each ball's first rebound is e^2 of its 0.72 m drop within 15 mm (16 x 50: 47, 182, 408, 587 mm)."""
        z = np.array(self._z)
        for ball, e in enumerate((0.0, 0.25, 0.5, 0.75, 0.9)):
            up = np.flatnonzero(np.diff(z[:, ball]) > 0)
            rebound = float(z[up[0] :, ball].max() - 0.08) if len(up) else 0.0
            assert abs(rebound - e * e * 0.72) < 0.015, (
                f"ball {ball} (e={e}) rebounds {1e3 * rebound:.0f} mm, e^2 h is {720 * e * e:.0f}"
            )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
