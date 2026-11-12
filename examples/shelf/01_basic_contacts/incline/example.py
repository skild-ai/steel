# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Incline
#
# Friction anchors: boxes and short stacks on a 22 deg incline with mu 0.5.
#
# Ported from steel-webgpu's tools/scenes.py (incline); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run incline
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
    camera = {'eye': (0.4, -1.9, 1.1), 'target': (0.0, 0.0, 0.4)}

    def build(self):
        """Friction anchors: boxes and short stacks on a 22 deg incline with mu 0.5."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        ang = math.radians(22)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -ang)
        n = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))
        b.add_shape_box(-1, xform=tf((0, 0, 0.35), q), hx=0.6, hy=0.45, hz=0.02, cfg=cfg(mu=0.5), color=GREY)
        for y, levels in ((-0.25, 1), (0.0, 2), (0.25, 3)):
            for k in range(levels):
                c = np.array((0, y, 0.35)) + n * (0.02 + 0.03 + 0.06 * k + 0.0005 * (k + 1))
                body = b.add_body(xform=tf(c, q))
                b.add_shape_box(body, hx=0.075, hy=0.075, hz=0.03, cfg=cfg(mu=0.5, density=400), color=[BLUE, TEAL, GOLD, RED][k])
        return b, {}, "Friction anchors: a box and stacks of 2 and 3 rest on a 22 deg incline (mu 0.5) without creeping."

    def test_post_step(self):
        self._frame = getattr(self, "_frame", 0) + 1
        if self._frame == 60:
            self._after_1s = self.state_0.body_q.numpy()[:, :3].copy()

    def check(self):
        """Nothing creeps: after the first second every box stays within 1 mm, and within 5 mm of where it was built."""
        q = self.state_0.body_q.numpy()[:, :3]
        creep = np.linalg.norm(q - self._after_1s, axis=1)
        moved = np.linalg.norm(q - self.model.body_q.numpy()[:, :3], axis=1)
        assert creep.max() < 0.001, f"box {creep.argmax()} crept {1e3 * creep.max():.1f} mm after the first second"
        assert moved.max() < 0.005, f"box {moved.argmax()} moved {1e3 * moved.max():.1f} mm"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
