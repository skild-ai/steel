# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Pinch grip
#
# Contact torsion: sphere-tipped grippers pinch a bar 8 cm from its center; only the gold pair has a torsion radius.
#
# Ported from steel-webgpu's tools/scenes.py (torsion); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run torsion
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
    camera = {'eye': (0.5, -1.0, 0.7), 'target': (0.05, 0.0, 0.45)}

    def build(self):
        """Contact torsion: sphere-tipped grippers pinch a bar 8 cm from its center; only the gold pair has a torsion radius."""
        b = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(b)
        b.add_ground_plane()
        for y, radius, color in ((0.25, 0.0, GREY), (-0.25, 0.01, GOLD)):
            tip_cfg = cfg(mu=0.8, density=2000)
            custom = {"steel:min_torsional_patch_radius": radius} if radius > 0 else None
            base = b.add_link(xform=tf((0.0, y, 0.5)))
            b.add_shape_box(base, xform=tf((0.0, 0.0, 0.08)), hx=0.02, hy=0.06, hz=0.01, cfg=cfg(density=500), color=GREY)
            joints = [b.add_joint_fixed(-1, base, parent_xform=tf((0.0, y, 0.5)))]
            for side in (1.0, -1.0):
                finger = b.add_link(xform=tf((0.0, y + side * 0.022, 0.5)))
                b.add_shape_box(finger, xform=tf((0.0, side * 0.01, 0.04)), hx=0.008, hy=0.004, hz=0.04, cfg=cfg(density=500), color=color)
                b.add_shape_sphere(finger, radius=0.012, cfg=tip_cfg, color=color, custom_attributes=custom)
                joints.append(b.add_joint_prismatic(base, finger, parent_xform=tf((0.0, side * 0.022, 0.0)), axis=(0, side, 0),
                                                    target_pos=-0.004, target_ke=3000.0, target_kd=50.0, effort_limit=30.0,
                                                    limit_lower=-0.02, limit_upper=0.03))
            b.add_articulation(joints)
            bar = b.add_body(xform=tf((0.08, y, 0.5)))
            b.add_shape_box(bar, hx=0.12, hy=0.01, hz=0.025, cfg=cfg(mu=0.8, density=500), color=[BLUE, TEAL][y < 0])
        return b, {}, "Contact torsion: sphere fingertips pinch a bar off its center; the gold tips (min_torsional_patch_radius 1 cm) hold it level, the grey ones let it swing."

    def _tilt(self):
        """Each bar's long axis against the horizontal [deg], grey then gold."""
        q = self.state_0.body_q.numpy()[[3, 7], 3:]
        x, y, z, w = q.T
        return np.degrees(np.arcsin(np.clip(2 * (x * z - w * y), -1.0, 1.0)))

    def test_post_step(self):
        t = self._tilt()
        self._grey_min = min(getattr(self, "_grey_min", 0.0), t[0])
        self._gold_max = max(getattr(self, "_gold_max", 0.0), abs(t[1]))

    def check(self):
        """The gold tips (torsion radius) hold their bar level; the grey ones let theirs swing down."""
        assert self._gold_max < 5.0, f"the gold bar tilted {self._gold_max:.1f} deg"
        assert self._grey_min < -45.0, f"the grey bar only swung to {self._grey_min:.1f} deg"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
