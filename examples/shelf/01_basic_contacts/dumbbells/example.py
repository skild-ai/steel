# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Dumbbells
#
# Articulated torsional and rolling friction (elliptic cone): a rolling and a spinning dumbbell.
#
# Ported from steel-webgpu's tools/scenes.py (dumbbells); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run dumbbells
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
    camera = {'eye': (0.0, -1.4, 0.6), 'target': (0.0, 0.0, 0.05)}

    def build(self):
        """Articulated torsional and rolling friction (elliptic cone): a rolling and a spinning dumbbell."""
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=cfg(mu=0.6, mu_torsional=0.01, mu_rolling=0.01))
        twists = {}
        ball = cfg(mu=0.6, mu_torsional=0.01, mu_rolling=0.01, density=800)
        for x, y, twist, color in ((-0.4, 0.2, (0.8, 0, 0, 0, 0.8 / 0.05, 0), BLUE), (0.2, -0.2, (0, 0, 0, 0, 0, 20.0), GOLD)):
            root = b.add_link(xform=tf((x, y - 0.07, 0.05)))
            b.add_shape_sphere(root, radius=0.05, cfg=ball, color=color)
            other = b.add_link(xform=tf((x, y + 0.07, 0.05)))
            b.add_shape_sphere(other, radius=0.05, cfg=ball, color=color)
            b.add_shape_capsule(other, xform=tf((0, -0.07, 0), wp.quat_rpy(math.pi / 2, 0.0, 0.0)), radius=0.012, half_height=0.05, cfg=cfg(density=300), color=GREY)
            j0 = b.add_joint_free(root)
            j1 = b.add_joint_fixed(root, other, parent_xform=tf((0, 0.14, 0)))
            b.add_articulation([j0, j1])
            twists[root] = twist
        return b, twists, "Articulated torsional and rolling friction: a rolling and a spinning dumbbell (floating-base articulations) stop under enable_torsional_rolling_friction."



if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
