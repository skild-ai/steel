# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Spinning and rolling friction
#
# Torsional and rolling friction on free bodies (steel:condim 6): per lane, a spinning ball and a rolling ball.
#
# Ported from steel-webgpu's tools/scenes.py (spin_roll); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run spin_roll
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
        """Torsional and rolling friction on free bodies (steel:condim 6): per lane, a spinning ball and a rolling ball."""
        b = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(b)
        b.add_ground_plane(cfg=cfg(mu=0.6))
        twists = {}
        # Free-body contacts take the larger condim-enabled coefficient of the pair (the ground stays condim 3).
        for mt, mr, y, color in ((0.0, 0.0, 0.3, GREY), (0.02, 0.004, -0.3, GOLD)):
            ball_cfg = cfg(mu=0.6, mu_torsional=mt, mu_rolling=mr, density=800)
            custom = {"steel:condim": 6} if mt > 0 else None
            spinner = b.add_body(xform=tf((-0.6, y, 0.05)))
            b.add_shape_sphere(spinner, radius=0.05, cfg=ball_cfg, color=color, custom_attributes=custom)
            twists[spinner] = (0, 0, 0, 0, 0, 25.0)
            roller = b.add_body(xform=tf((-0.4, y, 0.05)))
            b.add_shape_sphere(roller, radius=0.05, cfg=ball_cfg, color=color, custom_attributes=custom)
            twists[roller] = (0.8, 0, 0, 0, 0.8 / 0.05, 0)
        return b, twists, "Torsional and rolling friction (steel:condim 6 on the gold balls): the grey lane keeps spinning and rolling; the gold lane stops."



if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
