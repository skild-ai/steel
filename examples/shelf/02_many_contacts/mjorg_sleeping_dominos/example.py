# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org sleeping dominos
#
# 96 dominoes in five rows joined by U-turns. One tilted domino starts the wave; the rest
# sleep until it reaches them, and fall back asleep once they have toppled.
#
# Replicates the "Island Sleeping" scene of the MuJoCo 4.0 tour on mujoco.org from its
# MJCF (scene.xml, flattened from google-deepmind/mujoco model/sleep/dominos.xml), with
# Steel's island sleeping. See README.md for the gaps.
#
# Command: python -m shelf run mjorg_sleeping_dominos
#
###########################################################################

from pathlib import Path

import numpy as np

import newton
import newton.examples
from common.steel_scenes import SceneExample

# A domino has toppled when its long axis leans this far from vertical [deg].
TOPPLED_TILT_DEG = 45.0


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth 60, elevation -25 around the statistic centre
    camera = {"eye": (-0.34, -2.66, 1.43), "target": (1.2, 0.0, 0.0)}

    def build(self):
        builder = newton.ModelBuilder()
        self.preset.apply_model(builder)
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")))
        return builder, {}

    def check(self):
        """Every domino has toppled."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        q = self.state_0.body_q.numpy()
        x, y = q[:, 3], q[:, 4]
        tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
        standing = np.flatnonzero(tilt < TOPPLED_TILT_DEG)
        assert len(standing) == 0, f"{len(standing)} of {len(q)} dominoes still stand: bodies {standing.tolist()}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
