# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org falling chain
#
# Four interlocked chain links of 6 mm capsules drop onto a thin catch bar at a coarse
# time step and hang there without passing through each other.
#
# Replicates the "Speculative Contacts: Falling Chain" scene of the MuJoCo 4.0 tour on
# mujoco.org from its MJCF (scene.xml, flattened from google-deepmind/mujoco
# model/speculative/chain.xml). See README.md for the gaps.
#
# Command: python -m shelf run mjorg_falling_chain
#
###########################################################################

from pathlib import Path

import numpy as np

import newton
import newton.examples
from common.steel_scenes import SceneExample


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth 135, elevation -14 around the statistic centre
    camera = {"eye": (1.48, -1.48, 1.77), "target": (0.0, 0.0, 1.25)}

    def build(self):
        builder = newton.ModelBuilder()
        self.preset.apply_model(builder)
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")))
        return builder, {}

    def check(self):
        """The chain dropped onto the bar and hangs from it, each link stretched below the one above."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        q = self.state_0.body_q.numpy()
        assert 1.6 < q[0, 2] < 1.76, f"link1 is not hanging from the catch bar (z={q[0, 2]:.3f})"
        # links start 18 cm apart; hanging interlocked they are about 31 cm apart, and more once one slips out
        gaps = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
        assert (gaps > 0.25).all() and (gaps < 0.34).all(), f"links not hanging interlocked: gaps {np.round(gaps, 3)}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
