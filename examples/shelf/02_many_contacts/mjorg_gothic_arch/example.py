# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org gothic arch
#
# A dry-stone pointed arch: sixteen voussoirs and a keystone between two fixed springers,
# standing on friction alone with no mortar.
#
# Replicates the "Dry-Stone Gothic Arch" scene of the MuJoCo 4.0 tour on mujoco.org
# from its MJCF (scene.xml, flattened from google-deepmind/mujoco model/arch/gothic.xml).
# See README.md for the gaps.
#
# Command: python -m shelf run mjorg_gothic_arch
#
###########################################################################

from pathlib import Path

import numpy as np

import newton
import newton.examples
from common.steel_scenes import SceneExample

# How far the keystone may sink and any stone may move before the arch counts as fallen [m]; a few
# stones can slip several mm in the first seconds before the arch locks.
KEYSTONE_DROP_MAX = 0.01
STONE_SHIFT_MAX = 0.025
# How far any stone may move over the last 5 s of the check once the arch has settled [m].
SETTLED_SHIFT_MAX = 0.005
SETTLED_FRAME = 300


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth 140, elevation -20 around the statistic centre
    camera = {"eye": (2.63, -2.21, 2.26), "target": (0.0, 0.0, 1.01)}

    def build(self):
        builder = newton.ModelBuilder()
        self.preset.apply_model(builder)
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")))
        self.keystone = builder.body_label.index("Gothic arch/worldbody/keystone")
        self.start = np.array([list(builder.body_q[b])[:3] for b in range(builder.body_count)])
        self.frame = 0
        return builder, {}

    def test_post_step(self):
        self.frame += 1
        if self.frame == SETTLED_FRAME:
            self.settled = self.state_0.body_q.numpy()[:, :3].copy()

    def check(self):
        """The arch stands: the keystone holds its height, no stone slides out, and the settled arch stays put."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        q = self.state_0.body_q.numpy()
        drop = self.start[self.keystone, 2] - q[self.keystone, 2]
        shift = np.linalg.norm(q[:, :3] - self.start, axis=1).max()
        creep = np.linalg.norm(q[:, :3] - self.settled, axis=1).max()
        assert drop < KEYSTONE_DROP_MAX, f"the keystone sank {drop * 1000:.1f} mm"
        assert shift < STONE_SHIFT_MAX, f"a stone moved {shift * 1000:.1f} mm"
        assert creep < SETTLED_SHIFT_MAX, f"a stone moved {creep * 1000:.1f} mm after the arch settled"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
