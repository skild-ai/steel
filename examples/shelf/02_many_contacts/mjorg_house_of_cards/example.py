# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org house of cards
#
# A four-storey house of 26 playing cards, 0.7 mm thick, standing on friction alone.
#
# Replicates the "House of Cards" scene of the MuJoCo 4.0 tour on mujoco.org from its
# MJCF (scene.xml, flattened from google-deepmind/mujoco model/cards/house_of_cards.xml).
# See README.md for the gaps.
#
# Command: python -m shelf run mjorg_house_of_cards
#
###########################################################################

from pathlib import Path

import numpy as np

import newton
import newton.examples
from common.steel_scenes import SceneExample

# How far a card may move before the house counts as failing [m].
CARD_SHIFT_MAX = 0.005


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth 150, elevation -15 around the statistic centre
    camera = {"eye": (0.44, -0.25, 0.24), "target": (0.0, 0.0, 0.1)}

    def build(self):
        builder = newton.ModelBuilder()
        self.preset.apply_model(builder)
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")))
        self.start = np.array([list(builder.body_q[b])[:3] for b in range(builder.body_count)])
        return builder, {}

    def check(self):
        """Every card stays where it was built."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        q = self.state_0.body_q.numpy()
        shift = np.linalg.norm(q[:, :3] - self.start, axis=1)
        assert shift.max() < CARD_SHIFT_MAX, (
            f"{int((shift >= CARD_SHIFT_MAX).sum())} cards moved, up to {shift.max():.4f} m"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
