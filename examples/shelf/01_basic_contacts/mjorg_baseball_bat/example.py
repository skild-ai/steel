# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org baseball bat
#
# A spring plunger pitches a 4 cm ball and a spring-loaded 1.2 cm thin bat whips round
# and hits it back, at a 10 ms time step.
#
# Replicates the "Speculative Contacts: Baseball Bat" scene of the MuJoCo 4.0 tour on
# mujoco.org from its MJCF (scene.xml, flattened from google-deepmind/mujoco
# model/speculative/flipper.xml). See README.md for the gaps.
#
# Command: python -m shelf run mjorg_baseball_bat
#
###########################################################################

from pathlib import Path

import newton
import newton.examples
from common.steel_scenes import SceneExample


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth 118, elevation -23 around the statistic centre
    camera = {"eye": (1.09, -1.39, 0.85), "target": (0.4, -0.1, 0.23)}

    def build(self):
        builder = newton.ModelBuilder()
        self.preset.apply_model(builder)
        # the bat post is the scene's one visual-only geom; show the colliders next to it
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")), force_show_colliders=True)
        self.ball = builder.body_label.index("baseball_bat_ccd/worldbody/ball")
        return builder, {}

    def test_post_step(self):
        vx = float(self.state_0.body_qd.numpy()[self.ball, 0])
        self.pitch_vx = min(getattr(self, "pitch_vx", 0.0), vx)
        if self.pitch_vx < -1.0:
            self.return_vx = max(getattr(self, "return_vx", vx), vx)

    def check(self):
        """The plunger pitches the ball towards the bat and the bat sends it back."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        assert self.pitch_vx < -3.0, f"the ball was never pitched (min vx {self.pitch_vx:.2f} m/s)"
        assert self.return_vx > 1.0, f"the bat missed the ball (max vx after the pitch {self.return_vx:.2f} m/s)"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
