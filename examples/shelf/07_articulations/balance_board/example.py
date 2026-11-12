# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Balance board
#
# A spring-loaded balance beam tips because one counterweight is missing; the arm puts it back and the beam levels out.
#
# Rebuilt from steel-webgpu's exported scene (assets/scenes/balance_board: scene.json
# and scene.bin, copied next to this file); see common/exported_scene.py. The
# visual meshes draw when a steel-webgpu checkout sits next to the shelf.
#
# Command: python -m shelf run balance_board
#
###########################################################################

import numpy as np

import newton.examples
from common.exported_scene import ExportedSceneExample


class Example(ExportedSceneExample):
    file = __file__

    def test_post_step(self):
        pivot = float(self.state_0.joint_q.numpy()[self.model.joint_q_start.numpy()[16]])  # balance.pivot.q
        self._tip = min(getattr(self, "_tip", 0.0), pivot)
        self._pivots = (getattr(self, "_pivots", []) + [pivot])[-60:]

    def check(self):
        """The beam tips (about -0.28 rad), the arm places the missing weight and the beam levels out at +0.023 rad
        (a converged 16 x 32 run) and stays there."""
        assert self._tip < -0.2, f"the beam only tipped to {self._tip:.3f} rad"
        level = self._pivots[-1]
        assert abs(level - 0.023) < 0.01, f"the beam ends at {level:.4f} rad, converged +0.023"
        assert np.ptp(self._pivots) < 0.002, f"the beam still moves {np.ptp(self._pivots):.4f} rad in the last second"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
