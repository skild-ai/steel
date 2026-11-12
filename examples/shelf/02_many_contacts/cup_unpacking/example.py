# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

"""Clear lightweight packing, grasp a buried mug and place it outside its box.

Command: python -m shelf run cup_unpacking

Drag bodies to apply forces, pause or scrub the robot path, and reset the scene.
Only the robot joints are position-driven; object motion comes from contacts.
"""

import numpy as np
import warp as wp

import newton.examples
from common import presets

from newton.examples.robot import _cup_unpacking
from newton.examples.robot._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "cup_unpacking"
    robot_root = (-0.48, -0.25, 0)

    camera_position = (1.25, -1.65, 1.45)

    def __init__(self, viewer, args):
        self.preset = presets.load(__file__)
        # TabletopDemo (installed newton package) builds SolverSteel from these args, so fill them
        # from the preset before handing args over. Its frame rate is fixed at 60 Hz.
        args.substeps = self.preset.substeps
        args.iterations = self.preset.solver.need("iterations")
        args.solve_mode = self.preset.solver.need("solve_mode")
        args.contact_response = self.preset.solver.need("articulated_contact_response")
        super().__init__(viewer, args)

    @staticmethod
    def create_parser():
        # TabletopDemo.create_parser() without its solver-tuning arguments (--substeps, --iterations,
        # --solve-mode, --contact-response); those values now live in presets.toml.
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=900)
        return parser

    def populate(self, builder, info, ee, down, fingers):
        _cup_unpacking.populate(builder, info, ee, down, fingers, wp.get_device())

    def reset(self):
        super().reset()
        self.test_poses = [self.initial.body_q.numpy()] if self.test_mode else []

    def test_post_step(self):
        """Reject invalid states and record the physical mechanism trajectory."""
        super().test_post_step()
        self.test_poses.append(self.state_0.body_q.numpy())

    def test_final(self):
        """Validate the completed manipulation and mechanism stability."""
        self.check_capacity()
        result = _cup_unpacking.check(self.info, np.asarray(self.test_poses), self.fps)
        assert result["pass"], result


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
