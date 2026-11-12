# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Parallelogram gripper
#
# One prismatic actuator pulls two rods that swing the upper bars of two parallelogram fingers, so the
# jaws translate symmetrically and stay vertical. Four planar loops (revolute closures); 1 DOF.
#
# The mechanism comes from common/loop_mechanisms.py and runs through common/loop_linkage.py. check compares
# the tracked bodies with reference.json, an exact-constraint reference integration.
#
# Command: python -m shelf run loop_parallel_gripper
#
###########################################################################

from typing import ClassVar

import newton.examples

from common.loop_linkage import LoopLinkageExample


class Example(LoopLinkageExample):
    file = __file__
    mechanism = "gripper"
    period = 2.0  # [s] the trajectory repeats with this period once its one-second ramp is over
    camera: ClassVar = {"eye": (-0.152, -0.416, 0.578), "target": (0.0, 0.0, 0.5)}


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
