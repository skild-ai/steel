# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Stewart platform
#
# A 6-UPS hexapod: six universal-prismatic-spherical legs carry a 2 kg platform through heave, roll,
# pitch and yaw. Five spatial loops (ball closures) share the platform; 6 DOF, fully actuated.
#
# The mechanism comes from common/loop_mechanisms.py and runs through common/loop_linkage.py. check compares
# the tracked bodies with reference.json, an exact-constraint reference integration.
#
# Command: python -m shelf run loop_stewart_platform
#
###########################################################################

from typing import ClassVar

import newton.examples

from common.loop_linkage import LoopLinkageExample


class Example(LoopLinkageExample):
    file = __file__
    mechanism = "stewart"
    period = 10.0  # [s] the trajectory repeats with this period once its one-second ramp is over
    camera: ClassVar = {"eye": (-0.44, 0.761, 0.66), "target": (0.0, 0.0, 0.25)}


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
