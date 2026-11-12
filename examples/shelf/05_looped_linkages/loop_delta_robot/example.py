# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Delta robot
#
# Three driven upper arms carry a 0.3 kg platform through parallelogram forearms (ball joints at the
# elbows, universal joints at the platform). Five ball closures, nearly redundant rows; 3 DOF.
#
# The mechanism comes from common/loop_mechanisms.py and runs through common/loop_linkage.py. check compares
# the tracked bodies with reference.json, an exact-constraint reference integration.
#
# Command: python -m shelf run loop_delta_robot
#
###########################################################################

from typing import ClassVar

import newton.examples

from common.loop_linkage import LoopLinkageExample


class Example(LoopLinkageExample):
    file = __file__
    mechanism = "delta"
    period = 10.0  # [s] the trajectory repeats with this period once its one-second ramp is over
    camera: ClassVar = {"eye": (-0.248, -0.681, 0.814), "target": (0.0, 0.0, 0.62)}


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
