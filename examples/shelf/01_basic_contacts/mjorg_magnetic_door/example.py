# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org magnetic door
#
# A hinged fridge door with a magnet strip that faces a strip on the cabinet frame. The
# ajar door snaps shut, holds against a light tug on its handle, opens on a hard one, and
# its hinge spring swings it back until the magnets latch it again.
#
# Replicates the "Magnetic Adhesion" scene of the MuJoCo 4.0 tour on mujoco.org from its
# MJCF (scene.xml, google-deepmind/mujoco model/adhesion/fridge_door.xml), with Steel's
# contact adhesion: the geoms' `adhesion` imports as `steel:adhesion`. See README.md for
# the gaps.
#
# Command: python -m shelf run mjorg_magnetic_door
#
###########################################################################

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from common.steel_scenes import SceneExample
from newton.solvers import SolverSteel

AJAR_ANGLE = 0.006  # [rad] the XML's `ajar` keyframe
HANDLE_LEVER = 0.46  # [m] the handle's distance from the hinge axis
# (start [s], duration [s], pull on the handle [N]); the light tug stays below the latch's pull-off
TUGS = ((1.0, 0.3, 4.0), (2.0, 0.15, 25.0))
LATCHED_ANGLE = 0.002  # [rad] about 1 mm at the magnet strip
OPENED_ANGLE = math.radians(20.0)


@wp.kernel(enable_backward=False)
def tug_handle(dt: float, dof: int, time: wp.array[wp.float32], joint_f: wp.array[wp.float32]):
    """Write the handle tug at the current time as an opening hinge torque, then advance the time."""
    t = time[0]
    torque = float(0.0)
    if t >= wp.static(TUGS[0][0]) and t < wp.static(TUGS[0][0] + TUGS[0][1]):
        torque = wp.static(TUGS[0][2] * HANDLE_LEVER)
    if t >= wp.static(TUGS[1][0]) and t < wp.static(TUGS[1][0] + TUGS[1][1]):
        torque = wp.static(TUGS[1][2] * HANDLE_LEVER)
    joint_f[dof] = torque
    time[0] = t + dt


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth -120, elevation -20 around the statistic centre
    camera = {"eye": (0.96, 1.67, 1.10), "target": (0.0, 0.0, 0.4)}

    def build(self):
        builder = newton.ModelBuilder()
        SolverSteel.register_custom_attributes(builder)
        self.preset.apply_model(builder)
        # the hinge pin is the scene's one visual-only geom; show the colliders next to it
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")), force_show_colliders=True)
        hinge = builder.joint_label.index("Magnetic fridge door/worldbody/door/hinge")
        self.hinge_q = builder.joint_q_start[hinge]
        self.hinge_dof = builder.joint_qd_start[hinge]
        builder.joint_q[self.hinge_q] = AJAR_ANGLE
        self.time = wp.zeros(1, dtype=wp.float32)
        self.angles = []
        return builder, {}

    def before_substep(self, substep):
        wp.launch(tug_handle, dim=1, inputs=[self.sim_dt, self.hinge_dof], outputs=[self.time, self.control.joint_f])

    def test_post_step(self):
        self.angles.append((self.sim_time, float(self.state_0.joint_q.numpy()[self.hinge_q])))

    def check(self):
        """The door snaps shut, holds the light tug, opens on the hard tug and latches again."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        t, q = (np.asarray(v) for v in zip(*self.angles))
        light, hard = TUGS[0][0], TUGS[1][0]
        assert t[-1] >= hard + 4.5, f"run at least {hard + 4.5:g} s, not {t[-1]:.2f} s"
        closed = q[(t >= 0.5) & (t < light)]
        assert closed.max() <= LATCHED_ANGLE, f"the ajar door did not snap shut (max {closed.max():.4f} rad)"
        held = q[(t >= light) & (t < hard)]
        assert held.max() <= LATCHED_ANGLE, f"the light tug opened the door to {held.max():.4f} rad"
        opened = q[t >= hard].max()
        assert opened >= OPENED_ANGLE, f"the hard tug opened the door only to {math.degrees(opened):.1f} deg"
        end = q[t >= t[-1] - 0.5]
        assert end.max() <= LATCHED_ANGLE, f"the door is not latched again at the end (max {end.max():.4f} rad)"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
