# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Basic Pendulum
#
# Shows how to set up a simulation of a simple double pendulum using the
# newton.ModelBuilder() class.
#
# Command: python -m shelf run basic_pendulum --solver xpbd
#          python -m shelf run basic_pendulum --solver kamino
#
###########################################################################

import warp as wp

import newton
import newton.examples
from common import presets


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.preset = presets.load(__file__)
        self.solver_type = self.preset.name
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(builder)

        hx = 1.0
        hy = 0.1
        hz = 0.1

        # create first link
        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        # add joints
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            # rotate pendulum around the z-axis to appear sideways to the viewer
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        # Create articulation from joints
        builder.add_articulation([j0, j1], label="pendulum")

        # add ground plane
        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.solver = self.preset.make_solver(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if self.solver_type == "kamino":
            self.collision_pipeline = None
            self.contacts = None
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        contacts = None if self.solver_type == "kamino" else self.contacts
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)

            # swap states
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                # an odd count under graph capture: copy the last result into state_0 instead of swapping,
                # so every replay of the captured graph starts from the newest state
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # rough check that the pendulum links are in the correct area
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pendulum links in correct area",
            lambda q, qd: abs(q[0]) < 1e-5 and abs(q[1]) < 1.0 and q[2] < 5.0 and q[2] > 0.0,
            [0, 1],
        )

        def check_velocities(_, qd):
            # velocity outside the plane of the pendulum should be close to zero
            check = abs(qd[0]) < 1e-4 and abs(qd[6]) < 1e-4
            # velocity in the plane of the pendulum should be reasonable
            check = check and abs(qd[1]) < 10.0 and abs(qd[2]) < 5.0 and abs(qd[3]) < 10.0 and abs(qd[4]) < 10.0
            return check

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pendulum links have reasonable velocities",
            check_velocities,
            [0, 1],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    newton.examples.run(Example(viewer, args), args)
