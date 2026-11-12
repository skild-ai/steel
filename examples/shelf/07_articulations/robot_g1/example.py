# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Robot G1
#
# Shows how to set up a simulation of a G1 robot articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python -m shelf run robot_g1 --world-count 16
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from common import presets


class Example:
    def __init__(self, viewer, args):
        self.preset = presets.load(__file__)
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = self.preset.name

        self.viewer = viewer

        g1 = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(g1)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        g1.default_shape_cfg.ke = 1.0e3
        g1.default_shape_cfg.kd = 2.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")

        g1.add_usd(
            str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            xform=wp.transform(wp.vec3(0, 0, 0.2)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        for i in range(6, g1.joint_dof_count):
            g1.joint_target_ke[i] = 500.0
            g1.joint_target_kd[i] = 10.0
            g1.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # approximate meshes for faster collision detection
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(g1, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 2.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.use_mujoco_contacts = self.solver_type == "mujoco" and args.use_mujoco_contacts
        self.build_solver()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
        if self.solver_type == "mujoco":
            self.solver = self.preset.make_solver(self.model, use_mujoco_contacts=self.use_mujoco_contacts)
        else:
            self.solver = self.preset.make_solver(self.model)
        if self.use_mujoco_contacts:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model, rigid_contact_max=self.solver.get_max_contact_count(), soft_contact_max=0
            )
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver_observables = self.solver.observables(
            contacts=self.contacts, kinds={self.solver.ObservableKind.CONTACT_F}
        )

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        if not self.use_mujoco_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
                observables=self.solver_observables if substep == self.sim_substeps - 1 else None,
            )

            # swap states
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                # an odd substep count ends in state_1; copy it back so the captured graph keeps its buffers
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0, observables=self.solver_observables)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        # fmt: off
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all body velocities are small",
            lambda q, qd: max(abs(qd))
            < 0.015,  # Relaxed from 0.005 - G1 has higher residual velocities with collision pipeline
        )
        # fmt: on

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
