# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Robot H1
#
# Shows how to set up a simulation of a H1 articulation
# from a USD file using newton.ModelBuilder.add_usd().
#
# Command: python -m shelf run robot_h1 --world-count 16
#          python -m shelf run robot_h1 --solver kamino
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, JointType
from common import presets


class Example:
    def __init__(self, viewer, args):
        self.preset = presets.load(__file__)
        newton.use_coord_layout_targets = True
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = self.preset.name

        self.viewer = viewer

        self.device = wp.get_device()

        h1 = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(h1)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(h1)
        h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        h1.default_shape_cfg.ke = 2.0e3
        h1.default_shape_cfg.kd = 1.0e2
        h1.default_shape_cfg.kf = 1.0e3
        h1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_h1")
        asset_file = str(asset_path / "usd_structured" / "h1.usda")
        h1.add_usd(
            asset_file,
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        # approximate meshes for faster collision detection
        h1.approximate_meshes("bounding_box")

        for joint_id, joint_type in enumerate(h1.joint_type):
            if joint_type != JointType.REVOLUTE:
                continue
            dof_start = h1.joint_qd_start[joint_id]
            dof_end = h1.joint_qd_start[joint_id + 1] if joint_id + 1 < h1.joint_count else h1.joint_dof_count
            for dof_id in range(dof_start, dof_end):
                h1.joint_target_ke[dof_id] = self.preset.model.need("joint_target_ke")
                h1.joint_target_kd[dof_id] = self.preset.model.need("joint_target_kd")
                h1.joint_target_mode[dof_id] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder()
        builder.replicate(h1, self.world_count)

        self.root_dof_indices = []
        for root_joint_id in builder.articulation_start:
            root_dof_start = builder.joint_qd_start[root_joint_id]
            root_dof_end = builder.joint_qd_start[root_joint_id + 1]
            self.root_dof_indices.extend(range(root_dof_start, root_dof_end))

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
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
        self.viewer.set_world_offsets((3.0, 3.0, 0.0))

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
            self.contacts = self.collision_pipeline.contacts()
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
        """Verify the robot settles without actuating its floating base."""
        target_mode = self.model.joint_target_mode.numpy()
        target_ke = self.model.joint_target_ke.numpy()
        target_kd = self.model.joint_target_kd.numpy()
        if any(
            target_mode[dof_id] != int(JointTargetMode.NONE) or target_ke[dof_id] != 0.0 or target_kd[dof_id] != 0.0
            for dof_id in self.root_dof_indices
        ):
            raise AssertionError("floating-base DOFs must remain unactuated")

        velocity_limit = 0.015 if self.solver_type == "kamino" else 5e-3
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all body velocities are small",
            lambda q, qd: max(abs(qd)) < velocity_limit,
        )

    def test_post_step(self):
        """Verify Kamino remains settled after the initial eight seconds."""
        if self.solver_type == "kamino" and self.sim_time >= 8.0:
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities remain small after settling",
                lambda q, qd: max(abs(qd)) < 0.015,
            )

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
