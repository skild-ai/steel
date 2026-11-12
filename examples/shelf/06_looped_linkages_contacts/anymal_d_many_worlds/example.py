# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Robot Anymal D
#
# Shows how to simulate Anymal D with multiple worlds using SolverKamino.
#
# Command: python -m shelf run anymal_d_many_worlds --world-count 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.preset = presets.load(__file__)
        # Set simulation run-time configurations
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        # Kamino's own collision detector is a Kamino option; every other solver uses Newton's pipeline
        self.use_kamino_contacts = (args.use_kamino_contacts if args else False) and self.preset.name == "kamino"
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.preset.name == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0
        # Load the Anymal D USD and add it to the builder
        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        robot_builder.add_usd(
            asset_file,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize()

        # Create the Kamino solver for the given model
        self.build_solver()

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 1.0), q_b)
        self.base_q.assign([q_base] * self.world_count)
        if self.preset.name == "kamino":
            reset_config = newton.solvers.SolverKamino.ResetConfig(
                base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self.base_q),
            )
            self.solver.reset(state=self.state_0, config=reset_config)
        else:
            self._place_bases(self.base_q.numpy())

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(5.0, 0.0, 2.0)
            pitch = -15.0
            yaw = -180.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def _place_bases(self, q_base):
        """Put each world's floating base at its pose, as Kamino's reset does, through the free joints."""
        joint_q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        free = np.flatnonzero(self.model.joint_type.numpy() == int(newton.JointType.FREE))
        for w, j in enumerate(free):
            joint_q[q_start[j] : q_start[j] + 7] = q_base[w]
        self.model.joint_q.assign(joint_q)
        # reduced-coordinate solvers read the state's joint coordinates, so set those as well as the body poses
        self.state_0.joint_q.assign(joint_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
        if self.preset.name == "kamino":
            self.config = self.preset.make_config(self.model)
            self.config.use_collision_detector = self.use_kamino_contacts
            self.solver = newton.solvers.SolverKamino(self.model, config=self.config)
        else:
            self.solver = self.preset.make_solver(self.model)

        # Configure CD components based on whether we want to use Kamino's
        # internal contact solver or Newton's collision pipeline
        if not self.use_kamino_contacts:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.collision_pipeline = None
            self.contacts = newton.CollisionPipeline(self.model).contacts()
        self.solver_observables = self.solver.observables(
            contacts=self.contacts, kinds={self.solver.ObservableKind.CONTACT_F}
        )

    def capture(self):
        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if not self.use_kamino_contacts:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    self.contacts,
                    self.sim_dt,
                    observables=self.solver_observables,
                )
            else:
                self.solver.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    self.contacts,
                    self.sim_dt,
                    observables=self.solver_observables,
                )
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Contact points use the last substep's input body frames (before the swap).
        self.viewer.log_contacts(self.contacts, self.state_1, observables=self.solver_observables)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        # Only check velocities on CUDA, where example tests run enough frames to settle.
        # Short CPU smoke runs may still be falling when they finish.
        if self.device.is_cuda:
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: (
                    max(abs(qd)) < 0.25
                ),  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
