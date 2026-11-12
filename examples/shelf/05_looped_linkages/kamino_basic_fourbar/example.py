# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example for basic four-bar mechanism
#
# Shows how to simulate a basic four-bar linkage with multiple worlds using SolverKamino.
#
# Command: python -m shelf run kamino_basic_fourbar --world-count 16
#
###########################################################################

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets
from newton.tests import get_kamino_basics_asset
from newton.tests.utils import basics


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
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.preset.name == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0
        # Load the basic four-bar mechanism either from USD or by manually building it
        # with the builder API, depending on the command-line argument `--from-usd`
        if self.preset.name != "kamino":
            # Reduced-coordinate solvers (Steel) need the linkage as an articulation: a free joint on link 1,
            # the revolute tree, and the hinge that closes the loop. The USD asset carries no articulation,
            # so they build the same mechanism with the builder API and a floating base, matching the asset's
            # friction (1.0) and its hinge modes: 1 position-velocity driven, 3 effort, 2 and 4 passive.
            basics.build_boxes_fourbar(
                builder=robot_builder,
                floatingbase=True,
                use_custom_shape_cfg=True,
                friction=1.0,
            )
            hinge_modes = [
                newton.JointTargetMode.POSITION_VELOCITY,
                newton.JointTargetMode.NONE,
                newton.JointTargetMode.EFFORT,
                newton.JointTargetMode.NONE,
            ]
            hinges = [j for j, kind in enumerate(robot_builder.joint_type) if kind == newton.JointType.REVOLUTE]
            for j, mode in zip(hinges, hinge_modes, strict=True):
                robot_builder.joint_target_mode[robot_builder.joint_qd_start[j]] = int(mode)
        elif args is not None and args.from_usd:
            # Load the basic four-bar USD and add it to the builder
            asset_file = get_kamino_basics_asset("boxes_fourbar.usda")
            robot_builder.add_usd(
                asset_file,
                force_show_colliders=True,
                force_position_velocity_actuation=True,
                enable_self_collisions=False,
                hide_collision_shapes=False,
            )
        else:
            # Manually build the basic four-bar mechanism using the builder API
            basics.build_boxes_fourbar(builder=robot_builder)

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # Set ad-hoc softer joint PD gains to make the
        # four-bar more "squishy", i.e. more compliant
        self.model.joint_target_ke.fill_(1.0)
        self.model.joint_target_kd.fill_(0.001)
        if self.preset.name != "kamino":
            # Steel drives every DOF with nonzero gains whatever its target mode; keep the gains only where
            # Kamino applies them (position or velocity drives), not on passive, effort or free-base DOFs
            driven = np.isin(
                self.model.joint_target_mode.numpy(),
                [
                    int(newton.JointTargetMode.POSITION),
                    int(newton.JointTargetMode.VELOCITY),
                    int(newton.JointTargetMode.POSITION_VELOCITY),
                ],
            )
            self.model.joint_target_ke.assign(np.where(driven, self.model.joint_target_ke.numpy(), 0.0))
            self.model.joint_target_kd.assign(np.where(driven, self.model.joint_target_kd.numpy(), 0.0))

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
        q_base = wp.transformf((0.0, 0.0, 0.1), q_b)
        q_base = np.array(q_base)
        q_base = np.tile(q_base, (self.world_count, 1))
        for w in range(self.world_count):
            q_base[w, :3] += np.array([0.0, 0.0, 0.2]) * float(w)
        self.base_q.assign(q_base)
        if self.preset.name == "kamino":
            reset_config = newton.solvers.SolverKamino.ResetConfig(
                base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self.base_q),
            )
            self.solver.reset(state=self.state_0, config=reset_config)
        else:
            self._place_bases(q_base)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(-0.5, -1.0, 0.2)
            pitch = -5.0
            yaw = 70.0
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
            # Create and configure settings for SolverKamino and the collision detector
            solver_config = self.preset.make_config(self.model)
            solver_config.collision_detector.max_contacts = 32 * self.model.world_count
            self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)
        else:
            self.solver = self.preset.make_solver(self.model)

        # Create the contacts data container and the solver observables
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
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
            if self.preset.name != "kamino":
                # Kamino runs its own collision detector; the other solvers take Newton's pipeline contacts
                self.collision_pipeline.collide(self.state_0, self.contacts)
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
        # Since rendering is called after stepping the simulation, the previous and next
        # states correspond to self.state_1 and self.state_0 due to the reference swaps,
        # so contacts are rendered with self.state_1 to match the body positions at the
        # time of contact generation.
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1, observables=self.solver_observables)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--from-usd",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load the basic four-bar mechanism from USD (otherwise build it manually).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
