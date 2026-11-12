# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example for simulating all basic models as a single heterogeneous multi-world model with SolverKamino.
#
# Command: python -m shelf run loop_mixed_worlds
#
###########################################################################

import argparse

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
        self.viewer = viewer
        self.device = wp.get_device()

        # Define a helper function to load each basic model from USD and
        # add it to the builder, with consistent settings for all models
        def load_basic_asset_from_usd(asset_file: str) -> newton.ModelBuilder:
            asset_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            if self.preset.name == "kamino":
                newton.solvers.SolverKamino.register_custom_attributes(asset_builder)
            asset_builder.default_shape_cfg.margin = 0.0
            asset_builder.default_shape_cfg.gap = 0.0
            asset_builder.add_usd(
                asset_file,
                joint_ordering=None,
                force_show_colliders=True,
                force_position_velocity_actuation=True,
                enable_self_collisions=False,
                hide_collision_shapes=False,
            )
            return asset_builder

        # Load the heterogeneous basic models either from USD or manually using the
        # model builder API, depending on the command-line argument `--from-usd`
        builder = newton.ModelBuilder()
        if self.preset.name != "kamino":
            # Reduced-coordinate solvers (Steel) need every mechanism as an articulation. The USD four-bar has
            # none, so they build the same six worlds with the builder API and the four-bar on a floating base.
            # As in the USD assets, only the four-bar's first hinge is position/velocity driven.
            builder.add_world(
                basics.build_boxes_fourbar(ground=True, floatingbase=True, implicit_pd=True, new_world=True)
            )
            builder.add_world(basics.build_boxes_nunchaku(ground=True, new_world=True))
            builder.add_world(basics.build_boxes_hinged(ground=True, new_world=True))
            builder.add_world(basics.build_box_pendulum(ground=True, new_world=True))
            builder.add_world(basics.build_box_on_plane(ground=True, new_world=True))
            builder.add_world(basics.build_cartpole(z_offset=0.5, ground=True, new_world=True))
        elif args is not None and args.from_usd:
            # Load all basic USD assets and add them to the builder
            asset_names = [
                "boxes_fourbar",
                "boxes_nunchaku",
                "boxes_hinged",
                "box_pendulum",
                "box_on_plane",
                "cartpole",
            ]
            for asset_name in asset_names:
                asset_file = get_kamino_basics_asset(f"{asset_name}.usda")
                builder.add_world(builder=load_basic_asset_from_usd(asset_file))
        else:
            # Manually build the heterogeneous basic models using the builder API
            basics.make_basics_heterogeneous_builder(builder=builder, ground=True)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # Create the Kamino solver for the given model
        self.build_solver()

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets(spacing=(5.0, 5.0, 0.0))

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(0.0, -15.0, 1.6)
            pitch = -1.5
            yaw = 92.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

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
        parser.add_argument(
            "--from-usd",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load the heterogeneous basic models from USD (otherwise build them manually).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
