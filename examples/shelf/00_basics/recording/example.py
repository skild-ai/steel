# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Recording
#
# Shows how to record a simulation using ViewerFile for automatic recording
# of model structure and state data during simulation.
#
# Recording happens automatically - ViewerFile captures all logged states
# and saves them when the viewer is closed.
#
# Command: python -m shelf run recording
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets


class Example:
    def __init__(self, viewer, args):
        self.preset = presets.load(__file__)
        # Setup simulation parameters
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.world_count = args.world_count

        # Set numpy random seed for reproducibility
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        start_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*self.rng.uniform(-1.0, 1.0, size=3))), -wp.pi * 0.5)

        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.gap = 0.0
        articulation_builder.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )

        # Joint initial positions
        articulation_builder.joint_q[:7] = [0.0, 0.0, 1.5, *start_rot]

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.0
        for _i in range(self.world_count):
            articulation_builder.joint_q[7:] = self.rng.uniform(
                -1.0, 1.0, size=(len(articulation_builder.joint_q) - 7,)
            ).tolist()
            builder.add_world(articulation_builder)
        builder.add_ground_plane()

        # Finalize model
        self.model = builder.finalize()
        self.control = self.model.control()

        self.solver = self.preset.make_solver(self.model)
        # Steel takes its contacts from Newton's collision pipeline; MuJoCo and Kamino find their own
        self.collision_pipeline = newton.CollisionPipeline(self.model) if self.preset.name == "steel" else None
        self.contacts = self.collision_pipeline.contacts() if self.collision_pipeline else None

        self.state_0, self.state_1 = self.model.state(), self.model.state()

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

        # Set model in viewer (ViewerFile will automatically record it)
        self.viewer.set_model(self.model)

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                # an odd count under graph capture: copy the last result into state_0 instead of swapping,
                # so every replay of the captured graph starts from the newest state
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=False):
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)  # ViewerFile automatically records this
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body origins remain above the ground",
            lambda q, qd: q[2] > -0.1,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("recording_file", nargs="?", default="humanoid_recording.bin")
        parser.set_defaults(num_frames=1000, world_count=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    test_args, _ = parser.parse_known_args()

    if test_args.test:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
        raise SystemExit(0)

    args = parser.parse_args()
    print(f"Recording simulation to: {args.recording_file}")
    print("ViewerFile will automatically save when the simulation ends.")

    # Create ViewerFile with auto_save=False to only save at the end
    viewer = newton.viewer.ViewerFile(args.recording_file, auto_save=False)

    # Create example
    example = Example(viewer, args)

    # Run for a reasonable number of frames
    frame_count = 0

    while frame_count < args.num_frames:
        example.step()
        example.render()
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Frame {frame_count}/{args.num_frames}")

    # Close viewer (automatically saves recording)
    viewer.close()
    print(f"Recording completed: {args.recording_file}")
