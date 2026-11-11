# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Robot DR Legs
#
# Shows how to simulate DR Legs with multiple worlds using SolverKamino.
#
# Command: python -m shelf run kamino_robot_dr_legs --world-count 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets


def _articulated_stage(asset_file: str):
    """The DR Legs stage as one articulation, for reduced-coordinate solvers (Steel).

    The asset has no articulation root: Kamino takes its 36 hinges as loose maximal-coordinate joints. Here the
    stage is changed in memory only: an articulation root on the robot, and one parent joint kept per body (the
    driven one where a body has two) to form the tree from the pelvis; the other hinges are marked as loop
    closures (excluded from the articulation). Steel ignores the drive of a loop-closing joint, so every servo
    must stay in the tree.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(asset_file)
    joints = [UsdPhysics.Joint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    parent = [str(j.GetBody0Rel().GetTargets()[0]) for j in joints]
    child = [str(j.GetBody1Rel().GetTargets()[0]) for j in joints]
    driven = [any("DriveAPI" in s for s in j.GetPrim().GetAppliedSchemas()) for j in joints]
    tree = {}  # child body -> index of the joint kept as its tree parent
    for i in sorted(range(len(joints)), key=lambda i: not driven[i]):
        tree.setdefault(child[i], i)
    # the kept joints must reach every body from the root without a cycle
    roots = set(parent) - set(child)
    for body in tree:
        seen = set()
        while body not in roots:
            if body in seen:
                raise ValueError(f"DR Legs: the kept joints form a cycle at {body}")
            seen.add(body)
            body = parent[tree[body]]
    kept = set(tree.values())
    for i, joint in enumerate(joints):
        if i not in kept:
            if driven[i]:
                raise ValueError(f"DR Legs: driven joint {joint.GetPrim().GetPath()} would become a loop closure")
            joint.CreateExcludeFromArticulationAttr(True)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
    return stage


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.preset = presets.load(__file__)
        # Set simulation run-time configurations
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.kamino = self.preset.kind == "kamino"  # kamino and its kamino_dvi variant
        # Kamino's own collision detector is a Kamino option; every other solver uses Newton's pipeline
        self.use_kamino_contacts = (args.use_kamino_contacts if args else False) and self.kamino
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        # DVI benefits from early contact detection because it solves inequality
        # constraints slightly less accurately than PADMM. Contact forces remain
        # zero until the shapes overlap. The margin comes from the preset's model table.
        dvi_contact_margin = self.preset.model.need("shape_margin")
        self.viewer = viewer
        self.device = wp.get_device()
        self.animated = getattr(args, "animated", False) if args else False
        self.time = wp.zeros((self.world_count,), device=self.device)

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.kamino:
            newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = dvi_contact_margin
        robot_builder.default_shape_cfg.gap = 1e-2
        if self.animated:
            robot_builder.default_shape_cfg.mu = 0.1
        # Load the DR Legs USD and add it to the builder.
        # Pinned to an older revision: the current one is better for RL but tips over
        # differently in simulation, which the DVI contact regressions detect. Drop the
        # ref once that is understood.
        asset_path = newton.utils.download_asset("disneyresearch", ref="261cd1f429619d8ef4f546bd788ab9dea906b5e1")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")
        robot_builder.add_usd(
            asset_file if self.kamino else _articulated_stage(asset_file),
            # Kamino keeps the file's joint order; reduced-coordinate solvers (Steel) need a parent-first tree
            joint_ordering=None if self.kamino else "dfs",
            floating=None if self.kamino else True,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        if self.animated:
            # Increase P-gain for animation
            robot_builder.joint_target_ke = [150.0 if ke > 0.0 else 0.0 for ke in robot_builder.joint_target_ke]
        else:
            # Set joint armature and viscous damping for better
            # stability of the implicit joint-space PD controller
            # (on the hinges only: the free pelvis joint that reduced-coordinate solvers add gets neither)
            hinge = [
                mode != newton.JointTargetMode.NONE or not self._free_dof(robot_builder, d)
                for d, mode in enumerate(robot_builder.joint_target_mode)
            ]
            robot_builder.joint_armature = [0.011 if h else 0.0 for h in hinge]
            robot_builder.joint_damping = [0.044 if h else 0.0 for h in hinge]
            robot_builder.joint_target_ke = [
                10.0 if mode != newton.JointTargetMode.NONE else 0.0 for mode in robot_builder.joint_target_mode
            ]
            robot_builder.joint_target_kd = [
                2.0 if mode != newton.JointTargetMode.NONE else 0.0 for mode in robot_builder.joint_target_mode
            ]

        effort_limit_override = getattr(args, "joint_effort_limit", None) if args else None
        if effort_limit_override is not None:
            robot_builder.joint_effort_limit = [effort_limit_override] * robot_builder.joint_dof_count

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.margin = dvi_contact_margin
        builder.default_shape_cfg.gap = 1e-2
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 72 * self.world_count

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
        q_base = wp.transformf((0.0, 0.0, 0.4), q_b)
        self.base_q.assign([q_base] * self.world_count)
        if self.kamino:
            reset_config = newton.solvers.SolverKamino.ResetConfig(
                base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(base_q=self.base_q),
            )
            self.solver.reset(state=self.state_0, config=reset_config)
            self.solver.reset(state=self.state_1, config=reset_config)
        else:
            self._place_bases(self.base_q.numpy())

        # Load animation
        if self.animated:
            animation_asset = str(asset_path / "dr_legs" / "animation" / "dr_legs_animation_100fps.npy")
            self._init_animation(asset_file, animation_asset)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.graph = None
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(1.34, 0.0, 0.25)
            pitch = -7.0
            yaw = -180.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    @staticmethod
    def _free_dof(builder, dof: int) -> bool:
        """Whether a builder DOF belongs to a free joint."""
        for j, kind in enumerate(builder.joint_type):
            if kind == newton.JointType.FREE and builder.joint_qd_start[j] <= dof < builder.joint_qd_start[j] + 6:
                return True
        return False

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
        if self.kamino:
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

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.animated:
                self._advance_time()
                self._update_animation()
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
        self.viewer.log_contacts(self.contacts, self.state_1, observables=self.solver_observables)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    def _init_animation(self, model_asset: str, animation_asset: str):
        import numpy as np  # noqa: PLC0415
        from pxr import Usd  # noqa: PLC0415

        # Get names of animated joints
        stage = Usd.Stage.Open(model_asset)
        animation_joint_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            for schema in prim.GetAppliedSchemas()
            if "PhysicsDriveAPI" in schema
        ]

        # Match joints from USD to model joints
        joint_label = list(self.model.joint_label)
        joint_q_start = self.model.joint_q_start.numpy()
        try:
            channel_coords = np.array(
                [joint_q_start[joint_label.index(path)] for path in animation_joint_paths],
                dtype=np.int32,
            )
        except ValueError as e:
            raise RuntimeError(f"Animation joint not found in model.joint_label: {e}") from e

        #
        world_coords_offsets = np.arange(self.model.world_count, dtype=np.int64) * (
            self.model.joint_coord_count // self.model.world_count
        )
        self.animation_data = wp.array(
            np.load(animation_asset, allow_pickle=True),
            dtype=wp.float32,
            device=self.device,
        )
        self.animation_indices = wp.array(
            channel_coords[None, :] + world_coords_offsets[:, None],
            dtype=wp.int32,
            device=self.device,
        )
        self.animation_dt = 0.01

    def _advance_time(self):
        """Advances the current simulation time by ``dt``."""

        @wp.kernel
        def advance_time_kernel(dt: wp.float32, time: wp.array[wp.float32]):
            """Advance the time in each world."""
            wid = wp.tid()
            time[wid] += dt

        wp.launch(
            advance_time_kernel,
            dim=self.model.world_count,
            inputs=[self.sim_dt, self.time],
            device=self.device,
        )

    def _update_animation(self):
        """Update the animation target for each world."""

        @wp.kernel
        def animation_target_update_kernel(
            animation_dt: wp.float32,
            animation_data: wp.array2d[wp.float32],
            animation_indices: wp.array2d[wp.int32],
            time: wp.array[wp.float32],
            joint_target_q: wp.array[wp.float32],
        ):
            # Retrieve the world and channel index from the thread ID
            wid, cid = wp.tid()
            t = time[wid]  # Current time
            # Compute animation index based on animation fps, clamp by animation length
            anim_id = wp.min(wp.int32(wp.floor(t / animation_dt)), animation_data.shape[0] - 1)
            # Update animation target
            joint_target_q[animation_indices[wid, cid]] = animation_data[anim_id, cid]

        wp.launch(
            animation_target_update_kernel,
            dim=(self.model.world_count, self.animation_data.shape[1]),
            inputs=[
                self.animation_dt,
                self.animation_data,
                self.animation_indices,
                self.time,
                self.control.joint_target_q,
            ],
            device=self.device,
        )

    @staticmethod
    def create_parser():
        import argparse  # noqa: PLC0415

        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.add_argument(
            "--animated",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Animation the model based on imported motion.",
        )
        parser.add_argument(
            "--joint-effort-limit",
            type=float,
            default=None,
            help="Override effort limit for all joint DOFs (default: use USD values).",
        )
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
