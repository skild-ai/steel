# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Rigid-MPM Coupled Solver
#
# Rigid boxes driven by MuJoCo or Kamino fall into a granular MPM bed.  The rigid bodies
# are exposed to MPM as proxy colliders in the same shared model, and MPM
# grid impulses are harvested back into rigid-body wrenches.
#
# Command: python -m shelf run mujoco_mpm_coupled_solver
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverSteel
from common import presets

# A grain that leaves the fixed MPM grid gets runaway velocities. The cap clips MPM advection velocity (not collider
# impulses), and the padding is a conservative single-flight estimate: a grain launched at the cap from the bed or the
# settled stack top lands within |x| <= 7.2 m and peaks below z = 5.3 m, inside |x|, |y| <= 7.3 m and z <= 6.8 m.
_GRID_PADDING = 125
_PARTICLE_MAX_VELOCITY = 6.5


def _register_rigid_solver_custom_attributes(builder: newton.ModelBuilder, rigid_solver: str) -> None:
    if rigid_solver == "kamino":
        SolverKamino.register_custom_attributes(builder)


def _make_kamino_config(settings: dict[str, object]) -> SolverKamino.Config:
    # Start from SolverKamino.Config() (not Config.from_model) and apply the preset's
    # settings; nested tables set fields of the nested configs.
    config = SolverKamino.Config()
    for key, value in settings.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(getattr(config, key), sub_key, sub_value)
        else:
            setattr(config, key, value)
    return config


def _rigid_solver_entry_args(rigid_solver: str, settings: dict[str, object]):
    if rigid_solver == "kamino":
        return "kamino", SolverKamino, {"config": _make_kamino_config(settings)}
    if rigid_solver == "mujoco":
        return "mjc", SolverMuJoCo, dict(settings)
    if rigid_solver == "steel":
        return "steel", SolverSteel, dict(settings)
    raise ValueError(f"Unsupported rigid solver '{rigid_solver}'")


def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool = True):
    if not enabled:
        return None

    with wp.ScopedDevice(model.device):
        with wp.ScopedCapture() as capture:
            simulate()

    if capture.graph is None:
        raise RuntimeError(f"Graph capture failed on device {model.device}")
    return capture.graph


def _launch_frame_graph(model: newton.Model, graph) -> bool:
    if graph is None:
        return False

    with wp.ScopedDevice(model.device):
        wp.capture_launch(graph)
    return True


class Example:
    def __init__(self, viewer, args):
        self.preset = presets.load(__file__)
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        # Presets are named "<rigid>_implicit_mpm".
        self.rigid_solver = self.preset.name.split("_")[0]

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        _register_rigid_solver_custom_attributes(builder, self.rigid_solver)
        SolverImplicitMPM.register_custom_attributes(builder)

        self.rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)
        self.rigid_body_end = builder.body_count
        builder.add_ground_plane()

        self.voxel_size = 0.05
        self._emit_particles(builder, self.voxel_size)

        builder.particle_max_velocity = _PARTICLE_MAX_VELOCITY
        self.model = builder.finalize()

        self.build_solver()

        self.state_0 = self.model.state()
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = False

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.particle_render_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.graph = None
        self.capture()

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
        coupling = self.preset.solver
        rigid_entry = self.preset.entry(self.rigid_solver)
        mpm_config = self.preset.entry("implicit_mpm").make_config(self.model)
        mpm_config.voxel_size = self.voxel_size
        mpm_config.grid_padding = _GRID_PADDING

        rigid_name, rigid_solver, rigid_kwargs = _rigid_solver_entry_args(self.rigid_solver, rigid_entry.solver)
        rigid_body_indices = wp.array(
            list(range(self.rigid_body_start, self.rigid_body_end)),
            dtype=int,
            device=self.model.device,
        )
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name=rigid_name,
                    solver=lambda v: rigid_solver(model=v, **rigid_kwargs),
                    bodies=[int(i) for i in rigid_body_indices.numpy()],
                    joints=list(range(self.model.joint_count)),
                    substeps=rigid_entry.substeps,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                    particles=list(range(self.model.particle_count)),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source=rigid_name,
                        destination="mpm",
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        mass_scale=coupling.mass_scale,
                        mode=coupling.coupling_mode,
                        # MPM handles collider contact internally; no proxy
                        # collision pipeline should generate Contacts here.
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=coupling.proxy_iterations,
            ),
        )
        self.mpm_solver = self.solver.solver("mpm")

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate)

    def simulate(self):
        self.state_0.clear_forces()
        newton.examples.apply_coupled_viewer_forces(self, self.state_0)
        self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the sand",
            lambda q, qd: q[2] > 0.45,
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def render(self):
        render_state = newton.examples.get_coupled_view_state(self)
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)

        if render_state.particle_q is not None:
            self.viewer.log_points(
                "/sand",
                points=render_state.particle_q,
                radii=self.model.particle_radius,
                colors=self.particle_render_colors,
                hidden=not self.viewer.show_particles,
            )

        if self.show_impulses and render_state.particle_q is not None:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(render_state)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        drop_z = 2.0
        offsets_xy = [
            (0.00, 0.00),
            (0.10, 0.00),
            (-0.10, 0.00),
            (0.00, 0.10),
            (0.00, -0.10),
            (0.10, 0.10),
            (-0.10, 0.10),
            (0.10, -0.10),
            (-0.10, -0.10),
            (0.15, 0.00),
            (-0.15, 0.00),
            (0.00, 0.15),
        ]
        boxes = [
            (0.25, 0.35, 0.25),
            (0.25, 0.25, 0.25),
            (0.3, 0.2, 0.2),
            (0.25, 0.35, 0.25),
            (0.25, 0.25, 0.25),
            (0.3, 0.2, 0.2),
        ]

        for box_index, (hx, hy, hz) in enumerate(boxes):
            ox, oy = offsets_xy[box_index % len(offsets_xy)]
            pz = drop_z + float(box_index) * 0.6
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(float(ox), float(oy), pz), q=wp.quat_identity()),
                mass=75.0,
            )
            builder.add_shape_box(body, hx=float(hx), hy=float(hy), hz=float(hz))

    def _emit_particles(self, builder: newton.ModelBuilder, voxel_size: float):
        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([-1.0, -1.0, 0.0])
        bed_hi = np.array([1.0, 1.0, 0.5])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=bed_res[0] + 1,
            dim_y=bed_res[1] + 1,
            dim_z=bed_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.75},
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
