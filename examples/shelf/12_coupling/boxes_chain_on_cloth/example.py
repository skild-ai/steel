# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Rigid-XPBD Coupled Solver
#
# Rigid boxes and an articulated pendulum chain (driven by MuJoCo or Kamino)
# interact with a cloth sheet (simulated by XPBD). Contact forces from the
# deformable side are fed back to the rigid bodies through SolverCoupledProxy's
# proxy-body coupling path with momentum-based force harvesting.
#
# Pass ``--solver xpbd`` to run the same scene with a single XPBD solver
# (no coupling) as a reference baseline.
#
# Command: python -m shelf run boxes_chain_on_cloth
#          python -m shelf run boxes_chain_on_cloth --solver xpbd
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverKamino, SolverMuJoCo, SolverSteel
from common import presets


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
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        # Presets: "<rigid>_xpbd" couples a rigid solver with XPBD; "xpbd" is the pure-XPBD baseline.
        self.use_coupled = self.preset.name != "xpbd"

        self.rigid_solver = self.preset.name.split("_")[0]

        builder = newton.ModelBuilder()
        _register_rigid_solver_custom_attributes(builder, self.rigid_solver)
        builder.add_ground_plane()

        # ---- Rigid bodies (free-floating + articulated) ----
        rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)
        self._emit_articulated_chain(builder)
        rigid_body_end = builder.body_count

        # ---- Cloth ----
        self._emit_cloth(builder)

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0

        self.rigid_body_start = rigid_body_start
        self.rigid_body_end = rigid_body_end
        self.build_solver()

        # Simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
        rigid_body_start = self.rigid_body_start
        rigid_body_end = self.rigid_body_end
        if self.use_coupled:
            # ---------- Coupled path: rigid solver + XPBD ----------
            coupling = self.preset.solver
            xpbd_entry = self.preset.entry("xpbd")
            rigid_name, rigid_solver, rigid_kwargs = _rigid_solver_entry_args(
                self.rigid_solver, self.preset.entry(self.rigid_solver).solver
            )
            rigid_body_indices = wp.array(list(range(rigid_body_start, rigid_body_end)), dtype=int)
            xpbd_body_indices = wp.array(
                [i for i in range(self.model.body_count) if i < rigid_body_start or i >= rigid_body_end],
                dtype=int,
            )

            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name=rigid_name,
                        solver=lambda v: rigid_solver(model=v, **rigid_kwargs),
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        joints=list(range(self.model.joint_count)),
                    ),
                    SolverCoupledProxy.Entry(
                        name="xpbd",
                        solver=lambda v: xpbd_entry.make_solver(v),
                        bodies=[int(i) for i in xpbd_body_indices.numpy()],
                        particles=list(range(self.model.particle_count)),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source=rigid_name,
                            destination="xpbd",
                            bodies=[int(i) for i in rigid_body_indices.numpy()],
                            mass_scale=coupling.mass_scale,
                        )
                    ],
                    iterations=coupling.proxy_iterations,
                ),
            )
        else:
            # ---------- Pure-XPBD path (reference baseline) ----------
            self.solver = self.preset.make_solver(self.model)

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate)

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                # an odd count under graph capture: copy the last result into state_0 instead of swapping,
                # so every replay of the captured graph starts from the newest state
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all rigid bodies are above the ground",
            lambda q, qd: q[2] > -0.1,
        )
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 20.0, f"Bounding box exploded: size={bbox_size:.2f}"
        assert min_pos[2] > -0.5, f"Excessive penetration: z_min={min_pos[2]:.4f}"

    def test_post_step(self):
        # Falling from the highest start (z = 3.5 m) gives 8.3 m/s; a faster body was launched.
        speed = np.linalg.norm(self.state_0.body_qd.numpy()[:, :3], axis=1)
        assert speed.max() < 10.0, f"body {speed.argmax()} moves at {speed.max():.1f} m/s at t={self.sim_time:.2f} s"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        """Add a few rigid boxes above the cloth."""
        saved_density = builder.default_shape_cfg.density
        builder.default_shape_cfg.density = 100.0  # light boxes
        boxes = [
            (wp.vec3(0.0, 0.0, 2.0), (0.15, 0.15, 0.15)),
            (wp.vec3(0.3, 0.1, 2.5), (0.10, 0.20, 0.10)),
            (wp.vec3(-0.2, -0.1, 3.0), (0.12, 0.12, 0.12)),
        ]
        for pos, (hx, hy, hz) in boxes:
            body = builder.add_body(
                xform=wp.transform(p=pos, q=wp.quat_identity()),
            )
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
        builder.default_shape_cfg.density = saved_density

    def _emit_cloth(self, builder: newton.ModelBuilder):
        """Add a cloth sheet at z=1.0, fixed at left and right edges."""
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=30,
            dim_y=30,
            cell_x=1.0 / 30.0,
            cell_y=1.0 / 30.0,
            mass=0.1,
            add_springs=True,
            spring_ke=1.0e2,
            spring_kd=1.0,
            tri_ke=1.0e2,
            tri_ka=1.0e2,
            tri_kd=1.0e-1,
            edge_ke=1.0e2,
            edge_kd=1.0e-1,
            particle_radius=0.01,
        )

    def _emit_articulated_chain(self, builder: newton.ModelBuilder):
        """Add a free-falling 8-link chain."""
        n_links = 8
        hx, hy, hz = 0.06, 0.03, 0.03
        damping = 0.5
        start = wp.vec3(-0.3, 0.0, 3.5)

        saved_density = builder.default_shape_cfg.density
        builder.default_shape_cfg.density = 100.0

        prev_link = builder.add_link(
            xform=wp.transform(p=start, q=wp.quat_identity()),
        )
        builder.add_shape_box(prev_link, hx=hx, hy=hy, hz=hz)
        joints = [builder.add_joint_free(child=prev_link)]

        for _ in range(n_links - 1):
            link = builder.add_link()
            builder.add_shape_box(link, hx=hx, hy=hy, hz=hz)
            joints.append(
                builder.add_joint_revolute(
                    parent=prev_link,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    target_kd=damping,
                    parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                )
            )
            prev_link = link

        builder.add_articulation(joints, label="chain")
        builder.default_shape_cfg.density = saved_density

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
