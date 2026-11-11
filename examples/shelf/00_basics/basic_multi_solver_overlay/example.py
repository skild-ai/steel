# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Basic Multi-Solver Overlay
#
# Demonstrates the viewer "layer" system by running several simulations of
# the same quadruple-pendulum chain in a single viewer window, each driven
# by a different solver (XPBD, Featherstone, MuJoCo Warp, Kamino DVI, and Steel
# when the preset has a Steel entry). By default the
# layers overlay exactly so the per-solver divergence is obvious; bumping
# the ``spacing`` constant lays them out side-by-side along the world
# X-axis via ``viewer.set_layer_transform``. Layers can also be toggled
# independently via the "Layers" group in the viewer sidebar.
#
# Layers are created via ``viewer.activate(layer_id)``. Each layer owns its
# own model, state, and shape batches; the viewer accumulates the render
# objects from all layers and draws them together every frame. Each layer
# also captures its own CUDA graph so per-solver stepping stays fast.
# The per-layer transform applied via ``set_layer_transform`` is a pure
# render-time displacement: it shifts every drawn object in the layer
# without changing the underlying physics state, so passing
# ``wp.transform_identity()`` (or a zero-translation transform) makes the
# layers overlay exactly.
#
# Command: python -m shelf run basic_multi_solver_overlay
#
###########################################################################

import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets


def _build_pendulum_model(num_links: int = 4, register_solver_attributes=None) -> tuple[newton.Model, list[int]]:
    """Build an ``num_links``-link pendulum chain hanging from the world.

    Returns ``(model, link_shape_indices)``. ``link_shape_indices`` lists
    the shape indices for the pendulum links only (excluding the ground
    plane) so the caller can tint just the moving geometry per layer
    while keeping the shared ground plane neutral.
    """
    builder = newton.ModelBuilder()
    if register_solver_attributes is not None:
        register_solver_attributes(builder)

    hx = 1.0
    hy = 0.1
    hz = 0.1

    # Create one rigid link per chain segment with a single box shape each.
    links: list[int] = []
    link_shapes: list[int] = []
    for _ in range(num_links):
        link = builder.add_link()
        links.append(link)
        link_shapes.append(builder.add_shape_box(link, hx=hx, hy=hy, hz=hz))

    # Anchor joint: rotate around the world's Z-axis so the chain hangs
    # sideways from the viewer's default camera.
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
    joints: list[int] = []
    joints.append(
        builder.add_joint_revolute(
            parent=-1,
            child=links[0],
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
    )
    # Subsequent joints connect each link's +X end to the next link's -X end.
    for i in range(1, num_links):
        joints.append(
            builder.add_joint_revolute(
                parent=links[i - 1],
                child=links[i],
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )

    builder.add_articulation(joints, label="pendulum")
    builder.add_ground_plane(height=-3.0)
    return builder.finalize(), link_shapes


class _SolverLayer:
    """Bundle of (layer_id, model, solver, states, contacts) tracked together.

    Each layer captures its own CUDA graph (when running on a CUDA device)
    so the per-frame ``simulate`` step is a single graph replay per
    solver. Solvers that don't support capture (some MuJoCo configurations)
    transparently fall back to eager execution while the others keep their
    graphs.
    """

    def __init__(
        self,
        layer_id: str,
        color: tuple[float, float, float],
        entry: str,
        viewer,
        sim_substeps: int,
        sim_dt: float,
        register_solver_attributes=None,
        native_contacts: bool = False,
    ):
        self.layer_id = layer_id
        self.color = color
        self.entry = entry
        self.model, link_shape_indices = _build_pendulum_model(register_solver_attributes=register_solver_attributes)
        # Tint only the pendulum link shapes so each layer is visually
        # distinct, leaving the ground plane its default checkerboard.
        if self.model.shape_color is not None:
            color_vec = wp.vec3(*color)
            for s in link_shape_indices:
                self.model.shape_color[s : s + 1].fill_(color_vec)
        self.native_contacts = native_contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._viewer = viewer
        self._sim_substeps = sim_substeps
        self._sim_dt = sim_dt
        self.graph: wp.Graph | None = None

    def build_solver(self, solver_factory) -> None:
        """Build this layer's solver and everything bound to its instance."""
        self.solver = solver_factory(self.model)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver_observables = (
            self.solver.observables(contacts=self.contacts, kinds={self.solver.ObservableKind.CONTACT_F})
            if self.native_contacts
            else None
        )

    def _simulate(self) -> None:
        self._viewer.activate(self.layer_id)
        for substep in range(self._sim_substeps):
            self.state_0.clear_forces()
            self._viewer.apply_forces(self.state_0)
            if not self.native_contacts:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self._sim_dt,
                observables=self.solver_observables if substep == self._sim_substeps - 1 else None,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self) -> wp.Graph | None:
        if not wp.get_device().is_cuda:
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            return capture.graph
        except Exception as exc:
            warnings.warn(
                f"CUDA graph capture failed for layer '{self.layer_id}': {exc}",
                stacklevel=2,
            )
            return None

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()


class Example:
    def __init__(self, viewer, args):
        self.preset = presets.load(__file__)
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # The colors make it easy to tell the solvers apart when toggling layers in the
        # "Layers" group of the viewer sidebar. Each layer captures its own
        # CUDA graph after binding its model to the viewer (where supported).
        specs = [
            ("XPBD", (0.95, 0.45, 0.10), "xpbd", None, False),
            ("Featherstone", (0.20, 0.70, 0.95), "featherstone", None, False),
            ("MuJoCo Warp", (0.50, 0.85, 0.30), "mujoco", None, False),
            (
                "Kamino DVI",
                (0.75, 0.35, 0.90),
                "kamino",
                newton.solvers.SolverKamino.register_custom_attributes,
                True,
            ),
            ("Steel", (0.90, 0.20, 0.25), "steel", None, False),
        ]
        self.layers: list[_SolverLayer] = [
            _SolverLayer(
                name,
                color,
                entry,
                self.viewer,
                self.sim_substeps,
                self.sim_dt,
                register_attributes,
                native_contacts,
            )
            for name, color, entry, register_attributes, native_contacts in specs
            if entry in self.preset.entries
        ]
        self.build_solver()

        # Bind each layer to the viewer. Activating then calling ``set_model``
        # is all the per-solver wiring needed — the viewer auto-prefixes object
        # names so all four pendulums coexist in the same scene.
        # ``set_layer_transform`` positions each layer relative to the others.
        # The default ``spacing = 0`` overlays them so per-solver divergence
        # is obvious; raise ``spacing`` (each link is 2 m long, so ~9 m gives
        # a comfortable gap between neighboring swing planes) to lay them
        # out side-by-side along the world X-axis instead.
        spacing = 0.0
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            self.viewer.activate(layer.layer_id)
            self.viewer.set_model(layer.model)
            shift = (i - 0.5 * (n - 1)) * spacing
            self.viewer.set_layer_transform(
                layer.layer_id,
                wp.transform(wp.vec3(shift, 0.0, 0.0), wp.quat_identity()),
            )
            # Capture after binding the model so viewer-force kernels use
            # this layer's picking state, including picks made after capture.
            layer.graph = layer._capture()

    def build_solver(self):
        """Build every layer's solver and the observables bound to it; the tuner calls this again to swap them."""
        for layer in self.layers:
            layer.build_solver(self.preset.entry(layer.entry).make_solver)
            if layer.graph is not None:
                # A layer already stepping a graph captures it again so it drives the new solver.
                layer.graph = layer._capture()

    def step(self):
        for layer in self.layers:
            layer.step()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        # Steel and Featherstone integrate the same joint coordinates, so they move together until the chaotic
        # swing separates them, well after the first 2 s.
        by_entry = {layer.entry: layer for layer in self.layers}
        if not {"steel", "featherstone"} <= by_entry.keys() or self.sim_time > 2.0 + 0.5 * self.frame_dt:
            return
        steel_q = by_entry["steel"].state_0.body_q.numpy()[:, :3]
        featherstone_q = by_entry["featherstone"].state_0.body_q.numpy()[:, :3]
        gap = float(np.abs(steel_q - featherstone_q).max())
        assert gap < 0.01, f"the Steel pendulum is {gap:.4f} m from the Featherstone one at t={self.sim_time:.2f} s"

    def test_final(self):
        # The anchor is at z=5 and the chain has four 2 m links, so any
        # body must stay within the swept envelope and exactly in the
        # x=0 plane (the joints only rotate about y).
        for layer in self.layers:
            num_links = layer.model.body_count
            newton.examples.test_body_state(
                layer.model,
                layer.state_0,
                f"{layer.layer_id} pendulum links in correct area",
                lambda q, qd: abs(q[0]) < 1e-3 and -4.0 < q[2] < 6.0,
                list(range(num_links)),
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for layer in self.layers:
            self.viewer.activate(layer.layer_id)
            self.viewer.log_state(layer.state_0)
            if layer.contacts is not None:
                self.viewer.log_contacts(layer.contacts, layer.state_1, observables=layer.solver_observables)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
