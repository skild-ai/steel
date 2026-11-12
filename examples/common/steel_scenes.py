# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""The scenes ported from steel-webgpu: a shared base, so each example is just its scene.

steel-webgpu builds 22 feature scenes as Newton ``ModelBuilder`` code (``tools/scenes.py``) and ships 13
exported hero-montage scenes (``assets/scenes``). :class:`SceneExample` runs a scene the shelf's way: the
preset's solver (``[steel]`` by default), Newton's collision pipeline, a CUDA graph, the shelf's viewer,
and a finiteness test for ``check``. A builder scene subclasses it, sets ``file = __file__`` and
``camera``, and returns its builder, the initial body twists and any initial joint positions from
:meth:`build`; an exported scene subclasses :class:`common.exported_scene.ExportedSceneExample`.

A twist is ``(vx, vy, vz, wx, wy, wz)``: the body's centre-of-mass velocity and angular velocity in
the world frame, as :attr:`newton.State.body_qd` holds them. Kinematic bodies keep their twist every
substep; free bodies start with theirs. Joint velocities follow from the body velocities through
:func:`newton.eval_ik`, so a twist on a floating base reaches its free joint too.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets

# steel-webgpu's palette (tools/scenes.py)
BLUE, TEAL, GOLD, RED, GREY = (0.06, 0.38, 0.70), (0.06, 0.57, 0.43), (0.92, 0.66, 0.20), (0.80, 0.25, 0.20), (0.45, 0.5, 0.55)
# steel-webgpu's camera when a scene names none (packages/three/src/scenePreviewer.ts)
DEFAULT_CAMERA = {"eye": (1.25, -1.6, 1.25), "target": (0.1, 0.05, 0.15)}


def tf(p=(0.0, 0.0, 0.0), q=None) -> wp.transform:
    return wp.transform(wp.vec3(*p), wp.quat_identity() if q is None else q)


def cfg(**kw) -> newton.ModelBuilder.ShapeConfig:
    c = newton.ModelBuilder.ShapeConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def camera_from_eye(eye, target) -> tuple[wp.vec3, float, float]:
    """Newton's ``set_camera`` arguments (position, pitch, yaw in degrees) for a camera at ``eye`` looking at ``target``."""
    d = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return wp.vec3(*[float(x) for x in eye]), pitch, yaw


@wp.kernel(enable_backward=False)
def _set_twists(bodies: wp.array[wp.int32], twists: wp.array[wp.spatial_vector], body_qd: wp.array[wp.spatial_vector]):
    i = wp.tid()
    body_qd[bodies[i]] = twists[i]


@wp.kernel(enable_backward=False)
def _prescribe_kinematic(
    q_starts: wp.array[wp.int32],
    qd_starts: wp.array[wp.int32],
    coms: wp.array[wp.vec3],
    twists: wp.array[wp.spatial_vector],
    dt: float,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Move a kinematic body's free joint by its twist over ``dt`` (COM velocity and angular velocity, world frame)."""
    i = wp.tid()
    qs = q_starts[i]
    qds = qd_starts[i]
    t = twists[i]
    v = wp.spatial_top(t)
    w = wp.spatial_bottom(t)
    p = wp.vec3(joint_q[qs], joint_q[qs + 1], joint_q[qs + 2])
    r = wp.quat(joint_q[qs + 3], joint_q[qs + 4], joint_q[qs + 5], joint_q[qs + 6])
    com = p + wp.quat_rotate(r, coms[i]) + v * dt
    angle = wp.length(w) * dt
    if angle > 0.0:
        r = wp.normalize(wp.quat_from_axis_angle(wp.normalize(w), angle) * r)
    p = com - wp.quat_rotate(r, coms[i])
    for k in range(3):
        joint_q[qs + k] = p[k]
    for k in range(4):
        joint_q[qs + 3 + k] = r[k]
    for k in range(6):
        joint_qd[qds + k] = t[k]


class SceneExample:
    """A scene from steel-webgpu, run with the shelf's preset.

    Subclasses set ``file`` to their ``__file__`` (the preset loader finds presets.toml next to it) and
    ``camera`` (``{"eye", "target"}``), and implement :meth:`build`, which returns ``(builder, twists)``
    or ``(builder, twists, start)``, where ``twists`` maps body -> twist and ``start`` maps a name to
    ``(joint, values)``, the joint's initial coordinates. Any further tuple items (the builders' ``about``
    string) are ignored, so a builder from steel-webgpu's ``tools/scenes.py`` ports as it is.
    """

    file: str = __file__
    camera: dict[str, Any] = DEFAULT_CAMERA

    def build(self):
        raise NotImplementedError

    def check(self) -> None:
        """Scene-specific assertions for ``check`` (test mode), after the finiteness test."""

    def deepest_contact(self, bodies=None) -> float:
        """The deepest rigid contact [m] at the current poses (0 when none penetrate), for checks.

        Each contact's two support points at the post-step poses, along its normal, less both surface margins.
        ``bodies`` keeps only contacts that touch one of these body indices.
        """
        c = int(self.contacts.rigid_contact_count.numpy()[0])
        if not c:
            return 0.0
        body_q = self.state_0.body_q.numpy()
        shape_body = self.model.shape_body.numpy()
        s0, s1 = self.contacts.rigid_contact_shape0.numpy()[:c], self.contacts.rigid_contact_shape1.numpy()[:c]
        b0 = np.where(s0 >= 0, shape_body[np.maximum(s0, 0)], -1)
        b1 = np.where(s1 >= 0, shape_body[np.maximum(s1, 0)], -1)

        def world(b, p):
            out = p.copy()
            has = b >= 0
            q = body_q[b[has]]
            u, w = q[:, 3:6], q[:, 6:7]
            t = 2.0 * np.cross(u, p[has])
            out[has] = p[has] + w * t + np.cross(u, t) + q[:, :3]
            return out

        p0 = world(b0, self.contacts.rigid_contact_point0.numpy()[:c])
        p1 = world(b1, self.contacts.rigid_contact_point1.numpy()[:c])
        n = self.contacts.rigid_contact_normal.numpy()[:c]
        margin = self.contacts.rigid_contact_margin0.numpy()[:c] + self.contacts.rigid_contact_margin1.numpy()[:c]
        sep = np.einsum("ij,ij->i", n, p1 - p0) - margin
        keep = (s0 >= 0) & (s1 >= 0)
        if bodies is not None:
            keep &= np.isin(b0, bodies) | np.isin(b1, bodies)
        sep = sep[keep]
        return float(max(0.0, -sep.min())) if len(sep) else 0.0

    def __init__(self, viewer, args):
        self.preset = presets.load(self.file)
        self.viewer = viewer
        self.args = args
        self.test_mode = bool(getattr(args, "test", False))
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        built = self.build()
        builder, twists = built[0], built[1] or {}
        start = next((x for x in built[2:] if isinstance(x, dict)), None)
        # on the builder, so the model and every state made from it start there
        for joint, values in (start or {}).values():
            q_start = builder.joint_q_start[joint]
            builder.joint_q[q_start : q_start + len(values)] = values
        self.model = builder.finalize()
        self.build_solver()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        kinematic = self.model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC) if self.model.body_count else []
        self._kinematic = [b for b in twists if kinematic[b]]
        self._kinematic_twists = self._twist_arrays({b: twists[b] for b in self._kinematic})
        if self._kinematic:
            # Steel holds a kinematic joint's coordinates as given each step, so the scene moves them: the body's
            # free joint (world parent) advances by its twist every substep
            joint_child = self.model.joint_child.numpy()
            joint_type = self.model.joint_type.numpy()
            joints = [int(np.flatnonzero(joint_child == b)[0]) for b in self._kinematic]
            if any(joint_type[j] != int(newton.JointType.FREE) for j in joints):
                raise ValueError("a kinematic twist needs the body on a free joint")
            device = self.model.device
            self._kinematic_q_start = wp.array(self.model.joint_q_start.numpy()[joints], dtype=wp.int32, device=device)
            self._kinematic_qd_start = wp.array(self.model.joint_qd_start.numpy()[joints], dtype=wp.int32, device=device)
            self._kinematic_com = wp.array(self.model.body_com.numpy()[self._kinematic], dtype=wp.vec3, device=device)
        if twists:
            bodies, values = self._twist_arrays(twists)
            wp.launch(_set_twists, dim=len(twists), inputs=[bodies, values, self.state_0.body_qd], device=self.model.device)
            # joint velocities from the body velocities, so a twist on a floating base reaches its free joint
            newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            wp.launch(_set_twists, dim=len(twists), inputs=[bodies, values, self.state_0.body_qd], device=self.model.device)

        self.viewer.set_model(self.model)
        pos, pitch, yaw = camera_from_eye(self.camera["eye"], self.camera["target"])
        self.viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)
        self.capture()

    def _twist_arrays(self, twists: dict[int, Any]):
        device = self.model.device
        bodies = wp.array(list(twists), dtype=wp.int32, device=device)
        values = wp.array([wp.spatial_vector(*[float(v) for v in t]) for t in twists.values()], dtype=wp.spatial_vector, device=device)
        return bodies, values

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
        # the solver first: it registers the per-contact attributes it needs (compliance, warm start) on the model,
        # and the contacts buffer built afterwards carries them
        self.solver = self.preset.make_solver(self.model)
        # warm start matches contacts across steps, which the pipeline does only when asked
        matching = "latest" if self.preset.solver.get("contact_warmstart", False) else "disabled"
        # the preset's `collision` table holds CollisionPipeline arguments (box_box_sat, broad_phase, ...)
        collision = {"contact_matching": matching, **self.preset.collision}
        if isinstance(collision.get("body_pair_reduction"), dict):  # a TOML table: the BodyPairReductionConfig fields
            collision["body_pair_reduction"] = newton.CollisionPipeline.BodyPairReductionConfig(**collision["body_pair_reduction"])
        self.collision_pipeline = newton.CollisionPipeline(self.model, **collision)
        self.contacts = self.collision_pipeline.contacts()

    def capture(self):
        # one collide outside capture: pipeline stages that size themselves on first use do so here
        self.collision_pipeline.collide(self.state_0, self.contacts, dt=self.sim_dt)
        if wp.get_device().is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def before_substep(self, substep: int) -> None:
        """Hook for per-substep inputs (an exported scene's joint targets)."""

    def after_frame(self) -> None:
        """Hook run once per frame inside the captured graph, after the substeps."""

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self._kinematic:
                # the kinematic bodies' poses and twists from their joint coordinates, for collision and the solver
                newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, body_flag_filter=newton.BodyFlags.KINEMATIC)
            self.before_substep(substep)
            # dt sizes speculative contacts when the preset's collision table enables them
            self.collision_pipeline.collide(self.state_0, self.contacts, dt=self.sim_dt)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self._kinematic:
                _, values = self._kinematic_twists
                wp.launch(
                    _prescribe_kinematic, dim=len(self._kinematic),
                    inputs=[self._kinematic_q_start, self._kinematic_qd_start, self._kinematic_com, values, self.sim_dt],
                    outputs=[self.state_1.joint_q, self.state_1.joint_qd], device=self.model.device,
                )  # fmt: skip
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                # the graph starts from state_0 every frame: with an odd count, copy the last result back
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.after_frame()

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Every body and joint coordinate and velocity is finite; then the scene's own checks."""
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            array = getattr(self.state_0, name, None)
            if array is not None and len(array):
                values = array.numpy()
                assert np.isfinite(values).all(), f"{name} has non-finite entries"
        self.check()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()
