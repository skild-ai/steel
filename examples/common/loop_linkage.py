"""Closed-loop linkages run the shelf's way: a shared base, so each example is just its mechanism.

A mechanism (:mod:`common.loop_mechanisms`) is a set of bodies and joints at a reference pose where every loop is
closed, the joints flagged ``cut`` being the ones that close the loops. Here it becomes one articulation of the
non-cut joints (a spanning tree from the world) plus the cut joints as loop-closing joints, which Steel enforces as
bilateral rows (three per BALL closure, five per REVOLUTE closure). The actuated joints follow the mechanism's scripted
trajectory through PD targets written inside the CUDA graph every substep. There is gravity and no contact: the
loop handling is all there is to get right.

The trajectory ramps in over the first second and is periodic after that, so the target table covers the ramp plus
one period and the step counter wraps back to the period's start; the viewer can run as long as it likes.

``check`` runs the mechanism's first :attr:`LoopLinkageExample.check_frames` frames and compares the tracked bodies
(the platform, the jaws) with ``reference.json`` next to the example: positions per frame from an independent
maximal-coordinate integrator with exact constraints (SciPy DOP853 at rtol 1e-9).
It also bounds the worst loop gap over the run.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets
from common.loop_mechanisms import Mechanism, get_mechanism, joint_frames, quat_to_mat, tree_order
from common.steel_scenes import camera_from_eye


def build_model(mech: Mechanism) -> tuple[newton.Model, dict[int, int]]:
    """The mechanism as one articulation of its tree joints plus its cut joints as loop closures."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    for body in mech.bodies:
        bid = b.add_link(
            xform=wp.transform(wp.vec3(*body.pos), wp.quat(*body.quat)),
            mass=float(body.mass),
            inertia=wp.mat33(*body.inertia.ravel()),
            com=wp.vec3(0.0, 0.0, 0.0),
            lock_inertia=True,
            label=body.name,
        )
        for s in body.shapes:
            xf = wp.transform(wp.vec3(*s.pos), wp.quat(*s.quat))
            if s.kind == "capsule":
                b.add_shape_capsule(bid, xform=xf, radius=s.size[0], half_height=s.size[1], cfg=cfg)
            elif s.kind == "box":
                b.add_shape_box(bid, xform=xf, hx=s.size[0], hy=s.size[1], hz=s.size[2], cfg=cfg)
            else:
                b.add_shape_sphere(bid, xform=xf, radius=s.size[0], cfg=cfg)
    jid = {}
    for i in tree_order(mech) + [i for i, j in enumerate(mech.joints) if j.cut]:
        j = mech.joints[i]
        (pp, pq), (cp, cq) = joint_frames(mech, j)
        kw = dict(
            parent=j.parent,
            child=j.child,
            parent_xform=wp.transform(wp.vec3(*pp), wp.quat(*pq)),
            child_xform=wp.transform(wp.vec3(*cp), wp.quat(*cq)),
            label=j.name,
            collision_filter_parent=True,
        )
        if j.kind in ("revolute", "prismatic"):
            kw.update(axis=wp.vec3(*j.axis), armature=0.0, damping=0.0, friction=0.0)
            if j.drive is not None:
                kw.update(
                    target_ke=j.drive.ke,
                    target_kd=j.drive.kd,
                    effort_limit=j.drive.effort,
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                )
            else:
                kw.update(target_ke=0.0, target_kd=0.0, actuator_mode=newton.JointTargetMode.NONE)
            jid[i] = (b.add_joint_revolute if j.kind == "revolute" else b.add_joint_prismatic)(**kw)
        elif j.kind == "ball":
            jid[i] = b.add_joint_ball(armature=0.0, damping=0.0, friction=0.0, **kw)
        else:
            jid[i] = b.add_joint_fixed(**kw)
    b.add_articulation([jid[i] for i in tree_order(mech)], label=mech.name)
    return b.finalize(), jid


def loop_gap(mech: Mechanism, body_q: np.ndarray) -> float:
    """Worst distance [m] between the two anchors of any cut joint, from body poses (xyz + quat xyzw per body)."""
    worst = 0.0
    for j in (j for j in mech.joints if j.cut):
        ends = []
        for body in (j.parent, j.child):
            if body < 0:
                ends.append(j.anchor)
                continue
            ref = mech.bodies[body]
            local = quat_to_mat(ref.quat).T @ (j.anchor - ref.pos)
            ends.append(body_q[body, :3] + quat_to_mat(body_q[body, 3:]) @ local)
        worst = max(worst, float(np.linalg.norm(ends[0] - ends[1])))
    return worst


@wp.kernel(enable_backward=False)
def _write_targets(
    table_q: wp.array2d[float],
    table_qd: wp.array2d[float],
    coords: wp.array[int],
    dofs: wp.array[int],
    step: wp.array[int],
    ramp_rows: int,
    period_rows: int,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    i = wp.tid()
    k = step[0]
    if k >= ramp_rows + period_rows:
        k = ramp_rows + (k - ramp_rows) % period_rows
    target_q[coords[i]] = table_q[k, i]
    target_qd[dofs[i]] = table_qd[k, i]


@wp.kernel(enable_backward=False)
def _advance(step: wp.array[int]):
    step[0] = step[0] + 1


class LoopLinkageExample:
    """One closed-loop mechanism, run with the shelf's preset.

    Subclasses set ``file`` (their ``__file__``), ``mechanism`` (a name in
    :data:`common.loop_mechanisms.MECHANISMS`), ``period`` (seconds after which the trajectory repeats, once the
    one-second ramp is over), ``camera`` (``{"eye", "target"}``) and the check bounds.
    """

    file: str = __file__
    mechanism: str = ""
    ramp = 1.0  # [s] every mechanism's drive ramps in over its first second (common.loop_mechanisms.ramp)
    period = 2.0  # [s]
    camera: ClassVar[dict[str, Any]] = {"eye": (0.0, -1.5, 0.6), "target": (0.0, 0.0, 0.5)}
    check_frames = 200  # 4 s at 50 fps: the frames in reference.json
    # the presets are tuned to 0.5 mm tracked error and 0.25 mm gap; check allows twice that, for other GPUs
    max_tracked_error = 1e-3  # [m] tracked bodies vs reference.json, worst over the run
    max_gap = 5e-4  # [m] worst loop gap over the run

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

        self.mech = get_mechanism(self.mechanism)
        self.model, jid = build_model(self.mech)
        self.build_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # target table: one row per substep over the ramp and one period; the target is the one at the end of the substep
        ramp_rows = round(self.ramp * self.fps * self.sim_substeps)
        period_rows = round(self.period * self.fps * self.sim_substeps)
        times = (np.arange(ramp_rows + period_rows) + 1) * self.sim_dt
        tq, tqd = self.mech.target_table(times)
        device = self.model.device
        self._table_q = wp.array(tq.astype(np.float32), dtype=float, device=device)
        self._table_qd = wp.array(tqd.astype(np.float32), dtype=float, device=device)
        q_start, qd_start = self.model.joint_q_start.numpy(), self.model.joint_qd_start.numpy()
        self._coords = wp.array([int(q_start[jid[i]]) for i in self.mech.actuated], dtype=int, device=device)
        self._dofs = wp.array([int(qd_start[jid[i]]) for i in self.mech.actuated], dtype=int, device=device)
        self._rows = (ramp_rows, period_rows)
        self._step = wp.zeros(1, dtype=int, device=device)

        self._gap = 0.0
        self._track: list[np.ndarray] = []
        self._frame = 0

        self.viewer.set_model(self.model)
        pos, pitch, yaw = camera_from_eye(self.camera["eye"], self.camera["target"])
        self.viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)
        self.capture()

    def build_solver(self):
        """Build the solver; the tuner calls this again to swap it."""
        self.solver = self.preset.make_solver(self.model)

    def capture(self):
        if wp.get_device().is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _write_targets,
                dim=len(self.mech.actuated),
                inputs=[self._table_q, self._table_qd, self._coords, self._dofs, self._step, *self._rows],
                outputs=[self.control.joint_target_q, self.control.joint_target_qd],
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(_advance, dim=1, inputs=[self._step], device=self.model.device)
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                # the graph starts from state_0 every frame: with an odd count, copy the last result back
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if self.test_mode and self._frame < self.check_frames:
            body_q = self.state_0.body_q.numpy()
            self._gap = max(self._gap, loop_gap(self.mech, body_q))
            self._track.append(body_q[self.mech.tracked, :3].copy())
        self._frame += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Finite state; the loops stay closed; the tracked bodies follow the reference."""
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            values = getattr(self.state_0, name).numpy()
            assert np.isfinite(values).all(), f"{name} has non-finite entries"
        assert self._gap <= self.max_gap, f"worst loop gap {1e3 * self._gap:.3f} mm > {1e3 * self.max_gap:.3f} mm"
        with open(os.path.join(os.path.dirname(self.file), "reference.json")) as f:
            ref = np.asarray(json.load(f)["tracked_pos"])  # (frames, tracked bodies, 3) [m]
        n = min(len(self._track), len(ref))
        assert n > 0, "no frames recorded"
        err = float(np.linalg.norm(np.asarray(self._track[:n]) - ref[:n], axis=2).max())
        assert math.isfinite(err) and err <= self.max_tracked_error, (
            f"tracked bodies {1e3 * err:.2f} mm from the reference (bound {1e3 * self.max_tracked_error:.1f} mm)"
        )

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()
