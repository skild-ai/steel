# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Render one example with its Steel preset to an mp4 and record stability metrics.

usage (cwd = examples/, its own environment):
    python render_scene.py --plan plan.json --scene <name> --out <dir>

Writes <out>/<scene>/{clip_raw.mp4, poster.jpg, sheet.jpg, metrics.json}. The preset is used exactly as
presets.toml has it; the only addition is SolverSteel(row_watermark=True), a diagnostic counter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TREE = Path.cwd()
sys.path.insert(0, str(TREE))

import pyglet  # noqa: E402

pyglet.options["headless"] = True  # EGL, before anything imports pyglet.gl

STEEL_SOLVERS: list = []  # every SolverSteel built in this process
STEEL_KWARGS: list[dict] = []


def _patch_steel() -> None:
    import newton.solvers

    cls = newton.solvers.SolverSteel
    orig = cls.__init__

    def init(self, model, *args, **kwargs):
        kwargs.setdefault("row_watermark", True)
        orig(self, model, *args, **kwargs)
        STEEL_SOLVERS.append(self)
        STEEL_KWARGS.append({k: _plain(v) for k, v in kwargs.items()})

    cls.__init__ = init


def _plain(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (bool, int, float, str)) for x in v):
        return list(v)
    return f"<{type(v).__name__}>"


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (n,3) by warp quaternions q (n,4) in xyzw order."""
    xyz, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def find_attr(obj, cls, depth=2):
    """First attribute of obj (searching one level into containers/objects) that is an instance of cls."""
    seen = set()
    stack = [(obj, 0)]
    while stack:
        o, d = stack.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, cls) and o is not obj:
            return o
        if d >= depth:
            continue
        try:
            items = list(vars(o).values())
        except TypeError:
            items = list(o.values()) if isinstance(o, dict) else (list(o) if isinstance(o, (list, tuple)) else [])
        for it in items:
            if isinstance(it, (int, float, str, bytes, np.ndarray)) or it is None:
                continue
            stack.append((it, d + 1))
    return None


class Metrics:
    """Per-frame stability bookkeeping over bodies, particles, contacts, joints and Steel status."""

    def __init__(self, example, fps: float):
        import newton

        self.newton = newton
        layers = getattr(example, "layers", None)
        if not hasattr(example, "model") and layers:
            # One model per viewer layer (basic_multi_solver_overlay): measure the Steel layer.
            example = next((layer for layer in layers if getattr(layer, "entry", "") == "steel"), layers[0])
        self.ex = example
        self.fps = fps
        self.model = example.model
        m = self.model
        self.up = int(getattr(m, "up_axis", 2))
        g = m.gravity.numpy() if getattr(m, "gravity", None) is not None else np.array([[0, 0, -9.81]])
        self.gvec = np.asarray(g).reshape(-1, 3)[0].astype(np.float64)
        self.g = float(np.linalg.norm(self.gvec)) or 9.81
        self.nb = int(m.body_count)
        self.np_ = int(getattr(m, "particle_count", 0) or 0)
        c = getattr(example, "contacts", None)
        self.contacts = c if isinstance(c, newton.Contacts) else find_attr(example, newton.Contacts)
        if self.nb:
            self.mass = m.body_mass.numpy().astype(np.float64)
            self.inv_mass = m.body_inv_mass.numpy()
            self.inertia = m.body_inertia.numpy().astype(np.float64).reshape(-1, 3, 3)
            self.com = m.body_com.numpy().astype(np.float64)
            self.dyn = self.inv_mass > 0
        self.shape_body = m.shape_body.numpy() if m.shape_count else np.zeros(0, int)
        self.ground_h = None
        if m.shape_count:
            st = m.shape_type.numpy()
            planes = np.flatnonzero((st == int(newton.GeoType.PLANE)) & (self.shape_body == -1))
            if len(planes):
                xf = m.shape_transform.numpy()[planes]
                self.ground_h = float(np.max(xf[:, self.up]))
        self.frames = 0
        self.nonfinite_frame = None
        self.nonfinite_what = None
        self.max_v = 0.0
        self.max_v_frame = -1
        self.max_w = 0.0
        self.max_w_frame = -1
        self.max_pv = 0.0
        self.v_series = []
        self.ke_series = []
        self.e_series = []
        self.pen_series = []
        self.contact_series = []
        self.overflow_first_frame = None
        self.overflow_frames = 0
        self.invalid_contact_frames = 0
        self.invalid_contacts = 0
        self.invalid_contact_first_frame = None
        self.oob_bodies = set()
        self.oob_first_frame = None
        self.below_ground = set()
        self.below_first_frame = None
        self.joint_violation_max = 0.0
        self.joint_violation_frame = -1
        self.joint_violation_joint = None
        self.kinematic_moving = False
        self.ring: list[np.ndarray] = []
        self.ring_rot: list[np.ndarray] = []
        self.ring_n = max(2, int(round(fps)) + 1)
        self._init_bounds()
        self._init_joints()
        self._init_static_boxes()

    def _init_static_boxes(self):
        """World-frame colliding static boxes (decks, trays, ramps) for the sink-through check."""
        self.static_boxes = None
        self.inside_static = set()
        self.inside_static_first_frame = None
        self.inside_static_depth_max = 0.0
        self.inside_static_start = None
        m = self.model
        if not m.shape_count or not self.nb or not self.dyn.any():
            return
        st = m.shape_type.numpy()
        collide = (m.shape_flags.numpy() & int(self.newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        ids = np.flatnonzero((st == int(self.newton.GeoType.BOX)) & (self.shape_body == -1) & collide)
        if not len(ids):
            return
        xf = m.shape_transform.numpy()[ids].astype(np.float64)
        shape_world = m.shape_world.numpy()[ids] if getattr(m, "shape_world", None) is not None else np.full(len(ids), -1)
        body_world = m.body_world.numpy()[self.dyn] if getattr(m, "body_world", None) is not None else np.full(int(self.dyn.sum()), -1)
        # a box only bounds bodies it can collide with: global boxes (world -1), or the body's own world
        eligible = (shape_world[:, None] == -1) | (body_world[None, :] == -1) | (shape_world[:, None] == body_world[None, :])
        self.static_boxes = (xf[:, :3], xf[:, 3:7], m.shape_scale.numpy()[ids, :3].astype(np.float64), eligible)
        s = self._state()  # the scene as built, before the first step: bodies built inside a box do not count
        if s is not None and s.body_q is not None:
            q = s.body_q.numpy().astype(np.float64)[self.dyn]
            comw = q[:, :3] + quat_rotate(q[:, 3:7], self.com[self.dyn])
            self.inside_static_start = self._inside_static_boxes(comw)[0]

    def _inside_static_boxes(self, comw):
        """(inside, depth) of each dynamic body's COM against the boxes it can collide with."""
        t, r, h, eligible = self.static_boxes
        rinv = r * np.array([-1, -1, -1, 1])
        inside = np.zeros(len(comw), bool)
        depth = np.zeros(len(comw))
        for k in range(len(t)):
            local = quat_rotate(np.repeat(rinv[k : k + 1], len(comw), 0), comw - t[k])
            d = np.min(h[k] - np.abs(local), axis=1)  # > 0: inside, by this much
            hit = eligible[k] & np.isfinite(d) & (d > max(0.003, 0.002 * self.scale))
            inside |= hit
            depth = np.maximum(depth, np.where(hit, d, 0.0))
        return inside, depth

    def _static_containment(self, f, comw, ids):
        """Dynamic bodies whose COM lies inside a colliding static box (deeper than a few mm): sank through it."""
        inside, depth = self._inside_static_boxes(comw)
        new = inside if self.inside_static_start is None else inside & ~self.inside_static_start
        if new.any():
            self.inside_static.update(ids[new].tolist())
            self.inside_static_depth_max = max(self.inside_static_depth_max, float(depth[new].max()))
            if self.inside_static_first_frame is None:
                self.inside_static_first_frame = f

    def _state(self):
        s = getattr(self.ex, "state_0", None)
        return s if s is not None else getattr(self.ex, "state", None)

    def _init_bounds(self):
        s = self._state()
        pts = []
        self.v0 = 0.0
        self.w0 = 0.0
        if self.nb and s is not None and s.body_q is not None:
            q = s.body_q.numpy()
            qd = s.body_qd.numpy()
            if self.dyn.any():
                pts.append(q[self.dyn, :3])
                self.v0 = float(np.nanmax(np.linalg.norm(qd[self.dyn, :3], axis=1)))
                self.w0 = float(np.nanmax(np.linalg.norm(qd[self.dyn, 3:], axis=1)))
            else:
                pts.append(q[:, :3])
        if self.np_ and s is not None and s.particle_q is not None:
            pts.append(s.particle_q.numpy())
        if pts:
            p = pts[0] if self.nb and self.dyn.any() else np.concatenate(pts)
            p = p[np.isfinite(p).all(axis=1)]
            # robust box: parked or far-away outliers (unused particles, distant props) do not set the scale
            self.lo, self.hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
            self.start_up = p[:, self.up].copy() if self.nb and self.dyn.any() else None
        else:
            self.lo = self.hi = np.zeros(3)
            self.start_up = None
        self.scale = float(max(np.linalg.norm(self.hi - self.lo), 0.5))
        self.margin = max(3.0 * self.scale, 4.0)
        self.v_bound = max(10.0, 3.0 * math.sqrt(2 * self.g * max(self.scale, 1.0)), 2.0 * self.v0)
        self.w_bound = max(100.0, 2.0 * self.w0)

    def _init_joints(self):
        m = self.model
        self.jcheck = None
        if not m.joint_count or not self.nb:
            return
        jt = m.joint_type.numpy()
        qs = m.joint_q_start.numpy()
        ds = m.joint_qd_start.numpy()
        lo = m.joint_limit_lower.numpy()
        hi = m.joint_limit_upper.numpy()
        rev, pri = int(self.newton.JointType.REVOLUTE), int(self.newton.JointType.PRISMATIC)
        idx, qi, lows, highs, ang = [], [], [], [], []
        for j in range(len(jt)):
            if jt[j] not in (rev, pri):
                continue
            d = ds[j]
            if not (abs(lo[d]) < 1e4 and abs(hi[d]) < 1e4 and lo[d] < hi[d]):
                continue
            if jt[j] == rev and hi[d] - lo[d] >= 2 * math.pi:
                continue
            idx.append(j)
            qi.append(qs[j])
            lows.append(lo[d])
            highs.append(hi[d])
            ang.append(jt[j] == rev)
        if idx:
            import warp as wp

            self.jcheck = dict(joints=np.array(idx), qi=np.array(qi), lo=np.array(lows), hi=np.array(highs), ang=np.array(ang))
            self.jq = wp.zeros_like(m.joint_q)
            self.jqd = wp.zeros_like(m.joint_qd)

    def update(self, f: int, steel_solvers):
        self.frames = f + 1
        s = self._state()
        if s is None:
            return
        if self.nb and s.body_q is not None:
            q = s.body_q.numpy().astype(np.float64)
            qd = s.body_qd.numpy().astype(np.float64)
            if self.nonfinite_frame is None and not (np.isfinite(q).all() and np.isfinite(qd).all()):
                self.nonfinite_frame, self.nonfinite_what = f, "body_q/body_qd"
            v = np.linalg.norm(qd[:, :3], axis=1)
            w = np.linalg.norm(qd[:, 3:], axis=1)
            if (~self.dyn).any() and np.nanmax(np.r_[v[~self.dyn], w[~self.dyn]]) > 1e-6:
                self.kinematic_moving = True
            vd = np.where(self.dyn & np.isfinite(v), v, 0.0)
            wd = np.where(self.dyn & np.isfinite(w), w, 0.0)
            vm, wm = float(vd.max()), float(wd.max())
            self.v_series.append(vm)
            if vm > self.max_v:
                self.max_v, self.max_v_frame = vm, f
            if wm > self.max_w:
                self.max_w, self.max_w_frame = wm, f
            if self.dyn.any():
                qq = q[self.dyn]
                pos = qq[:, :3]
                rot = qq[:, 3:7]
                ok = np.isfinite(pos).all(axis=1)
                out = ok & (((pos < self.lo - self.margin) | (pos > self.hi + self.margin)).any(axis=1))
                ids = np.flatnonzero(self.dyn)[out]
                if len(ids):
                    self.oob_bodies.update(ids.tolist())
                    if self.oob_first_frame is None:
                        self.oob_first_frame = f
                if self.ground_h is not None:
                    below = ok & (pos[:, self.up] < self.ground_h - max(0.1, 0.05 * self.scale))
                    ids = np.flatnonzero(self.dyn)[below]
                    if len(ids):
                        self.below_ground.update(ids.tolist())
                        if self.below_first_frame is None:
                            self.below_first_frame = f
                # energy: kinetic (linear at the COM + angular) and gravitational
                m = self.mass[self.dyn]
                R_I = self.inertia[self.dyn]
                wdyn = qd[self.dyn, 3:]
                # world inertia = R I R^T; w . (R I R^T w) = (R^T w) . I (R^T w)
                rinv = rot * np.array([-1, -1, -1, 1])
                wb = quat_rotate(rinv, wdyn)
                ke = 0.5 * m * (qd[self.dyn, :3] ** 2).sum(1) + 0.5 * np.einsum("ni,nij,nj->n", wb, R_I, wb)
                comw = pos + quat_rotate(rot, self.com[self.dyn])
                if self.static_boxes is not None:
                    self._static_containment(f, comw, np.flatnonzero(self.dyn))
                pe = -m * (comw @ self.gvec)
                ke_t = float(np.nansum(ke))
                self.ke_series.append(ke_t)
                self.e_series.append(ke_t + float(np.nansum(pe)))
                self.ring.append(pos.copy())
                self.ring_rot.append(rot.copy())
                if len(self.ring) > self.ring_n:
                    self.ring.pop(0)
                    self.ring_rot.pop(0)
        if self.np_ and s.particle_q is not None:
            pq = s.particle_q.numpy()
            pqd = s.particle_qd.numpy()
            if self.nonfinite_frame is None and not (np.isfinite(pq).all() and np.isfinite(pqd).all()):
                self.nonfinite_frame, self.nonfinite_what = f, "particle_q/particle_qd"
            pv = np.linalg.norm(pqd, axis=1)
            pv = pv[np.isfinite(pv)]
            if len(pv):
                self.max_pv = max(self.max_pv, float(pv.max()))
        self._contacts(f, s)
        if self.jcheck is not None and f % 5 == 0:
            self._joints(f, s)
        for solver in steel_solvers:
            flags = solver.constraint_overflow.numpy()
            if flags.any():
                self.overflow_frames += 1
                if self.overflow_first_frame is None:
                    self.overflow_first_frame = f

    def _contacts(self, f, s):
        c = self.contacts
        if c is None or getattr(c, "rigid_contact_count", None) is None or not self.nb:
            self.pen_series.append(0.0)
            return
        n = int(c.rigid_contact_count.numpy()[0])
        n = min(n, int(c.rigid_contact_shape0.shape[0]))
        self.contact_series.append(n)
        if n == 0:
            self.pen_series.append(0.0)
            return
        s0 = c.rigid_contact_shape0.numpy()[:n]
        s1 = c.rigid_contact_shape1.numpy()[:n]
        valid = (s0 >= 0) & (s1 >= 0)
        s0, s1 = s0[valid], s1[valid]
        p0 = c.rigid_contact_point0.numpy()[:n][valid].astype(np.float64)
        p1 = c.rigid_contact_point1.numpy()[:n][valid].astype(np.float64)
        nrm = c.rigid_contact_normal.numpy()[:n][valid].astype(np.float64)
        m0 = c.rigid_contact_margin0.numpy()[:n][valid]
        m1 = c.rigid_contact_margin1.numpy()[:n][valid]
        q = s.body_q.numpy().astype(np.float64)
        ident = np.array([0, 0, 0, 0, 0, 0, 1.0])

        def world(sh, p):
            b = self.shape_body[sh]
            xf = np.where((b >= 0)[:, None], q[np.clip(b, 0, None)], ident)
            return xf[:, :3] + quat_rotate(xf[:, 3:7], p)

        d = ((world(s1, p1) - world(s0, p0)) * nrm).sum(1) - (m0 + m1)
        bad = ~np.isfinite(d) | ~np.isfinite(nrm).all(axis=1)
        if bad.any():
            # Non-finite contact geometry is a failure in its own right; penetration below covers the rest.
            self.invalid_contact_frames += 1
            self.invalid_contacts += int(bad.sum())
            if self.invalid_contact_first_frame is None:
                self.invalid_contact_first_frame = f
            s0, s1, m0, m1, d = s0[~bad], s1[~bad], m0[~bad], m1[~bad], d[~bad]
        if not len(d):
            self.pen_series.append(0.0)
            return
        i = int(d.argmin())
        pen = float(max(0.0, -d[i]))
        self.pen_series.append(pen)
        if pen > 0:
            pair = {"t": round(f / self.fps, 2), "depth_m": round(pen, 5), "shape0": int(s0[i]), "shape1": int(s1[i]),
                    "body0": int(self.shape_body[s0[i]]), "body1": int(self.shape_body[s1[i]]),
                    "margins_m": [round(float(m0[i]), 5), round(float(m1[i]), 5)]}  # fmt: skip
            self.pen_last_pair = pair
            if pen >= getattr(self, "pen_peak_pair", {}).get("depth_m", 0):
                self.pen_peak_pair = pair

    def _joints(self, f, s):
        jc = self.jcheck
        try:
            self.newton.eval_ik(self.model, s, self.jq, self.jqd)
        except Exception:  # an exotic joint set eval_ik cannot invert: skip the check for this scene
            self.jcheck = None
            return
        qv = self.jq.numpy()[jc["qi"]]
        lo, hi = jc["lo"], jc["hi"]
        viol = np.maximum(lo - qv, qv - hi).clip(min=0)
        a = jc["ang"]
        if a.any():  # angles: the smallest violation over 2*pi wraps
            best = viol.copy()
            for k in (-1, 1):
                qk = qv + 2 * math.pi * k
                best = np.where(a, np.minimum(best, np.maximum(lo - qk, qk - hi).clip(min=0)), best)
            viol = best
        viol = np.where(np.isfinite(viol), viol, 0.0)
        if len(viol) and viol.max() > self.joint_violation_max:
            i = int(viol.argmax())
            self.joint_violation_max = float(viol[i])
            self.joint_violation_frame = f
            self.joint_violation_joint = int(jc["joints"][i])

    def summary(self, steel_solvers) -> dict:
        fps = self.fps
        out = {
            "frames": self.frames,
            "bodies": self.nb,
            "dynamic_bodies": int(self.dyn.sum()) if self.nb else 0,
            "particles": self.np_,
            "scene_scale_m": round(self.scale, 3),
            "has_ground_plane": self.ground_h is not None,
            "nonfinite_frame": self.nonfinite_frame,
            "nonfinite_what": self.nonfinite_what,
            "max_lin_speed": round(self.max_v, 3),
            "max_lin_speed_t": round(self.max_v_frame / fps, 2) if self.max_v_frame >= 0 else None,
            "lin_speed_bound": round(self.v_bound, 2),
            "max_ang_speed": round(self.max_w, 2),
            "max_ang_speed_t": round(self.max_w_frame / fps, 2) if self.max_w_frame >= 0 else None,
            "ang_speed_bound": round(self.w_bound, 1),
            "initial_max_lin_speed": round(self.v0, 3),
            "max_particle_speed": round(self.max_pv, 3),
            "bodies_out_of_bounds": len(self.oob_bodies),
            "out_of_bounds_t": round(self.oob_first_frame / fps, 2) if self.oob_first_frame is not None else None,
            "bodies_below_ground": len(self.below_ground),
            "below_ground_t": round(self.below_first_frame / fps, 2) if self.below_first_frame is not None else None,
            "bodies_inside_static": len(self.inside_static),
            "inside_static_t": round(self.inside_static_first_frame / fps, 2) if self.inside_static_first_frame is not None else None,
            "inside_static_depth_max_m": round(self.inside_static_depth_max, 4),
            "static_boxes": 0 if self.static_boxes is None else len(self.static_boxes[0]),
            "kinematic_bodies_move": self.kinematic_moving,
            "has_joint_drives": bool(
                self.model.joint_count
                and getattr(self.model, "joint_target_ke", None) is not None
                and np.abs(self.model.joint_target_ke.numpy()).max(initial=0) > 0
            ),
            "joint_limit_violation_max": round(self.joint_violation_max, 4),
            "joint_limit_violation_t": round(self.joint_violation_frame / fps, 2) if self.joint_violation_frame >= 0 else None,
            "joint_limit_violation_joint": self.joint_violation_joint,
            "joint_limits_checked": int(len(self.jcheck["joints"])) if self.jcheck is not None else 0,
            "overflow_frames": self.overflow_frames,
            "invalid_contact_frames": self.invalid_contact_frames,
            "invalid_contacts": self.invalid_contacts,
            "invalid_contact_first_t": round(self.invalid_contact_first_frame / fps, 2)
            if self.invalid_contact_first_frame is not None
            else None,
            "overflow_first_t": round(self.overflow_first_frame / fps, 2) if self.overflow_first_frame is not None else None,
            "contacts_found": self.contacts is not None,
        }
        pen = np.array(self.pen_series) if self.pen_series else np.zeros(1)
        half = pen[len(pen) // 2 :]
        out["penetration_peak_m"] = round(float(pen.max()), 5)
        out["penetration_peak_t"] = round(float(pen.argmax()) / fps, 2)
        out["penetration_sustained_m"] = round(float(np.median(half)), 5)
        out["penetration_p95_m"] = round(float(np.percentile(pen, 95)), 5)
        out["penetration_peak_pair"] = getattr(self, "pen_peak_pair", None)
        out["penetration_last_pair"] = getattr(self, "pen_last_pair", None)
        if self.contact_series:
            out["contacts_max"] = int(max(self.contact_series))
        if self.e_series:
            e = np.array(self.e_series)
            ke = np.array(self.ke_series)
            ok = np.isfinite(e)
            e = e[ok]
            ke = ke[np.isfinite(ke)]
            mtot = float(self.mass[self.dyn].sum())
            escale = max(mtot * self.g * self.scale, float(ke.max(initial=0)), 1e-9)
            gain = float(np.max(e - np.minimum.accumulate(e))) if len(e) else 0.0
            out["energy_scale_J"] = round(escale, 4)
            out["energy_gain_ratio"] = round(gain / escale, 4)
            out["energy_gain_t"] = round(float(np.argmax(e - np.minimum.accumulate(e))) / fps, 2) if len(e) else None
            last = ke[-max(1, int(fps)) :]
            out["ke_last_s_mean_ratio"] = round(float(last.mean()) / escale, 5) if len(last) else None
            out["ke_series_1hz"] = [round(float(x), 4) for x in ke[:: max(1, int(fps))]]
        if self.ring and self.start_up is not None and len(self.ring[-1]) == len(self.start_up):
            end_up = self.ring[-1][:, self.up]
            floor = float(np.min(self.start_up))
            sank = np.isfinite(end_up) & (end_up < floor - max(0.05, 0.05 * self.scale))
            out["bodies_sank_below_start"] = int(sank.sum())
            out["dynamic_bodies_considered"] = int(len(end_up))
        if len(self.ring) >= 2:
            P = np.stack(self.ring)
            net = np.linalg.norm(P[-1] - P[0], axis=1)
            path = np.linalg.norm(np.diff(P, axis=0), axis=2).sum(0)
            Q = np.stack(self.ring_rot)
            dots = np.abs((Q[1:] * Q[:-1]).sum(2)).clip(max=1.0)
            apath = (2 * np.arccos(dots)).sum(0)
            anet = 2 * np.arccos(np.abs((Q[-1] * Q[0]).sum(1)).clip(max=1.0))
            rest = np.isfinite(net) & (net < 0.002) & (anet < 0.02)
            out["rest_bodies_last_s"] = int(rest.sum())
            jit = rest & ((path > max(0.02, 0.01 * self.scale)) | (apath > 0.5))
            out["jittering_bodies_last_s"] = int(jit.sum())
            out["rest_path_max_m"] = round(float(path[rest].max()), 4) if rest.any() else 0.0
            out["rest_angpath_max_rad"] = round(float(apath[rest].max()), 4) if rest.any() else 0.0
        if self.v_series:
            v = np.array(self.v_series)
            out["lin_speed_series_1hz"] = [round(float(v[i : i + int(fps)].max()), 3) for i in range(0, len(v), max(1, int(fps)))]
        out["steel"] = []
        for solver, kw in zip(steel_solvers, STEEL_KWARGS):
            entry = {"ctor_kwargs": kw}
            try:
                entry["row_watermarks"] = solver.constraint_row_watermarks()
            except Exception as e:
                entry["row_watermarks_error"] = str(e)[:200]
            try:
                solver.check_constraint_capacity()
                entry["capacity_check"] = "ok"
            except Exception as e:
                entry["capacity_check"] = str(e)[:400]
            out["steel"].append(entry)
        return out


class RenderLoop:
    """Replaces newton.examples.run: steps the example, pipes frames to ffmpeg, keeps metrics."""

    def __init__(self, entry: dict, out_dir: Path, ffmpeg: str, wall_budget_s: float):
        self.entry, self.out_dir, self.ffmpeg, self.wall_budget = entry, out_dir, ffmpeg, wall_budget_s
        self.ran = False
        self.result: dict = {}

    def __call__(self, example, args) -> None:
        import warp as wp
        from PIL import Image, ImageDraw

        self.ran = True
        e = self.entry
        viewer = example.viewer
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        fps = float(getattr(example, "fps", 0) or 0) or (1.0 / float(getattr(example, "frame_dt", 1 / 60)))
        frames = int(round(e["sim_seconds"] * fps))
        speed = e["speed"]
        skip = max(1, math.ceil(fps * speed / 60.0))
        vfps = fps * speed / skip
        sheet_at = set(int(round(x)) for x in np.linspace(0, frames - 1, 12))
        poster_at = int(min(frames - 1, max(0, e["poster_frac"] * frames)))
        metrics = Metrics(example, fps)
        proc = None
        sheet = []
        step_ms = []
        error = None
        stop_reason = None
        t0 = time.perf_counter()
        mp4 = self.out_dir / "clip_raw.mp4"
        f = -1
        try:
            for f in range(frames):
                wp.synchronize_device()
                ts = time.perf_counter()
                example.step()
                wp.synchronize_device()
                step_ms.append((time.perf_counter() - ts) * 1000)
                example.render()
                need = (f % skip == 0) or f in sheet_at or f == poster_at
                if need:
                    img = viewer.get_frame().numpy()
                    if proc is None:
                        h, w = img.shape[:2]
                        proc = subprocess.Popen(
                            [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-framerate", f"{vfps:g}", "-i", "-",
                             "-vf", "scale=854:-2:flags=lanczos", "-c:v", "libx264", "-preset", "medium",
                             "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                            stdin=subprocess.PIPE,
                        )  # fmt: skip
                    if f % skip == 0:
                        proc.stdin.write(img.tobytes())
                    if f in sheet_at:
                        sheet.append((f, Image.fromarray(img).resize((320, 180), Image.LANCZOS)))
                    if f == poster_at:
                        Image.fromarray(img).resize((854, 480), Image.LANCZOS).save(self.out_dir / "poster.jpg", quality=82)
                metrics.update(f, STEEL_SOLVERS)
                if metrics.nonfinite_frame is not None and f > metrics.nonfinite_frame + 30:
                    stop_reason = "non-finite state"
                    break
                if time.perf_counter() - t0 > self.wall_budget:
                    stop_reason = f"wall budget {self.wall_budget:.0f}s reached"
                    break
        except Exception as ex:
            error = "".join(traceback.format_exception_only(type(ex), ex)).strip()[-600:]
            traceback.print_exc()
        if proc is not None:
            proc.stdin.close()
            if proc.wait() != 0:
                error = error or f"ffmpeg exited {proc.returncode}"
        if sheet:
            W = Image.new("RGB", (4 * 320, 3 * 180), (0, 0, 0))
            dr = ImageDraw.Draw(W)
            for i, (fr, im) in enumerate(sheet):
                x, y = (i % 4) * 320, (i // 4) * 180
                W.paste(im, (x, y))
                dr.rectangle([x, y, x + 64, y + 16], fill=(0, 0, 0))
                dr.text((x + 3, y + 2), f"t={fr / fps:.2f}s", fill=(255, 255, 0))
            W.save(self.out_dir / "sheet.jpg", quality=85)
        st = np.array(step_ms) if step_ms else np.zeros(1)
        warm = st[min(len(st) - 1, 5) :]  # the first steps include CUDA graph capture and JIT
        self.result = {
            "fps": fps,
            "frames_planned": frames,
            "frames_done": f + 1 if error is None else max(f, 0),
            "video_fps": round(vfps, 3),
            "video_speed": speed,
            "wall_s": round(time.perf_counter() - t0, 1),
            "step_ms_mean": round(float(warm.mean()), 3),
            "step_ms_p95": round(float(np.percentile(warm, 95)), 3),
            "step_ms_first": round(float(st[0]), 1),
            "error": error,
            "stop_reason": stop_reason,
            "metrics": metrics.summary(STEEL_SOLVERS),
        }
        try:
            viewer.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wall-budget", type=float, default=2400)
    p.add_argument("--no-headless-arg", action="store_true", help="for examples that swap to ViewerNull under --headless")
    p.add_argument("--attempt", default="", help="render_one.sh's id for this run, echoed into metrics.json")
    a = p.parse_args()
    entry = next(e for e in json.loads(a.plan.read_text())["render"] if e["scene"] == a.scene)
    out_dir = a.out / a.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in ("clip_raw.mp4", "poster.jpg", "sheet.jpg", "metrics.json"):
        (out_dir / old).unlink(missing_ok=True)

    import warp as wp

    wp.config.quiet = True
    _patch_steel()
    from common import cli, modes, presets, thumbs

    path = cli.find(entry["scene"])
    committed = select_committed(entry["preset"])
    data = presets.read(path)
    table = data.get(entry["preset"], {})
    preset_settings = {k: table[k] for k in ("sim", "collision", "solver", "model", "entries") if k in table}
    # The check's scene arguments (world count, sizes) apply; its --set overrides cheapen the solver for a
    # short test, so the render keeps the preset as presets.toml has it and only records them.
    extra, check_overrides = thumbs._split_check_args(cli.check_args(data, entry["preset"]))
    overrides = {}
    os.environ["SHELF_SOLVER"] = entry["preset"]
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    loop = RenderLoop(entry, out_dir, ffmpeg, a.wall_budget)
    t0 = time.perf_counter()
    build_error = None
    try:
        modes.run_example(path, ["--viewer", "gl", *([] if a.no_headless_arg else ["--headless"]), "--quiet", *extra], loop=loop, look="steel")
    except BaseException as ex:  # SystemExit from an example counts as a failed build too
        build_error = "".join(traceback.format_exception_only(type(ex), ex)).strip()[-600:]
        traceback.print_exc()
    res = dict(loop.result)
    res.update({
        "scene": entry["scene"], "preset": entry["preset"], "attempt": a.attempt, "args": extra, "overrides": overrides,
        "env_overrides_cleared": committed, "preset_settings": preset_settings,
        "settings_match_plan": preset_settings == entry["settings"], "check_overrides_not_applied": check_overrides,
        "build_error": build_error if not loop.ran else None,
        "loop_ran": loop.ran, "total_wall_s": round(time.perf_counter() - t0, 1),
        "newton_commit": cli.newton_commit(), "gpu": cli.gpu_name(),
    })  # fmt: skip
    if loop.ran and build_error and not res.get("error"):
        res["error"] = build_error
    if not res["settings_match_plan"] and not res.get("error"):
        res["error"] = "presets.toml no longer matches plan.json for this preset; plan again"
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    ok = loop.ran and not res.get("error")
    print("RENDERED" if ok else "RENDER_FAILED", entry["scene"], flush=True)
    sys.exit(0 if ok else 1)


def select_committed(preset: str) -> dict:
    """Select ``preset`` exactly as presets.toml has it: drop the environment overrides the loader would apply."""
    from common import presets

    cleared = {k: os.environ.pop(k) for k in ("SHELF_SET", "SHELF_PRESETS") if k in os.environ}
    presets.select(preset, {})
    return cleared


if __name__ == "__main__":
    main()
