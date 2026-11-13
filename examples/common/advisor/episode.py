# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Episode probe: run a scene for its whole episode and record the facts the stability rules need.

Numpy on host copies, one sample per frame, no torch: it is meant for scenes of one to a few hundred worlds
(these examples, a standalone Newton script), not 16k-world training runs. On top of the task-health
probe (``health.py``) it records:

- the start state: whether ``state.body_q`` / ``body_qd`` agree with forward kinematics of ``joint_q`` /
  ``joint_qd``. Steel integrates joint coordinates, so a pose or velocity written only to the body arrays is lost;
- Steel's row high-water marks and overflow counters per row family (``row_watermark=True``), the capacity
  check, and the contact high-water mark against the contact buffer;
- contact islands: free bodies joined by contacts, with the heaviest-to-lightest mass ratio of any island
  (a heavy body resting on light ones is the slow case for Gauss-Seidel);
- per-frame penetration (peak and the median of the second half), speeds, joint-limit excess;
- the measured frame time with the example's own stepping (CUDA graphs when it captures them).
"""

from __future__ import annotations

import time

import numpy as np

from .health import HealthProbe
from .model import JOINT_NAMES, _ModelView, _torsion_facts

FAMILIES = ("articulation", "free_body", "propagation")
CAP_KEYS = {
    "articulation": "articulation_constraint_max",
    "free_body": "free_body_constraint_max",
    "propagation": "propagation_constraint_max",
}
LIMITED_JOINT_TYPES = (0, 1, 6)  # prismatic, revolute, D6: one coordinate per DOF
START_POS_TOL, START_ROT_TOL, START_VEL_TOL = 1e-4, 1e-3, 1e-3  # m, rad, m/s or rad/s


def solver_capacities(solver) -> dict:
    """The constructed row capacities (propagation resolved to its effective value)."""
    out = {}
    for fam, key in CAP_KEYS.items():
        for name in (key, "_" + key):
            v = getattr(solver, name, None)
            if isinstance(v, (int, np.integer)):
                out[fam] = int(v)
                break
    if "propagation" not in out and {"articulation", "free_body"} <= out.keys():
        out["propagation"] = out["articulation"] + out["free_body"]  # SolverSteel's default when unset
    return out


def solver_structure(solver) -> dict:
    """Loop closures, mimic rows and the paths that decide where they are solved (private fields, read defensively)."""
    g = lambda name, default=None: getattr(solver, name, default)  # noqa: E731
    return {
        "connect_rows_joints": int(g("_connect_count", 0) or 0),
        "mimic_constraints": int(g("_mimic_count", 0) or 0),
        "propagation": bool(g("_propagation_active", False)),
        "solve_mode": g("_solve_mode"),
        "contact_torsion": bool(g("_contact_torsion_enabled", False)),
        "contact_torsion_device": g("_device_torsion") is not None,
        "friction_patches": bool(g("_friction_anchors_enabled", False)),
    }


def start_state_facts(model, state) -> dict:
    """Compare the body arrays of ``state`` with forward kinematics of its joint coordinates."""
    import newton
    import warp as wp

    out = {"checked": False}
    if int(model.joint_count) == 0 or state is None or getattr(state, "joint_q", None) is None:
        return out
    fk = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, fk)
    wp.synchronize_device()
    bq, fq = state.body_q.numpy(), fk.body_q.numpy()
    bqd, fqd = state.body_qd.numpy(), fk.body_qd.numpy()
    if not (np.isfinite(bq).all() and np.isfinite(fq).all()):
        return {"checked": True, "nonfinite": True}
    m = _ModelView(model)
    jointed = np.zeros(int(model.body_count), dtype=bool)
    jointed[m.joint_child[m.joint_child >= 0]] = True
    dp = np.linalg.norm(bq[:, :3] - fq[:, :3], axis=1)
    dot = np.abs((bq[:, 3:7] * fq[:, 3:7]).sum(1)).clip(max=1.0)
    dr = 2 * np.arccos(dot)
    dv = np.linalg.norm(bqd - fqd, axis=1)
    pose = jointed & ((dp > START_POS_TOL) | (dr > START_ROT_TOL))
    vel = jointed & (dv > START_VEL_TOL)
    # Velocity written to body_qd only: the joint coordinates are at rest while the bodies are not.
    body_only_vel = vel & (np.linalg.norm(fqd, axis=1) < START_VEL_TOL)
    lab = lambda b: m.short(m.body_labels, int(b))  # noqa: E731
    return {
        "checked": True,
        "bodies": int(jointed.sum()),
        "pose_mismatch_bodies": int(pose.sum()),
        "pose_mismatch_max_m": float(dp[jointed].max(initial=0.0)),
        "pose_mismatch_max_rad": float(dr[jointed].max(initial=0.0)),
        "velocity_mismatch_bodies": int(vel.sum()),
        "velocity_mismatch_max": float(dv[jointed].max(initial=0.0)),
        "velocity_in_body_qd_only_bodies": int(body_only_vel.sum()),
        "examples": [lab(b) for b in np.nonzero(pose | vel)[0][:6]],
    }


class EpisodeProbe:
    """Call :meth:`update` once per frame after stepping; :meth:`finish` returns the episode facts."""

    def __init__(self, model, solver, substep_dt: float, collision_dt: float, match_threshold: float | None):
        self.model, self.solver = model, solver
        self.health = HealthProbe(model, solver, substep_dt, collision_dt, match_threshold)
        self.m = self.health.m
        mass = model.body_mass.numpy() if getattr(model, "body_mass", None) is not None else np.zeros(model.body_count)
        flags = (
            model.body_flags.numpy() if getattr(model, "body_flags", None) is not None else np.zeros(model.body_count)
        )
        import newton

        kinematic = (flags & int(newton.BodyFlags.KINEMATIC)) != 0
        self.mass = np.where(kinematic | (mass <= 0) | self.m.fixed_root, 0.0, mass)  # 0: infinite (no response)
        self.island_ratio_max, self.island_pair = 1.0, (-1, -1)
        self.pair_ratio_max = 1.0
        self.pen_series: list[float] = []
        self._pen_max: dict[str, float] = {}
        self._first_pen: dict[str, float] | None = None
        self.contact_count_max = 0
        self.limit_excess_max, self.limit_excess_joint = 0.0, -1
        self.speed_max, self.ang_speed_max = 0.0, 0.0
        self.frame_ms: list[float] = []
        self._limited = self._limited_coords()
        self.loop_gap_max, self.loop_gap_joint = 0.0, -1
        self._loops = self._loop_joints()
        self._prev_bq = None
        self._frozen_run = True
        self.initial_speed_max = 0.0
        self.unchanged_frames = 0  # frames, from the second on, that repeat the previous frame's poses exactly
        self.sank: set[int] = set()
        self.sank_depth_max, self.sank_first_frame = 0.0, None
        self._boxes = self._static_boxes()
        self._inside0 = None

    def _limited_coords(self):
        """(q index, lower, upper, joint) per limited coordinate: the DOFs ``_limited_dofs`` counts.

        Prismatic, revolute and D6 joints have one coordinate per DOF, so DOF k of joint j is coordinate
        ``joint_q_start[j] + k``."""
        m = self.m
        if m.lower is None or getattr(self.model, "joint_q_start", None) is None:
            return None
        q_start = self.model.joint_q_start.numpy()
        qi, dofs, joints = [], [], []
        for j in range(len(m.joint_type)):
            if int(m.joint_type[j]) not in LIMITED_JOINT_TYPES or q_start[j + 1] - q_start[j] != m.joint_dofs[j]:
                continue
            for k in range(int(m.joint_dofs[j])):
                d = int(m.joint_qd_start[j]) + k
                if np.isfinite(m.lower[d]) and np.isfinite(m.upper[d]) and m.upper[d] - m.lower[d] < 1e5:
                    qi.append(int(q_start[j]) + k)
                    dofs.append(d)
                    joints.append(j)
        if not qi:
            return None
        return np.asarray(qi), m.lower[dofs], m.upper[dofs], np.asarray(joints)

    def observe_initial(self, state) -> None:
        """Record the state as built, before the first step: the containment and frozen-state baselines."""
        bq = state.body_q.numpy()
        if not np.isfinite(bq).all():
            return
        bqd = state.body_qd.numpy()
        dyn = self.mass > 0
        if dyn.any() and np.isfinite(bqd).all():
            self.initial_speed_max = float(np.abs(bqd[dyn]).max())
        if self._boxes is not None:
            self._inside0 = self._inside(bq)[0]

    def _static_boxes(self):
        """Colliding static boxes (decks, trays, ramps, the ground as a box) and which dynamic bodies they bound."""
        import newton

        m, model = self.m, self.model
        dyn = np.flatnonzero(self.mass > 0)
        if not len(dyn) or m.shape_type is None:
            return None
        ids = np.flatnonzero((m.shape_type == int(newton.GeoType.BOX)) & (m.shape_body == -1) & m.collide)
        if not len(ids):
            return None
        xf = model.shape_transform.numpy()[ids].astype(np.float64)
        half = model.shape_scale.numpy()[ids, :3].astype(np.float64)
        sw = m.shape_world[ids]
        bw = m.body_world[dyn]
        # A box bounds the bodies it can collide with: global boxes, global bodies, or the same world.
        eligible = (sw[:, None] == -1) | (bw[None, :] == -1) | (sw[:, None] == bw[None, :])
        com = model.body_com.numpy().astype(np.float64)[dyn]
        return dyn, xf, half, eligible, com

    def _inside(self, bq: np.ndarray):
        """Per dynamic body: is its centre of mass inside an eligible static box, and how deep."""
        dyn, xf, half, eligible, com = self._boxes
        q = bq[dyn].astype(np.float64)
        comw = q[:, :3] + _rotate(q[:, 3:7], com)
        inside, depth = np.zeros(len(dyn), bool), np.zeros(len(dyn))
        for k in range(len(xf)):
            qinv = np.repeat(xf[k : k + 1, 3:7] * np.array([-1, -1, -1, 1.0]), len(dyn), 0)
            local = _rotate(qinv, comw - xf[k, :3])
            d = np.min(half[k] - np.abs(local), axis=1)
            hit = eligible[k] & np.isfinite(d) & (d > 0.003)
            inside |= hit
            depth = np.maximum(depth, np.where(hit, d, 0.0))
        return inside, depth

    def _loop_joints(self):
        """(joints, parent, child, parent anchor, child anchor) of loop-closing joints, from Steel's plan."""
        plan = getattr(self.solver, "_model_plan", None)
        art = getattr(plan, "loop_joint_articulation", None)
        if art is None or getattr(self.model, "joint_X_p", None) is None:
            return None
        js = np.flatnonzero(np.asarray(art) >= 0)
        if js.size == 0:
            return None
        xp, xc = self.model.joint_X_p.numpy()[js, :3], self.model.joint_X_c.numpy()[js, :3]
        return js, self.m.joint_parent[js], self.m.joint_child[js], xp, xc

    def update(self, state, contacts, frame_ms: float | None = None) -> None:
        if frame_ms is not None:
            self.frame_ms.append(frame_ms)
        self.health.update(state, contacts)
        if self._first_pen is None:
            self._first_pen = dict(self.health.pen)
        bqd = state.body_qd.numpy()
        if np.isfinite(bqd).all():
            dyn = self.mass > 0
            if dyn.any():
                self.speed_max = max(self.speed_max, float(np.linalg.norm(bqd[dyn, :3], axis=1).max()))
                self.ang_speed_max = max(self.ang_speed_max, float(np.linalg.norm(bqd[dyn, 3:], axis=1).max()))
        if self._limited is not None and getattr(state, "joint_q", None) is not None:
            qi, lo, hi, js = self._limited
            q = state.joint_q.numpy()[qi]
            if np.isfinite(q).all():
                ex = np.maximum(lo - q, q - hi)
                i = int(np.argmax(ex))
                if ex[i] > self.limit_excess_max:
                    self.limit_excess_max, self.limit_excess_joint = float(ex[i]), int(js[i])
        bq_now = state.body_q.numpy()
        # Frozen from the start: every frame since the first repeats the one before it (late settling does not count).
        if self._prev_bq is not None and self._frozen_run and (self.mass > 0).any():
            if np.array_equal(bq_now, self._prev_bq):
                self.unchanged_frames += 1
            else:
                self._frozen_run = False
        self._prev_bq = bq_now.copy()
        if self._boxes is not None:
            bq0 = bq_now
            if np.isfinite(bq0).all():
                inside, depth = self._inside(bq0)
                if self._inside0 is None:
                    self._inside0 = inside  # no observe_initial(): the first observed frame is the baseline
                new = inside & ~self._inside0
                if new.any():
                    self.sank.update(self._boxes[0][new].tolist())
                    self.sank_depth_max = max(self.sank_depth_max, float(depth[new].max()))
                    if self.sank_first_frame is None:
                        self.sank_first_frame = self.health.frames
        if self._loops is not None:
            js, par, chi, xp, xc = self._loops
            bq = state.body_q.numpy()
            if np.isfinite(bq).all():
                a = np.where(par[:, None] >= 0, _point(bq, par, xp), xp)
                b = _point(bq, chi, xc)
                gap = np.linalg.norm(a - b, axis=1)
                i = int(np.argmax(gap))
                if gap[i] > self.loop_gap_max:
                    self.loop_gap_max, self.loop_gap_joint = float(gap[i]), int(js[i])
        if contacts is None:
            return
        n = int(contacts.rigid_contact_count.numpy()[0])
        self.contact_count_max = max(self.contact_count_max, n)
        pen = self.health.pen
        self.pen_series.append(max((v for k, v in pen.items() if k != "support_support"), default=0.0))
        for k, v in pen.items():  # the running max, kept here because the health probe's dict is cleared per frame
            self._pen_max[k] = max(self._pen_max.get(k, 0.0), v)
        pen.clear()
        if n == 0:
            return
        s0 = contacts.rigid_contact_shape0.numpy()[:n]
        s1 = contacts.rigid_contact_shape1.numpy()[:n]
        ok = (s0 >= 0) & (s1 >= 0)
        b0, b1 = self.m.shape_body[s0[ok]], self.m.shape_body[s1[ok]]
        both = (b0 >= 0) & (b1 >= 0)
        b0, b1 = b0[both], b1[both]
        m0, m1 = self.mass[b0], self.mass[b1]
        # Free bodies only: a robot link's response comes from its whole articulation, and grasps have their own rules.
        dyn = (m0 > 0) & (m1 > 0) & self.health.free[b0] & self.health.free[b1]
        if not dyn.any():
            return
        b0, b1, m0, m1 = b0[dyn], b1[dyn], m0[dyn], m1[dyn]
        self.pair_ratio_max = max(self.pair_ratio_max, float((np.maximum(m0, m1) / np.minimum(m0, m1)).max()))
        ratio, pair = _island_ratio(b0, b1, self.mass)
        if ratio > self.island_ratio_max:
            self.island_ratio_max, self.island_pair = ratio, pair

    def finish(self) -> dict:
        h = self.health.finish()
        h["penetration_max_by_class"] = dict(self._pen_max)
        h["first_frame_penetration_by_class"] = self._first_pen or {}
        m = self.m
        lab = lambda b: m.short(m.body_labels, int(b)) if b >= 0 else "none"  # noqa: E731
        pen = np.array(self.pen_series) if self.pen_series else np.zeros(1)
        ft = np.array(self.frame_ms[5:] if len(self.frame_ms) > 10 else self.frame_ms or [0.0])
        out = {
            "health": h,
            "frames": h["frames"],
            "island_mass_ratio_max": round(self.island_ratio_max, 2),
            "island_mass_ratio_bodies": [lab(self.island_pair[0]), lab(self.island_pair[1])],
            "pair_mass_ratio_max": round(self.pair_ratio_max, 2),
            "penetration_peak_m": float(pen.max()),
            "penetration_sustained_m": float(np.median(pen[len(pen) // 2 :])),
            "penetration_p95_m": float(np.percentile(pen, 95)),
            "body_speed_max": self.speed_max,
            "body_ang_speed_max": self.ang_speed_max,
            "joint_limit_excess_max": self.limit_excess_max,
            "joint_limit_excess_joint": (
                m.short(m.joint_labels, self.limit_excess_joint) if self.limit_excess_joint >= 0 else None
            ),
            "loop_gap_max_m": self.loop_gap_max if self._loops is not None else None,
            "loop_gap_joint": m.short(m.joint_labels, self.loop_gap_joint) if self.loop_gap_joint >= 0 else None,
            "unchanged_frames": self.unchanged_frames,
            "initial_speed_max": self.initial_speed_max,
            "bodies_inside_static": len(self.sank),
            "inside_static_depth_max_m": self.sank_depth_max,
            "inside_static_first_frame": self.sank_first_frame,
            "contact_count_max": self.contact_count_max,
            "rigid_contact_max": int(getattr(self.model, "rigid_contact_max", 0) or 0),
            "frame_ms_mean": float(ft.mean()),
            "frame_ms_median": float(np.median(ft)),
        }
        out["capacities"] = solver_capacities(self.solver)
        out["structure"] = solver_structure(self.solver)
        try:
            out["row_watermarks"] = self.solver.constraint_row_watermarks()
        except Exception as exc:  # noqa: BLE001 - row_watermark=False or an older Newton
            out["row_watermarks"] = f"{type(exc).__name__}: {exc}"
        try:
            self.solver.check_constraint_capacity()
            out["capacity_check"] = "ok"
        except Exception as exc:  # noqa: BLE001 - the check raises RuntimeError with the reason
            out["capacity_check"] = str(exc)[:400]
        return out


def model_facts(model, solver) -> dict:
    """Static facts the stability rules read: bodies, masses, joints, torsion radii."""
    m = _ModelView(model)
    mass = model.body_mass.numpy() if getattr(model, "body_mass", None) is not None else np.zeros(model.body_count)
    dyn = (mass > 0) & ~m.fixed_root
    w0 = m.body_in_w0 & dyn
    jt = [JOINT_NAMES.get(int(t), str(t)) for t in m.joint_type[m.joint_in_w0]]
    free_w0 = m.free_body_w0 & dyn
    fm = mass[free_w0]
    return {
        "world_count": m.W,
        "bodies_world0": int(w0.sum()),
        "free_bodies_world0": int(free_w0.sum()),
        "articulations_world0": len(m.arts_w0),
        "articulated_dofs_world0": int(
            sum(m.joint_dofs[j] for j in np.nonzero(m.joint_in_w0)[0] if not m.free_body[m.joint_child[j]])
        ),
        "joint_types_world0": {t: jt.count(t) for t in sorted(set(jt))},
        "driven_dofs_world0": int(((m.ke if m.ke is not None else np.zeros(0)) > 0)[m.dof_in_w0[: len(m.ke)]].sum())
        if m.ke is not None
        else 0,
        "limited_dofs_world0": _limited_dofs(m),
        "free_mass_ratio_world0": float(fm.max() / fm.min()) if len(fm) > 1 and fm.min() > 0 else 1.0,
        "torsion": _torsion_facts(model, m),
    }


def _limited_dofs(m: _ModelView) -> int:
    """World-0 DOFs of prismatic, revolute and D6 joints with a finite range."""
    if m.lower is None:
        return 0
    rng = m.upper - m.lower
    finite = np.isfinite(m.lower) & np.isfinite(m.upper) & (rng < 1e5)
    jt = m.joint_type[m.dof_joint]
    return int((finite & m.dof_in_w0 & np.isin(jt, LIMITED_JOINT_TYPES)).sum())


def _island_ratio(b0: np.ndarray, b1: np.ndarray, mass: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Largest heaviest/lightest mass ratio of any island of contact-connected dynamic bodies."""
    nodes, inv = np.unique(np.concatenate([b0, b1]), return_inverse=True)
    parent = np.arange(len(nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n = len(b0)
    for a, b in zip(inv[:n], inv[n:], strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(nodes))])
    best, pair = 1.0, (-1, -1)
    mn = mass[nodes]
    for r in np.unique(roots):
        sel = roots == r
        if sel.sum() < 2:
            continue
        hi, lo = int(np.argmax(np.where(sel, mn, -np.inf))), int(np.argmin(np.where(sel, mn, np.inf)))
        ratio = float(mn[hi] / mn[lo])
        if ratio > best:
            best, pair = ratio, (int(nodes[hi]), int(nodes[lo]))
    return best, pair


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (n, 3) by quaternions ``q`` (n, 4, xyzw)."""
    u, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _point(bq: np.ndarray, body: np.ndarray, p: np.ndarray) -> np.ndarray:
    """World positions of body-frame points ``p`` on bodies ``body`` (body_q rows are p, q xyzw)."""
    t = bq[np.maximum(body, 0)]
    u, w = t[:, 3:6], t[:, 6:7]
    rot = (
        2.0 * (u * p).sum(1, keepdims=True) * u + (w * w - (u * u).sum(1, keepdims=True)) * p + 2.0 * w * np.cross(u, p)
    )
    return t[:, :3] + rot


def time_step(step) -> float:
    """Wall time of one ``step()`` call in ms, synchronized on both sides."""
    import warp as wp

    wp.synchronize_device()
    t = time.perf_counter()
    step()
    wp.synchronize_device()
    return 1e3 * (time.perf_counter() - t)
