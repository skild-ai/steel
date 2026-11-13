# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Task-health probe: contact heuristics a throughput probe cannot see.

Runs over a whole scripted episode (or a long random-action window) and records, per frame:

- non-finite state: any body pose, orientation or velocity that is not finite. Such a frame makes every other
  metric unknown, and the report says so instead of treating it as zero;
- warm-start matching: the share of contacts that found their previous-step partner, by health pair class, when
  ``contact_warmstart`` is on, overall and for a cohort of contacts on fast bodies (linear speed x collision
  interval above the pipeline's ``contact_matching_pos_threshold``). The cohort is a heuristic: Newton compares
  contact midpoints, which rotation moves and COM speed does not measure;
- travel against the detection margin: per free body, ``|v| x substep dt`` against its smallest shape gap plus the
  smallest static gap. A conservative translational indicator, not measured contact travel or proof of tunnelling;
- deep penetration of objects into static or fixed supports, against the object's thinnest half-extent (capsule
  radius, sphere radius or box half-size; hull-only bodies are listed as unknown);
- velocity bursts: a free body whose linear speed jumps by more than ``BURST_DV`` in one frame (all occurrences
  are counted; a few are kept as examples);
- grasp slip: the pose drift of each object relative to the actuated link it stays in contact with, counted only
  while the object touches no static or fixed support.

Roles are the probe's own: ``robot`` is a link of an articulation with a driven DOF (``ke > 0``), ``support`` is a
static shape or a zero-DOF link welded to the world, and ``object`` is every other body (free bodies and passive
articulated objects alike). Everything is numpy on host copies, one sample per frame, so it is meant for the shelf
(one to 64 worlds), not training runs of thousands of worlds. An empty flag list is not task acceptance.
"""

from __future__ import annotations

import collections

import numpy as np

from .model import _ModelView

BURST_DV = 2.0  # m/s gained in one frame by a free body
SLIP_MM, SLIP_DEG = 10.0, 10.0  # grasp-slip warn thresholds
HOLD_FRAMES = 30  # a robot-object contact streak shorter than this is a touch, not a hold
BURST_EXAMPLES = 5

ROLE_SUPPORT, ROLE_OBJECT, ROLE_ROBOT = 0, 1, 2
CLASSES = ("robot_robot", "robot_object", "robot_support", "object_object", "object_support", "support_support")


class HealthProbe:
    """Call :meth:`update` once per frame after stepping; :meth:`finish` returns the facts dict."""

    def __init__(self, model, solver, substep_dt: float, collision_dt: float, match_threshold: float | None):
        self.m = _ModelView(model)
        self.model, self.solver = model, solver
        self.substep_dt, self.collision_dt = float(substep_dt), float(collision_dt)
        self.match_threshold = match_threshold
        m = self.m
        nb = int(model.body_count)
        gap = m.shape_gap if m.shape_gap is not None else np.zeros(len(m.shape_body))
        body = m.shape_body
        static_gap = float(gap[body < 0].min()) if (body < 0).any() else 0.0  # the thinnest static margin
        self.body_gap = np.full(nb, np.inf)
        np.minimum.at(self.body_gap, body[body >= 0], gap[body >= 0])
        self.body_margin_static = self.body_gap + static_gap
        scale = model.shape_scale.numpy()
        stype = m.shape_type
        # A body's thinnest half-extent: the depth at which an overlap passes the shape's middle.
        half = np.where(np.isin(stype, (3, 4)), scale[:, 0], scale.min(axis=1))  # sphere/capsule radius, else min
        half = np.where(np.isin(stype, (7, 3, 4)), half, np.nan)  # boxes, spheres, capsules; hulls are unknown
        self.body_half = np.full(nb, np.inf)
        ok = (body >= 0) & np.isfinite(half)
        np.minimum.at(self.body_half, body[ok], half[ok])
        self.free = m.free_body.copy()
        driven = np.zeros(nb, dtype=bool)
        ke = m.ke if m.ke is not None else np.zeros(0)
        for js in m.art_joints.values():
            dofs = [d for j in js for d in range(m.joint_qd_start[j], m.joint_qd_start[j] + m.joint_dofs[j])]
            if any(ke[d] > 0 for d in dofs if d < len(ke)):
                driven[m.joint_child[js]] = True
        role = np.full(nb + 1, ROLE_SUPPORT, dtype=np.int8)  # index nb stands for a static shape (body -1)
        role[:nb] = ROLE_OBJECT
        role[:nb][driven & ~self.free] = ROLE_ROBOT
        role[:nb][m.fixed_root] = ROLE_SUPPORT
        self.role = role
        self.frames = 0
        self.nonfinite_frames = 0
        self.nonfinite_first: tuple[int, int] | None = None  # (frame, body)
        self.nonfinite_worlds: set[int] = set()
        self.unobserved_contact_frames = 0
        self.invalid_contact_frames = 0
        self.invalid_contacts = 0
        self.invalid_contact_first: tuple[int, int, int, int] | None = None  # (frame, shape0, shape1, body)
        self.match = collections.defaultdict(lambda: [0, 0])  # class -> [matched, total]
        self.match_fast = collections.defaultdict(lambda: [0, 0])
        self.travel_ratio_max, self.travel_body, self.travel_over = 0.0, -1, 0
        self.speed_max, self.speed_body = 0.0, -1
        self.pen = collections.defaultdict(float)
        self.pen_over_half, self.pen_over_half_body = 0.0, -1
        self.pen_unknown_bodies: set[int] = set()
        self.bursts = 0
        self.burst_worlds: set[int] = set()
        self.burst_examples: list[tuple[int, int, float]] = []
        self.prev_speed = None
        self.streak: dict[tuple[int, int], list] = {}
        self.slip = []  # (robot body, object body, frames held, max mm, max deg)

    def update(self, state, contacts) -> None:
        self.frames += 1
        f = self.frames
        bq = state.body_q.numpy()
        bqd = state.body_qd.numpy()
        finite = np.isfinite(bq).all(axis=1) & np.isfinite(bqd).all(axis=1)
        if not finite.all():
            # Every derived metric of this frame is unknown; record it and skip the frame.
            bad = np.nonzero(~finite)[0]
            self.nonfinite_frames += 1
            if self.nonfinite_first is None:
                self.nonfinite_first = (f, int(bad[0]))
            self.nonfinite_worlds.update(int(w) for w in np.maximum(self.m.body_world[bad], 0))
            self.prev_speed = None
            self._close_streaks(set(), f)
            return
        speed = np.linalg.norm(bqd[:, :3], axis=1)
        free = self.free
        if free.any():
            s = np.where(free, speed, 0.0)
            b = int(np.argmax(s))
            if s[b] > self.speed_max:
                self.speed_max, self.speed_body = float(s[b]), b
            ratio = s * self.substep_dt / np.maximum(self.body_margin_static, 1e-9)
            r = int(np.argmax(ratio))
            self.travel_over = max(self.travel_over, int((ratio > 1.0).sum()))
            if ratio[r] > self.travel_ratio_max:
                self.travel_ratio_max, self.travel_body = float(ratio[r]), r
            if self.prev_speed is not None:
                jump = np.where(free, speed - self.prev_speed, 0.0)
                hit = np.nonzero(jump > BURST_DV)[0]
                self.bursts += len(hit)
                self.burst_worlds.update(int(w) for w in np.maximum(self.m.body_world[hit], 0))
                for b in hit[: max(BURST_EXAMPLES - len(self.burst_examples), 0)]:
                    self.burst_examples.append((f, int(b), float(jump[b])))
        self.prev_speed = speed
        if contacts is None:
            # No contact data this frame: hold streaks cannot be continued across it.
            self.unobserved_contact_frames += 1
            self._close_streaks(set(), f)
            return
        n = int(contacts.rigid_contact_count.numpy()[0])
        s0 = contacts.rigid_contact_shape0.numpy()[:n]
        s1 = contacts.rigid_contact_shape1.numpy()[:n]
        ok = (s0 >= 0) & (s1 >= 0)
        if not ok.any():
            self._close_streaks(set(), f)
            return
        s0, s1 = s0[ok], s1[ok]
        b0, b1 = self.m.shape_body[s0], self.m.shape_body[s1]
        nb = len(self.free)
        r0, r1 = self.role[np.where(b0 >= 0, b0, nb)], self.role[np.where(b1 >= 0, b1, nb)]
        cls = _classes(r0, r1)
        mi = getattr(contacts, "rigid_contact_match_index", None)
        mi = None if mi is None else mi.numpy()[:n][ok]
        p0 = contacts.rigid_contact_point0.numpy()[:n][ok]
        p1 = contacts.rigid_contact_point1.numpy()[:n][ok]
        nrm = contacts.rigid_contact_normal.numpy()[:n][ok]
        m0 = contacts.rigid_contact_margin0.numpy()[:n][ok]
        m1 = contacts.rigid_contact_margin1.numpy()[:n][ok]
        gap = ((_world(bq, b1, p1) - _world(bq, b0, p0)) * nrm).sum(1) - m0 - m1
        valid = np.isfinite(gap) & np.isfinite(nrm).all(axis=1)
        if not valid.all():
            # Invalid contact geometry: record it, drop those contacts, and end every streak (holds are unknown).
            bad = int(np.nonzero(~valid)[0][0])
            self.invalid_contact_frames += 1
            self.invalid_contacts += int((~valid).sum())
            if self.invalid_contact_first is None:
                body = int(b0[bad] if b0[bad] >= 0 else b1[bad])
                self.invalid_contact_first = (f, int(s0[bad]), int(s1[bad]), body)
            self._close_streaks(set(), f)
            s0, s1, b0, b1, r0, r1, cls = s0[valid], s1[valid], b0[valid], b1[valid], r0[valid], r1[valid], cls[valid]
            gap = gap[valid]
            if mi is not None:
                mi = mi[valid]
            if not valid.any():
                return
        if mi is not None:
            fast = None
            if self.match_threshold:
                v0 = np.where(b0 >= 0, speed[np.maximum(b0, 0)], 0.0)
                v1 = np.where(b1 >= 0, speed[np.maximum(b1, 0)], 0.0)
                fast = np.maximum(v0, v1) * self.collision_dt > self.match_threshold
            for k in np.unique(cls):
                sel = cls == k
                rec = self.match[CLASSES[k]]
                rec[0] += int((mi[sel] >= 0).sum())
                rec[1] += int(sel.sum())
                if fast is not None:
                    rec = self.match_fast[CLASSES[k]]
                    rec[0] += int((mi[sel & fast] >= 0).sum())
                    rec[1] += int((sel & fast).sum())
        depth = np.clip(-gap, 0.0, None)
        for k in np.unique(cls):
            self.pen[CLASSES[k]] = max(self.pen[CLASSES[k]], float(depth[cls == k].max()))
        # The object side of each object-support contact, chosen by role in either shape order.
        os_ = cls == CLASSES.index("object_support")
        obj = np.where(r0 == ROLE_OBJECT, b0, b1)
        if os_.any():
            ob = obj[os_]
            half = self.body_half[ob]
            known = np.isfinite(half)
            self.pen_unknown_bodies.update(int(x) for x in np.unique(ob[~known & (depth[os_] > 0)]))
            if known.any():
                rel = depth[os_][known] / half[known]
                i = int(np.argmax(rel))
                if rel[i] > self.pen_over_half:
                    self.pen_over_half, self.pen_over_half_body = float(rel[i]), int(ob[known][i])
        grounded = set(np.unique(obj[os_]).tolist())
        ro = cls == CLASSES.index("robot_object")
        rob = np.where(r0 == ROLE_ROBOT, b0, b1)
        now = {(int(r), int(o)) for r, o in zip(rob[ro], obj[ro], strict=True)}
        for key in now:
            rel = _relative(bq[key[0]], bq[key[1]])
            if key not in self.streak:
                self.streak[key] = [f, rel, 0.0, 0.0]
            elif key[1] in grounded:
                self.streak[key][1] = rel  # re-anchor: placement or release, not slip
            else:
                st = self.streak[key]
                st[2] = max(st[2], 1e3 * float(np.linalg.norm(rel[:3] - st[1][:3])))
                st[3] = max(st[3], float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(rel[3:], st[1][3:]))))))))
        self._close_streaks(now, f)

    def _close_streaks(self, now: set, f: int) -> None:
        """End the streaks of pairs not in contact this frame (an unobserved frame ends every streak)."""
        for key in [k for k in self.streak if k not in now]:
            start, _, mm, deg = self.streak.pop(key)
            if f - start >= HOLD_FRAMES:
                self.slip.append((key[0], key[1], f - start, mm, deg))

    def finish(self) -> dict:
        self._close_streaks(set(), self.frames + 1)
        m = self.m
        lab = lambda b: m.short(m.body_labels, b) if b >= 0 else "none"  # noqa: E731
        slip = sorted(self.slip, key=lambda s: -max(s[3] / SLIP_MM, s[4] / SLIP_DEG))
        rate = lambda d: {k: {"matched_frac": v[0] / v[1] if v[1] else None, "contacts": v[1]} for k, v in d.items()}  # noqa: E731
        return {
            "frames": self.frames,
            "nonfinite_frames": self.nonfinite_frames,
            "nonfinite_first": None
            if self.nonfinite_first is None
            else {"frame": self.nonfinite_first[0], "body": lab(self.nonfinite_first[1])},
            "nonfinite_worlds": len(self.nonfinite_worlds),
            "unobserved_contact_frames": self.unobserved_contact_frames,
            "invalid_contact_frames": self.invalid_contact_frames,
            "invalid_contacts": self.invalid_contacts,
            "invalid_contact_first": None
            if self.invalid_contact_first is None
            else {
                "frame": self.invalid_contact_first[0],
                "shape0": self.invalid_contact_first[1],
                "shape1": self.invalid_contact_first[2],
                "body": lab(self.invalid_contact_first[3]),
                "world": int(max(m.body_world[self.invalid_contact_first[3]], 0))
                if self.invalid_contact_first[3] >= 0
                else -1,
            },
            "warmstart_match": rate(self.match),
            "warmstart_match_fast": rate(self.match_fast),
            "match_threshold_m": self.match_threshold,
            "free_body_speed_max": self.speed_max,
            "free_body_speed_body": lab(self.speed_body),
            "travel_over_margin_max": self.travel_ratio_max,
            "travel_over_margin_body": lab(self.travel_body),
            "travel_over_margin_bodies_max": self.travel_over,
            "margin_of_that_body_m": float(self.body_margin_static[self.travel_body])
            if self.travel_body >= 0
            else None,
            "substep_dt": self.substep_dt,
            "penetration_max_by_class": dict(self.pen),
            "object_support_penetration_over_half_thickness": self.pen_over_half,
            "object_support_penetration_body": lab(self.pen_over_half_body),
            "penetration_unknown_thickness_bodies": sorted(lab(b) for b in self.pen_unknown_bodies)[:8],
            "bursts": self.bursts,
            "burst_worlds": len(self.burst_worlds),
            "burst_examples": [(f, lab(b), round(dv, 2)) for f, b, dv in self.burst_examples],
            "grasp_holds": [
                {
                    "robot": lab(r),
                    "object": lab(o),
                    "frames": n,
                    "slip_mm": round(mm, 1),
                    "slip_deg": round(deg, 1),
                    "object_free": bool(self.free[o]),
                }
                for r, o, n, mm, deg in slip[:8]
            ],
        }


def health_flags(h: dict, solver_cfg: dict) -> list[tuple[str, str]]:
    """(level, message) flags from the health facts. Heuristics: an empty list is not task acceptance."""
    out = []
    if h.get("nonfinite_frames"):
        first = h.get("nonfinite_first") or {}
        out.append(
            (
                "critical",
                f"Non-finite state in {h['nonfinite_frames']} of {h['frames']} frames ({h['nonfinite_worlds']} "
                f"worlds), first at frame {first.get('frame')} on {first.get('body')}. Those frames are "
                "excluded, so every other health metric is incomplete.",
            )
        )
    if h.get("invalid_contact_frames"):
        first = h.get("invalid_contact_first") or {}
        out.append(
            (
                "critical",
                f"{h['invalid_contacts']} contacts with non-finite geometry (point, normal or margin) in "
                f"{h['invalid_contact_frames']} frames, first at frame {first.get('frame')}: shapes "
                f"{first.get('shape0')}-{first.get('shape1')} on {first.get('body')} (world {first.get('world')}). "
                "They are excluded and end every grasp hold, so contact metrics are incomplete.",
            )
        )
    ws = solver_cfg.get("contact_warmstart")
    ro = (h.get("warmstart_match_fast") or {}).get("robot_object") or {}
    if ws and ro.get("contacts") and (ro.get("matched_frac") or 0.0) < 0.5:
        out.append(
            (
                "warn",
                f"Warm start is on, but only {100 * ro['matched_frac']:.0f}% of robot-object contacts in the "
                f"fast-body cohort (linear speed x collision interval above the {h.get('match_threshold_m')} m "
                f"match threshold; {ro['contacts']} contact-frames) matched their previous step. Newton's "
                "world-space midpoint matching may contribute: inspect midpoint displacement, normal changes and "
                "competing matches before relying on warm start to hold the grasp or changing the threshold.",
            )
        )
    fast = {k: v for k, v in (h.get("warmstart_match_fast") or {}).items() if v.get("contacts")}
    if ws and fast:
        out.append(
            (
                "info",
                "Warm-start match rate in the fast-body cohort: "
                + ", ".join(f"{k} {100 * (v['matched_frac'] or 0):.0f}% of {v['contacts']}" for k, v in fast.items()),
            )
        )
    if (h.get("travel_over_margin_max") or 0.0) > 1.0:
        out.append(
            (
                "warn",
                f"Free body {h['travel_over_margin_body']} moves up to {h['travel_over_margin_max']:.0f}x its static "
                f"detection margin ({1e3 * (h['margin_of_that_body_m'] or 0):.2f} mm) per substep "
                f"({h['free_body_speed_max']:.2f} m/s max, linear COM speed). It can reach a surface before a "
                "contact row exists; compare with the penetration below before changing gaps or speculative "
                "contacts.",
            )
        )
    depth = h.get("object_support_penetration_over_half_thickness") or 0.0
    if depth > 1.0:
        out.append(
            (
                "critical" if depth > 1.5 else "warn",
                f"An object ({h['object_support_penetration_body']}) penetrated a static or fixed support by {depth:.1f}x "
                "its thinnest half-extent: past the middle, its contact normal can flip and push it through or wedge "
                "it between supports.",
            )
        )
    if h.get("penetration_unknown_thickness_bodies"):
        out.append(
            (
                "info",
                "Penetration of hull-only objects is not normalized (unknown thickness): "
                + ", ".join(h["penetration_unknown_thickness_bodies"]),
            )
        )
    if h.get("bursts"):
        out.append(
            (
                "critical",
                f"{h['bursts']} velocity bursts (a free body gaining > {BURST_DV} m/s in one frame) in "
                f"{h['burst_worlds']} worlds, e.g. {h['burst_examples'][:3]}. Inspect the bursting body's "
                "contacts in the frame before; the jump alone does not identify the cause.",
            )
        )
    for g in h.get("grasp_holds") or []:
        if g["slip_mm"] > SLIP_MM or g["slip_deg"] > SLIP_DEG:
            out.append(
                (
                    "warn",
                    f"Grasp slip: {g['object']} drifted {g['slip_mm']} mm / {g['slip_deg']} deg relative to "
                    f"{g['robot']} while carried over a {g['frames']}-frame hold. Check friction convergence "
                    "(iterations, friction_anchor_beta, warm-start matching) before trusting task results.",
                )
            )
            break
    if h.get("unobserved_contact_frames"):
        out.append(("info", f"{h['unobserved_contact_frames']} frames had no contact buffer; grasp holds end at each."))
    return out


def _classes(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Index into CLASSES for each pair of roles (order-invariant)."""
    lo, hi = np.minimum(r0, r1), np.maximum(r0, r1)
    out = np.full(len(r0), CLASSES.index("support_support"), dtype=np.int8)
    out[(lo == ROLE_ROBOT) & (hi == ROLE_ROBOT)] = CLASSES.index("robot_robot")
    out[(lo == ROLE_OBJECT) & (hi == ROLE_ROBOT)] = CLASSES.index("robot_object")
    out[(lo == ROLE_SUPPORT) & (hi == ROLE_ROBOT)] = CLASSES.index("robot_support")
    out[(lo == ROLE_OBJECT) & (hi == ROLE_OBJECT)] = CLASSES.index("object_object")
    out[(lo == ROLE_SUPPORT) & (hi == ROLE_OBJECT)] = CLASSES.index("object_support")
    return out


def _world(bq, body, p):
    out = p.copy()
    has = body >= 0
    t = bq[body[has]]
    q = t[:, 3:7]
    u, w = q[:, :3], q[:, 3:4]
    v = p[has]
    out[has] = (
        t[:, :3]
        + 2.0 * (u * v).sum(1, keepdims=True) * u
        + (w * w - (u * u).sum(1, keepdims=True)) * v
        + 2.0 * w * np.cross(u, v)
    )
    return out


def _relative(a, b):
    """b's pose in a's frame: (p, q xyzw)."""
    qa = a[3:7] * np.array([-1, -1, -1, 1.0])
    d = b[:3] - a[:3]
    return np.concatenate([_rot(qa, d), _mul(qa, b[3:7])])


def _rot(q, v):
    u, w = q[:3], q[3]
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def _mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
