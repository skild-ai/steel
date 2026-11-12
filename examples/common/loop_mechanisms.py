# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop linkage library: the mechanisms behind the shelf's ``loop_*`` examples.

The examples run these mechanisms through :mod:`common.loop_linkage`.

Each mechanism is described in world space at a reference pose in which every loop is closed exactly: bodies carry
their world pose (origin at the centre of mass), mass and inertia; joints carry a world anchor point and, where
needed, a world axis. Joint frames are world-aligned at the reference pose, so every joint coordinate is zero there.
Joints flagged ``cut`` are the loop-closing joints a reduced-coordinate solver enforces as constraints; the remaining
joints form a spanning tree rooted at the world. Actuated joints carry a PD drive and a target trajectory.

The same description feeds the Newton model builders (tree + loop joints, or the raw joint graph) and the
independent maximal-coordinate reference integrator in :mod:`loop_reference`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# small rotation helpers (quaternions are x, y, z, w like Warp)


def quat_from_two_vectors(a, b) -> np.ndarray:
    """Shortest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -1.0 + 1e-12:
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return np.array([perp[0], perp[1], perp[2], 0.0])
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)])


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def smooth_step(t: float, period: float) -> float:
    """0 -> 1 -> 0 over one period with zero slope at the ends: (1 - cos(2 pi t / period)) / 2."""
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period))


def ramp(t: float, t_ramp: float = 1.0) -> float:
    """Smooth 0 -> 1 ramp over ``t_ramp`` seconds."""
    return 1.0 if t >= t_ramp else 0.5 * (1.0 - math.cos(math.pi * t / t_ramp))


# ---------------------------------------------------------------------------------------------------------------------
# description types


@dataclass
class Shape:
    """Visual shape in the body frame. ``kind`` is capsule (radius, half_height, along local z), box (half extents) or sphere (radius)."""

    kind: str
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    color: tuple = (0.6, 0.6, 0.6)


@dataclass
class Body:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    mass: float
    inertia: np.ndarray
    shapes: list = field(default_factory=list)


@dataclass
class Drive:
    """PD drive on a revolute or prismatic joint: ``f = ke (q* - q) + kd (qd* - qd)``."""

    ke: float
    kd: float
    effort: float = 1.0e6


@dataclass
class Joint:
    name: str
    kind: str  # revolute | prismatic | ball | fixed
    parent: int  # body index, -1 = world
    child: int
    anchor: np.ndarray
    axis: np.ndarray | None = None
    cut: bool = False
    drive: Drive | None = None


@dataclass
class Mechanism:
    name: str
    title: str
    summary: str
    stresses: str
    bodies: list
    joints: list
    actuated: list  # joint indices, in target order
    targets: Callable[[float], tuple]  # t -> (q*, qd*) arrays over ``actuated``
    duration: float = 4.0
    tracked: list = field(default_factory=list)  # body indices whose positions are the headline error
    dof: int = 0

    def target_table(self, times) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.targets(t)[0] for t in times], dtype=np.float64)
        qd = np.array([self.targets(t)[1] for t in times], dtype=np.float64)
        return q, qd


def numeric_rate(fn: Callable[[float], np.ndarray], t: float, h: float = 1e-5) -> np.ndarray:
    return (np.asarray(fn(t + h)) - np.asarray(fn(max(t - h, 0.0)))) / (t + h - max(t - h, 0.0))


# ---------------------------------------------------------------------------------------------------------------------
# body helpers

PALETTE = {
    "base": (0.42, 0.45, 0.50),
    "blue": (0.20, 0.45, 0.80),
    "orange": (0.93, 0.55, 0.20),
    "green": (0.30, 0.68, 0.35),
    "red": (0.85, 0.30, 0.28),
    "purple": (0.55, 0.40, 0.75),
    "grey": (0.70, 0.72, 0.75),
    "gold": (0.85, 0.70, 0.25),
}


def rod_body(name, p0, p1, radius, mass, color="blue", extra_shapes=()) -> Body:
    """Slender rod from ``p0`` to ``p1``: COM at the midpoint, local z along the rod."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    q = quat_from_two_vectors([0.0, 0.0, 1.0], d)
    ixx = mass * (3.0 * radius * radius + length * length) / 12.0
    izz = 0.5 * mass * radius * radius
    shapes = [Shape("capsule", (radius, 0.5 * length), color=PALETTE.get(color, color))]
    shapes += list(extra_shapes)
    return Body(name, 0.5 * (p0 + p1), q, mass, np.diag([ixx, ixx, izz]), shapes)


def box_body(name, pos, half, mass, color="base", quat=(0.0, 0.0, 0.0, 1.0)) -> Body:
    hx, hy, hz = half
    inertia = np.diag(
        [mass * (hy * hy + hz * hz) / 3.0, mass * (hx * hx + hz * hz) / 3.0, mass * (hx * hx + hy * hy) / 3.0]
    )
    return Body(name, np.asarray(pos, float), np.asarray(quat, float), mass, inertia, [Shape("box", tuple(half), color=PALETTE.get(color, color))])


def node_body(name, pos, mass, radius=0.012, color="grey") -> Body:
    """Small spherical body (universal-joint cross, end-effector puck)."""
    i = 0.4 * mass * radius * radius
    return Body(name, np.asarray(pos, float), np.array([0.0, 0.0, 0.0, 1.0]), mass, np.diag([i, i, i]), [Shape("sphere", (radius,), color=PALETTE.get(color, color))])


def local_shape(body: Body, kind, size, world_pos, color, world_quat=(0.0, 0.0, 0.0, 1.0)) -> Shape:
    """Shape placed at a world pose on ``body`` (at its reference pose)."""
    r = quat_to_mat(body.quat)
    pos = r.T @ (np.asarray(world_pos, float) - body.pos)
    qinv = np.array([-body.quat[0], -body.quat[1], -body.quat[2], body.quat[3]])
    return Shape(kind, tuple(size), tuple(pos), tuple(quat_mul(qinv, np.asarray(world_quat, float))), PALETTE.get(color, color))


def circle_intersection_xz(c0, r0, c1, r1, pick: str):
    """Intersection of two circles in the x-z plane (y = 0); ``pick`` = 'low' | 'high' | 'left' | 'right'."""
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    d = float(np.linalg.norm(c1 - c0))
    if d > r0 + r1 or d < abs(r0 - r1):
        raise ValueError("circles do not intersect")
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    m = c0 + a * (c1 - c0) / d
    perp = np.array([-(c1 - c0)[2], 0.0, (c1 - c0)[0]]) / d
    p, q = m + h * perp, m - h * perp
    key = {"low": lambda v: v[2], "high": lambda v: -v[2], "left": lambda v: v[0], "right": lambda v: -v[0]}[pick]
    return p if key(p) < key(q) else q


Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def planar_angle(v) -> float:
    """atan2(z, x) of a vector in the x-z plane."""
    return math.atan2(v[2], v[0])


# A rotation by +q about +y decreases atan2(z, x) by q, so joint coordinates about +y are minus the in-plane angle.


# ---------------------------------------------------------------------------------------------------------------------
# 1. Stewart platform (6-UPS)


def stewart_platform() -> Mechanism:
    bodies, joints = [], []
    bodies.append(box_body("base", [0.0, 0.0, 0.02], [0.30, 0.30, 0.02], 20.0, "base"))
    joints.append(Joint("base_fix", "fixed", -1, 0, np.array([0.0, 0.0, 0.02])))
    zb, zp, rb, rp = 0.06, 0.42, 0.22, 0.13
    base_ang = [-15, 15, 105, 135, 225, 255]
    plat_ang = [-45, 45, 75, 165, 195, 285]
    B = [np.array([rb * math.cos(math.radians(a)), rb * math.sin(math.radians(a)), zb]) for a in base_ang]
    P = [np.array([rp * math.cos(math.radians(a)), rp * math.sin(math.radians(a)), zp]) for a in plat_ang]
    plat_mass, plat_r, plat_h = 2.0, 0.15, 0.015
    ip = np.diag([plat_mass * (plat_r**2 / 4 + plat_h**2 / 3), plat_mass * (plat_r**2 / 4 + plat_h**2 / 3), plat_mass * plat_r**2 / 2])
    platform = Body("platform", np.array([0.0, 0.0, zp]), np.array([0.0, 0.0, 0.0, 1.0]), plat_mass, ip, [Shape("box", (0.15, 0.15, plat_h), color=PALETTE["blue"])])
    colors = ["blue", "orange", "green", "orange", "blue", "green"]
    leg_bodies = []
    for k in range(6):
        u = unit(P[k] - B[k])
        length = float(np.linalg.norm(P[k] - B[k]))
        cross = node_body(f"cross_{k}", B[k], 0.05, 0.016, "grey")
        cyl = rod_body(f"cylinder_{k}", B[k], B[k] + 0.55 * length * u, 0.022, 0.4, colors[k])
        rod = rod_body(f"rod_{k}", P[k] - 0.55 * length * u, P[k], 0.012, 0.2, "grey", ())
        leg_bodies.append((cross, cyl, rod, u, length))
    # body order: base, legs (cross, cylinder, rod) x 6, platform
    for cross, cyl, rod, _, _ in leg_bodies:
        bodies += [cross, cyl, rod]
    bodies.append(platform)
    ip_idx = len(bodies) - 1
    act = []
    for k, (_cross, _cyl, _rod, u, length) in enumerate(leg_bodies):
        ic, iy, ir = 1 + 3 * k, 2 + 3 * k, 3 + 3 * k
        a1 = unit(np.cross(Z, u))
        a2 = unit(np.cross(u, a1))
        joints.append(Joint(f"u{k}_a", "revolute", 0, ic, B[k], a1))
        joints.append(Joint(f"u{k}_b", "revolute", ic, iy, B[k], a2))
        joints.append(Joint(f"leg{k}", "prismatic", iy, ir, B[k] + 0.5 * length * u, u, drive=Drive(20000.0, 60.0)))
        act.append(len(joints) - 1)
        joints.append(Joint(f"s{k}", "ball", ir, ip_idx, P[k], cut=(k > 0)))
    p_local = [p - np.array([0.0, 0.0, zp]) for p in P]

    def pose(t):
        r = ramp(t, 1.0)
        x = 0.02 * r * math.sin(2 * math.pi * 0.6 * t)
        z = 0.03 * smooth_step(t, 2.0)
        roll = math.radians(6.0) * r * math.sin(2 * math.pi * 0.7 * t)
        pitch = math.radians(4.0) * r * math.sin(2 * math.pi * 0.5 * t + 1.0)
        yaw = math.radians(8.0) * r * math.sin(2 * math.pi * 0.4 * t)
        q = quat_mul(quat_axis_angle(Z, yaw), quat_mul(quat_axis_angle(Y, pitch), quat_axis_angle([1, 0, 0], roll)))
        return np.array([x, 0.0, zp + z]), quat_to_mat(q)

    lengths0 = np.array([float(np.linalg.norm(P[k] - B[k])) for k in range(6)])

    def leg_q(t):
        c, r = pose(t)
        return np.array([float(np.linalg.norm(c + r @ p_local[k] - B[k])) for k in range(6)]) - lengths0

    def targets(t):
        return leg_q(t), numeric_rate(leg_q, t)

    return Mechanism(
        "stewart", "Stewart platform (6-UPS hexapod)",
        "Six universal-prismatic-spherical legs between a fixed base and a 2 kg platform; the prismatic legs are position-driven along a heave/roll/pitch/yaw trajectory computed by inverse kinematics.",
        "Five spatial loops sharing one platform: strongly coupled closure rows, all ball cuts, 6 DOF fully actuated.",
        bodies, joints, act, targets, 4.0, tracked=[ip_idx], dof=6,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 2. Delta robot


def delta_robot() -> Mechanism:
    bodies, joints = [], []
    zb, rb, la, lf, rp, w, alpha = 0.80, 0.14, 0.12, 0.30, 0.05, 0.06, math.radians(30.0)
    bodies.append(box_body("base", [0.0, 0.0, zb + 0.02], [0.18, 0.18, 0.015], 10.0, "base"))
    joints.append(Joint("base_fix", "fixed", -1, 0, np.array([0.0, 0.0, zb + 0.02])))
    phis = [math.radians(90.0 + 120.0 * i) for i in range(3)]
    dh = rb + la * math.cos(alpha) - rp
    ez = zb - la * math.sin(alpha)
    z0 = ez - math.sqrt(lf * lf - dh * dh)
    platform_pos = np.array([0.0, 0.0, z0])
    colors = ["blue", "orange", "green"]
    upper_idx, elbow, tang, attach = [], [], [], []
    for i, phi in enumerate(phis):
        r_hat = np.array([math.cos(phi), math.sin(phi), 0.0])
        t_hat = np.array([-math.sin(phi), math.cos(phi), 0.0])
        bpt = np.array([0.0, 0.0, zb]) + rb * r_hat
        e = bpt + la * (math.cos(alpha) * r_hat - math.sin(alpha) * Z)
        upper = rod_body(f"upper_{i}", bpt, e, 0.012, 0.15, colors[i])
        upper.shapes.append(local_shape(upper, "capsule", (0.008, 0.5 * w), e, colors[i], quat_from_two_vectors(Z, t_hat)))
        bodies.append(upper)
        upper_idx.append(len(bodies) - 1)
        joints.append(Joint(f"shoulder_{i}", "revolute", 0, len(bodies) - 1, bpt, t_hat, drive=Drive(20.0, 0.6)))
        elbow.append(e)
        tang.append(t_hat)
        attach.append(platform_pos + rp * r_hat)
    platform = box_body("platform", platform_pos, [0.06, 0.06, 0.01], 0.3, "red")
    rods = []
    for i in range(3):
        for s in (-1.0, 1.0):
            top = elbow[i] + 0.5 * w * s * tang[i]
            bot = attach[i] + 0.5 * w * s * tang[i]
            rods.append((i, s, top, bot))
    # body order: base, 3 upper arms, platform, then per rod (rod, cross)
    bodies.append(platform)
    ip = len(bodies) - 1
    act = [1, 2, 3]
    act = [j for j, jt in enumerate(joints) if jt.drive is not None]
    first = True
    for i, s, top, bot in rods:
        rod = rod_body(f"rod_{i}{'ab'[int(s > 0)]}", top, bot, 0.006, 0.04, "grey")
        cross = node_body(f"cross_{i}{'ab'[int(s > 0)]}", bot, 0.01, 0.009, "grey")
        bodies += [rod, cross]
        irod, icross = len(bodies) - 2, len(bodies) - 1
        a1 = tang[i]
        a2 = unit(np.cross(unit(bot - top), a1))
        if first:
            # tree: upper arm -> rod (ball) -> cross -> platform
            joints.append(Joint(f"elbow_{i}{'ab'[int(s > 0)]}", "ball", upper_idx[i], irod, top))
            joints.append(Joint(f"wrist_{i}{'ab'[int(s > 0)]}_a", "revolute", irod, icross, bot, a2))
            joints.append(Joint(f"wrist_{i}{'ab'[int(s > 0)]}_b", "revolute", icross, ip, bot, a1))
            first = False
        else:
            joints.append(Joint(f"wrist_{i}{'ab'[int(s > 0)]}_b", "revolute", ip, icross, bot, a1))
            joints.append(Joint(f"wrist_{i}{'ab'[int(s > 0)]}_a", "revolute", icross, irod, bot, a2))
            joints.append(Joint(f"elbow_{i}{'ab'[int(s > 0)]}", "ball", upper_idx[i], irod, top, cut=True))

    def targets(t):
        r = ramp(t, 1.0)
        a = math.radians(12.0)
        q = np.array([a * r * math.sin(2 * math.pi * 0.8 * t + 2 * math.pi * i / 3) + math.radians(6.0) * smooth_step(t, 2.0) for i in range(3)])
        return q, numeric_rate(lambda tt: np.array([a * ramp(tt, 1.0) * math.sin(2 * math.pi * 0.8 * tt + 2 * math.pi * i / 3) + math.radians(6.0) * smooth_step(tt, 2.0) for i in range(3)]), t)

    return Mechanism(
        "delta", "Delta robot",
        "Three driven upper arms carry a 0.3 kg platform through three parallelogram forearms (ball joints at the elbows, universal joints at the platform); the shoulders trace phase-shifted sines so the platform circles while it heaves.",
        "Three parallel chains with five ball cuts; parallelograms keep the platform level, so the closure rows are nearly redundant.",
        bodies, joints, act, targets, 4.0, tracked=[ip], dof=3,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 3. Five-bar planar manipulator


def five_bar() -> Mechanism:
    a1, a2 = np.array([-0.08, 0.0, 0.6]), np.array([0.08, 0.0, 0.6])
    l1, l2 = 0.15, 0.25
    b1 = a1 + l1 * np.array([math.cos(math.radians(225)), 0.0, math.sin(math.radians(225))])
    b2 = a2 + l1 * np.array([math.cos(math.radians(-45)), 0.0, math.sin(math.radians(-45))])
    p = circle_intersection_xz(b1, l2, b2, l2, "low")
    bodies = [box_body("base", [0.0, 0.0, 0.62], [0.13, 0.02, 0.02], 5.0, "base")]
    bodies.append(rod_body("crank_1", a1, b1, 0.012, 0.12, "orange"))
    bodies.append(rod_body("crank_2", a2, b2, 0.012, 0.12, "green"))
    d1 = rod_body("distal_1", b1, p, 0.010, 0.20, "blue")
    d1.shapes.append(local_shape(d1, "sphere", (0.02,), p, "red"))
    bodies.append(d1)
    bodies.append(rod_body("distal_2", b2, p, 0.010, 0.20, "green"))
    joints = [
        Joint("base_fix", "fixed", -1, 0, np.array([0.0, 0.0, 0.62])),
        Joint("theta_1", "revolute", 0, 1, a1, Y, drive=Drive(30.0, 0.8)),
        Joint("knee_1", "revolute", 1, 3, b1, Y),
        Joint("tip", "revolute", 3, 4, p, Y),
        Joint("theta_2", "revolute", 0, 2, a2, Y, drive=Drive(30.0, 0.8)),
        Joint("knee_2", "revolute", 2, 4, b2, Y, cut=True),
    ]

    def q_of(t):
        s = smooth_step(t, 2.0)
        r = ramp(t, 1.0)
        a, b = math.radians(22.0), math.radians(10.0)
        return np.array([-a * s + b * r * math.sin(2 * math.pi * 0.9 * t), a * s + b * r * math.sin(2 * math.pi * 0.9 * t + 0.5 * math.pi)])

    return Mechanism(
        "five_bar", "Five-bar planar manipulator",
        "Two driven cranks on a fixed base and two distal links meeting at the end-effector; the cranks swing the end-effector down toward the fully stretched (singular) configuration and back, with a sideways weave.",
        "Planar loop with a revolute cut (three of its five rows redundant); approaches the stretched singularity where the reflected inertia grows.",
        bodies, joints, [1, 4], lambda t: (q_of(t), numeric_rate(q_of, t)), 4.0, tracked=[3], dof=2,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 4. Watt six-bar walking leg


def watt_six_bar() -> Mechanism:
    o1, o2 = np.array([0.0, 0.0, 0.70]), np.array([0.13, 0.0, 0.72])
    rc, l_ab, l_ob = 0.045, 0.13, 0.10
    a = o1 + np.array([rc, 0.0, 0.0])
    b = circle_intersection_xz(a, l_ab, o2, l_ob, "low")
    nab = unit(b - a)
    perp = np.array([nab[2], 0.0, -nab[0]])  # rotate -90 deg in x-z: points below the coupler
    if perp[2] > 0:
        perp = -perp
    c = a + 0.45 * (b - a) + 0.07 * perp
    d = o2 + 1.7 * (b - o2)
    e = np.array([0.02, 0.0, 0.36])
    bodies = [box_body("base", [0.065, 0.0, 0.74], [0.10, 0.02, 0.015], 5.0, "base")]
    bodies.append(rod_body("crank", o1, a, 0.010, 0.08, "red"))
    cpl = rod_body("coupler", a, b, 0.010, 0.18, "orange")
    cpl.shapes.append(local_shape(cpl, "capsule", (0.007, 0.5 * float(np.linalg.norm(c - a))), 0.5 * (a + c), "orange", quat_from_two_vectors(Z, c - a)))
    cpl.shapes.append(local_shape(cpl, "capsule", (0.007, 0.5 * float(np.linalg.norm(c - b))), 0.5 * (b + c), "orange", quat_from_two_vectors(Z, c - b)))
    bodies.append(cpl)
    bodies.append(rod_body("rocker", o2, d, 0.010, 0.16, "green"))
    bodies.append(rod_body("link_ce", c, e, 0.009, 0.15, "purple"))
    foot = rod_body("link_de", d, e, 0.009, 0.15, "green")
    foot.shapes.append(local_shape(foot, "sphere", (0.018,), e, "base"))
    bodies.append(foot)
    joints = [
        Joint("base_fix", "fixed", -1, 0, np.array([0.065, 0.0, 0.74])),
        Joint("crank", "revolute", 0, 1, o1, Y, drive=Drive(20.0, 0.5)),
        Joint("crank_coupler", "revolute", 1, 2, a, Y),
        Joint("coupler_ce", "revolute", 2, 4, c, Y),
        Joint("foot", "revolute", 4, 5, e, Y),
        Joint("rocker", "revolute", 0, 3, o2, Y),
        Joint("coupler_rocker", "revolute", 2, 3, b, Y, cut=True),
        Joint("rocker_de", "revolute", 3, 5, d, Y, cut=True),
    ]
    omega, tr = 2.0 * math.pi * 0.6, 1.0

    def q_of(t):
        th = omega * (0.5 * t - tr / (2 * math.pi) * math.sin(math.pi * t / tr)) if t < tr else omega * (0.5 * tr + t - tr)
        return np.array([-th])  # crank turns counter-clockwise in the x-z view

    def qd_of(t):
        return np.array([-(omega * 0.5 * (1 - math.cos(math.pi * t / tr)) if t < tr else omega)])

    return Mechanism(
        "six_bar", "Watt six-bar walking leg",
        "A Grashof crank-rocker drives a Watt-I six-bar whose dyad carries the foot; the crank is driven at 0.6 rev/s after a 1 s ramp, so the foot traces its closed walking path more than twice.",
        "Two coupled planar loops (two revolute cuts) through a full crank rotation; ternary links transmit the loop forces.",
        bodies, joints, [1], lambda t: (q_of(t), qd_of(t)), 4.0, tracked=[5], dof=1,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 5. Parallelogram parallel-jaw gripper


def parallel_gripper() -> Mechanism:
    bodies = [box_body("palm", [0.0, 0.0, 0.60], [0.06, 0.02, 0.02], 1.0, "base")]
    joints = [Joint("palm_fix", "fixed", -1, 0, np.array([0.0, 0.0, 0.60]))]
    bodies.append(box_body("slider", [0.0, 0.0, 0.51], [0.016, 0.012, 0.02], 0.05, "grey"))
    joints.append(Joint("actuator", "prismatic", 0, 1, np.array([0.0, 0.0, 0.51]), Z, drive=Drive(3000.0, 40.0)))
    for sgn, side, color in ((-1.0, "l", "blue"), (1.0, "r", "green")):
        u0, u1 = np.array([sgn * 0.035, 0.0, 0.58]), np.array([sgn * 0.10, 0.0, 0.53])
        l0, l1 = np.array([sgn * 0.035, 0.0, 0.54]), np.array([sgn * 0.10, 0.0, 0.49])
        tip = np.array([sgn * 0.10, 0.0, 0.40])
        sp = np.array([sgn * 0.012, 0.0, 0.50])
        m = 0.5 * (u0 + u1)
        bodies.append(rod_body(f"upper_{side}", u0, u1, 0.007, 0.03, color))
        iu = len(bodies) - 1
        bodies.append(rod_body(f"lower_{side}", l0, l1, 0.007, 0.03, color))
        il = len(bodies) - 1
        jaw = rod_body(f"jaw_{side}", np.array([sgn * 0.10, 0.0, 0.535]), tip, 0.008, 0.10, "red")
        jaw.shapes.append(local_shape(jaw, "box", (0.006, 0.012, 0.025), tip + np.array([-sgn * 0.008, 0.0, 0.02]), "red"))
        bodies.append(jaw)
        ij = len(bodies) - 1
        bodies.append(rod_body(f"rod_{side}", sp, m, 0.005, 0.02, "grey"))
        ir = len(bodies) - 1
        joints += [
            Joint(f"upper_{side}", "revolute", 0, iu, u0, Y),
            Joint(f"jaw_{side}", "revolute", iu, ij, u1, Y),
            Joint(f"lower_{side}", "revolute", 0, il, l0, Y),
            Joint(f"rod_{side}", "revolute", 1, ir, sp, Y),
            Joint(f"jaw_lower_{side}", "revolute", il, ij, l1, Y, cut=True),
            Joint(f"rod_upper_{side}", "revolute", ir, iu, m, Y, cut=True),
        ]

    def q_of(t):
        return np.array([-0.016 * smooth_step(t, 2.0)])

    return Mechanism(
        "gripper", "Parallelogram parallel-jaw gripper",
        "A prismatic actuator pulls two connecting rods that swing the upper bars of two parallelogram fingers, so the jaws translate symmetrically while staying vertical; two close-open cycles.",
        "Four planar loops (two per finger) with revolute cuts; symmetric motion must stay symmetric and the jaws must stay parallel.",
        bodies, joints, [1], lambda t: (q_of(t), numeric_rate(q_of, t)), 4.0, tracked=[4, 8], dof=1,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 6. Four-bar prosthetic knee (crossed / polycentric)


def four_bar_knee() -> Mechanism:
    t1, t2 = np.array([0.035, 0.0, 0.52]), np.array([-0.030, 0.0, 0.53])
    s1, s2 = np.array([0.030, 0.0, 0.44]), np.array([-0.025, 0.0, 0.45])
    thigh = rod_body("thigh", np.array([0.0, 0.0, 0.95]), np.array([0.0, 0.0, 0.56]), 0.035, 3.0, "base")
    thigh.shapes.append(local_shape(thigh, "box", (0.045, 0.025, 0.02), np.array([0.0, 0.0, 0.54]), "base"))
    s_mid = 0.5 * (s1 + s2)
    ankle = np.array([0.0, 0.0, 0.06])
    shank = rod_body("shank", s_mid + np.array([0.0, 0.0, -0.01]), ankle, 0.025, 2.0, "green")
    shank.shapes.append(local_shape(shank, "box", (0.035, 0.022, 0.012), s_mid, "green"))
    shank.shapes.append(local_shape(shank, "box", (0.09, 0.04, 0.02), ankle + np.array([0.04, 0.0, -0.02]), "grey"))
    bodies = [thigh, rod_body("link_front", t1, s2, 0.007, 0.05, "orange"), rod_body("link_back", t2, s1, 0.007, 0.05, "blue"), shank]
    joints = [
        Joint("thigh_fix", "fixed", -1, 0, np.array([0.0, 0.0, 0.95])),
        Joint("front_thigh", "revolute", 0, 1, t1, Y, drive=Drive(200.0, 10.0)),
        Joint("front_shank", "revolute", 1, 3, s2, Y),
        Joint("back_thigh", "revolute", 0, 2, t2, Y),
        Joint("back_shank", "revolute", 2, 3, s1, Y, cut=True),
    ]

    def q_of(t):
        return np.array([math.radians(35.0) * smooth_step(t, 2.0)])

    return Mechanism(
        "knee", "Four-bar prosthetic knee",
        "A crossed four-bar (polycentric) knee: the thigh is fixed, the front bar is driven through two flexion cycles and the 2 kg shank with its foot swings on the moving instantaneous centre.",
        "One planar loop under a heavy, long-lever load; the instantaneous centre of rotation moves through the cycle.",
        bodies, joints, [1], lambda t: (q_of(t), numeric_rate(q_of, t)), 4.0, tracked=[3], dof=1,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 7. Closed-chain dual-arm manipulation


def dual_arm() -> Mechanism:
    l1, l2, l3 = 0.32, 0.30, 0.08
    sl, sr = np.array([-0.45, 0.0, 0.15]), np.array([0.45, 0.0, 0.15])
    o0 = np.array([0.0, 0.0, 0.55])
    obj_half = (0.15, 0.05, 0.05)

    def arm_points(obj_pos, phi, side):
        # side -1 left, +1 right; gripper point on the object face, wrist 0.08 outboard along the object x axis
        ox = np.array([math.cos(phi), 0.0, -math.sin(phi)])  # object x axis after pitch phi about +y
        g = obj_pos + side * obj_half[0] * ox
        wr = g + side * l3 * ox
        s = sl if side < 0 else sr
        e = circle_intersection_xz(s, l1, wr, l2, "high")
        return s, e, wr, g

    bodies = [box_body("base", [0.0, 0.0, 0.05], [0.55, 0.08, 0.05], 30.0, "base")]
    joints = [Joint("base_fix", "fixed", -1, 0, np.array([0.0, 0.0, 0.05]))]
    pts = {}
    for side, name, color in ((-1, "l", "blue"), (1, "r", "orange")):
        s, e, wr, g = arm_points(o0, 0.0, side)
        pts[name] = (s, e, wr, g)
        bodies.append(rod_body(f"upper_{name}", s, e, 0.02, 0.6, color))
        bodies.append(rod_body(f"fore_{name}", e, wr, 0.017, 0.5, color))
        hand = rod_body(f"hand_{name}", wr, g, 0.015, 0.2, color)
        hand.shapes.append(local_shape(hand, "box", (0.012, 0.03, 0.03), g + np.array([side * 0.01, 0.0, 0.0]), "grey"))
        bodies.append(hand)
    bodies.append(box_body("object", o0, obj_half, 1.0, "gold"))
    io = len(bodies) - 1
    gains = [Drive(400.0, 20.0), Drive(250.0, 10.0), Drive(60.0, 2.0)]
    s, e, wr, g = pts["l"]
    joints += [
        Joint("shoulder_l", "revolute", 0, 1, s, Y, drive=gains[0]),
        Joint("elbow_l", "revolute", 1, 2, e, Y, drive=gains[1]),
        Joint("wrist_l", "revolute", 2, 3, wr, Y, drive=gains[2]),
        Joint("grip_l", "fixed", 3, io, g),
    ]
    s, e, wr, g = pts["r"]
    joints += [
        Joint("shoulder_r", "revolute", 0, 4, s, Y, drive=gains[0]),
        Joint("elbow_r", "revolute", 4, 5, e, Y, drive=gains[1]),
        Joint("wrist_r", "revolute", 5, 6, wr, Y, drive=gains[2]),
        Joint("grip_r", "revolute", 6, io, g, Y, cut=True),
    ]
    act = [1, 2, 3, 5, 6, 7]

    def link_angles(obj_pos, phi):
        out = []
        for side in (-1, 1):
            s, e, wr, g = arm_points(obj_pos, phi, side)
            a1, a2, a3 = planar_angle(e - s), planar_angle(wr - e), planar_angle(g - wr)
            rel = np.array([a1, a2 - a1, a3 - a2])
            out.append(-np.unwrap(rel))
        return np.concatenate(out)

    q_ref = link_angles(o0, 0.0)

    def q_of(t):
        r = ramp(t, 1.0)
        pos = o0 + np.array([0.04 * r * math.sin(2 * math.pi * 0.4 * t), 0.0, 0.06 * smooth_step(t, 2.0)])
        phi = math.radians(8.0) * r * math.sin(2 * math.pi * 0.3 * t)
        d = link_angles(pos, phi) - q_ref
        return (d + math.pi) % (2 * math.pi) - math.pi

    return Mechanism(
        "dual_arm", "Closed-chain dual-arm manipulation",
        "Two planar 3R arms hold a 1 kg box, the left one welded to it and the right one pinned; all six joints are driven to the inverse kinematics of a box trajectory (lift, sway, tilt), so any drift sets the arms against each other.",
        "Loop through the object; six drives on a three-DOF closed chain, so the drives fight unless the loop is held exactly (operational-space consistency).",
        bodies, joints, act, lambda t: (q_of(t), numeric_rate(q_of, t)), 4.0, tracked=[io], dof=3,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 8. Closed-chain legged robot leg with the foot pinned to the ground


def closed_chain_leg() -> Mechanism:
    hip, knee, ankle = np.array([0.0, 0.0, 0.76]), np.array([0.06, 0.0, 0.46]), np.array([0.0, 0.0, 0.10])
    c0, c1 = np.array([-0.06, 0.0, 0.76]), np.array([-0.11, 0.0, 0.73])
    q_pt = np.array([0.035, 0.0, 0.40])
    torso = box_body("torso", [0.0, 0.0, 0.80], [0.10, 0.06, 0.05], 4.0, "base")
    shank = rod_body("shank", knee, ankle, 0.018, 0.5, "green")
    shank.shapes.append(local_shape(shank, "box", (0.06, 0.035, 0.012), ankle + np.array([0.02, 0.0, -0.012]), "grey"))
    shank.shapes.append(local_shape(shank, "capsule", (0.006, 0.5 * float(np.linalg.norm(q_pt - knee))), 0.5 * (q_pt + knee), "green", quat_from_two_vectors(Z, q_pt - knee)))
    bodies = [
        torso,
        rod_body("thigh", hip, knee, 0.02, 0.8, "blue"),
        shank,
        rod_body("crank", c0, c1, 0.01, 0.1, "red"),
        rod_body("pushrod", c1, q_pt, 0.007, 0.08, "orange"),
    ]
    joints = [
        Joint("torso_slide", "prismatic", -1, 0, np.array([0.0, 0.0, 0.80]), Z),
        Joint("hip", "revolute", 0, 1, hip, Y),
        Joint("knee", "revolute", 1, 2, knee, Y),
        Joint("crank", "revolute", 0, 3, c0, Y, drive=Drive(300.0, 10.0)),
        Joint("crank_rod", "revolute", 3, 4, c1, Y),
        Joint("rod_shank", "revolute", 4, 2, q_pt, Y, cut=True),
        Joint("foot_pin", "revolute", -1, 2, ankle, Y, cut=True),
    ]

    def q_of(t):
        return np.array([math.radians(20.0) * smooth_step(t, 2.0)])

    return Mechanism(
        "leg", "Closed-chain leg with the foot on the ground",
        "A 4 kg torso on a vertical slide stands on one leg whose knee is driven through a crank and pushrod; the foot is pinned to the ground (a contact loop), so turning the crank makes the robot squat and rise twice.",
        "A transmission loop plus a ground loop through a world-parent pin; the whole torso weight passes through the closures.",
        bodies, joints, [3], lambda t: (q_of(t), numeric_rate(q_of, t)), 4.0, tracked=[0, 2], dof=1,
    )


MECHANISMS = {
    "stewart": stewart_platform,
    "delta": delta_robot,
    "five_bar": five_bar,
    "six_bar": watt_six_bar,
    "gripper": parallel_gripper,
    "knee": four_bar_knee,
    "dual_arm": dual_arm,
    "leg": closed_chain_leg,
}


def get_mechanism(name: str) -> Mechanism:
    return MECHANISMS[name]()


def joint_frames(mech: Mechanism, j: Joint):
    """Parent and child joint transforms (position, quaternion) in their body frames; joint frames are world-aligned at the reference pose."""

    def frame(body_idx):
        if body_idx < 0:
            return j.anchor.copy(), np.array([0.0, 0.0, 0.0, 1.0])
        b = mech.bodies[body_idx]
        r = quat_to_mat(b.quat)
        return r.T @ (j.anchor - b.pos), np.array([-b.quat[0], -b.quat[1], -b.quat[2], b.quat[3]])

    return frame(j.parent), frame(j.child)


def tree_order(mech: Mechanism) -> list[int]:
    """Indices of the non-cut joints in breadth-first order from the world."""
    tree = [i for i, j in enumerate(mech.joints) if not j.cut]
    order, reached = [], {-1}
    while len(order) < len(tree):
        grew = False
        for i in tree:
            if i not in order and mech.joints[i].parent in reached:
                order.append(i)
                reached.add(mech.joints[i].child)
                grew = True
        if not grew:
            raise ValueError(f"{mech.name}: tree joints do not reach every body")
    return order


def check_mechanism(mech: Mechanism) -> dict:
    """Structural checks: one tree joint per body, every body reached, loops counted, Gruebler mobility."""
    tree = tree_order(mech)
    children = [mech.joints[i].child for i in tree]
    assert sorted(children) == list(range(len(mech.bodies))), f"{mech.name}: each body needs exactly one tree joint"
    cons = {"revolute": 5, "prismatic": 5, "ball": 3, "fixed": 6}
    mobility = 6 * len(mech.bodies) - sum(cons[j.kind] for j in mech.joints)
    return dict(bodies=len(mech.bodies), joints=len(mech.joints), loops=sum(j.cut for j in mech.joints), gruebler_6d=mobility, dof=mech.dof)
