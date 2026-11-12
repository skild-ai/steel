# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Hoberman Sphere
#
# A Hoberman sphere: six rings of ten scissor units on the great circles of an icosidodecahedron, 120 angulated
# bars and 30 hub connectors joined by 240 revolute joints. It has one degree of freedom and 91 independent loops.
# A drive on one hub opens it from a scissor angle of 120 degrees to 30 and closes it again every 4 s, and the
# whole sphere has to follow. The hub radius has a closed form, a cos(phi / 2) / sin(pi / 10).
#
# For Steel the joint graph becomes a free root, a spanning tree grown from one connector (the driven hub joint
# kept in the tree) and the other 91 hinges as REVOLUTE loop closures. Zero gravity and no contact, so all forces
# are internal and momentum must stay zero. MuJoCo gets the same tree, with each loop hinge as two CONNECT
# equality constraints along its axis (add_loop_closures_mujoco). (A variant on the ground is left out: the
# one-hub drive of 2 N m/rad cannot open the sphere against gravity and floor friction, and the contact rows cost ~4x.)
#
# Kamino takes the same model unchanged: it assumes no tree, so the free root adds no rows and the tree hinges and
# the loop closures are all the same revolute constraints. The hub drives are the same implicit PD (POSITION mode,
# same gains and targets), which Kamino solves as a compliant row of the joint.
#
# The geometry and the exact ground truth (ground_truth.py) live in this folder. check bounds the joint gaps, the hub-radius error against the
# closed form, the drive tracking and the momentum.
#
# Command: python -m shelf run loop_hoberman_sphere
#
###########################################################################

import math
import types

import numpy as np
import warp as wp

import newton
import newton.examples
from common import presets

# the sphere's parameters: arm length, scissor range, drive and bar sizes
PARAMS = types.SimpleNamespace(
    shape="sphere",
    segments=10,
    arm=0.05,  # [m] hub to the end of a bar
    phi_min=30.0,  # [deg] scissor angle when open
    phi_max=120.0,  # [deg] when closed (the build pose)
    start="closed",
    period=4.0,  # [s] closed -> open -> closed
    ke=2.0,  # [N m/rad] drive stiffness
    kd=0.05,  # [N m s/rad]
    density=1200.0,  # [kg/m^3] bars and connectors
    bar_radius=0.0025,
    connector_radius=0.005,
    mu=0.5,
)


def ring_nodes(n: int, a: float, phi: float):
    """Hub, outer and inner node positions of one ring in its plane (2D), for scissor angle ``phi`` [rad]."""
    al = 2.0 * math.pi / n
    s, c = math.sin(al / 2), math.cos(al / 2)
    rho = a * math.cos(phi / 2) / s
    d = a * math.sin(phi / 2)  # half of the U-V diagonal
    H, U, V = [], [], []
    for k in range(n):
        th = k * al
        H.append(rho * np.array([math.cos(th), math.sin(th)]))
        dm = np.array([math.cos(th + al / 2), math.sin(th + al / 2)])
        U.append((rho * c + d) * dm)
        V.append((rho * c - d) * dm)
    return np.array(H), np.array(U), np.array(V), rho


def bar_angles(n: int, a: float, phi: float):
    """In-plane orientation [rad] of each bar's right arm: A_k points at U_k, B_k at V_k."""
    H, U, V, _ = ring_nodes(n, a, phi)
    pa = np.arctan2(U[:, 1] - H[:, 1], U[:, 0] - H[:, 0])
    pb = np.arctan2(V[:, 1] - H[:, 1], V[:, 0] - H[:, 0])
    return pa, pb


def icosidodecahedron_rings():
    """Six great circles (unit normal, ordered unit hub directions, global vertex ids) of an icosidodecahedron."""
    g = (1 + 5**0.5) / 2
    ico = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            ico += [(0, s1, s2 * g), (s1, s2 * g, 0), (s2 * g, 0, s1)]
    ico = np.array(ico, dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    verts = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.dot(ico[i], ico[j]) - 1 / 5**0.5) < 1e-6:  # icosahedron edge
                m = ico[i] + ico[j]
                verts.append(m / np.linalg.norm(m))
    verts = np.array(verts)
    assert len(verts) == 30
    axes = [
        ico[i]
        for i in range(12)
        if all(np.dot(ico[i], x) < 1 - 1e-6 for x in ico[:i]) and all(np.dot(ico[i], -x) < 1 - 1e-6 for x in ico[:i])
    ]
    rings = []
    for nrm in axes[:6]:
        ids = [i for i in range(30) if abs(np.dot(verts[i], nrm)) < 1e-6]
        assert len(ids) == 10, len(ids)
        u = verts[ids[0]]
        v = np.cross(nrm, u)
        ids.sort(key=lambda i: (math.atan2(np.dot(verts[i], v), np.dot(verts[i], u)) + 1e-9) % (2 * math.pi))
        u = verts[ids[0]]  # the reference vertex must be hub 0 after sorting
        v = np.cross(nrm, u)
        rings.append((nrm, u, v, ids))
    count = np.zeros(30, int)
    for r in rings:
        count[r[3]] += 1
    assert (count == 2).all(), count
    return verts, rings


def mat_to_quat(R: np.ndarray):
    """Rotation matrix with columns as frame axes -> warp quaternion (x, y, z, w)."""
    return wp.quat_from_matrix(wp.mat33(*R.flatten().tolist()))


# ----------------------------------------------------------------------------------------------- joint graph


class Graph:
    """Bodies and revolute joints of the mechanism at the build pose (scissor angle ``phi0``)."""

    def __init__(self, args):
        self.args = args
        a, phi0 = args.arm, math.radians(args.phi_max if args.start == "closed" else args.phi_min)
        self.phi0 = phi0
        self.bodies = []  # dict(pos, R, kind, shapes=[...])
        self.joints = []  # dict(p, c, anchor_world, axis_world, R_frame, kind)
        self.hub_points = []  # per hub: (body, local point) used to measure hub radius
        self.scissors = []  # per hub: (bar A body, bar B body, local right-arm vectors) to measure phi
        if args.shape == "ring":
            n = args.segments
            u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])  # vertical ring in the x-z plane
            nrm = np.cross(u, v)  # right-handed ring frame
            self.n = n
            self._add_ring(n, a, phi0, nrm, u, v, ring_index=0, connectors=None)
        else:
            n = 10
            self.n = n
            verts, rings = icosidodecahedron_rings()
            rho0 = a * math.cos(phi0 / 2) / math.sin(math.pi / n)
            conn = []
            for i in range(30):
                conn.append(self._add_body(rho0 * verts[i], np.eye(3), "connector"))
                self.hub_points.append((conn[-1], np.zeros(3)))
            self.verts = verts
            for r, (nrm, u, v, ids) in enumerate(rings):
                self._add_ring(n, a, phi0, nrm, u, v, ring_index=r, connectors=[conn[i] for i in ids])
        self.center0 = np.mean([self.bodies[b]["pos"] + self.bodies[b]["R"] @ p for b, p in self.hub_points], axis=0)

    def _add_body(self, pos, R, kind, shapes=None):
        self.bodies.append(dict(pos=np.asarray(pos, float), R=np.asarray(R, float), kind=kind, shapes=shapes or []))
        return len(self.bodies) - 1

    def _add_joint(self, p, c, point_world, R_frame, kind):
        self.joints.append(
            dict(p=p, c=c, point=np.asarray(point_world, float), R=np.asarray(R_frame, float), kind=kind)
        )
        return len(self.joints) - 1

    def _add_ring(self, n, a, phi0, nrm, u, v, ring_index, connectors):
        R = np.column_stack([u, v, nrm])  # ring frame: x = u, y = v, z = normal
        H2, U2, V2, rho = ring_nodes(n, a, phi0)
        to3 = lambda p2: p2[0] * u + p2[1] * v
        H = [to3(p) for p in H2]
        U = [to3(p) for p in U2]
        V = [to3(p) for p in V2]
        A, B = [], []
        for k in range(n):
            # body frame = ring frame at the hub; arms as capsules from the hub to each end (local coordinates)
            arms_a = [R.T @ (U[k] - H[k]), R.T @ (V[(k - 1) % n] - H[k])]
            arms_b = [R.T @ (V[k] - H[k]), R.T @ (U[(k - 1) % n] - H[k])]
            A.append(self._add_body(H[k], R, "bar", shapes=arms_a))
            B.append(self._add_body(H[k], R, "bar", shapes=arms_b))
        for k in range(n):
            if connectors is None:
                self.joint_hub = self._add_joint(A[k], B[k], H[k], R, "hub")
                self.hub_points.append((A[k], np.zeros(3)))
            else:
                self._add_joint(connectors[k], A[k], H[k], R, "hub")
                self._add_joint(connectors[k], B[k], H[k], R, "hub")
            self.scissors.append((A[k], B[k], R.T @ (U[k] - H[k]), R.T @ (V[k] - H[k])))
            self._add_joint(A[k], B[(k + 1) % n], U[k], R, "outer")
            self._add_joint(B[k], A[(k + 1) % n], V[k], R, "inner")
        self.ring_bodies = getattr(self, "ring_bodies", []) + [(A, B)]
        self.ring_frames = getattr(self, "ring_frames", []) + [R]

    def analytic_body_q(self, phi: float) -> np.ndarray:
        """Exact pose (x, y, z, qx, qy, qz, qw) of every body at scissor angle phi, centred like the build pose."""
        n, a = self.n, self.args.arm
        al = 2 * math.pi / n
        rho = a * math.cos(phi / 2) / math.sin(al / 2)
        pa, pb = bar_angles(n, a, phi)
        pa0, pb0 = bar_angles(n, a, self.phi0)
        da, db = pa[0] - pa0[0], pb[0] - pb0[0]
        out = np.zeros((len(self.bodies), 7))
        for i, B in enumerate(self.bodies):  # connectors: slide along their ray, no rotation
            if B["kind"] == "connector":
                out[i, :3] = rho * B["pos"] / np.linalg.norm(B["pos"])
                out[i, 6] = 1.0
        for (A, Bb), R in zip(self.ring_bodies, self.ring_frames):
            for k in range(n):
                hub = rho * (math.cos(k * al) * R[:, 0] + math.sin(k * al) * R[:, 1])
                for body, d in ((A[k], da), (Bb[k], db)):
                    Rz = np.array([[math.cos(d), -math.sin(d), 0.0], [math.sin(d), math.cos(d), 0.0], [0.0, 0.0, 1.0]])
                    q = mat_to_quat(R @ Rz)
                    out[body, :3] = hub
                    out[body, 3:] = [q[0], q[1], q[2], q[3]]
        return out

    def driven_joints(self, mode):
        """Hub joints that carry a drive: the first hub joint, or every hub joint of the graph."""
        hubs = [j for j, J in enumerate(self.joints) if J["kind"] == "hub"]
        if mode == "one":
            return hubs[:1]
        if self.args.shape == "sphere":
            return [j for j in hubs if self._is_a(self.joints[j]["c"])]  # connector -> A bar of every hub
        return hubs

    def joint_q_of_phi(self, j, phi):
        """Coordinate of hub joint ``j`` (child relative to parent about the ring normal) at scissor angle ``phi``."""
        n, a = self.n, self.args.arm
        pa, pb = bar_angles(n, a, phi)
        pa0, pb0 = bar_angles(n, a, self.phi0)
        da, db = pa[0] - pa0[0], pb[0] - pb0[0]  # every A bar rotates by da, every B bar by db (symmetry)
        J = self.joints[j]
        pk, ck = self.bodies[J["p"]]["kind"], self.bodies[J["c"]]["kind"]
        if pk == "connector":  # connector -> A (or B) bar
            return da if self._is_a(J["c"]) else db
        return db - da  # ring hub joint A -> B

    def _is_a(self, b):
        for A, B in self.ring_bodies:
            if b in A:
                return True
            if b in B:
                return False
        raise KeyError(b)


def shape_cfg() -> newton.ModelBuilder.ShapeConfig:
    """Bars and connectors carry mass (from their density) but take no contact."""
    cfg = newton.ModelBuilder.ShapeConfig(density=PARAMS.density, mu=PARAMS.mu, margin=1e-6, gap=1e-2)
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    return cfg


def add_loop_closures_mujoco(b: newton.ModelBuilder, G: Graph, loops, ids, model_cfg):
    """MuJoCo: each loop hinge as two CONNECT equalities at +-``hinge_half_span`` along its axis.

    SolverMuJoCo turns a REVOLUTE joint outside the articulation into the same CONNECT pair by itself, but with
    MuJoCo's default solref / solimp (0.02 s, impedance 0.9-0.95), soft enough that the 91 loops open by 2 mm at
    4 substeps and fly apart at fewer. Explicit equalities take the preset's ``eq_solref`` / ``eq_solimp``.
    """
    from newton.solvers import mujoco as mjc

    half = float(model_cfg.need("hinge_half_span"))
    solref, solimp = model_cfg.need("eq_solref"), model_cfg.need("eq_solimp")
    for j in loops:
        J = G.joints[j]
        Bp = G.bodies[J["p"]]
        anchor = Bp["R"].T @ (J["point"] - Bp["pos"])  # hinge point and axis in the parent body's frame
        axis = Bp["R"].T @ J["R"][:, 2]
        for side in (-1.0, 1.0):
            mjc.add_equality_connect(
                b,
                ids[J["p"]],
                ids[J["c"]],
                anchor=wp.vec3(*(anchor + side * half * axis)),
                label=f"j{j}_{J['kind']}_loop_{'a' if side < 0 else 'b'}",
                solref=solref,
                solimp=solimp,
            )


def build_model(G: Graph, drive_mode: str, preset=None):
    """Free root + spanning tree (driven joints first) + revolute loop closures: the ``tree`` form.

    The loop closures are REVOLUTE joints outside the articulation, except for MuJoCo (``preset.kind``), which gets
    CONNECT equalities from :func:`add_loop_closures_mujoco`.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    drive = set(G.driven_joints(drive_mode))
    driven_bodies = {G.joints[j]["c"] for j in drive} | {G.joints[j]["p"] for j in drive}
    ring_of = {b_: r for r, (A, Bb) in enumerate(G.ring_bodies) for b_ in A + Bb}
    palette = [
        (0.23, 0.49, 0.63),
        (0.75, 0.38, 0.23),
        (0.42, 0.60, 0.24),
        (0.54, 0.36, 0.69),
        (0.77, 0.60, 0.17),
        (0.23, 0.60, 0.56),
    ]
    amber = (1.0, 0.62, 0.11)
    ids = []
    for B in G.bodies:
        bi = b.add_link(xform=wp.transform(wp.vec3(*B["pos"]), mat_to_quat(B["R"])), label=f"{B['kind']}_{len(ids)}")
        ids.append(bi)
        if B["kind"] == "connector":
            on = len(ids) - 1 in driven_bodies
            b.add_shape_sphere(
                bi,
                radius=PARAMS.connector_radius * (1.5 if on else 1.0),
                cfg=shape_cfg(),
                color=amber if on else (0.55, 0.58, 0.61),
            )
            continue
        for arm in B["shapes"]:
            L = float(np.linalg.norm(arm))
            zhat = arm / L
            axis = np.cross([0.0, 0.0, 1.0], zhat)
            ang = math.acos(max(-1.0, min(1.0, zhat[2])))
            qa = wp.quat_from_axis_angle(
                wp.vec3(*(axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else [1.0, 0.0, 0.0])), ang
            )
            color = amber if len(ids) - 1 in driven_bodies else palette[ring_of.get(len(ids) - 1, 0) % 6]
            b.add_shape_capsule(
                bi,
                xform=wp.transform(wp.vec3(*(arm / 2)), qa),
                radius=PARAMS.bar_radius,
                half_height=L / 2,
                cfg=shape_cfg(),
                color=color,
            )

    def frames(j, flip):
        J = G.joints[j]
        p, c = (J["c"], J["p"]) if flip else (J["p"], J["c"])
        Bp, Bc = G.bodies[p], G.bodies[c]
        xp = wp.transform(wp.vec3(*(Bp["R"].T @ (J["point"] - Bp["pos"]))), mat_to_quat(Bp["R"].T @ J["R"]))
        xc = wp.transform(wp.vec3(*(Bc["R"].T @ (J["point"] - Bc["pos"]))), mat_to_quat(Bc["R"].T @ J["R"]))
        return p, c, xp, xc

    def add_rev(j, flip, loop=False):
        p, c, xp, xc = frames(j, flip)
        driven = j in drive and not loop
        kw = dict(
            target_pos=0.0,
            target_ke=PARAMS.ke if driven else 0.0,
            target_kd=PARAMS.kd if driven else 0.0,
            armature=0.0,
            label=f"j{j}_{G.joints[j]['kind']}{'_loop' if loop else ''}",
        )
        if driven:
            kw["actuator_mode"] = newton.JointTargetMode.POSITION
        return b.add_joint_revolute(
            parent=ids[p], child=ids[c], parent_xform=xp, child_xform=xc, axis=(0.0, 0.0, 1.0), **kw
        )

    nb = len(G.bodies)
    in_tree, used, order = {0}, set(), []
    while len(in_tree) < nb:  # grow the tree greedily, driven joints first, so drives stay tree joints
        frontier = [j for j, J in enumerate(G.joints) if j not in used and ((J["p"] in in_tree) != (J["c"] in in_tree))]
        frontier.sort(key=lambda j: (j not in drive, j))
        j = frontier[0]
        used.add(j)
        J = G.joints[j]
        flip = J["c"] in in_tree
        in_tree.add(J["p"] if flip else J["c"])
        order.append((j, flip))
    sign, coord_joint = {}, {}
    jid = [b.add_joint_free(child=ids[0], label="root")]
    for j, flip in order:
        coord_joint[j] = add_rev(j, flip)
        jid.append(coord_joint[j])
        sign[j] = -1.0 if flip else 1.0
    b.add_articulation(jid, label="hoberman")
    loops = [j for j in range(len(G.joints)) if j not in used]
    if preset is not None and preset.kind == "mujoco":
        add_loop_closures_mujoco(b, G, loops, ids, preset.model)
    else:
        for j in loops:
            add_rev(j, False, loop=True)
    model = b.finalize()
    return model, dict(sign=sign, coord_joint=coord_joint, loops=loops, drive=sorted(drive))


def quat_rot(q, v):
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class Metrics:
    """Per frame, from body poses: worst joint gap [mm] and axis misalignment [deg] over every hinge, the driven
    scissor angle, and the hub radii against the closed form at the measured angle; momentum from body velocities."""

    def __init__(self, G: Graph, model):
        self.jp = np.array([J["p"] for J in G.joints])
        self.jc = np.array([J["c"] for J in G.joints])
        P = [G.bodies[J["p"]] for J in G.joints]
        C = [G.bodies[J["c"]] for J in G.joints]
        self.ap = np.array([Bp["R"].T @ (J["point"] - Bp["pos"]) for J, Bp in zip(G.joints, P, strict=True)])
        self.ac = np.array([Bc["R"].T @ (J["point"] - Bc["pos"]) for J, Bc in zip(G.joints, C, strict=True)])
        self.xp = np.array([Bp["R"].T @ J["R"][:, 2] for J, Bp in zip(G.joints, P, strict=True)])
        self.xc = np.array([Bc["R"].T @ J["R"][:, 2] for J, Bc in zip(G.joints, C, strict=True)])
        self.hub_b = np.array([h[0] for h in G.hub_points])
        self.hub_p = np.array([h[1] for h in G.hub_points])
        self.sa = np.array([s[0] for s in G.scissors])
        self.sb = np.array([s[1] for s in G.scissors])
        self.va = np.array([s[2] for s in G.scissors])
        self.vb = np.array([s[3] for s in G.scissors])
        self.mass = model.body_mass.numpy().astype(np.float64)
        self.com = model.body_com.numpy().astype(np.float64)
        self.s_half = math.sin(math.pi / G.n)

    def __call__(self, bq, bqd):
        bq = bq.astype(np.float64)
        pos, q = bq[:, :3], bq[:, 3:7]
        gap = 1e3 * np.linalg.norm(
            pos[self.jp] + quat_rot(q[self.jp], self.ap) - pos[self.jc] - quat_rot(q[self.jc], self.ac), axis=1
        )
        xa, xb = quat_rot(q[self.jp], self.xp), quat_rot(q[self.jc], self.xc)
        mis = np.degrees(np.arccos(np.clip(np.sum(xa * xb, axis=1), -1, 1)))
        hubs = pos[self.hub_b] + quat_rot(q[self.hub_b], self.hub_p)
        r = np.linalg.norm(hubs - hubs.mean(axis=0), axis=1)
        wa, wb = quat_rot(q[self.sa], self.va), quat_rot(q[self.sb], self.vb)
        phi = np.arccos(
            np.clip(np.sum(wa * wb, axis=1) / (np.linalg.norm(wa, axis=1) * np.linalg.norm(wb, axis=1)), -1, 1)
        )
        rho = PARAMS.arm * np.cos(phi[0] / 2) / self.s_half
        momentum = float(np.linalg.norm((self.mass[:, None] * bqd[:, :3].astype(np.float64)).sum(axis=0)))
        return dict(
            gap_mm=float(gap.max()),
            mis_deg=float(mis.max()),
            phi_deg=float(np.degrees(phi[0])),
            r_err_mm=float(1e3 * np.abs(r - rho).max()),
            momentum=momentum,
        )


@wp.kernel(enable_backward=False)
def _set_targets(
    coords: wp.array[int],
    q_open: wp.array[float],
    q_closed: wp.array[float],
    tbuf: wp.array[float],
    dt: float,
    period: float,
    target_q: wp.array[float],
):
    i = wp.tid()
    t = tbuf[0] + dt  # the target at the end of this substep
    s = 0.5 - 0.5 * wp.cos(2.0 * wp.pi * t / period)  # 0 (closed) at t = 0 -> 1 (open) at half a period -> 0
    target_q[coords[i]] = q_closed[i] + s * (q_open[i] - q_closed[i])


@wp.kernel(enable_backward=False)
def _advance(tbuf: wp.array[float], dt: float, period: float):
    t = tbuf[0] + dt
    if t >= period:
        t = t - period
    tbuf[0] = t


class Example:
    def __init__(self, viewer, args=None):
        self.preset = presets.load(__file__)
        self.viewer = viewer
        self.test_mode = bool(getattr(args, "test", False))
        self.fps = self.preset.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.preset.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.graph_desc = Graph(PARAMS)
        self.drive = getattr(args, "drive", "one")
        self.model, self.info = build_model(self.graph_desc, self.drive, self.preset)
        self.build_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        G, info = self.graph_desc, self.info
        q_start = self.model.joint_q_start.numpy()
        phi_closed, phi_open = math.radians(PARAMS.phi_max), math.radians(PARAMS.phi_min)
        device = self.model.device
        self._coords = wp.array([int(q_start[info["coord_joint"][j]]) for j in info["drive"]], dtype=int, device=device)
        self._q_closed = wp.array(
            [info["sign"][j] * G.joint_q_of_phi(j, phi_closed) for j in info["drive"]], dtype=float, device=device
        )
        self._q_open = wp.array(
            [info["sign"][j] * G.joint_q_of_phi(j, phi_open) for j in info["drive"]], dtype=float, device=device
        )
        self._time = wp.zeros(1, dtype=float, device=device)

        self.metrics = Metrics(G, self.model)
        self._worst = {"gap_mm": 0.0, "mis_deg": 0.0, "r_err_mm": 0.0, "momentum": 0.0, "phi_err_deg": 0.0}
        self._frame = 0
        self.body_q_frames = []  # test mode: body_q after every frame, scored against ground_truth.py

        self.viewer.set_model(self.model)
        # the open sphere is 0.32 m across
        self.viewer.set_camera(pos=wp.vec3(-0.42, -0.42, 0.18), pitch=-15.0, yaw=45.0)
        self.capture()

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
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
            wp.launch(
                _set_targets,
                dim=len(self.info["drive"]),
                inputs=[self._coords, self._q_open, self._q_closed, self._time, self.sim_dt, PARAMS.period],
                outputs=[self.control.joint_target_q],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(_advance, dim=1, inputs=[self._time, self.sim_dt, PARAMS.period], device=self.model.device)
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
        self._frame += 1
        if self.test_mode:
            self.body_q_frames.append(self.state_0.body_q.numpy())
            m = self.metrics(self.body_q_frames[-1], self.state_0.body_qd.numpy())
            s = 0.5 - 0.5 * math.cos(2.0 * math.pi * self.sim_time / PARAMS.period)
            m["phi_err_deg"] = abs(m["phi_deg"] - (PARAMS.phi_max + s * (PARAMS.phi_min - PARAMS.phi_max)))
            for k in self._worst:
                self._worst[k] = max(self._worst[k], m[k])

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Finite state; every hinge closed; the hubs on the closed-form radius; the drive tracked; momentum zero."""
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            assert np.isfinite(getattr(self.state_0, name).numpy()).all(), f"{name} has non-finite entries"
        w = self._worst
        assert w["gap_mm"] < 0.5, f"worst joint gap {w['gap_mm']:.3f} mm"
        assert w["mis_deg"] < 1.0, f"worst hinge-axis misalignment {w['mis_deg']:.3f} deg"
        assert w["r_err_mm"] < 1.0, f"hub radius {w['r_err_mm']:.3f} mm from a cos(phi/2)/sin(pi/10)"
        assert w["phi_err_deg"] < 5.0, f"driven scissor angle {w['phi_err_deg']:.2f} deg from its target"
        assert w["momentum"] < 1e-3, f"linear momentum {w['momentum']:.2e} kg m/s with no external force"
        gt = self.ground_truth_score()
        if gt is not None:
            print("ground truth: " + ", ".join(f"{k} {v:.4g}" for k, v in gt.items()))
            # the mechanism against the exact trajectory; the free-body drift (orient/com) is reported, not bounded
            assert gt["phi_max_err_deg"] < 1.0, f"scissor angle {gt['phi_max_err_deg']:.3f} deg from the exact phi(t)"
            assert gt["shape_pos_max_mm"] < 2.0, f"shape {gt['shape_pos_max_mm']:.3f} mm from the exact one"

    def ground_truth_score(self):
        """The recorded frames scored against the exact trajectory (ground_truth.py); None at another frame rate."""
        import ground_truth  # next to this file, which the runner puts on sys.path

        if not self.body_q_frames or self.fps != ground_truth.FPS:
            return None
        return ground_truth.score(np.array(self.body_q_frames[: ground_truth.FRAMES]), self.drive)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--drive", choices=["one", "all"], default="one", help="drive one hub joint, or every hub")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
