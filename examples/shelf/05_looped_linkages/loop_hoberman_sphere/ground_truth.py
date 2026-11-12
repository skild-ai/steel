# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Ground truth for the Hoberman sphere (loop_hoberman_sphere/example.py)
#
# The sphere is a one-degree-of-freedom mechanism (scissor angle phi) that floats freely, so its exact motion is a
# 7-DOF ODE: phi plus the rigid pose (x0, R0) of the "structure frame" in which every body has its closed-form pose
#
#     hub_i(phi) = rho(phi) e_i,   rho = arm cos(phi/2) / sin(pi/10),   R_i(phi) = R_i0 Rz(sigma_i (phi - phi0) / 2)
#
# (sigma = -1 for A bars, +1 for B bars, 0 for connectors, which slide along their ray without turning; the example's
# Graph.analytic_body_q). Body masses, COMs and inertias come from the finalized Newton model, which differ between
# the drive modes (a driven connector is drawn 1.5x larger and is 3.4x heavier). The drive is the example's: the
# PD tau = ke (q_target(t) - q) - kd qdot on each driven hub joint (target_qd is 0, effort limit 1e6 never reached),
# with q_target(t) the kernel's raised cosine sampled continuously. The equations are Kane's (virtual power) in the
# quasi-velocities (phidot, v0, w0), integrated with SciPy's DOP853 at rtol 1e-12.
#
# Checks (``python ground_truth.py --check``): kinetic energy against the drive's work, linear and angular momentum,
# the closure gaps of the reconstructed poses against the model's own joint frames, the null space of the full
# 1200-row constraint Jacobian (exactly 7 = 1 + 6, and the closed-form velocities lie in it), the drive coordinate
# against Newton's eval_ik, a tolerance study, and (``--maximal``) a second formulation in maximal coordinates
# (150 Newton-Euler bodies, 1200 revolute rows, the joint graph only, no closed form) that must agree.
# Results (2026-11-07, both drives): |KE - work| / KE < 2e-13, |P| < 3e-18 kg m/s, |L| < 1e-19 kg m^2/s; null space
# 7 (singular values 1.7e-15 against 9.6e-3 for the 8th); closure on the model's float32 joint frames 2e-9 m;
# rtol 1e-10 and Radau (rtol 1e-11) within 6e-13 m of the reference; maximal coordinates within 4e-13 m (one, 4 s)
# and 7e-16 m (all, 2 s). Costs: --drive one 25 s, --drive all 6 min (the 60 drives make it stiff); --check adds
# a few minutes, --maximal about 2 min per second of --drive one and 6 min per second of --drive all.
#
# The drive's damping acts on the joint velocity against a zero target velocity, so phi trails the raised cosine by
# about kd / ke = 25 ms (up to 1.8 deg) in both modes; --drive one is underdamped (zeta 0.23-0.64) and overshoots
# 120 deg by 0.3 deg. The free body never turns (the shape change carries no angular momentum, coupling < 1e-16), and
# under --drive one it shifts by up to 0.35 mm, because the heavier driven connector moves the COM in the shape.
#
# Usage:
#     python shelf/05_looped_linkages/loop_hoberman_sphere/ground_truth.py            # write both references
#     python shelf/05_looped_linkages/loop_hoberman_sphere/ground_truth.py --check    # and validate them
#     from ground_truth import score; score(body_q_frames, "one")                     # score a solver run
#
# Frame k of a reference is the state at t = (k + 1) / 50 s, i.e. after k + 1 calls of Example.step().
#
###########################################################################

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FPS = 50
FRAMES = 1000  # 20 s, five open-close cycles
REFERENCE = {"one": HERE / "reference_drive_one.npz", "all": HERE / "reference_drive_all.npz"}


# ----------------------------------------------------------------------------------------------- small helpers


def skew(v):
    v = np.asarray(v)
    z = np.zeros(v.shape[:-1])
    return np.stack(
        [
            np.stack([z, -v[..., 2], v[..., 1]], -1),
            np.stack([v[..., 2], z, -v[..., 0]], -1),
            np.stack([-v[..., 1], v[..., 0], z], -1),
        ],
        -2,
    )


def quat_to_mat(q):
    """Quaternion(s) (x, y, z, w) -> rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def mat_to_quat(R):
    """Rotation matrices -> quaternions (x, y, z, w) with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.zeros((len(R), 4))
    tr = np.trace(R, axis1=1, axis2=2)
    diag = np.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2], tr], 1)
    k = diag.argmax(1)
    for i in range(len(R)):
        r = R[i]
        if k[i] == 3:
            s = 2.0 * math.sqrt(1.0 + tr[i])
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif k[i] == 0:
            s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif k[i] == 1:
            s = 2.0 * math.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2])
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = 2.0 * math.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2])
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q *= np.where(q[:, 3:4] < 0.0, -1.0, 1.0)
    return q.reshape(*shp, 4)


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def quat_angle(a, b):
    """Angle [rad] of the rotation between quaternions a and b (sign-insensitive, accurate near zero)."""
    a = np.asarray(a, dtype=np.float64)
    a = a * np.array([-1.0, -1.0, -1.0, 1.0])  # conjugate
    d = quat_mul(a, np.asarray(b, dtype=np.float64))
    return 2.0 * np.arctan2(np.linalg.norm(d[..., :3], axis=-1), np.abs(d[..., 3]))


def raised_cosine(t, period):
    """The example's drive phase: 0 (closed) at t = 0, 1 (open) at half a period (``_set_targets``)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * t / period)


def load_example():
    """The example module, imported from its file (``from common import presets`` needs the repo root)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("hoberman_example", HERE / "example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- the mechanism


class Mechanism:
    """Closed-form kinematics in phi, the model's mass properties and the drive, for one drive mode."""

    def __init__(self, ex, drive: str):
        P = ex.PARAMS
        self.ex, self.drive, self.P = ex, drive, P
        G = ex.Graph(P)
        model, info = ex.build_model(G, drive)
        self.G, self.model, self.info = G, model, info
        self.n, self.arm, self.phi0 = G.n, P.arm, G.phi0
        self.s_half = math.sin(math.pi / G.n)
        nb = len(G.bodies)
        self.nb = nb
        # static body data: hub direction e, build rotation R0, ring normal nrm, sigma (-1 A, +1 B, 0 connector)
        self.e = np.array([B["pos"] / np.linalg.norm(B["pos"]) for B in G.bodies])
        self.R0 = np.array([B["R"] for B in G.bodies])
        self.sigma = np.array(
            [0.0 if B["kind"] == "connector" else (-1.0 if G._is_a(i) else 1.0) for i, B in enumerate(G.bodies)]
        )
        self.nrm = self.R0[:, :, 2] * (self.sigma != 0.0)[:, None]
        self.w = 0.5 * self.sigma[:, None] * self.nrm  # body angular velocity per unit phidot (structure frame)
        # mass properties of the finalized model (float32 there; inertia symmetrized)
        self.mass = model.body_mass.numpy().astype(np.float64)
        self.com = model.body_com.numpy().astype(np.float64)
        inertia = model.body_inertia.numpy().astype(np.float64)
        self.inertia = 0.5 * (inertia + np.swapaxes(inertia, 1, 2))
        self.M = float(self.mass.sum())
        # the model's joints, by graph joint index (labels are "j{index}_{kind}[_loop]"), with float64 frames
        labels = list(model.joint_label)
        jp, jc = model.joint_parent.numpy(), model.joint_child.numpy()
        self.mjoint = {}  # graph joint -> (model joint, model parent, model child)
        for mj, lab in enumerate(labels):
            name = lab.split("/")[-1]
            if name.startswith("j"):
                self.mjoint[int(name[1:].split("_")[0])] = (mj, int(jp[mj]), int(jc[mj]))
        assert len(self.mjoint) == len(G.joints)
        # drives: every joint with a target gain; coordinate and target exactly as the example computes them
        ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        qd_start = model.joint_qd_start.numpy()
        eff = model.joint_effort_limit.numpy()
        phi_closed, phi_open = math.radians(P.phi_max), math.radians(P.phi_min)
        self.drives = []
        for j in info["drive"]:
            mj, p, c = self.mjoint[j]
            dof = int(qd_start[mj])
            self.drives.append(
                {
                    "j": j,
                    "p": p,
                    "c": c,
                    "ke": float(ke[dof]),
                    "kd": float(kd[dof]),
                    "effort": float(eff[dof]),
                    "q_closed": info["sign"][j] * G.joint_q_of_phi(j, phi_closed),
                    "q_open": info["sign"][j] * G.joint_q_of_phi(j, phi_open),
                    "Xp": self.joint_frame(j, p),
                    "Xc": self.joint_frame(j, c),
                }
            )
        driven = set(np.nonzero(ke > 0)[0])
        assert driven == {int(qd_start[self.mjoint[d["j"]][0]]) for d in self.drives}, "drives differ from the model"
        # dq/dphi of each drive: axis . (w_child - w_parent); the joint angle is linear in phi
        self.kappa = np.array([self.drive_axis0(d) @ (self.w[d["c"]] - self.w[d["p"]]) for d in self.drives])
        self.ke = np.array([d["ke"] for d in self.drives])
        self.kd = np.array([d["kd"] for d in self.drives])
        self.q_closed = np.array([d["q_closed"] for d in self.drives])
        self.q_open = np.array([d["q_open"] for d in self.drives])
        # the scissor of hub 0 (the driven hub in --drive one), measured like the example's Metrics
        self.scissor = G.scissors[0]

    def joint_frame(self, j, body):
        """Rotation of graph joint j's frame in ``body``'s frame (float64, as build_model computes it)."""
        J, B = self.G.joints[j], self.G.bodies[body]
        return B["R"].T @ J["R"]

    def drive_axis0(self, d):
        return self.R0[d["p"]] @ d["Xp"][:, 2]

    # -------------------------------------------------------------------------------- closed-form kinematics

    def rho(self, phi):
        return self.arm * np.cos(phi / 2) / self.s_half

    def shape(self, phi):
        """Body origins and rotations in the structure frame at scissor angle phi."""
        d = 0.5 * self.sigma * (phi - self.phi0)
        c, s = np.cos(d), np.sin(d)
        Rz = np.zeros((self.nb, 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
        return self.rho(phi) * self.e, self.R0 @ Rz

    def body_q(self, phi, x0=None, q0=None):
        """Newton body_q (x, y, z, qx, qy, qz, qw) of every body for scissor angle phi and rigid pose (x0, q0)."""
        pos, R = self.shape(phi)
        if q0 is not None:
            R0 = quat_to_mat(q0)
            pos, R = pos @ R0.T, R0 @ R
        if x0 is not None:
            pos = pos + x0
        return np.concatenate([pos, mat_to_quat(R)], axis=1)

    def drive_q(self, phi):
        """Joint coordinates of the driven joints from the poses (relative rotation of the joint frames)."""
        _, R = self.shape(phi)
        out = []
        for d in self.drives:
            Rrel = (R[d["p"]] @ d["Xp"]).T @ (R[d["c"]] @ d["Xc"])
            out.append(math.atan2(Rrel[1, 0], Rrel[0, 0]))
        return np.array(out)

    def drive_target(self, t):
        return self.q_closed + raised_cosine(t, self.P.period) * (self.q_open - self.q_closed)

    def drive_force(self, t, phi, phid):
        """Generalized drive force on phi, and the per-joint torques."""
        q = self.kappa * (phi - self.phi0)  # == drive_q(phi), checked in validate()
        tau = self.ke * (self.drive_target(t) - q) - self.kd * self.kappa * phid
        eff = np.array([d["effort"] for d in self.drives])
        tau = np.clip(tau, -eff, eff)
        return float(np.sum(tau * self.kappa)), tau

    # -------------------------------------------------------------------------------- reduced dynamics (Kane)

    def terms(self, phi, R0, nu):
        """Mass matrix (7x7) and velocity-dependent bias for quasi-velocities nu = (phidot, v0, w0)."""
        phid, w0 = nu[0], nu[4:7]
        pos, Q = self.shape(phi)
        drho, ddrho = -self.arm * math.sin(phi / 2) / (2 * self.s_half), -0.25 * self.rho(phi)
        qc = np.einsum("bij,bj->bi", Q, self.com)  # COM offset in the structure frame
        c = pos + qc  # body COMs, and their first and second derivatives in phi
        n_x_qc = np.cross(self.nrm, qc)
        dc = drho * self.e + 0.5 * self.sigma[:, None] * n_x_qc
        ddc = ddrho * self.e + 0.25 * self.sigma[:, None] ** 2 * np.cross(self.nrm, n_x_qc)
        r, rdc, rddc, rw = c @ R0.T, dc @ R0.T, ddc @ R0.T, self.w @ R0.T
        Rw = R0 @ Q
        Iw = Rw @ self.inertia @ np.swapaxes(Rw, 1, 2)
        nb = self.nb
        Jv = np.zeros((nb, 3, 7))
        Jv[:, :, 0] = rdc
        Jv[:, :, 1:4] = np.eye(3)
        Jv[:, :, 4:7] = -skew(r)
        Jw = np.zeros((nb, 3, 7))
        Jw[:, :, 0] = rw
        Jw[:, :, 4:7] = np.eye(3)
        Mm = np.einsum("b,bik,bil->kl", self.mass, Jv, Jv) + np.einsum("bik,bij,bjl->kl", Jw, Iw, Jw)
        Om = w0 + rw * phid
        a_b = np.cross(w0, np.cross(w0, r)) + 2.0 * np.cross(w0, rdc * phid) + rddc * phid**2
        al_b = np.cross(w0, rw * phid)
        IwOm = np.einsum("bij,bj->bi", Iw, Om)
        tq = np.einsum("bij,bj->bi", Iw, al_b) + np.cross(Om, IwOm)
        bias = np.einsum("bik,bi->k", Jv, self.mass[:, None] * a_b) + np.einsum("bik,bi->k", Jw, tq)
        return Mm, bias, {"r": r, "V": np.einsum("bik,k->bi", Jv, nu), "Om": Om, "Iw": Iw, "x": r}

    def rhs(self, t, y):
        phi, _x0, q0, nu = y[0], y[1:4], y[4:8], y[8:15]
        R0 = quat_to_mat(q0)
        Mm, bias, _ = self.terms(phi, R0, nu)
        Qphi, _ = self.drive_force(t, phi, nu[0])
        gen = np.zeros(7)
        gen[0] = Qphi
        nud = np.linalg.solve(Mm, gen - bias)
        w0 = nu[4:7]
        qd = 0.5 * quat_mul(np.array([w0[0], w0[1], w0[2], 0.0]), q0)
        return np.concatenate([[nu[0]], nu[1:4], qd, nud, [Qphi * nu[0]]])

    def y0(self):
        y = np.zeros(16)
        y[0] = self.phi0
        y[7] = 1.0
        return y

    def solve(self, t_end, t_eval, rtol=1e-12, atol=1e-14):
        from scipy.integrate import solve_ivp

        return solve_ivp(self.rhs, (0.0, t_end), self.y0(), method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)

    def diagnostics(self, t, y):
        """Kinetic energy, drive work, linear momentum (about origin) and angular momentum of a state."""
        phi, x0, q0, nu = y[0], y[1:4], y[4:8], y[8:15]
        Mm, _, k = self.terms(phi, quat_to_mat(q0), nu)
        X = x0 + k["r"]
        P = (self.mass[:, None] * k["V"]).sum(0)
        L = (np.cross(X, self.mass[:, None] * k["V"]) + np.einsum("bij,bj->bi", k["Iw"], k["Om"])).sum(0)
        return 0.5 * nu @ Mm @ nu, y[15], P, L

    def com_world(self, body_q):
        """Mass-weighted centre of a body_q array (frames x bodies x 7)."""
        R = quat_to_mat(body_q[..., 3:7])
        c = body_q[..., :3] + np.einsum("...ij,...j->...i", R, np.broadcast_to(self.com, body_q[..., :3].shape))
        return np.einsum("b,...bi->...i", self.mass, c) / self.M


# ----------------------------------------------------------------------------------------------- maximal coordinates


class Maximal:
    """Second formulation for validation: 150 free bodies (Newton-Euler at the COM) and the 240 revolute joints of
    the graph as 1200 acceleration-level constraint rows (point + two axis-alignment rows each, rank 893), with mild
    Baumgarte stabilization. (A fixed independent subset of rows does not work: one picked at the build pose turns
    singular below phi = 55 deg, while the full Jacobian stays well conditioned.) It uses only the joint graph (float64 frames), the
    model's mass properties and the drives, none of the closed form."""

    def __init__(self, mech: Mechanism, baumgarte=20.0):
        self.m = mech
        G = mech.G
        self.alpha = baumgarte
        nb = mech.nb
        self.nb = nb
        jp, jc, ap, ac, zp, t1, t2 = [], [], [], [], [], [], []
        for J in G.joints:
            Bp, Bc = G.bodies[J["p"]], G.bodies[J["c"]]
            jp.append(J["p"])
            jc.append(J["c"])
            ap.append(Bp["R"].T @ (J["point"] - Bp["pos"]) - mech.com[J["p"]])
            ac.append(Bc["R"].T @ (J["point"] - Bc["pos"]) - mech.com[J["c"]])
            zp.append(Bp["R"].T @ J["R"][:, 2])
            t1.append(Bc["R"].T @ J["R"][:, 0])
            t2.append(Bc["R"].T @ J["R"][:, 1])
        self.jp, self.jc = np.array(jp), np.array(jc)
        self.ap, self.ac, self.zp, self.t1, self.t2 = map(np.array, (ap, ac, zp, t1, t2))
        self.nj = len(jp)
        # drive torques act on the model's parent/child with the model's joint frames
        R0 = mech.R0
        x = np.array([B["pos"] for B in G.bodies]) + np.einsum("bij,bj->bi", R0, mech.com)
        self.state0 = np.concatenate([x, mat_to_quat(R0), np.zeros((nb, 6))], axis=1).reshape(-1)

    def constraints(self, x, R, v, w):
        """Constraint Jacobian (sparse), acceleration bias and position/velocity errors."""
        from scipy.sparse import coo_matrix

        jp, jc, nj, nb = self.jp, self.jc, self.nj, self.nb
        rp = np.einsum("jab,jb->ja", R[jp], self.ap)
        rc = np.einsum("jab,jb->ja", R[jc], self.ac)
        u = np.einsum("jab,jb->ja", R[jp], self.zp)
        wk = [np.einsum("jab,jb->ja", R[jc], t) for t in (self.t1, self.t2)]
        wp_, wc_ = w[jp], w[jc]
        rows, cols, vals = [], [], []

        def put(r0, body, col0, block):  # block: (nj, k, 3)
            k = block.shape[1]
            rr = (r0[:, None, None] + np.arange(k)[None, :, None]) * np.ones((1, 1, 3), int)
            cc = (6 * body[:, None, None] + col0 + np.arange(3)[None, None, :]) * np.ones((1, k, 1), int)
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(block.ravel())

        base = 5 * np.arange(nj)
        eye = np.broadcast_to(np.eye(3), (nj, 3, 3))
        put(base, jp, 0, eye)
        put(base, jp, 3, -skew(rp))
        put(base, jc, 0, -eye)
        put(base, jc, 3, skew(rc))
        C = [x[jp] + rp - x[jc] - rc]
        Cd = [v[jp] + np.cross(wp_, rp) - v[jc] - np.cross(wc_, rc)]
        bias = [np.cross(wp_, np.cross(wp_, rp)) - np.cross(wc_, np.cross(wc_, rc))]
        for k, wv in enumerate(wk):
            uxw = np.cross(u, wv)
            put(base + 3 + k, jp, 3, uxw[:, None, :])
            put(base + 3 + k, jc, 3, -uxw[:, None, :])
            C.append(np.sum(u * wv, 1)[:, None])
            Cd.append(np.sum(uxw * (wp_ - wc_), 1)[:, None])
            dd = np.cross(np.cross(wp_, u), wv) + np.cross(u, np.cross(wc_, wv))
            bias.append(np.sum(dd * (wp_ - wc_), 1)[:, None])
        Gm = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(5 * nj, 6 * nb)
        ).tocsr()
        cat = lambda a: np.concatenate(a, axis=1).reshape(-1)
        return Gm, cat(bias), (cat(C), cat(Cd))

    def unpack(self, y):
        s = y.reshape(self.nb, 13)
        return s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13]

    def drive_torques(self, t, R, w):
        """World torque on every body from the drives, and the drive coordinates."""
        m = self.m
        tq = np.zeros((self.nb, 3))
        qs = []
        for d, qt, ke, kd in zip(m.drives, m.drive_target(t), m.ke, m.kd, strict=True):
            Fp, Fc = R[d["p"]] @ d["Xp"], R[d["c"]] @ d["Xc"]
            Rrel = Fp.T @ Fc
            q = math.atan2(Rrel[1, 0], Rrel[0, 0])
            axis = 0.5 * (Fp[:, 2] + Fc[:, 2])
            axis /= np.linalg.norm(axis)
            qd = axis @ (w[d["c"]] - w[d["p"]])
            tau = float(np.clip(ke * (qt - q) - kd * qd, -d["effort"], d["effort"]))
            tq[d["c"]] += tau * axis
            tq[d["p"]] -= tau * axis
            qs.append(q)
        return tq, np.array(qs)

    def rhs(self, t, y):
        from scipy.sparse import bsr_matrix, identity
        from scipy.sparse.linalg import splu

        self.nfev = getattr(self, "nfev", 0) + 1
        if self.nfev % 5000 == 0:
            print(f"    (maximal: t = {t:.3f} s after {self.nfev} evaluations)", flush=True)
        x, q, v, w = self.unpack(y)
        R = quat_to_mat(q)
        m = self.m
        Iw = R @ m.inertia @ np.swapaxes(R, 1, 2)
        Iinv = np.linalg.inv(Iw)
        tq, _ = self.drive_torques(t, R, w)
        f = np.zeros((self.nb, 6))
        f[:, 3:] = tq - np.cross(w, np.einsum("bij,bj->bi", Iw, w))
        blocks = np.zeros((self.nb, 6, 6))
        blocks[:, :3, :3] = np.eye(3) / m.mass[:, None, None]
        blocks[:, 3:, 3:] = Iinv
        Minv = bsr_matrix(
            (blocks, np.arange(self.nb), np.arange(self.nb + 1)), shape=(6 * self.nb, 6 * self.nb)
        ).tocsr()
        Gm, bias, (C, Cd) = self.constraints(x, R, v, w)
        a = self.alpha
        fv = f.reshape(-1)
        b = -bias - 2.0 * a * Cd - a * a * C - Gm @ (Minv @ fv)
        # all 1200 rows (rank 893): the acceleration is unique although the multipliers are not, since the null
        # space of G M^-1 G^T is that of G^T. Solve the slightly regularized system and refine it to convergence.
        A = (Gm @ Minv @ Gm.T).tocsc()
        eps = 1e-9 * A.diagonal().max()
        lu = splu((A + eps * identity(A.shape[0], format="csc")).tocsc())
        lam = lu.solve(b)
        acc = Minv @ (fv + Gm.T @ lam)
        for _ in range(50):
            dl = lu.solve(b - A @ lam)
            da = Minv @ (Gm.T @ dl)
            lam += dl
            acc += da
            if np.abs(da).max() <= 1e-15 * np.abs(acc).max():
                break
        acc = acc.reshape(self.nb, 6)
        qd = 0.5 * quat_mul(np.concatenate([w, np.zeros((self.nb, 1))], 1), q)
        return np.concatenate([v, qd, acc[:, :3], acc[:, 3:]], axis=1).reshape(-1)

    def body_q(self, y):
        x, q, _, _ = self.unpack(y)
        R = quat_to_mat(q)
        return np.concatenate(
            [x - np.einsum("bij,bj->bi", R, self.m.com), q / np.linalg.norm(q, axis=1, keepdims=True)], 1
        )


# ----------------------------------------------------------------------------------------------- reference files


def generate(drive: str, frames: int = FRAMES, rtol: float = 1e-12, atol: float = 1e-14, ex=None, path=None):
    """Integrate the reduced equations and write the reference for one drive mode."""
    ex = ex or load_example()
    mech = Mechanism(ex, drive)
    t_eval = (np.arange(frames) + 1) / FPS
    t0 = time.time()
    sol = mech.solve(t_eval[-1], t_eval, rtol=rtol, atol=atol)
    assert sol.success, sol.message
    Y = sol.y.T
    E, W, P, L = zip(*(mech.diagnostics(t, y) for t, y in zip(t_eval, Y, strict=True)), strict=True)
    _Qphi, tau = zip(*(mech.drive_force(t, y[0], y[8]) for t, y in zip(t_eval, Y, strict=True)), strict=True)
    G = mech.G
    data = {
        "drive": drive,
        "fps": FPS,
        "t": t_eval,
        "phi": Y[:, 0],
        "phi_dot": Y[:, 8],
        "phi_target": np.radians(
            ex.PARAMS.phi_max + raised_cosine(t_eval, ex.PARAMS.period) * (ex.PARAMS.phi_min - ex.PARAMS.phi_max)
        ),
        "hub_radius": mech.rho(Y[:, 0]),
        "rigid_pos": Y[:, 1:4],
        "rigid_quat": Y[:, 4:8] / np.linalg.norm(Y[:, 4:8], axis=1, keepdims=True),
        "rigid_vel": Y[:, 9:12],
        "rigid_omega": Y[:, 12:15],
        "drive_q": np.array([mech.kappa * (p - mech.phi0) for p in Y[:, 0]]),
        "drive_q_target": np.array([mech.drive_target(t) for t in t_eval]),
        "drive_tau": np.array(tau),
        "kinetic_energy": np.array(E),
        "drive_work": np.array(W),
        "momentum": np.array(P),
        "angular_momentum": np.array(L),
        # static data to rebuild every body's pose: body_q = rigid o (rho(phi) e_i, R0_i Rz(sigma_i (phi - phi0) / 2))
        "body_dir": mech.e,
        "body_quat0": mat_to_quat(mech.R0),
        "body_sigma": mech.sigma,
        "body_mass": mech.mass,
        "body_com": mech.com,
        "arm": mech.arm,
        "segments": mech.n,
        "phi0": mech.phi0,
        "scissor_bodies": np.array(G.scissors[0][:2]),
        "scissor_arms": np.array(G.scissors[0][2:]),
        "connectors": np.array([i for i, B in enumerate(G.bodies) if B["kind"] == "connector"]),
        "rtol": rtol,
        "atol": atol,
    }
    path = path or REFERENCE[drive]
    np.savez_compressed(path, **data)
    print(
        f"[{drive}] {frames} frames in {time.time() - t0:.1f} s, nfev {sol.nfev} -> {path} ({path.stat().st_size / 1e3:.0f} kB)"
    )
    return mech, data


# ----------------------------------------------------------------------------------------------- scoring


class Reference:
    """A reference trajectory read back from its .npz, with the body poses rebuilt exactly (float64)."""

    def __init__(self, drive: str = "one", path=None):
        z = np.load(path or REFERENCE[drive])
        self.data = {k: z[k] for k in z.files}
        for k, v in self.data.items():
            setattr(self, k, v)
        self.frames = len(self.t)
        self.s_half = math.sin(math.pi / int(self.segments))
        self.R0 = quat_to_mat(self.body_quat0)
        self.nrm = self.R0[:, :, 2]
        self.M = float(self.body_mass.sum())

    def body_q(self, k=None) -> np.ndarray:
        """body_q (frames x bodies x 7) of frames k (an index, slice or None for all)."""
        sl = np.arange(self.frames)[k if k is not None else slice(None)]
        sl = np.atleast_1d(sl)
        phi = self.phi[sl]
        rho = float(self.arm) * np.cos(phi / 2) / self.s_half
        d = 0.5 * self.body_sigma[None, :] * (phi[:, None] - float(self.phi0))
        zero = np.zeros_like(d)
        rz = np.stack([zero, zero, np.sin(d / 2), np.cos(d / 2)], -1)
        q_shape = quat_mul(self.body_quat0[None], rz)  # R0_i Rz(d)
        q = quat_mul(self.rigid_quat[sl][:, None, :], q_shape)
        Rr = quat_to_mat(self.rigid_quat[sl])
        pos = np.einsum("fij,fbj->fbi", Rr, rho[:, None, None] * self.body_dir[None]) + self.rigid_pos[sl][:, None, :]
        q *= np.where(q[..., 3:4] < 0.0, -1.0, 1.0)
        return np.concatenate([pos, q], -1)

    def com_start(self) -> np.ndarray:
        """Centre of mass at t = 0 (the build pose), where it stays for all time."""
        phi0 = float(self.phi0)
        pos = float(self.arm) * math.cos(phi0 / 2) / self.s_half * self.body_dir
        c = pos + np.einsum("bij,bj->bi", self.R0, self.body_com)
        return (self.body_mass[:, None] * c).sum(0) / self.M


def scissor_phi(body_q: np.ndarray, bodies, arms) -> np.ndarray:
    """Scissor angle [rad] between the right arms of bars A and B of one hub (the example's Metrics), per frame."""
    a = np.einsum("fij,j->fi", quat_to_mat(body_q[:, bodies[0], 3:7]), arms[0])
    b = np.einsum("fij,j->fi", quat_to_mat(body_q[:, bodies[1], 3:7]), arms[1])
    c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.arccos(np.clip(c, -1.0, 1.0))


def rigid_quat(body_q: np.ndarray, connectors) -> np.ndarray:
    """Orientation of the structure frame per frame: the connectors never turn in it, so it is their chordal mean."""
    q = body_q[:, connectors, 3:7].astype(np.float64)
    q = q * np.where(np.sum(q * q[:, :1], -1, keepdims=True) < 0.0, -1.0, 1.0)
    m = q.mean(1)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def rigid_fit_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per point distance between point sets a and b (frames x points x 3) after the best rigid fit of a onto b."""
    ca, cb = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
    H = np.einsum("fpi,fpj->fij", a - ca, b - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.ones((len(a), 3))
    D[:, 2] = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", U, Vt)))
    R = np.einsum("fji,fj,fkj->fik", Vt, D, U)  # V diag(D) U^T
    fit = np.einsum("fij,fpj->fpi", R, a - ca) + cb
    return np.linalg.norm(fit - b, axis=-1)


def score(body_q_frames, drive: str = "one", reference: Reference | None = None) -> dict:
    """Errors of a solver run against the exact reference.

    Args:
        body_q_frames: (frames, 150, 7) array of Newton ``state.body_q`` (x, y, z, qx, qy, qz, qw); frame k is the
            state after k + 1 calls of ``Example.step()`` (t = (k + 1) / 50 s). At most the reference's length.
        drive: ``"one"`` or ``"all"`` (the example's ``--drive``).
        reference: an already loaded :class:`Reference` (optional).

    Returns:
        dict of floats: ``pos_max_mm`` / ``pos_rms_mm`` (body origins, over all frames and bodies),
        ``rot_max_deg`` (body orientations), ``phi_max_err_deg`` (scissor angle of the driven hub),
        ``final_pos_max_mm`` / ``final_pos_rms_mm`` / ``final_phi_err_deg`` (last frame),
        ``orient_max_err_deg`` / ``final_orient_err_deg`` (rigid orientation of the structure frame, from the
        connectors), ``com_drift_mm`` (largest COM displacement from its start; the reference's is 0),
        ``shape_pos_max_mm`` / ``shape_pos_rms_mm`` (body origins after the best rigid fit per frame: the mechanism's
        error with the free-body drift taken out), ``frames``.
    """
    ref = reference or Reference(drive)
    if str(ref.drive) != drive:
        raise ValueError(f"reference is for --drive {ref.drive}, not {drive}")
    bq = np.asarray(body_q_frames, dtype=np.float64)
    if bq.ndim == 2:
        bq = bq[None]
    F = len(bq)
    if F > ref.frames:
        raise ValueError(f"{F} frames but the reference has {ref.frames}")
    rq = ref.body_q(slice(0, F))
    dpos = np.linalg.norm(bq[..., :3] - rq[..., :3], axis=-1)
    drot = quat_angle(bq[..., 3:7], rq[..., 3:7])
    phi = scissor_phi(bq, ref.scissor_bodies, ref.scissor_arms)
    dphi = np.abs(phi - ref.phi[:F])
    dor = quat_angle(rigid_quat(bq, ref.connectors), ref.rigid_quat[:F])
    R = quat_to_mat(bq[..., 3:7])
    com = (bq[..., :3] + np.einsum("fbij,bj->fbi", R, ref.body_com)) * ref.body_mass[None, :, None]
    com = com.sum(1) / ref.M
    com0 = ref.com_start()  # zero momentum: the reference's COM never moves
    shape = rigid_fit_residual(bq[..., :3], rq[..., :3])
    return {
        "frames": F,
        "pos_max_mm": float(1e3 * dpos.max()),
        "pos_rms_mm": float(1e3 * np.sqrt(np.mean(dpos**2))),
        "rot_max_deg": float(np.degrees(drot.max())),
        "phi_max_err_deg": float(np.degrees(dphi.max())),
        "final_pos_max_mm": float(1e3 * dpos[-1].max()),
        "final_pos_rms_mm": float(1e3 * np.sqrt(np.mean(dpos[-1] ** 2))),
        "final_phi_err_deg": float(np.degrees(dphi[-1])),
        "orient_max_err_deg": float(np.degrees(dor.max())),
        "final_orient_err_deg": float(np.degrees(dor[-1])),
        "com_drift_mm": float(1e3 * np.linalg.norm(com - com0, axis=-1).max()),
        "shape_pos_max_mm": float(1e3 * shape.max()),
        "shape_pos_rms_mm": float(1e3 * np.sqrt(np.mean(shape**2))),
    }


# ----------------------------------------------------------------------------------------------- validation


def validate(ex, mech: Mechanism, data: dict, maximal_seconds: float = 0.0):
    """Independent checks of one reference; prints and returns the numbers."""
    print(f"[{mech.drive}] validation", flush=True)
    out = _Printed()
    G, model = mech.G, mech.model
    # 1. closed form == the example's analytic_body_q, and the drive coordinate == the example's joint_q_of_phi
    phis = np.radians(np.linspace(25.0, 125.0, 11))
    out["closed form vs example's analytic_body_q: pos [m]"] = max(
        np.abs(mech.body_q(p)[:, :3] - G.analytic_body_q(p)[:, :3]).max() for p in phis
    )
    out["closed form vs example's analytic_body_q: rot [rad] (float32 quats there)"] = max(
        quat_angle(mech.body_q(p)[:, 3:], G.analytic_body_q(p)[:, 3:]).max() for p in phis
    )
    e_q = max(np.abs(mech.drive_q(p) - mech.kappa * (p - mech.phi0)).max() for p in phis)
    e_q2 = max(
        abs(mech.drive_q(p)[i] - mech.info["sign"][d["j"]] * G.joint_q_of_phi(d["j"], p))
        for p in phis
        for i, d in enumerate(mech.drives)
    )
    out["drive q: poses vs kappa (phi - phi0) vs example's joint_q_of_phi [rad]"] = max(e_q, e_q2)
    # 2. Newton's own inverse kinematics on a reconstructed pose: the driven coordinate in model.joint_q
    import newton
    import warp as wp

    st = model.state()
    phi_t = math.radians(55.0)
    st.body_q.assign(
        wp.array(
            mech.body_q(phi_t, np.array([0.01, -0.02, 0.03]), mat_to_quat(quat_to_mat([0.1, 0.2, 0.3, 0.9]))).astype(
                np.float32
            ),
            dtype=wp.transform,
            device=model.device,
        )
    )
    jq = wp.zeros_like(model.joint_q)
    jqd = wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, st, jq, jqd)
    qs = model.joint_q_start.numpy()
    jqn = jq.numpy()
    dq = np.array(
        [jqn[qs[mech.mjoint[d["j"]][0]]] - mech.kappa[i] * (phi_t - mech.phi0) for i, d in enumerate(mech.drives)]
    )
    out["drive q vs Newton eval_ik, mod 2 pi [rad] (float32)"] = float(np.abs((dq + np.pi) % (2 * np.pi) - np.pi).max())
    # 3. closure: every hinge of the model (its float32 frames) on reconstructed poses, over the whole reference
    Xp, Xc = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    jp_, jc_ = model.joint_parent.numpy(), model.joint_child.numpy()
    rev = [mj for mj, _, _ in mech.mjoint.values()]
    worst_gap, worst_ang = 0.0, 0.0
    for k in range(0, len(data["t"]), 25):
        bq = mech.body_q(data["phi"][k], data["rigid_pos"][k], data["rigid_quat"][k])
        Rb = quat_to_mat(bq[:, 3:])
        a = bq[jp_[rev], :3] + np.einsum("jab,jb->ja", Rb[jp_[rev]], Xp[rev, :3])
        b = bq[jc_[rev], :3] + np.einsum("jab,jb->ja", Rb[jc_[rev]], Xc[rev, :3])
        za = (Rb[jp_[rev]] @ quat_to_mat(Xp[rev, 3:]))[:, :, 2]
        zb = (Rb[jc_[rev]] @ quat_to_mat(Xc[rev, 3:]))[:, :, 2]
        worst_gap = max(worst_gap, np.linalg.norm(a - b, axis=1).max())
        worst_ang = max(worst_ang, np.arccos(np.clip(np.sum(za * zb, 1), -1, 1)).max())
    out["closure_gap_m (model float32 frames)"] = worst_gap
    out["closure_axis_rad"] = worst_ang
    # 4. constraint Jacobian of the graph (float64 frames): null space is exactly 7, and holds the closed-form velocities
    mx = Maximal(mech)
    worst_null, gaps = 0.0, []
    for p in np.radians([30.0, 75.0, 120.0]):
        pos, Rb = mech.shape(p)
        x = pos + np.einsum("bij,bj->bi", Rb, mech.com)
        Gm, _, (C, _) = mx.constraints(x, Rb, np.zeros_like(x), np.zeros_like(x))
        sv = np.linalg.svd(Gm.toarray(), compute_uv=False)
        gaps.append((sv[-8], sv[-7]))
        nu_cols = []
        for col in range(7):
            nu = np.zeros(7)
            nu[col] = 1.0
            _, _, k = mech.terms(p, np.eye(3), nu)
            nu_cols.append(np.concatenate([k["V"], k["Om"]], 1).reshape(-1))
        resid = Gm @ np.array(nu_cols).T
        worst_null = max(worst_null, np.abs(resid).max(), np.abs(C).max())
    out["svd: 8th / 7th smallest singular value"] = [(float(a), float(b)) for a, b in gaps]
    out["closed-form velocities in null space (max |G v|), closure"] = worst_null
    # 5. energy and momentum
    E, W, P, L = data["kinetic_energy"], data["drive_work"], data["momentum"], data["angular_momentum"]
    out["max |KE - work| / max KE"] = float(np.abs(E - W).max() / E.max())
    out["max KE [J]"] = float(E.max())
    out["max |P| [kg m/s]"] = float(np.linalg.norm(P, axis=1).max())
    out["max |L| [kg m^2/s]"] = float(np.linalg.norm(L, axis=1).max())
    # 6. tolerance study: rtol 1e-10 against the reference (rtol 1e-12)
    t_eval = data["t"]
    sol = mech.solve(t_eval[-1], t_eval, rtol=1e-10, atol=1e-12)
    ref = Reference(mech.drive)
    bq_ref = ref.body_q()
    bq_lo = np.array([mech.body_q(y[0], y[1:4], y[4:8]) for y in sol.y.T])
    out["rtol 1e-10 vs 1e-12: max pos diff [m]"] = float(
        np.linalg.norm(bq_lo[..., :3] - bq_ref[..., :3], axis=-1).max()
    )
    out["rtol 1e-10 vs 1e-12: max phi diff [rad]"] = float(np.abs(sol.y[0] - data["phi"]).max())
    from scipy.integrate import solve_ivp

    sol = solve_ivp(mech.rhs, (0.0, t_eval[-1]), mech.y0(), method="Radau", t_eval=t_eval, rtol=1e-11, atol=1e-13)
    bq_ra = np.array([mech.body_q(y[0], y[1:4], y[4:8]) for y in sol.y.T])
    out["Radau rtol 1e-11 vs DOP853: max pos diff [m]"] = float(
        np.linalg.norm(bq_ra[..., :3] - bq_ref[..., :3], axis=-1).max()
    )
    # 7. the reference's own reconstruction == the Mechanism's (the scoring path), and score(ref) == 0
    bq_m = np.array(
        [mech.body_q(p, x, q) for p, x, q in zip(data["phi"], data["rigid_pos"], data["rigid_quat"], strict=True)]
    )
    out["Reference.body_q vs Mechanism.body_q [m]"] = float(np.abs(bq_m[..., :3] - bq_ref[..., :3]).max())
    out["score(reference)"] = score(bq_ref, mech.drive)
    # 8. maximal coordinates
    if maximal_seconds > 0.0:
        nf = round(maximal_seconds * FPS)
        te = t_eval[:nf]
        t0 = time.time()
        sm = solve_ivp(mx.rhs, (0.0, te[-1]), mx.state0, method="DOP853", t_eval=te, rtol=1e-10, atol=1e-12)
        bq_mx = np.array([mx.body_q(y) for y in sm.y.T])
        d = np.linalg.norm(bq_mx[..., :3] - bq_ref[:nf, ..., :3], axis=-1)
        out["maximal: seconds, nfev, wall"] = (float(te[-1]), int(sm.nfev), round(time.time() - t0, 1))
        out["maximal vs reduced: max pos diff [m]"] = float(d.max())
        out["maximal vs reduced: max rot diff [rad]"] = float(quat_angle(bq_mx[..., 3:], bq_ref[:nf, ..., 3:]).max())
        out["maximal: score"] = score(bq_mx, mech.drive)
        last = sm.y.T[-1]
        x, q, v, w = mx.unpack(last)
        _, _, (C, _Cd) = mx.constraints(x, quat_to_mat(q), v, w)
        out["maximal: final constraint error"] = float(np.abs(C).max())
    return dict(out)


class _Printed(dict):
    """A dict that prints every entry as it is set (validation steps take minutes)."""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        print(f"    {key}: {value}", flush=True)


def summary(mech: Mechanism, data: dict):
    """The drive as a 1-DOF oscillator in phi, the rigid coupling, and how far phi lags its target."""
    meff, coup, lin = [], [], []
    for p in np.radians(np.linspace(30.0, 120.0, 19)):
        Mm, _, _ = mech.terms(p, np.eye(3), np.zeros(7))
        meff.append(Mm[0, 0] - Mm[0, 1:] @ np.linalg.solve(Mm[1:, 1:], Mm[1:, 0]))  # locked (zero-momentum) inertia
        coup.append(np.abs(np.linalg.solve(Mm[1:, 1:], Mm[1:, 0])[3:]).max())  # rigid spin per unit phidot
        lin.append(np.abs(np.linalg.solve(Mm[1:, 1:], Mm[1:, 0])[:3]).max())
    k = float(np.sum(mech.ke * mech.kappa**2))
    c = float(np.sum(mech.kd * mech.kappa**2))
    m0, m1 = min(meff), max(meff)
    lag = np.degrees(data["phi"] - data["phi_target"])
    print(
        f"    m_eff {m0:.4e}..{m1:.4e} kg m^2, k_phi {k:.3f} N m/rad, c_phi {c:.4f} N m s/rad, "
        f"w_n {math.sqrt(k / m1):.1f}..{math.sqrt(k / m0):.1f} rad/s, zeta {c / (2 * math.sqrt(k * m1)):.2f}..{c / (2 * math.sqrt(k * m0)):.2f}"
    )
    print(f"    rigid coupling per unit phidot: spin {max(coup):.2e} rad/rad, shift {max(lin):.3e} m/rad")
    print(
        f"    phi - target: {lag.min():.3f}..{lag.max():.3f} deg (kd/ke * max target rate = "
        f"{np.degrees(mech.kd[0] / mech.ke[0] * math.pi / mech.P.period * math.radians(mech.P.phi_max - mech.P.phi_min)):.3f} deg)"
    )
    print(
        f"    phi {np.degrees(data['phi'].min()):.4f}..{np.degrees(data['phi'].max()):.4f} deg, hub radius "
        f"{1e3 * data['hub_radius'].min():.4f}..{1e3 * data['hub_radius'].max():.4f} mm, rigid turn max "
        f"{np.degrees(quat_angle(data['rigid_quat'], np.array([0, 0, 0, 1.0])).max()):.2e} deg, rigid shift max "
        f"{1e3 * np.linalg.norm(data['rigid_pos'], axis=1).max():.4f} mm"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", choices=["one", "all", "both"], default="both")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--check", action="store_true", help="validate the references after writing them")
    parser.add_argument("--maximal", type=float, default=0.0, help="seconds to cross-check in maximal coordinates")
    args = parser.parse_args()
    ex = load_example()
    for drive in ("one", "all") if args.drive == "both" else (args.drive,):
        mech, data = generate(drive, args.frames, args.rtol, ex=ex)
        summary(mech, data)
        if args.check or args.maximal > 0.0:
            validate(ex, mech, data, args.maximal)


if __name__ == "__main__":
    main()
