#!/usr/bin/env python3
"""Measure where SolverSteel's contact warm start helps and where it hurts.

Each scene runs many worlds with ``contact_warmstart`` off and on. Both arms use the same collision pipeline
(contact matching on), so the contact order is identical and only the warm start differs.

    python warmstart_experiment.py --scene stack --worlds 128 --out results.jsonl
    python warmstart_experiment.py --scene all --quick

Results of one run are in ``warmstart_results.jsonl`` next to this script.

Scenes: ``stack`` (ten equal boxes), ``ratio`` (six boxes, each three times heavier than the one below),
``pinch_hold`` and ``pinch_carry`` (a two-finger pinch held still or carried up and down), ``pile`` (small
boxes poured into a bin). One JSON line per arm.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import warp as wp

import newton

FPS = 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="all", choices=["all", "stack", "ratio", "pinch_hold", "pinch_carry", "pile"])
    ap.add_argument("--worlds", type=int, default=0, help="override the scene's world count")
    ap.add_argument("--quick", action="store_true", help="few worlds and frames, for a smoke test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    scenes = ["stack", "ratio", "pinch_hold", "pinch_carry", "pile"] if args.scene == "all" else [args.scene]
    out = open(args.out, "a") if args.out else None
    for scene in scenes:
        for arm in ARMS[scene]:
            result = run_arm(scene, dict(arm), args)
            line = json.dumps(result)
            print(line, flush=True)
            if out:
                out.write(line + "\n")
                out.flush()


# Arms per scene: solver options plus scene parameters. Warm start is toggled inside run_arm.
ARMS = {
    "stack": [{"iterations": i, "substeps": 2} for i in (8, 16, 24, 32, 48)],
    "ratio": [{"iterations": i, "substeps": s} for s, i in ((2, 96), (4, 24), (4, 48), (8, 24), (8, 48))],
    "pinch_hold": [
        {"iterations": 8, "substeps": 3, "friction_anchor_beta": beta, "body_frame": bf}
        for beta in (0.0, None)
        for bf in (False, True)
    ],
    "pinch_carry": [
        {"iterations": 8, "substeps": 3, "friction_anchor_beta": beta, "body_frame": bf, "lift_speed": v}
        for v in (0.05, 0.2)
        for beta in (0.0, None)
        for bf in (False, True)
    ],
    "pile": [{"iterations": i, "substeps": 2} for i in (4, 8, 16)],
}


def run_arm(scene: str, arm: dict, args) -> dict:
    results = {"scene": scene, **{k: v for k, v in arm.items()}}
    for warm in (False, True):
        r = simulate(scene, arm, warm, args)
        for k, v in r.items():
            results[f"{'on' if warm else 'off'}_{k}"] = v
    return results


def simulate(scene: str, arm: dict, warm: bool, args) -> dict:
    worlds = args.worlds or {"stack": 128, "ratio": 128, "pinch_hold": 64, "pinch_carry": 64, "pile": 64}[scene]
    seconds = {"stack": 5.0, "ratio": 5.0, "pinch_hold": 5.0, "pinch_carry": 4.0, "pile": 5.0}[scene]
    if args.quick:
        worlds, seconds = min(worlds, 4), min(seconds, 0.5)
    substeps, iterations = arm["substeps"], arm["iterations"]
    dt = 1.0 / (FPS * substeps)
    if scene in ("stack", "ratio"):
        model, info = build_stack(worlds, 10 if scene == "stack" else 6, 1.0 if scene == "stack" else 3.0)
    elif scene == "pile":
        model, info = build_pile(worlds)
    else:
        model, info = build_pinch(worlds, lift=scene == "pinch_carry")
    solver_kwargs = dict(iterations=iterations, contact_warmstart=warm, row_watermark=True)
    if "friction_anchor_beta" in arm:
        solver_kwargs["friction_anchor_beta"] = arm["friction_anchor_beta"]
    if scene.startswith("pinch"):
        solver_kwargs.update(enable_joint_limits=True, articulation_constraint_max=256)
    if scene == "pile":
        solver_kwargs.update(solve_mode="colored", articulated_contact_response="propagation",
                             propagation_constraint_max=8192)
    solver = newton.solvers.SolverSteel(model, **solver_kwargs)
    pipeline = newton.CollisionPipeline(
        model, contact_matching="latest", contact_matching_body_frame=bool(arm.get("body_frame", False))
    )
    contacts = pipeline.contacts()
    control = model.control()
    state_0, state_1 = model.state(), model.state()
    if scene.startswith("pinch"):
        info["setup"](model, control, state_0)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    frames = int(round(seconds * FPS))
    bodies = info["bodies"]
    lift = arm.get("lift_speed", 0.0)
    period = (seconds - 0.5) / 2
    matched = total = 0
    burst_frames = 0
    peak_late_speed = 0.0
    rel0 = None
    frame_ms = []
    collapse_time = np.full(worlds, np.inf)
    for f in range(frames):
        t = (f + 1) / FPS
        if scene == "pinch_carry":
            u = t - 0.5
            if u > 0.0:
                z = lift * period / math.pi * (1.0 - math.cos(math.pi * min(u, 2 * period) / period))
                info["set_hand"](control, z)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if f % 6 == 0 or f == frames - 1:
            n = int(contacts.rigid_contact_count.numpy()[0])
            if n:
                idx = contacts.rigid_contact_match_index.numpy()[:n]
                matched += int((idx >= 0).sum())
                total += n
        if scene == "pile" and t > 1.5:
            qd = state_0.body_qd.numpy().reshape(worlds, bodies, 6)
            speed = np.linalg.norm(qd[:, info["free"], :3], axis=2).max(axis=1)
            burst_frames += int((speed > 2.0).sum())
            peak_late_speed = max(peak_late_speed, float(speed.max()))
        if scene in ("stack", "ratio") and f % 6 == 0:
            top = state_0.body_q.numpy().reshape(worlds, bodies, 7)[:, -1, 2]
            collapse_time[(info["rest_top"] - top > 0.05) & np.isinf(collapse_time)] = t
        if scene == "pinch_carry" and f == 29:
            rel0 = info["box_in_hand"](state_0)
    try:
        solver.check_constraint_capacity()
        capacity_ok = True
    except RuntimeError:
        capacity_ok = False
    q = state_0.body_q.numpy().reshape(worlds, bodies, 7)
    qd = state_0.body_qd.numpy().reshape(worlds, bodies, 6)
    out = {
        "match_rate": round(matched / total, 4) if total else None,
        "ms_per_frame_median": round(float(np.median(frame_ms[len(frame_ms) // 4:])), 3),
        "nonfinite_worlds": int((~np.isfinite(q).all(axis=(1, 2))).sum()),
        "capacity_ok": capacity_ok,
    }
    if scene in ("stack", "ratio"):
        sag = (info["rest_top"] - q[:, -1, 2]) * 1e3
        out.update(
            top_sag_mm_median=round(float(np.median(sag)), 3),
            top_sag_mm_max=round(float(np.max(sag)), 3),
            standing_worlds=int(np.isinf(collapse_time).sum()),
            collapse_time_median_s=(round(float(np.median(collapse_time)), 3) if np.isfinite(np.median(collapse_time))
                                    else None),
            final_speed_mean=round(float(np.linalg.norm(qd[:, :, :3], axis=2).mean()), 5),
        )
    elif scene == "pile":
        out.update(burst_world_frames=burst_frames, peak_speed_after_1_5s=round(peak_late_speed, 3),
                   below_floor=int((q[:, info["free"], 2] < 0.0).sum()))
    else:
        rel = info["box_in_hand"](state_0)
        start = rel0 if scene == "pinch_carry" else info["rel_start"]
        twist = np.degrees(2.0 * np.arccos(np.clip(np.abs((rel[0] * start[0]).sum(axis=1)), 0.0, 1.0)))
        slip = np.linalg.norm(rel[1] - start[1], axis=1) * 1e3
        out.update(twist_deg_median=round(float(np.median(twist)), 3), twist_deg_max=round(float(np.max(twist)), 3),
                   slip_mm_median=round(float(np.median(slip)), 3), slip_mm_max=round(float(np.max(slip)), 3),
                   dropped_worlds=int((slip > 20.0).sum()))
    return out


def build_stack(worlds: int, count: int, ratio: float):
    """``count`` 10 cm cubes on the ground; each cube ``ratio`` times as heavy as the one below."""
    b = newton.ModelBuilder()
    b.add_ground_plane()
    h = 0.05
    for i in range(count):
        cfg = newton.ModelBuilder.ShapeConfig(density=500.0 * ratio**i, mu=0.6)
        body = b.add_body(xform=wp.transform((0.0, 0.0, h + 2 * h * i + 0.0005 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=h, hy=h, hz=h, cfg=cfg)
    top = newton.ModelBuilder()
    top.replicate(b, worlds)
    return top.finalize(), {"bodies": b.body_count, "rest_top": h + 2 * h * (count - 1)}


def build_pile(worlds: int):
    """40 small boxes poured into an open bin of static walls."""
    b = newton.ModelBuilder()
    b.add_ground_plane()
    wall = newton.ModelBuilder.ShapeConfig(mu=0.5)
    for x, y, hx, hy in ((0.0, 0.1, 0.1, 0.005), (0.0, -0.1, 0.1, 0.005), (0.1, 0.0, 0.005, 0.1), (-0.1, 0.0, 0.005, 0.1)):
        b.add_shape_box(-1, xform=wp.transform((x, y, 0.06), wp.quat_identity()), hx=hx, hy=hy, hz=0.06, cfg=wall)
    rng = np.random.default_rng(0)
    piece = newton.ModelBuilder.ShapeConfig(density=800.0, mu=0.5)
    for i in range(40):
        p = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), 0.05 + 0.03 * i)
        q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0, math.pi)))
        body = b.add_body(xform=wp.transform(p, q))
        b.add_shape_box(body, hx=0.012, hy=0.008, hz=0.004, cfg=piece)
    top = newton.ModelBuilder()
    top.replicate(b, worlds)
    return top.finalize(), {"bodies": b.body_count, "free": np.arange(b.body_count)}


def build_pinch(worlds: int, lift: bool):
    """A two-finger pinch of a box whose centre of mass sits off the pinch axis, so only pad friction holds it."""
    half_width, squeeze = 0.039, 0.004
    b = newton.ModelBuilder()
    pad = newton.ModelBuilder.ShapeConfig(mu=1.1, gap=0.001, density=0.0)
    wall = newton.ModelBuilder.ShapeConfig(mu=0.45, gap=0.00015, density=0.0)
    hand_z = 0.4
    hand_xform = wp.transform(wp.vec3(0.0, 0.0, hand_z), wp.quat_identity())
    hand = b.add_link(xform=hand_xform, mass=0.65, inertia=wp.mat33(np.eye(3) * 1e-3))
    if lift:
        slider = newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), target_ke=1.0e5, target_kd=2.0e3,
                                                    armature=0.1, effort_limit=1.0e6,
                                                    actuator_mode=newton.JointTargetMode.POSITION)
        joints = [b.add_joint_prismatic(-1, hand, parent_xform=hand_xform, child_xform=wp.transform_identity(),
                                        axis=slider)]
    else:
        joints = [b.add_joint_fixed(-1, hand, parent_xform=hand_xform)]
    for side in (-1.0, 1.0):
        finger = b.add_link(mass=0.03, inertia=wp.mat33(np.eye(3) * 1e-5))
        for dx in (-0.006, 0.006):
            for dz in (-0.012, 0.012):
                b.add_shape_box(finger, xform=wp.transform(wp.vec3(dx, side * 0.0075, dz), wp.quat_identity()),
                                hx=0.005, hy=0.0075, hz=0.01, cfg=pad)
        dof = newton.ModelBuilder.JointDofConfig(axis=(0.0, side, 0.0), limit_lower=0.0, limit_upper=0.04,
                                                 target_q=0.04, target_ke=2500.0, target_kd=30.0, armature=0.08,
                                                 effort_limit=35.0, actuator_mode=newton.JointTargetMode.POSITION)
        joints.append(b.add_joint_prismatic(hand, finger,
                                            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                                            child_xform=wp.transform_identity(), axis=dof))
    b.add_articulation(joints)
    mass, hx, hy, hz, t = 0.16, 0.065, half_width, 0.1, 0.0015
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, hand_z - 0.08), wp.quat_identity()), mass=mass,
                     com=wp.vec3(0.03, 0.0, -0.06),
                     inertia=wp.mat33(np.diag([mass * (hy**2 + hz**2) / 3, mass * (hx**2 + hz**2) / 3,
                                               mass * (hx**2 + hy**2) / 3])))
    for xf, he in (((0, 0, -hz + t), (hx, hy, t)), ((-hx + t, 0, 0), (t, hy, hz)), ((hx - t, 0, 0), (t, hy, hz)),
                   ((0, -hy + t, 0), (hx, t, hz)), ((0, hy - t, 0), (hx, t, hz))):
        b.add_shape_box(box, xform=wp.transform(wp.vec3(*xf), wp.quat_identity()), hx=he[0], hy=he[1], hz=he[2],
                        cfg=wall)
    b.add_ground_plane()
    bodies, coords = b.body_count, len(b.joint_q)
    top = newton.ModelBuilder()
    top.replicate(b, worlds)
    model = top.finalize()
    first = 1 if lift else 0

    def setup(model, control, state):
        q = model.joint_q.numpy()
        target = control.joint_target_q.numpy()
        for w in range(worlds):
            for k in (0, 1):
                q[w * coords + first + k] = half_width + 0.0005
                target[w * coords + first + k] = half_width - squeeze
        state.joint_q.assign(q)
        control.joint_target_q.assign(target)
        info["target"] = target

    def set_hand(control, z):
        target = info["target"]
        target[[w * coords for w in range(worlds)]] = z
        control.joint_target_q.assign(target)

    def box_in_hand(state):
        """Box orientation and position in the hand frame, per world."""
        bq = state.body_q.numpy().reshape(worlds, bodies, 7)
        qh, qb = bq[:, 0, 3:7], bq[:, box, 3:7]
        inv = qh * np.array([-1.0, -1.0, -1.0, 1.0])
        rel_q = quat_mul(inv, qb)
        rel_p = quat_rotate(inv, bq[:, box, :3] - bq[:, 0, :3])
        return rel_q, rel_p

    info = {"bodies": bodies, "setup": setup, "set_hand": set_hand, "box_in_hand": box_in_hand}
    probe = model.state()
    q = model.joint_q.numpy()
    for w in range(worlds):
        for k in (0, 1):
            q[w * coords + first + k] = half_width + 0.0005
    probe.joint_q.assign(q)
    newton.eval_fk(model, probe.joint_q, probe.joint_qd, probe)
    info["rel_start"] = box_in_hand(probe)
    return model, info


def quat_mul(a, b):
    x1, y1, z1, w1 = a.T
    x2, y2, z2, w2 = b.T
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=1)


def quat_rotate(q, v):
    qv = np.concatenate([v, np.zeros((len(v), 1))], axis=1)
    conj = q * np.array([-1.0, -1.0, -1.0, 1.0])
    return quat_mul(quat_mul(q, qv), conj)[:, :3]


if __name__ == "__main__":
    main()
