# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Random pushes: kick each world's base through state.joint_qd, or push it with state.body_f."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
PUSH_TIMES = (1.0, 2.5)  # s
PULSE = 0.1  # s, duration of a force push


def build():
    base = newton.ModelBuilder()
    body = base.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
    base.add_shape_box(body, hx=0.08, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    builder = newton.ModelBuilder()
    builder.replicate(base, WORLDS, spacing=(0.0, 0.6, 0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    return builder.finalize()


def simulate(rec, label, kicks, method):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[:, :2].copy()
    mass = model.body_mass.numpy()

    run = rec.variant(label, model)
    push_frames = {round(t * rec.fps): k for k, t in enumerate(PUSH_TIMES)}
    force, push_end = np.zeros((WORLDS, 6), np.float32), -1
    for frame in range(rec.frames(4.0)):
        k = push_frames.get(frame)
        if k is not None and method == "joint_qd":
            # Steel's state is joint_q/joint_qd; a free joint's qd is (v_com, omega) in world frame
            qd = state_0.joint_qd.numpy().reshape(WORLDS, 6)
            qd[:, :2] += kicks[k]
            state_0.joint_qd.assign(qd.reshape(-1))
        elif k is not None and method == "body_qd":
            qd = state_0.body_qd.numpy()  # an output for Steel, recomputed from joint_qd
            qd[:, :2] += kicks[k]
            state_0.body_qd.assign(qd)
        elif k is not None and method == "body_f":
            force[:, :2] = kicks[k] * mass[:, None] / PULSE  # same impulse as the kick
            push_end = frame + round(PULSE * rec.fps)
        slide = np.linalg.norm(state_0.body_q.numpy()[:, :2] - start, axis=1)
        run.frame(state_0, **{f"world {w}": d for w, d in enumerate(slide)})
        for _ in range(SUBSTEPS):
            # world-frame (force, torque) at the COM; set it on the input state of every step
            state_0.body_f.assign(force if frame < push_end else np.zeros_like(force))
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    angle = rng.uniform(0.0, 2.0 * np.pi, (len(PUSH_TIMES), WORLDS))
    speed = rng.uniform(0.5, 1.5, (len(PUSH_TIMES), WORLDS))  # m/s
    kicks = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * speed[..., None]
    rec = Recorder()
    simulate(rec, "velocity kick via state.joint_qd", kicks, "joint_qd")
    simulate(rec, "force pulse via state.body_f", kicks, "body_f")
    simulate(rec, "velocity kick via state.body_qd: ignored", kicks, "body_qd")
    rec.save()
