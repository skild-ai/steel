"""Drive gain randomization: per-world joint_target_ke/kd + JOINT_DOF_FORCE_PROPERTIES."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    arm = newton.ModelBuilder()
    link = arm.add_link(xform=tf(0.0, 0.0, 0.6))
    arm.add_shape_box(link, xform=tf(0.15, 0.0, 0.0), hx=0.15, hy=0.02, hz=0.02)
    cfg = newton.ModelBuilder.ShapeConfig(density=8000.0)
    arm.add_shape_sphere(link, xform=tf(0.3, 0.0, 0.0), radius=0.04, cfg=cfg)
    j = arm.add_joint_revolute(
        -1, link, axis=newton.Axis.Z, parent_xform=tf(0.0, 0.0, 0.6), target_ke=20.0, target_kd=2.0
    )
    arm.add_articulation([j])
    builder = newton.ModelBuilder()
    builder.replicate(arm, WORLDS, spacing=(0.0, 0.45, 0.0))
    return builder.finalize()


def simulate(rec, label, ke, kd):
    model = build()
    solver = SolverSteel(model)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for frame in range(rec.frames(5.0)):
        if frame == rec.fps:  # randomize at 1 s, then step the target to 90 degrees
            model.joint_target_ke.assign(ke.astype(np.float32))  # one DOF per world
            model.joint_target_kd.assign(kd.astype(np.float32))
            # the drives are folded into the mass matrix, which this refreshes
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            control.joint_target_q.fill_(math.radians(90.0))
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"ke {p:.1f} kd {d:.2f}": v for p, d, v in zip(ke, kd, q)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    ke = rng.uniform(2.0, 20.0, WORLDS)
    kd = rng.uniform(0.05, 0.5, WORLDS)
    order = np.argsort(ke)
    ke, kd = ke[order], kd[order]
    rec = Recorder()
    simulate(rec, "gains sampled per world", ke, kd)
    rec.save()
