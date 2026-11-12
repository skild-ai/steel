"""Joint friction randomization: per-world joint_friction, enabled at construction."""

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
    pendulum = newton.ModelBuilder()
    link = pendulum.add_link(xform=tf(0.0, 0.0, 0.8))
    pendulum.add_shape_capsule(link, xform=tf(0.0, 0.0, -0.2), radius=0.015, half_height=0.18)
    pendulum.add_shape_sphere(link, xform=tf(0.0, 0.0, -0.4), radius=0.05)
    j = pendulum.add_joint_revolute(-1, link, axis=newton.Axis.Y, parent_xform=tf(0.0, 0.0, 0.8))
    pendulum.add_articulation([j])
    pendulum.joint_q[-1] = math.radians(70.0)
    builder = newton.ModelBuilder()
    builder.replicate(pendulum, WORLDS, spacing=(0.0, 0.3, 0.0))
    return builder.finalize()


def simulate(rec, label, friction, enable):
    model = build()
    # the friction rows exist only if they are enabled here; this cannot be switched on later
    solver = SolverSteel(model, enable_joint_friction=enable)
    model.joint_friction.assign(friction.astype(np.float32))  # N m, one DOF per world
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(5.0)):
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"friction {f:.3f}": a for f, a in zip(friction, q)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    friction = np.sort(rng.uniform(0.0, 0.6, WORLDS))
    rec = Recorder()
    simulate(rec, "enable_joint_friction=True", friction, True)
    simulate(rec, "enable_joint_friction=False (default): ignored", friction, False)
    rec.save()
