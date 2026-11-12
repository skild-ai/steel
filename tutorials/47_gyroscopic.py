"""Free-body gyroscopics: a T-handle spun about its middle axis flips (Dzhanibekov effect)."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.6), wp.quat_identity()))
    along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)
    along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
    builder.add_shape_capsule(
        body, xform=wp.transform((0.0, 0.0, 0.0), along_x), radius=0.02, half_height=0.18
    )
    builder.add_shape_capsule(
        body, xform=wp.transform((0.0, 0.09, 0.0), along_y), radius=0.025, half_height=0.07
    )
    model = builder.finalize()
    model.steel.rigid_body_angular_damping.fill_(0.0)  # default 0.05 1/s, as in PhysX
    # spin about the handle's middle axis
    model.joint_qd.assign(np.array([0.0, 0.0, 0.0, 0.02, 8.0, 0.01], dtype=np.float32))
    return model


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(8.0)):
        x, y, z, w = state_0.body_q.numpy()[0, 3:]
        handle_y = 1.0 - 2.0 * (x * x + z * z)  # world y of the body's y axis
        run.frame(state_0, handle_y=handle_y, omega=np.linalg.norm(state_0.body_qd.numpy()[0, 3:]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "spin about the intermediate axis")
    rec.save()
