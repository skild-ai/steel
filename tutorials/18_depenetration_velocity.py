"""Cap how fast overlapping bodies are pushed apart: steel:rigid_body_max_depenetration_velocity."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build(max_speed):
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    builder.add_ground_plane()
    # three boxes spawned 3 cm into each other and 2 cm into the ground
    for i in range(3):
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.03 + 0.07 * i), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.08, hy=0.08, hz=0.05)
    model = builder.finalize()
    model.steel.rigid_body_max_depenetration_velocity.fill_(max_speed)
    return model


def simulate(rec, label, max_speed, **options):
    model = build(max_speed)
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(1.5)):
        q, qd = state_0.body_q.numpy(), state_0.body_qd.numpy()
        run.frame(state_0, top_z=q[2, 2], top_speed=abs(qd[2, 2]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "unbounded (default)", float("inf"))
    simulate(rec, "max 0.2 m/s", 0.2)
    rec.save()
