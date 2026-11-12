"""restitution_velocity_threshold: the slowest impact that still bounces."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(restitution=0.8)
    builder.add_ground_plane(cfg=cfg)
    ball = builder.add_body(xform=wp.transform((0.0, 0.0, 0.56), wp.quat_identity()))
    builder.add_shape_sphere(ball, radius=0.06, cfg=cfg)
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        run.frame(state_0, height=state_0.body_q.numpy()[0, 2] - 0.06)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "threshold 0.5 m/s (default)")
    simulate(rec, "threshold 0.05 m/s", restitution_velocity_threshold=0.05)
    simulate(rec, "threshold 1.5 m/s", restitution_velocity_threshold=1.5)
    simulate(rec, "enable_restitution=False", enable_restitution=False)
    rec.save()
