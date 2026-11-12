"""Penetration recovery: Baumgarte bias adds velocity; velocity_iterations remove it."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(3):
        # each box starts 2 cm into the ground
        box = builder.add_body(xform=wp.transform((0.25 * i - 0.25, 0.0, 0.03), wp.quat_identity()))
        builder.add_shape_box(
            box, hx=0.08, hy=0.08, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5)
        )
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(1.5)):
        run.frame(state_0, z_mm=1000.0 * (state_0.body_q.numpy()[1, 2] - 0.05))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "baumgarte=0.2 (default)")
    simulate(rec, "baumgarte=0.8", baumgarte=0.8)
    simulate(rec, "baumgarte=0.8, velocity_iterations=4", baumgarte=0.8, velocity_iterations=4)
    rec.save()
