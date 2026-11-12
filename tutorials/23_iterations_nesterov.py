"""Iterations and Nesterov momentum on a tall stack."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
N, H = 8, 0.05  # eight 10 cm boxes


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    for i in range(N):
        box = builder.add_body(
            xform=wp.transform((0.0, 0.0, H + 2 * H * i + 0.0002 * i), wp.quat_identity())
        )
        builder.add_shape_box(box, hx=H, hy=H, hz=H, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        top = state_0.body_q.numpy()[N - 1, 2]
        run.frame(state_0, sag_mm=1000.0 * (H * (2 * N - 1) - top))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "iterations=6", iterations=6)
    simulate(rec, "iterations=6, nesterov_momentum=True", iterations=6, nesterov_momentum=True)
    simulate(rec, "iterations=12 (default)")
    simulate(rec, "iterations=12, nesterov_momentum=True", nesterov_momentum=True)
    rec.save()
