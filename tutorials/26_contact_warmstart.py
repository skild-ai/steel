"""contact_warmstart: start each step from the matched contacts' previous impulses."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
N, H = 8, 0.05


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    for i in range(N):
        box = builder.add_body(
            xform=wp.transform((0.0, 0.0, H + 2 * H * i + 0.0002 * i), wp.quat_identity())
        )
        builder.add_shape_box(box, hx=H, hy=H, hz=H, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    return builder.finalize()


def simulate(rec, label, warmstart, iterations, decay=1.0):
    model = build()
    solver = SolverSteel(
        model, iterations=iterations, contact_warmstart=warmstart, contact_warmstart_decay=decay
    )
    # warm start finds last step's impulses through the pipeline's contact matching
    pipeline = newton.CollisionPipeline(
        model, contact_matching="latest" if warmstart else "disabled"
    )
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
    simulate(rec, "iterations=4, cold start (default)", False, 4)
    simulate(rec, "iterations=4, contact_warmstart=True", True, 4)
    simulate(rec, "iterations=4, warm start, decay 0.9", True, 4, decay=0.9)
    simulate(rec, "iterations=12, cold start", False, 12)
    rec.save()
