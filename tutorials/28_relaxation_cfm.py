# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""relaxation over-relaxes each Gauss-Seidel update; cfm shifts each row's diagonal."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
N, H = 8, 0.05  # eight 10 cm boxes
R = 0.05  # m, a 1 kg sphere: its contact row's unit response is about 1 / (1 kg)


def build_stack():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    for i in range(N):
        box = builder.add_body(
            xform=wp.transform((0.0, 0.0, H + 2 * H * i + 0.0002 * i), wp.quat_identity())
        )
        builder.add_shape_box(box, hx=H, hy=H, hz=H, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    return builder.finalize()


def build_ball():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(restitution=0.8)
    builder.add_ground_plane(cfg=cfg)
    ball = builder.add_body(xform=wp.transform((0.0, 0.0, 0.56), wp.quat_identity()))
    builder.add_shape_sphere(ball, radius=0.06, cfg=cfg)
    return builder.finalize()


def build_sphere():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    sphere = builder.add_body(xform=wp.transform((0.0, 0.0, R), wp.quat_identity()))
    density = 1.0 / (4.0 / 3.0 * 3.14159265 * R**3)
    builder.add_shape_sphere(sphere, radius=R, cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return builder.finalize()


def simulate(rec, label, scene, **options):
    model = {"stack": build_stack, "ball": build_ball, "rest": build_sphere}[scene]()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model, scene=scene)
    for _ in range(rec.frames(4.0 if scene == "stack" else 2.0)):
        z = state_0.body_q.numpy()[:, 2]
        if scene == "stack":
            run.frame(state_0, sag_mm=1000.0 * (H * (2 * N - 1) - z[N - 1]))
        elif scene == "rest":
            run.frame(state_0, penetration_mm=1000.0 * (R - z[0]))
        else:
            run.frame(state_0, height=z[0] - 0.06)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    for relaxation in [1.0, 1.4, 1.8]:
        default = " (default)" if relaxation == 1.0 else ""
        simulate(
            rec,
            f"stack, relaxation={relaxation}{default}",
            "stack",
            iterations=6,
            relaxation=relaxation,
        )
    simulate(rec, "stack, cfm=1e-2", "stack", iterations=6, cfm=1.0e-2)
    for relaxation in [1.0, 1.8]:
        simulate(rec, f"bounce, relaxation={relaxation}", "ball", relaxation=relaxation)
    # cfm moves the resting point only of a regularized row: the residual is -g (A + cfm) lambda
    simulate(rec, "rest, cfm=1", "rest", cfm=1.0)
    for cfm in [1.0e-6, 1.0]:
        simulate(
            rec, f"rest, regularized, cfm={cfm:g}", "rest", cfm=cfm, contact_regularization=0.5
        )
    rec.save()
