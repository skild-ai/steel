# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Joint reaction wrenches from the body_parent_f observable."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
LINKS = 6


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for i in range(LINKS):
        link = builder.add_link(xform=tf(0.0, 0.0, 1.0 - 0.1 * i - 0.05))
        builder.add_shape_capsule(
            link, radius=0.02, half_height=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0)
        )
        anchor = tf(0.0, 0.0, 1.0) if parent < 0 else tf(0.0, 0.0, -0.05)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=anchor,
                child_xform=tf(0.0, 0.0, 0.05),
                damping=0.005,  # explicit: keep it small for light links
            )
        )
        builder.joint_q[-1] = 0.6 if i == 0 else 0.0
        parent = link
    builder.add_articulation(joints)
    return builder.finalize()


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observed = solver.observables({"body_parent_f"})
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    weight = model.body_mass.numpy().sum() * 9.81

    run = rec.variant(label, model)
    for _ in range(rec.frames(5.0)):
        root = observed.body_parent_f.numpy()[0]  # (force, torque) the world applies to link 0
        run.frame(state_0, root_force_z=root[2], root_force_x=root[0], chain_weight=weight)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT, observables=observed)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "six-link chain released at 0.6 rad")
    rec.save()
