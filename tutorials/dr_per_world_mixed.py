# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Several randomizations on a quadruped: one notify with OR-ed flags, then a partial reset."""

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 4
STANCE = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
FLAGS = (
    newton.ModelFlags.BODY_INERTIAL_PROPERTIES
    | newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    | newton.ModelFlags.SHAPE_PROPERTIES
    | newton.ModelFlags.MODEL_PROPERTIES
)


def build():
    robot = newton.ModelBuilder()
    robot.default_joint_cfg.target_ke = 40.0
    robot.default_joint_cfg.target_kd = 2.0
    robot.default_joint_cfg.armature = 0.01
    robot.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform((0.0, 0.0, 0.62), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )
    robot.joint_q[-12:] = STANCE
    robot.joint_target_q[-12:] = STANCE
    builder = newton.ModelBuilder()
    builder.replicate(robot, WORLDS, spacing=(0.0, 1.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize()


def randomize(model, worlds, rng, nominal):
    """Sample new values for the listed worlds only; every other world keeps its values."""
    # *_world_start[w] is the first index of world w in the flat arrays
    body_start = model.body_world_start.numpy()
    dof_start = model.joint_dof_world_start.numpy()
    shape_start = model.shape_world_start.numpy()
    mass, ke = model.body_mass.numpy(), model.joint_target_ke.numpy()
    mu, gravity = model.shape_material_mu.numpy(), model.gravity.numpy()
    for w in worlds:
        base = body_start[w]  # the floating base is the first body of each world
        mass[base] = nominal["mass"][base] * rng.uniform(0.7, 2.0)
        legs = slice(dof_start[w] + 6, dof_start[w + 1])  # skip the base's free joint
        ke[legs] = nominal["ke"][legs] * rng.uniform(0.5, 1.5)
        mu[shape_start[w] : shape_start[w + 1]] = rng.uniform(0.3, 1.0)
        gravity[w] = (0.0, 0.0, -rng.uniform(8.0, 11.0))
    model.body_mass.assign(mass)
    model.body_inv_mass.assign(1.0 / mass)
    model.joint_target_ke.assign(ke)
    model.shape_material_mu.assign(mu)
    model.gravity.assign(gravity)


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model, articulation_constraint_max=256)  # four feet, several contacts each
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    nominal = {"mass": model.body_mass.numpy(), "ke": model.joint_target_ke.numpy()}
    rng = np.random.default_rng(13)
    base = model.body_world_start.numpy()[:WORLDS]

    run = rec.variant(label, model)
    for frame in range(rec.frames(5.0)):
        if frame == rec.fps:  # first reset: every world
            randomize(model, range(WORLDS), rng, nominal)
            solver.notify_model_changed(FLAGS)
        if frame == 3 * rec.fps:  # partial reset: worlds 1 and 3 only
            randomize(model, [1, 3], rng, nominal)
            mask = wp.array(
                [False, True, False, True, False], dtype=wp.bool
            )  # world_count + 1 entries
            solver.notify_model_changed(FLAGS, world_mask=mask)
        height = state_0.body_q.numpy()[base, 2]
        run.frame(state_0, **{f"world {w}": h for w, h in enumerate(height)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "mass, gains, friction and gravity per world")
    rec.save()
