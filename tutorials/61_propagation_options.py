# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Propagation options: same-articulation rows, cached responses and the colored contact order."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
LINKS = 12
BASE = {"solve_mode": "colored", "articulated_contact_response": "propagation"}
OPTIONS = {
    "colored + propagation (defaults)": {},
    "propagation_same_articulation_rows=True": {"propagation_same_articulation_rows": True},
    "propagation_cached_response=False": {"propagation_cached_response": False},
    'propagation_color_order="rotating"': {"propagation_color_order": "rotating"},
    # the 96 dense self-contact rows would not fit the default 32; as propagation rows they do
    "same_articulation_rows, articulation_constraint_max=32": {
        "articulation_constraint_max": 32,
        "propagation_same_articulation_rows": True,
    },
}


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    """A twelve-link chain hung from one end; it swings down and folds onto itself."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    parent, joints = -1, []
    for i in range(LINKS):
        link = builder.add_link(xform=tf(0.08 * i + 0.04, 0.0, 0.5))
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)
        builder.add_shape_capsule(
            link, xform=wp.transform((0.0, 0.0, 0.0), rot), radius=0.025, half_height=0.02
        )
        anchor = tf(0.0, 0.0, 0.5) if parent < 0 else tf(0.04, 0.0, 0.0)
        axis = (0.0, 1.0, 0.0) if i % 3 else (0.0, 0.3, 1.0)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=axis,
                parent_xform=anchor,
                child_xform=tf(-0.04, 0.0, 0.0),
                armature=0.002,
                damping=0.002,
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize()


def simulate(rec, label):
    model = build()
    options = dict(
        BASE, articulation_constraint_max=256, free_body_constraint_max=256, row_watermark=True
    )
    solver = SolverSteel(model, **dict(options, **OPTIONS[label]))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        marks = solver.constraint_row_watermarks()
        run.frame(
            state_0,
            dense_rows=marks["articulation_raw_high_water"],
            propagation_rows=marks["propagation_raw_high_water"],
            tip_z=state_0.body_q.numpy()[-1, 2],
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    for label in OPTIONS:
        simulate(rec, label)
    rec.save()
