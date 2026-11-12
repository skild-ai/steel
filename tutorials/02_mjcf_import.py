# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Steel attributes from an MJCF file: geom adhesion and the friction triple carry into Steel."""

import newton
import numpy as np
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8

# A box under a ceiling and a ball rolling on the floor. friction="slide torsional rolling".
MJCF = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" friction="0.6 0.01 0.004"/>
    <geom name="ceiling" type="box" size="0.2 0.2 0.05" pos="0 0 1.05" {adhesion}/>
    <body name="box" pos="0 0 0.95"><freejoint/>
      <geom type="box" size="0.05 0.05 0.05" density="1000" gap="0.004"/>
    </body>
    <body name="ball" pos="0.5 0 0.08"><freejoint/>
      <geom type="sphere" size="0.08" density="800" friction="0.6 0.01 0.004"/>
    </body>
  </worldbody>
</mujoco>"""


def build(adhesion, condim):
    builder = newton.ModelBuilder()
    # register first, so the importer fills model.steel.adhesion from the geom's adhesion attribute
    SolverSteel.register_custom_attributes(builder)
    builder.add_mjcf(
        MJCF.format(adhesion=f'adhesion="{adhesion}" gap="0.004"' if adhesion else 'gap="0.004"')
    )
    model = builder.finalize()
    # steel:condim cannot be authored in MJCF; set it after import to use the friction triple
    model.steel.condim.fill_(condim)
    qd = model.joint_qd.numpy()
    qd[6] = 1.0  # roll the ball at 1 m/s
    model.joint_qd.assign(qd)
    return model


def simulate(rec, label, adhesion, condim):
    model = build(adhesion, condim)
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        q, qd = state_0.body_q.numpy(), state_0.body_qd.numpy()
        run.frame(
            state_0, box_drop_mm=1000.0 * (0.95 - q[0, 2]), ball_speed=np.linalg.norm(qd[1, :3])
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "MJCF without adhesion, condim 3", adhesion=0, condim=3)
    # 4 corner contacts x 20 N = 80 N against the box's 9.8 N weight
    simulate(rec, 'MJCF adhesion="20", condim 6 after import', adhesion=20, condim=6)
    rec.save()
