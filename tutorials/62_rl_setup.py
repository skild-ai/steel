"""An RL loop: many worlds, one graph launch per frame, device-side terminations and resets."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
EPISODE_STEPS = wp.constant(240 * 3)  # time limit: 3 s
STEPS_PER_FRAME = wp.constant(SUBSTEPS)


def task():
    """One world: a box tossed onto a tilted pad; done when it leaves the pad or times out."""
    scene = newton.ModelBuilder()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.15)
    scene.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.05), tilt), hx=0.25, hy=0.25, hz=0.02)
    box = scene.add_body(xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()))
    scene.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(mu=0.4))
    return scene


def build(worlds):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(task(), worlds, spacing=(0.8, 0.8, 0.0))
    return builder.finalize()


@wp.kernel
def find_done(
    body_q: wp.array(dtype=wp.transform),
    overflow: wp.array(dtype=wp.bool),
    origin: wp.array(dtype=wp.vec3),
    steps: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.bool),
):
    w = wp.tid()  # one box per world, so body w is world w's box
    p = wp.transform_get_translation(body_q[w]) - origin[w]
    off_pad = wp.abs(p[0]) > 0.3 or wp.abs(p[1]) > 0.3 or p[2] < 0.03
    # a world that dropped rows is terminated too: its physics is no longer trustworthy
    done[w] = off_pad or overflow[w] or steps[w] >= EPISODE_STEPS
    steps[w] = wp.where(done[w], 0, steps[w] + STEPS_PER_FRAME)


@wp.kernel
def reset_worlds(
    done: wp.array(dtype=wp.bool),
    start_q: wp.array(dtype=wp.float32),
    seed: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    w = wp.tid()  # one free joint per world: 7 coordinates, 6 velocities
    if not done[w]:
        return
    for k in range(7):
        joint_q[w * 7 + k] = start_q[w * 7 + k]
    rng = wp.rand_init(seed[0], w)
    joint_qd[w * 6 + 0] = wp.randf(rng, -1.5, 1.5)
    joint_qd[w * 6 + 1] = wp.randf(rng, -1.5, 1.5)
    for k in range(2, 6):
        joint_qd[w * 6 + k] = wp.randf(rng, -2.0, 2.0)


@wp.kernel
def next_seed(seed: wp.array(dtype=wp.int32)):
    seed[0] = seed[0] + 1


class Env:
    """All worlds stepped together; the done mask and resets live on the device."""

    def __init__(self, worlds):
        self.worlds = worlds
        self.model = build(worlds)
        self.solver = SolverSteel(self.model)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        origin = self.model.body_q.numpy()[:, :3] - np.array([0.0, 0.0, 0.25])
        self.origin = wp.array(origin, dtype=wp.vec3)
        self.steps = wp.zeros(worlds, dtype=wp.int32)
        self.seed = wp.zeros(1, dtype=wp.int32)
        # world_mask for reset() has one more entry for the global world, which never resets
        self.mask = wp.zeros(worlds + 1, dtype=wp.bool)
        self.done = self.mask[:worlds]
        self.graph = None
        # start every world with its own random toss
        self.done.fill_(True)
        self.reset()
        self.done.fill_(False)

    def reset(self):
        """Restart the worlds flagged in ``done``; everything stays on the device."""
        wp.launch(
            reset_worlds,
            self.worlds,
            [self.done, self.model.joint_q, self.seed],
            [self.state_0.joint_q, self.state_0.joint_qd],
        )
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, mask=self.done
        )
        self.solver.reset(self.state_0, world_mask=self.mask)
        wp.launch(next_seed, 1, [self.seed])

    def frame(self):
        for _ in range(SUBSTEPS):
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            find_done,
            self.worlds,
            [self.state_0.body_q, self.solver.constraint_overflow, self.origin, self.steps],
            [self.done],
        )
        self.reset()

    def run_frame(self):
        if self.graph is None:
            self.frame()  # compile everything once, eagerly
            with wp.ScopedCapture() as capture:
                self.frame()
            self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)


def simulate(rec, worlds):
    env = Env(worlds)
    run = rec.variant(f"{worlds} worlds, one graph launch per frame", env.model)
    resets = 0
    for _ in range(rec.frames(6.0)):
        run.frame(env.state_0, resets=resets)
        env.run_frame()
        resets += int(env.done.numpy().sum())
    run.check(env.solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, 16)
    rec.save()
