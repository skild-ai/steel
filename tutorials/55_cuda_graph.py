"""CUDA graph capture: record one frame of collide + step calls once, then replay it."""

import time

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8  # even, so the state buffers end each frame where they started


def build():
    scene = newton.ModelBuilder()
    for i in range(6):
        body = scene.add_body(
            xform=wp.transform(
                (0.02 * (i % 2), 0.0, 0.1 + 0.15 * i), wp.quat_rpy(0.2 * i, 0.1, 0.0)
            )
        )
        scene.add_shape_box(body, hx=0.05, hy=0.04, hz=0.04)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(scene, 16, spacing=(0.35, 0.35, 0.0))
    return builder.finalize()


def simulate(rec, label, use_graph):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    def frame_steps():
        nonlocal state_0, state_1
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0

    graph = None
    if use_graph:
        frame_steps()  # warm up: compile kernels outside the capture
        state_0.assign(model.state())  # restart from the initial state
        solver.reset(state_0)
        with wp.ScopedCapture() as capture:
            frame_steps()
        graph = capture.graph

    run = rec.variant(label, model)
    elapsed, frames = 0.0, rec.frames(3.0)
    for frame in range(frames):
        run.frame(state_0, top_z=state_0.body_q.numpy()[:, 2].max())
        wp.synchronize()
        start = time.perf_counter()
        if graph is not None:
            wp.capture_launch(graph)
        else:
            frame_steps()
        wp.synchronize()
        elapsed += time.perf_counter() - start if frame > 0 else 0.0
    run.check(solver)
    print(f"{label}: {1000.0 * elapsed / (frames - 1):.3f} ms per frame, {SUBSTEPS} steps")


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "eager steps", False)
    simulate(rec, "captured graph", True)
    rec.save()
