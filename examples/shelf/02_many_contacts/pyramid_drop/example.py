# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Pyramid drop
#
# A ball is held 0.3 m above the apex of a 285-block stepped pyramid (square layers of 0.1 m
# cubes, 9x9 down to 1x1) and released at 1.5 s, once the pyramid has stood on its own.
# Four worlds side by side drop a 1, 10, 100 and 1000 kg ball: the three lighter balls come
# to rest on the apex and the 1000 kg ball collapses the pyramid. The 100 kg ball rests in
# unstable balance and rolls off the apex a few seconds later.
#
# Command: python -m shelf run pyramid_drop
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from common.steel_scenes import GOLD, GREY, SceneExample, cfg, tf

BLOCK = 0.1  # cube edge [m]
BLOCK_DENSITY = 500.0  # [kg/m^3], 0.5 kg per block
LAYERS = 9
GAP = 0.001  # horizontal gap between neighbouring blocks of a layer [m]
FRICTION = 0.5
BALL_RADIUS = {1.0: 0.06, 10.0: 0.08, 100.0: 0.12, 1000.0: 0.20}  # [m] per ball mass [kg], one world each
DROP_GAP = 0.3  # ball bottom above the apex top until the release [m]
RELEASE_TIME = 1.5  # [s]
WORLD_SPACING = 3.0  # [m] along x between the worlds as drawn; each world simulates at the origin
# A block counts as displaced past this distance from its start [m].
DISPLACED = 0.02
# The heaviest ball collapses its pyramid: at least this share of its blocks is displaced.
COLLAPSE_MASS = 1000.0
COLLAPSED_SHARE = 0.75


@wp.kernel(enable_backward=False)
def _hold_balls(
    balls: wp.array[wp.int32],
    q_starts: wp.array[wp.int32],
    qd_starts: wp.array[wp.int32],
    poses: wp.array[wp.transform],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    pose = poses[i]
    p = wp.transform_get_translation(pose)
    r = wp.transform_get_rotation(pose)
    qs = q_starts[i]
    for k in range(3):
        joint_q[qs + k] = p[k]
    for k in range(4):
        joint_q[qs + 3 + k] = r[k]
    for k in range(6):
        joint_qd[qd_starts[i] + k] = 0.0
    body_q[balls[i]] = pose
    body_qd[balls[i]] = wp.spatial_vector()


class Example(SceneExample):
    file = __file__
    camera = {"eye": (0.5, -7.6, 2.3), "target": (0.5, 0.0, 0.55)}

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.viewer.set_world_offsets((WORLD_SPACING, 0.0, 0.0))

    def build(self):
        block_cfg = cfg(mu=FRICTION, restitution=0.0, density=BLOCK_DENSITY)
        scene = newton.ModelBuilder()
        self.block_ids, self.ball_ids, ball_joints = [], [], []
        for w, (mass, radius) in enumerate(BALL_RADIUS.items()):
            world = newton.ModelBuilder()
            self.preset.apply_model(world)
            # each world has its own ground; only the first is drawn, as the viewer's offsets would z-fight them
            world.add_ground_plane(cfg=cfg(mu=FRICTION, restitution=0.0, density=BLOCK_DENSITY, is_visible=w == 0))
            for p in _pyramid_positions():
                body = world.add_body(xform=tf(p))
                world.add_shape_box(body, hx=0.5 * BLOCK, hy=0.5 * BLOCK, hz=0.5 * BLOCK, cfg=block_cfg, color=GREY)
            ball_cfg = cfg(mu=FRICTION, restitution=0.0, density=mass / (4.0 / 3.0 * np.pi * radius**3))
            ball = world.add_body(xform=tf((0.0, 0.0, LAYERS * BLOCK + DROP_GAP + radius)))
            world.add_shape_sphere(ball, radius=radius, cfg=ball_cfg, color=GOLD)
            first = scene.body_count
            scene.add_world(world)
            self.block_ids.append(np.arange(first, first + ball))
            self.ball_ids.append(first + ball)
            ball_joints.append(scene.joint_count - 1)
        self.frame = 0
        device = wp.get_device()
        self._balls = wp.array(self.ball_ids, dtype=wp.int32, device=device)
        self._ball_q_starts = wp.array([scene.joint_q_start[j] for j in ball_joints], dtype=wp.int32, device=device)
        self._ball_qd_starts = wp.array([scene.joint_qd_start[j] for j in ball_joints], dtype=wp.int32, device=device)
        self._ball_poses = wp.array([scene.body_q[b] for b in self.ball_ids], dtype=wp.transform, device=device)
        self.start = np.array([list(scene.body_q[b])[:3] for b in range(scene.body_count)])
        return scene, {}

    def build_solver(self):
        """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
        # Steel sizes its contact scratch from the model's capacity, so it is set before the solver
        self.model.rigid_contact_max = self.preset.collision.need("rigid_contact_max")
        super().build_solver()

    def step(self):
        if self.frame * self.frame_dt < RELEASE_TIME:
            state = self.state_0
            wp.launch(
                _hold_balls,
                dim=len(self.ball_ids),
                inputs=[self._balls, self._ball_q_starts, self._ball_qd_starts, self._ball_poses],
                outputs=[state.joint_q, state.joint_qd, state.body_q, state.body_qd],
                device=self.model.device,
            )
        super().step()
        self.frame += 1

    def test_post_step(self):
        if self.frame == round(RELEASE_TIME * self.fps):
            self.at_release = self.state_0.body_q.numpy()[:, :3].copy()

    def check(self):
        """Every pyramid stands until the release; the lighter balls leave theirs standing, 1000 kg topples it."""
        q = self.state_0.body_q.numpy()[:, :3]
        for blocks, ball, mass in zip(self.block_ids, self.ball_ids, BALL_RADIUS, strict=True):
            before = np.linalg.norm(self.at_release[blocks] - self.start[blocks], axis=1).max()
            assert before < DISPLACED, f"{mass:g} kg: a block moved {before * 1000:.1f} mm before the release"
            fall = self.start[ball, 2] - q[ball, 2]
            assert fall > 0.8 * DROP_GAP, f"{mass:g} kg: the ball fell only {fall * 1000:.0f} mm"
            displaced = int((np.linalg.norm(q[blocks] - self.start[blocks], axis=1) > DISPLACED).sum())
            if mass < COLLAPSE_MASS:
                assert displaced == 0, f"{mass:g} kg: {displaced} of {len(blocks)} blocks displaced"
            else:
                assert displaced >= COLLAPSED_SHARE * len(blocks), f"{mass:g} kg: only {displaced} blocks displaced"


def _pyramid_positions() -> np.ndarray:
    """Block centers, bottom layer first."""
    pitch = BLOCK + GAP
    centers = []
    for layer in range(LAYERS):
        n = LAYERS - layer
        offsets = (np.arange(n) - 0.5 * (n - 1)) * pitch
        for x in offsets:
            for y in offsets:
                centers.append((x, y, 0.5 * BLOCK + layer * BLOCK))
    return np.array(centers)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
