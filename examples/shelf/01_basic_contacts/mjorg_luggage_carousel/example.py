# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example mujoco.org luggage carousel
#
# Three conveyor belts carry six bags down a cascade and drop them onto a
# baggage-claim carousel that carries them around. The belts, their rollers and the
# ring never move: their surfaces do, through Steel's mesh surface velocity.
#
# Replicates the "Surface Velocity: Luggage Carousel" scene of the MuJoCo 4.0 tour
# on mujoco.org from its MJCF (scene.xml, flattened from google-deepmind/mujoco
# model/surfacevel/carousel.xml). See README.md for the gaps.
#
# Command: python -m shelf run mjorg_luggage_carousel
#
###########################################################################

from pathlib import Path

import numpy as np

import newton
import newton.examples
from common.steel_scenes import SceneExample

# MuJoCo 4.0's surfacevel per geom: linear then angular velocity in the geom frame [m/s, rad/s].
SURFACE_VELOCITY = {
    "carousel": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.4)),
    **{f"{belt}belt": ((0.7, 0.0, 0.0), (0.0, 0.0, 0.0)) for belt in ("", "top_", "up_")},
    **{f"{belt}belt_roller{i}": ((0.0, 0.0, 0.0), (0.0, 0.0, -11.667)) for belt in ("", "top_", "up_") for i in (1, 2)},
}
# A bag is on the carousel when its centre is within this radius band of the ring's axis and above this
# height [m]; it has ridden it once it has gone this far round while on it [rad].
RING_RADII = (1.0, 2.1)
RING_HEIGHT_MIN = 0.35
RIDE_MIN = 0.5 * np.pi
# Bags that must end on the carousel having ridden it; a bag can tip off a belt transfer (see README.md).
RIDING_MIN = 4


class Example(SceneExample):
    file = __file__
    # MuJoCo's free camera for the scene: azimuth 90, elevation -45 around the statistic centre
    camera = {"eye": (0.0, -10.1, 9.07), "target": (0.0, -1.27, 0.24)}

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        # the meshes exist once the model is finalized; the captured graph reads these arrays in place
        for mesh, velocities in self.surfaces:
            mesh.mesh.velocities.assign(velocities)

    def build(self):
        builder = newton.ModelBuilder()
        self.preset.apply_model(builder)
        builder.add_mjcf(str(Path(__file__).with_name("scene.xml")))
        # MuJoCo collides a mesh geom as its convex hull; the bags' superspheres are convex
        for shape, body in enumerate(builder.shape_body):
            if body >= 0 and builder.shape_type[shape] == newton.GeoType.MESH:
                builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
        self.surfaces = [
            _moving_surface(builder, shape, *SURFACE_VELOCITY[label.rsplit("/", 1)[-1]])
            for shape, label in enumerate(builder.shape_label)
            if label.rsplit("/", 1)[-1] in SURFACE_VELOCITY
        ]
        assert len(self.surfaces) == len(SURFACE_VELOCITY), "a moving surface is missing from scene.xml"
        self.angle = None
        return builder, {}

    def test_post_step(self):
        q = self.state_0.body_q.numpy()
        angle = np.arctan2(q[:, 1], q[:, 0])
        if self.angle is None:
            self.ridden = np.zeros(len(q))
        else:
            self.ridden += np.where(_on_ring(q), np.angle(np.exp(1j * (angle - self.angle))), 0.0)
        self.angle = angle

    def check(self):
        """The bags come down the belts and ride round on the carousel."""
        self.solver.check_constraint_capacity()  # dropped contact rows invalidate the run
        riding = _on_ring(self.state_0.body_q.numpy()) & (self.ridden > RIDE_MIN)
        assert riding.sum() >= RIDING_MIN, f"{int(riding.sum())} bags rode the carousel: {np.round(self.ridden, 1)}"


def _on_ring(q):
    radius = np.hypot(q[:, 0], q[:, 1])
    return (radius > RING_RADII[0]) & (radius < RING_RADII[1]) & (q[:, 2] > RING_HEIGHT_MIN)


def _moving_surface(builder, shape, linear, angular):
    """Make a static box, cylinder or mesh shape a mesh whose vertices carry ``linear + angular x r``."""
    shape_type = builder.shape_type[shape]
    hx, hy, hz = builder.shape_scale[shape]
    if shape_type == newton.GeoType.BOX:
        mesh = newton.Mesh.create_box(hx, hy, hz, compute_inertia=False)
    elif shape_type == newton.GeoType.CYLINDER:
        mesh = newton.Mesh.create_cylinder(hx, hy, up_axis=newton.Axis.Z, compute_inertia=False)
    else:
        mesh = builder.shape_source[shape]
    if shape_type != newton.GeoType.MESH:
        builder.shape_type[shape] = newton.GeoType.MESH
        builder.shape_scale[shape] = (1.0, 1.0, 1.0)
        builder.shape_source[shape] = mesh
        builder.shape_collision_radius[shape] = float(np.linalg.norm(mesh.vertices, axis=1).max())
    mesh.enable_surface_velocity = True
    velocities = np.asarray(linear) + np.cross(np.asarray(angular), mesh.vertices)
    return mesh, velocities.astype(np.float32)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
