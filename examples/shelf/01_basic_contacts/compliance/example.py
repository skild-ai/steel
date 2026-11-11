# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Soft contacts
#
# Contact compliance: 1 kg blocks on springs of 1e3, 1e4 and 1e5 N/m (shape ke, series with a stiff ground).
#
# Steel's contact_compliance reads a per-contact stiffness and damping from the contacts buffer; Newton's
# collision pipeline fills them only for hydroelastic contacts. The scene fills them for its point contacts
# after every collision pass, from the two shapes' ke and kd (see _contact_material).
#
# Ported from steel-webgpu's tools/scenes.py (compliance); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run compliance
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from common.steel_scenes import BLUE, GOLD, GREY, RED, TEAL, SceneExample, cfg, tf


@wp.kernel(enable_backward=False)
def _contact_material(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    ke: wp.array[float],
    kd: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    friction: wp.array[float],
):
    """Per-contact stiffness [N/m] and damping [N s/m] of a point contact from its shapes' ke and kd.

    Each shape is a spring and damper in parallel; the two in series give k = ka kb / (ka + kb), and each damper
    sees its spring's share k / ki of the closing speed, so c = ca (k / ka)^2 + cb (k / kb)^2. The pair's springs
    are split evenly over its contact points, so a box resting on its face carries k in total.
    """
    i = wp.tid()
    n = count[0]
    if i >= n:
        return
    a = shape0[i]
    b = shape1[i]
    points = wp.int32(0)
    for j in range(n):
        if shape0[j] == a and shape1[j] == b:
            points += 1
    ka = ke[a]
    kb = ke[b]
    k = wp.float32(0.0)
    c = wp.float32(0.0)
    if ka > 0.0 and kb > 0.0:
        k = ka * kb / (ka + kb)
        c = kd[a] * (k / ka) * (k / ka) + kd[b] * (k / kb) * (k / kb)
    stiffness[i] = k / float(points)
    damping[i] = c / float(points)
    friction[i] = 0.0  # zero keeps the pair's own friction coefficient


class _MaterialPipeline:
    """Newton's collision pipeline, then the contacts' material from the shapes (what hydroelastic contacts carry)."""

    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def contacts(self):
        contacts = self.pipeline.contacts()
        if contacts.rigid_contact_stiffness is None:
            n = contacts.rigid_contact_max
            device = self.model.device
            contacts.per_contact_shape_properties = True
            contacts.rigid_contact_stiffness = wp.zeros(n, dtype=wp.float32, device=device)
            contacts.rigid_contact_damping = wp.zeros(n, dtype=wp.float32, device=device)
            contacts.rigid_contact_friction = wp.zeros(n, dtype=wp.float32, device=device)
        return contacts

    def collide(self, state, contacts, **kwargs):
        self.pipeline.collide(state, contacts, **kwargs)  # dt and the rest, as SceneExample passes them
        wp.launch(
            _contact_material, dim=contacts.rigid_contact_max,
            inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
                    self.model.shape_material_ke, self.model.shape_material_kd],
            outputs=[contacts.rigid_contact_stiffness, contacts.rigid_contact_damping, contacts.rigid_contact_friction],
            device=self.model.device,
        )  # fmt: skip


class Example(SceneExample):
    file = __file__
    camera = {'eye': (0.3, -1.0, 0.35), 'target': (0.0, 0.0, 0.05)}

    def build(self):
        """Contact compliance: 1 kg blocks on springs of 1e3, 1e4 and 1e5 N/m (shape ke, series with a stiff ground)."""
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=cfg(ke=1.0e7, kd=0.0))
        for i, (ke, kd) in enumerate(((1.0e3, 20.0), (1.0e4, 60.0), (1.0e5, 200.0))):
            body = b.add_body(xform=tf(((i - 1) * 0.25, 0.0, 0.3)))
            b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg(ke=ke, kd=kd, density=1000), color=[RED, GOLD, TEAL][i])
        return b, {}, "Contact compliance: 1 kg blocks with contact stiffness 1e3, 1e4 and 1e5 N/m drop and settle about 10, 1 and 0.1 mm into the ground."

    def build_solver(self):
        super().build_solver()
        if self.preset.solver.get("contact_compliance", False):
            self.collision_pipeline = _MaterialPipeline(self.model, self.collision_pipeline)
            self.contacts = self.collision_pipeline.contacts()



if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
