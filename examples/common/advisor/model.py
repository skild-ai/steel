# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Numpy views of a Newton model for the episode probe: world-0 masks, body roles, torsion radii."""

from __future__ import annotations

import collections

import numpy as np

JOINT_NAMES = {0: "prismatic", 1: "revolute", 2: "ball", 3: "fixed", 4: "free", 5: "distance", 6: "d6", 7: "rod"}


class _ModelView:
    """Numpy copies of the model arrays the analysis needs, plus world-0 masks and body roles."""

    def __init__(self, model):
        self.model = model
        self.W = int(model.world_count)
        g = lambda name: getattr(model, name).numpy() if getattr(model, name, None) is not None else None  # noqa: E731
        self.shape_world, self.shape_body, self.shape_type = g("shape_world"), g("shape_body"), g("shape_type")
        self.shape_flags = g("shape_flags")
        self.shape_gap, self.shape_margin = g("shape_gap"), g("shape_margin")
        self.body_world = g("body_world")
        self.joint_type, self.joint_parent, self.joint_child = g("joint_type"), g("joint_parent"), g("joint_child")
        self.joint_world, self.joint_articulation = g("joint_world"), g("joint_articulation")
        self.joint_qd_start = g("joint_qd_start")
        self.lower, self.upper = g("joint_limit_lower"), g("joint_limit_upper")
        self.vel_limit, self.effort_limit = g("joint_velocity_limit"), g("joint_effort_limit")
        self.ke, self.kd, self.armature = g("joint_target_ke"), g("joint_target_kd"), g("joint_armature")
        self.passive_ke = g("joint_stiffness")
        self.target_mode = g("joint_target_mode")
        self.art_start = g("articulation_start")
        from newton import ShapeFlags

        self.collide = (self.shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        self.shape_labels = list(getattr(model, "shape_label", None) or getattr(model, "shape_key", None) or [])
        self.body_labels = list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])
        self.joint_labels = list(getattr(model, "joint_label", None) or getattr(model, "joint_key", None) or [])
        nb, nj = int(model.body_count), int(model.joint_count)
        self.joint_dofs = (self.joint_qd_start[1:] - self.joint_qd_start[:-1])[:nj]
        self.dof_joint = np.repeat(np.arange(nj), self.joint_dofs)
        w0 = lambda arr: (arr == 0) | (self.W == 1) & (arr <= 0)  # noqa: E731
        self.body_in_w0 = w0(self.body_world)
        self.joint_in_w0 = w0(self.joint_world)
        self.shape_in_w0 = w0(self.shape_world)
        self.dof_in_w0 = self.joint_in_w0[self.dof_joint]
        self.inbound = np.full(nb, -1)
        self.inbound[self.joint_child] = np.arange(nj)
        # Body roles: a free body is the only body of an articulation whose one joint is FREE.
        art_of_joint = self.joint_articulation
        self.art_count = int(model.articulation_count)
        art_joints = collections.defaultdict(list)
        for j in range(nj):
            art_joints[int(art_of_joint[j])].append(j)
        self.art_joints = art_joints
        self.body_art = np.full(nb, -1)
        for a, js in art_joints.items():
            self.body_art[self.joint_child[js]] = a
        self.free_body = np.zeros(nb, dtype=bool)
        for a, js in art_joints.items():
            if len(js) == 1 and self.joint_type[js[0]] in (4, 5) and self.joint_parent[js[0]] < 0:
                self.free_body[self.joint_child[js[0]]] = True
        # Zero-DOF bodies welded to the world (a fixed-base robot's root link) collide like static geometry.
        self.fixed_root = np.zeros(nb, dtype=bool)
        roots = (self.joint_parent < 0) & (self.joint_dofs == 0)
        self.fixed_root[self.joint_child[roots]] = True
        self.free_body_w0 = self.free_body & self.body_in_w0
        self.arts_w0 = sorted(
            {
                int(art_of_joint[j])
                for j in np.nonzero(self.joint_in_w0)[0]
                if art_of_joint[j] >= 0 and not self.free_body[self.joint_child[j]]
            }
        )

    def body_role(self, b: int) -> str:
        if b < 0:
            return "static"
        if self.free_body[b]:
            return "object"
        if self.fixed_root[b]:
            return "fixed_root"
        return "robot"

    def short(self, labels, i) -> str:
        return labels[i].split("/")[-1] if labels and 0 <= i < len(labels) else str(i)


def _torsion_facts(model, m: _ModelView) -> dict:
    steel = getattr(model, "steel", None)
    r = getattr(steel, "min_torsional_patch_radius", None) if steel is not None else None
    if r is None:
        return {"radius_max": 0.0, "shapes_with_radius_world0": 0, **_reference_contact_law(model, m)}
    r = r.numpy()
    sel = m.shape_in_w0 & m.collide
    out = {
        "radius_max": float(r.max()) if len(r) else 0.0,
        "shapes_with_radius_world0": int((r[sel] > 0).sum()),
        "examples": [m.short(m.shape_labels, int(i)) for i in np.nonzero(sel & (r > 0))[0][:6]],
    }
    out.update(_reference_contact_law(model, m))
    return out


def _reference_contact_law(model, m: "_ModelView") -> dict:
    """MuJoCo condim the reference would use: per-shape condim and explicit pair overrides, all worlds.

    condim 4 or 6 adds torsional (and rolling) friction. Unknown unless SolverMuJoCo attributes are registered.
    """
    mj = getattr(model, "mujoco", None)
    condim = getattr(mj, "condim", None)
    if condim is None:
        return {"mujoco_condim_max": None, "mujoco_pair_condim_max": None, "reference_torsion": "unknown"}
    shape_max = int(condim.numpy()[m.collide].max()) if m.collide.any() else 3
    pair_world = getattr(mj, "pair_world", None)
    pairs = 0 if pair_world is None else int(pair_world.shape[0])
    pair_condim = getattr(mj, "pair_condim", None)
    pair_max = int(pair_condim.numpy().max()) if pairs and pair_condim is not None else (3 if not pairs else None)
    if pair_max is None:
        law = "unknown"
    else:
        law = "torsion" if max(shape_max, pair_max) > 3 else "no_torsion"
    return {
        "mujoco_condim_max": shape_max,
        "mujoco_pair_condim_max": pair_max,
        "mujoco_explicit_pairs": pairs,
        "reference_torsion": law,
    }
