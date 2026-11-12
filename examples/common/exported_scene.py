"""Exported steel-webgpu scenes rebuilt as Newton models: the hero-montage demos.

steel-webgpu's ``tools/newton_export.py`` writes a Newton model as ``scene.json`` and ``scene.bin``:
bodies with mass, inertia and pose; collision shapes as convex hulls (spheres and capsules keep their
form); joints with their limits, drives and articulations; collision filters; a scripted path of joint
targets at 60 Hz for the first articulation; and visual meshes stored once under ``assets/scenes/shared``.
The builders of the hero-montage scenes live in Newton's ``scripts/benchmarks/hero_montage``, which the
shelf does not have, so these examples rebuild each scene from its export. The physics is complete. The
visual meshes draw when a steel-webgpu checkout sits next to the shelf, or ``STEEL_WEBGPU_ASSETS`` names
its ``assets/scenes/shared`` folder; otherwise the collision hulls draw.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from common.steel_scenes import DEFAULT_CAMERA, SceneExample, cfg

ROOT = Path(__file__).resolve().parents[1]


def shared_assets() -> Path | None:
    """steel-webgpu's ``assets/scenes/shared`` (visual meshes), if it can be found."""
    named = os.environ.get("STEEL_WEBGPU_ASSETS")
    candidates = [Path(named)] if named else []
    candidates.extend(parent / "steel-webgpu" / "assets" / "scenes" / "shared"
                      for parent in (ROOT.parent, ROOT.parent.parent))
    return next((p for p in candidates if (p / "meshes").is_dir()), None)


def _view(blob: bytes, ref: dict, dtype) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype, count=int(ref["n"]), offset=int(ref["off"]))


def _transform(xf) -> wp.transform:
    return wp.transform(wp.vec3(*[float(x) for x in xf[:3]]), wp.quat(*[float(x) for x in xf[3:7]]))


def _hull_mesh(blob: bytes, hull: dict) -> tuple[np.ndarray, np.ndarray]:
    """A hull's vertices and fan-triangulated faces."""
    verts = _view(blob, hull["verts"], np.float32).reshape(-1, 3)
    start = _view(blob, hull["face_start"], np.uint32)
    face_verts = _view(blob, hull["face_verts"], np.uint32)
    tris = []
    for f in range(len(start) - 1):
        poly = face_verts[start[f] : start[f + 1]]
        for k in range(1, len(poly) - 1):
            tris += [int(poly[0]), int(poly[k]), int(poly[k + 1])]
    return verts.astype(np.float32), np.asarray(tris, dtype=np.int32)


def _box_half_extents(verts: np.ndarray) -> tuple[float, float, float] | None:
    """The half extents when the hull is an axis-aligned box centred on the origin (its box is a better collider)."""
    if len(verts) != 8:
        return None
    hi = verts.max(axis=0)
    if not (np.allclose(np.abs(verts), hi, atol=1e-6) and np.all(hi > 0)):
        return None
    return tuple(float(h) for h in hi)


def _shared_mesh(shared: Path, entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """A visual mesh from the shared store: positions (f32), then u16 or u32 indices, each padded to 4 bytes."""
    data = (shared / "meshes" / Path(entry["file"]).name).read_bytes()
    nv, ni = int(entry["nv"]), int(entry["ni"])
    verts = np.frombuffer(data, dtype=np.float32, count=nv * 3).reshape(-1, 3)
    offset = nv * 12
    offset += -offset % 4
    index_type = np.uint16 if entry.get("index", "u32") == "u16" else np.uint32
    indices = np.frombuffer(data, dtype=index_type, count=ni, offset=offset).astype(np.int32)
    return verts.astype(np.float32), indices


class ExportedScene:
    """A ``scene.json`` + ``scene.bin`` pair as a :class:`newton.ModelBuilder`, with what the loop needs."""

    def __init__(self, folder: Path, visuals: bool = True):
        self.folder = Path(folder)
        self.scene: dict[str, Any] = json.loads((self.folder / "scene.json").read_text())
        blob = (self.folder / "scene.bin").read_bytes()
        scene = self.scene
        shared = shared_assets() if visuals else None
        self.visuals_drawn = shared is not None
        b = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(b)

        # bodies: every body is a link; the export lists every joint, free joints included
        bodies = scene["bodies"]
        per_body = {
            "max_linear_velocity": "steel:rigid_body_max_linear_velocity",
            "max_angular_velocity": "steel:rigid_body_max_angular_velocity",
            "angular_damping": "steel:rigid_body_angular_damping",
        }
        registered = set(b.custom_attributes)
        for i in range(bodies["count"]):
            custom = {}
            for key, attr in per_body.items():
                values = bodies.get(key)
                if values and values[i] is not None and attr in registered:
                    custom[attr] = float(values[i])
            inertia = np.asarray(bodies["inertia"][i], dtype=np.float32).reshape(3, 3)
            b.add_link(
                xform=_transform(bodies["q"][i]),
                mass=float(bodies["mass"][i]),
                com=wp.vec3(*[float(x) for x in bodies["com"][i]]),
                inertia=wp.mat33(*inertia.flatten().tolist()),
                is_kinematic=bool(bodies["kinematic"][i]) if bodies.get("kinematic") else False,
                disable_gravity=bool(bodies["no_gravity"][i]) if bodies.get("no_gravity") else False,
                label=bodies["label"][i] or None,
                custom_attributes=custom or None,
            )

        # collision shapes, in the export's order so the filter pairs index them
        shape_ids = []
        for s in scene["shapes"]:
            material = cfg(
                density=0.0,  # the bodies carry their exported mass and inertia
                mu=s["mu"], restitution=s["restitution"], mu_torsional=s["mu_torsional"], mu_rolling=s["mu_rolling"],
                ke=s["ke"], kd=s["kd"], gap=s["gap"], collision_group=s["group"],
            )  # fmt: skip
            custom = {}
            if s.get("condim", 3) != 3:
                custom["steel:condim"] = int(s["condim"])
            if s.get("torsion_radius", 0.0) > 0.0 and "steel:min_torsional_patch_radius" in registered:
                custom["steel:min_torsional_patch_radius"] = float(s["torsion_radius"])
            color = wp.vec3(*[float(c) for c in s["color"]])
            xform = _transform(s["xf"])
            if s.get("plane"):
                shape_ids.append(b.add_shape_plane(xform=xform, width=0.0, length=0.0, body=s["body"], cfg=material, color=color))
                continue
            # a shape with a visual mesh draws through it; without the shared store, its hull shows instead
            material.is_visible = bool(s.get("visible", True)) or shared is None
            prim = s.get("prim")
            if prim and prim["kind"] == 1:
                shape_ids.append(b.add_shape_sphere(s["body"], xform=xform, radius=prim["r"], cfg=material, color=color, custom_attributes=custom or None))
                continue
            if prim and prim["kind"] == 2:
                shape_ids.append(b.add_shape_capsule(s["body"], xform=xform, radius=prim["r"], half_height=prim["h"], cfg=material, color=color, custom_attributes=custom or None))
                continue
            verts, tris = _hull_mesh(blob, s["hull"])
            box = _box_half_extents(verts)
            if box is not None:
                shape_ids.append(b.add_shape_box(s["body"], xform=xform, hx=box[0], hy=box[1], hz=box[2], cfg=material, color=color, custom_attributes=custom or None))
            else:
                mesh = newton.Mesh(verts, tris, compute_inertia=False)
                shape_ids.append(b.add_shape_convex_hull(s["body"], xform=xform, mesh=mesh, cfg=material, color=color, custom_attributes=custom or None))
        for a, c in scene.get("filters", []):
            b.add_shape_collision_filter_pair(shape_ids[a], shape_ids[c])

        # visual meshes: drawn, never collided
        if shared is not None:
            look = cfg(density=0.0, has_shape_collision=False, has_particle_collision=False)
            for v in scene.get("visuals", []):
                if "mesh" in v and "file" in v["mesh"]:
                    verts, tris = _shared_mesh(shared, v["mesh"])
                elif "hull" in v:
                    verts, tris = _hull_mesh(blob, v["hull"])
                else:
                    continue
                mesh = newton.Mesh(verts, tris, compute_inertia=False)
                b.add_shape_mesh(v["body"], xform=_transform(v["xf"]), mesh=mesh, cfg=look, color=wp.vec3(*[float(c) for c in v["color"]]))

        # joints, with their dof settings, then the articulations that group them
        J = scene["joints"]
        D = newton.ModelBuilder.JointDofConfig
        modes = J.get("target_mode") or []
        load = J.get("friction_load_coefficient") or []
        springs = J.get("spring_stiffness") or []
        spring_refs = J.get("spring_ref") or []
        passive = J.get("passive_damping") or []
        groups: dict[int, list[int]] = {}
        for j in range(J["count"]):
            jtype = newton.JointType(J["type"][j])
            parent, child = J["parent"][j], J["child"][j]
            X_p, X_c = _transform(J["X_p"][j]), _transform(J["X_c"][j])
            n_lin, n_ang = J["dof_dim"][j]
            d0, q0 = J["qd_start"][j], J["q_start"][j]
            custom = {}
            if load and load[j] and "steel:joint_friction_load_coefficient" in registered:
                custom["steel:joint_friction_load_coefficient"] = float(load[j])

            def dof(k: int, coord: bool) -> Any:
                d = d0 + k
                return D(
                    axis=tuple(float(x) for x in J["axis"][d]),
                    limit_lower=float(J["lower"][d]), limit_upper=float(J["upper"][d]),
                    target_pos=float(J["target_q"][q0 + k]) if coord else 0.0, target_vel=float(J["target_qd"][d]),
                    target_ke=float(J["ke"][d]), target_kd=float(J["kd"][d]),
                    damping=float(passive[d]) if passive else 0.0,
                    spring_stiffness=float(springs[d]) if springs else 0.0, spring_ref=float(spring_refs[d]) if spring_refs else 0.0,
                    armature=float(J["armature"][d]), effort_limit=float(J["effort"][d]), velocity_limit=float(J["vel_limit"][d]),
                    friction=float(J["friction"][d]),
                    actuator_mode=newton.JointTargetMode(int(modes[d])) if modes else None,
                )  # fmt: skip

            if jtype == newton.JointType.FIXED:
                b.add_joint_fixed(parent, child, parent_xform=X_p, child_xform=X_c, custom_attributes=custom or None)
            elif jtype == newton.JointType.FREE:
                b.add_joint_free(child, parent=parent, parent_xform=X_p, child_xform=X_c)
            elif jtype == newton.JointType.BALL:
                b.add_joint_ball(
                    parent, child, parent_xform=X_p, child_xform=X_c, armature=float(J["armature"][d0]),
                    friction=float(J["friction"][d0]), damping=float(passive[d0]) if passive else None,
                    custom_attributes=custom or None,
                )  # fmt: skip
            elif jtype == newton.JointType.REVOLUTE:
                b.add_joint_revolute(parent, child, parent_xform=X_p, child_xform=X_c, axis=dof(0, True), custom_attributes=custom or None)
            elif jtype == newton.JointType.PRISMATIC:
                b.add_joint_prismatic(parent, child, parent_xform=X_p, child_xform=X_c, axis=dof(0, True), custom_attributes=custom or None)
            elif jtype == newton.JointType.D6:
                b.add_joint_d6(
                    parent, child, parent_xform=X_p, child_xform=X_c,
                    linear_axes=[dof(k, True) for k in range(n_lin)],
                    angular_axes=[dof(n_lin + k, True) for k in range(n_ang)],
                    custom_attributes=custom or None,
                )  # fmt: skip
            else:
                raise ValueError(f"{self.folder.name}: joint {j} has type {jtype.name}, which the loader does not rebuild")
            art = J["articulation"][j]
            if art >= 0:
                groups.setdefault(art, []).append(j)
        for art in sorted(groups):
            b.add_articulation(groups[art])
        # joint mimics (Model.joint_mimic_joint: q[follower] = c0 + c1 * q[leader]), e.g. a gripper's second finger
        coeffs = J.get("mimic_coeffs") or []
        for follower, leader in enumerate(J.get("mimic_joint") or []):
            if leader >= 0:
                b.set_joint_mimic(follower, leader, coeffs=tuple(float(c) for c in coeffs[follower]))
        for m in scene.get("mimics", []):
            if m.get("enabled", True):
                b.set_joint_mimic(m["follower"], m["leader"], coeffs=(float(m["c0"]), float(m["c1"])))
        if len(b.joint_q) == len(J["q"]):
            b.joint_q = [float(x) for x in J["q"]]
        else:
            print(f"[shelf] {self.folder.name}: joint coordinates differ in length ({len(b.joint_q)} vs {len(J['q'])}); keeping the builder's")

        self.builder = b
        self.twists = {i: tuple(t) for i, t in enumerate(bodies.get("twist", [])) if any(t)}
        frames, driven = int(scene.get("target_frames", 0)), int(scene.get("driven", 0))
        self.targets = _view(blob, scene["targets"], np.float32).reshape(frames, driven).copy() if frames and driven else None
        self.driven = driven
        self.targets_per_substep = bool(scene.get("targets_per_substep", False))
        first_joint = J["articulation_start"][0] if J.get("articulation_start") else 0
        self.target_offset = int(J["q_start"][first_joint]) if J["count"] else 0
        info = scene.get("info", {})
        self.camera = info.get("camera") or DEFAULT_CAMERA
        self.about = info.get("about", "")


@wp.kernel(enable_backward=False)
def _set_targets(targets: wp.array2d[float], frame: wp.array[wp.int32], frac: float, offset: int, out: wp.array[float]):
    i = wp.tid()
    last = targets.shape[0] - 1
    f0 = wp.min(frame[0], last)
    f1 = wp.min(frame[0] + 1, last)
    out[offset + i] = targets[f0, i] + (targets[f1, i] - targets[f0, i]) * frac


@wp.kernel(enable_backward=False)
def _advance(frame: wp.array[wp.int32]):
    frame[0] = frame[0] + 1


class ExportedSceneExample(SceneExample):
    """A hero-montage scene: ``scene.json`` and ``scene.bin`` next to the example's file.

    The scripted joint targets (60 Hz) drive the first articulation's first ``driven`` coordinates,
    held per frame or interpolated per substep as the export says, and held at the last row after.
    """

    def build(self):
        self.exported = ExportedScene(Path(self.file).parent)
        self.camera = self.exported.camera
        if self.exported.targets is not None:
            device = wp.get_device()
            self._targets = wp.array(self.exported.targets, dtype=float, device=device)
            self._frame = wp.zeros(1, dtype=wp.int32, device=device)
        else:
            self._targets = None
        return self.exported.builder, self.exported.twists

    def before_substep(self, substep: int) -> None:
        if self._targets is None:
            return
        frac = substep / self.sim_substeps if self.exported.targets_per_substep else 0.0
        wp.launch(
            _set_targets, dim=self.exported.driven,
            inputs=[self._targets, self._frame, float(frac), self.exported.target_offset, self.control.joint_target_q],
            device=self.model.device,
        )  # fmt: skip

    def after_frame(self) -> None:
        if self._targets is not None:
            wp.launch(_advance, dim=1, inputs=[self._frame], device=self.model.device)
