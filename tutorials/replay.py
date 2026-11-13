# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Record Steel tutorial runs as replay JSON for the tutorials page, or print a summary.

Every tutorial imports this module. ``python NN_name.py`` runs the variants and prints
the final metric of each; ``--record out.json`` also writes body poses, shapes and metric
series that the page replays in 3D.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time

import newton
import numpy as np
import warp as wp

from source_hash import source_sha256

_RENDERED_TYPES = {
    newton.GeoType.PLANE,
    newton.GeoType.SPHERE,
    newton.GeoType.CAPSULE,
    newton.GeoType.ELLIPSOID,
    newton.GeoType.CYLINDER,
    newton.GeoType.BOX,
    newton.GeoType.MESH,
    newton.GeoType.CONE,
    newton.GeoType.CONVEX_MESH,
}


class Recorder:
    """Collect the variants of one tutorial; ``fps`` is the replay frame rate."""

    def __init__(self, fps: int = 30):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--record", metavar="OUT.json", help="write a replay file for the tutorials page"
        )
        parser.add_argument("--device", default=None, help="Warp device, for example cuda:0")
        self.args = parser.parse_args()
        if self.args.device:
            wp.set_device(self.args.device)
        self.fps = fps
        self.script = os.path.abspath(sys.argv[0])
        self.variants: list[Variant] = []

    def frames(self, seconds: float) -> int:
        """Number of replay frames in ``seconds``, including the first."""
        return int(round(seconds * self.fps)) + 1

    def variant(
        self,
        label: str,
        model: newton.Model,
        *,
        ghost: bool = False,
        slot: int | None = None,
        scene: str | None = None,
    ) -> "Variant":
        """Start one run; ``ghost`` overlays run ``slot``, ``scene`` groups runs for charts."""
        variant = Variant(label, model, self.fps, ghost, slot, scene)
        self.variants.append(variant)
        return variant

    def save(self) -> None:
        """Print a summary and, with ``--record``, write the replay file."""
        for variant in self.variants:
            last = variant.metrics[-1] if variant.metrics else {}
            flag = "  (rows dropped)" if variant.overflow else ""
            print(f"{variant.label:<48} {json.dumps(last)}  {variant.wall:.2f} s{flag}")
        if not self.args.record:
            return
        source_sha = source_sha256(self.script)
        doc = {
            "script": os.path.basename(self.script),
            "script_sha256": source_sha,
            "newton_commit": _newton_commit(),
            "newton_version": newton.__version__,
            "warp_version": wp.config.version,
            "gpu": wp.get_device().name,
            "fps": self.fps,
            "variants": [v.to_json() for v in self.variants],
        }
        with open(self.args.record, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        print(f"wrote {self.args.record}")


class Variant:
    """One recorded run: a pose per frame for every body, plus named metric values."""

    def __init__(self, label, model, fps, ghost, slot, scene):
        self.label, self.model, self.fps, self.ghost, self.slot = label, model, fps, ghost, slot
        self.scene = scene
        self.poses: list[np.ndarray] = []
        self.metrics: list[dict] = []
        self.awake: list[list[int]] = []
        self.particles: list[np.ndarray] = []
        self.overflow = False
        self.start = time.perf_counter()
        self.wall = 0.0

    def frame(self, state: newton.State, awake=None, particles=None, **metrics) -> None:
        """Record ``state.body_q``, optional awake flags, particle positions and metric values."""
        self.poses.append(state.body_q.numpy().astype(np.float32).copy())
        if particles is not None:
            self.particles.append(particles.numpy().astype(np.float32).copy())
        self.metrics.append({k: _round(v) for k, v in metrics.items()})
        if awake is not None:
            self.awake.append([int(x) for x in awake.numpy()])
        self.wall = time.perf_counter() - self.start

    def check(self, solver) -> None:
        """Note whether the solver dropped rows; call once after the run."""
        try:
            solver.check_constraint_capacity()
        except RuntimeError:
            self.overflow = True

    def to_json(self) -> dict:
        poses = (
            np.stack(self.poses)
            if self.poses
            else np.zeros((0, self.model.body_count, 7), np.float32)
        )
        return {
            "label": self.label,
            "fps": self.fps,
            "ghost": self.ghost,
            "slot": self.slot,
            "scene": self.scene,
            "bodies": int(self.model.body_count),
            "frames": int(poses.shape[0]),
            "poses": base64.b64encode(poses.astype(np.float32).tobytes()).decode(),
            "shapes": _shape_table(self.model),
            "metric": self.metrics,
            "awake": self.awake or None,
            "particles": self._particles_json(),
            "wall_s": round(self.wall, 3),
            "overflow": self.overflow,
        }

    def _particles_json(self):
        if not self.particles:
            return None
        frames = np.stack(self.particles)
        tris = self.model.tri_indices
        radius = self.model.particle_radius.numpy()
        return {
            "count": int(frames.shape[1]),
            "frames": base64.b64encode(frames.astype(np.float32).tobytes()).decode(),
            "radius": round(float(radius.mean()), 5) if radius.size else 0.0,
            "tris": [int(v) for v in tris.numpy().reshape(-1)]
            if tris is not None and tris.shape[0]
            else None,
        }


def _shape_table(model):
    types = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    xforms = model.shape_transform.numpy()
    bodies = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    shapes = []
    for i, t in enumerate(types):
        if int(t) not in _RENDERED_TYPES or not flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        shape = {
            "b": int(bodies[i]),
            "t": int(t),
            "s": [round(float(v), 5) for v in scales[i]],
            "x": [round(float(v), 6) for v in xforms[i]],
        }
        if int(t) in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            mesh = model.shape_source[i]
            shape["v"] = [round(float(v), 5) for v in np.asarray(mesh.vertices).reshape(-1)]
            shape["i"] = [int(v) for v in np.asarray(mesh.indices).reshape(-1)]
        shapes.append(shape)
    return shapes


def _round(value):
    value = float(value)
    return round(value, 5) if np.isfinite(value) else None


def _newton_commit():
    if os.environ.get("NEWTON_COMMIT"):
        return os.environ["NEWTON_COMMIT"]
    try:
        root = os.path.dirname(os.path.dirname(newton.__file__))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
