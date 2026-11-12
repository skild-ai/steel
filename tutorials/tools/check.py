#!/usr/bin/env python3
"""Check tutorials/: metadata fields, index order, and that every recording matches its script.

usage: python tutorials/tools/check.py [--poses]

Fails when a <script>.toml misses a field or is not listed in index.toml, when a recording is
missing or was made from a different version of its script (the site shows the committed script next
to the replay), or when a recording dropped rows without ``allow_dropped_rows = true``. ``--poses``
also decodes every recorded pose and fails on non-finite values. Needs the standard library, plus
numpy for --poses.
"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import tomllib
from timing_groups import runners

TUTORIALS = Path(__file__).resolve().parent.parent
TIMING_RUNNERS = runners()
REQUIRED = {"id": str, "title": str, "about": str, "look": list, "try": list, "param": dict}
OPTIONAL = {
    "spacing": list,
    "speed": (int, float),
    "fit": str,
    "metric": dict,
    "concept": dict,
    "timing": dict,
    "notify_flag": str,
    "update_route": str,
    "allow_dropped_rows": bool,
    "in_review": list,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--poses", action="store_true", help="decode recorded poses and check they are finite"
    )
    args = ap.parse_args()
    errors = []
    index = tomllib.loads((TUTORIALS / "index.toml").read_text())
    listed = [i for group in index["group"] for i in group["tutorials"]]
    for dup in {i for i in listed if listed.count(i) > 1}:
        errors.append(f"index.toml lists {dup} more than once")
    seen = set()
    for path in sorted(TUTORIALS.glob("*.toml")):
        if path.name in ("index.toml", "ruff.toml"):
            continue
        name = path.stem
        try:
            meta = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            errors.append(f"{path.name}: {e}")
            continue
        errors += [f"{path.name}: {e}" for e in check_meta(meta)]
        tid = meta.get("id", name)
        seen.add(tid)
        if tid not in listed:
            errors.append(f"{path.name}: {tid} is not listed in index.toml")
        script = TUTORIALS / f"{name}.py"
        if not script.exists():
            errors.append(f"{path.name}: no {name}.py next to it")
            continue
        recording = TUTORIALS / "recordings" / f"{tid}.json"
        if not recording.exists():
            errors.append(f"{tid}: no recording; run tools/record_all.py --only {tid}")
            continue
        rec = json.loads(recording.read_text())
        sha = hashlib.sha256(script.read_bytes()).hexdigest()
        if rec.get("script_sha256") != sha:
            errors.append(f"{tid}: {name}.py changed after it was recorded; record it again")
        if any(v.get("overflow") for v in rec["variants"]) and not meta.get("allow_dropped_rows"):
            errors.append(f"{tid}: the recording dropped rows")
        if "timing" in meta and meta["timing"].get("group") not in TIMING_RUNNERS:
            errors.append(
                f"{tid}: no tools/timing*.py runner defines group {meta['timing'].get('group')}"
            )
        if "timing" in meta and "timing" not in rec:
            errors.append(
                f"{tid}: [timing] is set but the recording has no cost rows; record with --timing"
            )
        if args.poses:
            errors += [f"{tid}: {e}" for e in check_poses(rec)]
    for missing in sorted(set(listed) - seen):
        errors.append(f"index.toml lists {missing}, which has no <script>.toml")
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} problem(s)")
    print(f"{len(seen)} tutorials OK")


def check_meta(meta):
    for key, kind in REQUIRED.items():
        if not isinstance(meta.get(key), kind):
            yield f"missing or invalid {key}"
    for key, value in meta.items():
        if key not in REQUIRED and key not in OPTIONAL:
            yield f"unknown field {key}"
        elif key in OPTIONAL and not isinstance(value, OPTIONAL[key]):
            yield f"{key} has the wrong type"
    if not {"name", "text"} <= set(meta.get("param", {})):
        yield "param needs name and text"
    metric = meta.get("metric")
    if metric is not None:
        for chart in metric.get("charts", [metric]):
            if "label" not in chart or not ("key" in chart or "multi" in chart):
                yield "every chart needs a label and a key or multi"
    for entry in meta.get("in_review", []):
        if not (isinstance(entry.get("pr"), int) and isinstance(entry.get("text"), str)):
            yield "every in_review entry needs an integer pr and a text"
    concept = meta.get("concept")
    if concept is not None:
        if not {"name", "what", "here", "consider"} <= set(concept):
            yield "concept needs name, what, here and consider"
        elif not all({"term", "text"} <= set(c) for c in concept["consider"]):
            yield "every concept.consider needs term and text"


def check_poses(rec):
    import numpy as np

    for v in rec["variants"]:
        poses = np.frombuffer(base64.b64decode(v["poses"]), dtype=np.float32)
        if poses.size != v["frames"] * v["bodies"] * 7:
            yield f"{v['label']}: pose count does not match frames x bodies"
        elif not np.isfinite(poses).all():
            yield f"{v['label']}: non-finite poses"
        particles = v.get("particles")
        if particles:
            n = particles["count"]
            frames = np.frombuffer(base64.b64decode(particles["frames"]), dtype=np.float32)
            if frames.size != v["frames"] * n * 3:
                yield f"{v['label']}: particle count does not match frames x particles"
            elif not np.isfinite(frames).all():
                yield f"{v['label']}: non-finite particle positions"
            tris = particles.get("tris") or []
            if len(tris) % 3 or any(not 0 <= i < n for i in tris):
                yield f"{v['label']}: particle triangle indices out of range"


if __name__ == "__main__":
    main()
