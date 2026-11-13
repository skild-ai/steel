# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Compress the clips of a run, grade every scene from its metrics and check, write results.json.

usage: python3 collect.py <run dir> [--previous <earlier run dir>]

The run dir holds plan.json and out/<scene>/ (render_one.sh writes them). Reads <run dir>/visual_review.json if
present ({scene: {"grade": ..., "note": ...}}): the reviewer's visual verdict can raise a grade (never lower an
automatic unstable) and its note joins the reasons.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
MAX_BYTES = 1_500_000
RANK = {"ok": 0, "suspect": 1, "unstable": 2, "failed": 3}


def grade(entry: dict, res: dict | None, check: dict) -> tuple[str, list[str]]:
    """Automatic grade and its reasons from one scene's metrics.json and check result."""
    if res is None:
        return "failed", ["no metrics.json: the render task died"]
    if not res.get("loop_ran"):
        why = res.get("build_error") or "the example runs its own loop"
        return "failed", [f"did not render: {why[-200:]}"]
    m = res.get("metrics", {})
    bad, sus = [], []
    if res.get("error"):
        bad.append(f"step raised: {res['error'][-160:]}")
    if m.get("nonfinite_frame") is not None:
        bad.append(f"non-finite {m.get('nonfinite_what')} at t={m['nonfinite_frame'] / res['fps']:.2f}s")
    for s in m.get("steel", []):
        cap = s.get("capacity_check", "ok")
        if cap != "ok":
            wm = s.get("row_watermarks", {})
            fam = [f"{k.split('_')[0]} excess {wm[k]}" for k in wm if k.endswith("overflow_excess_high_water") and wm[k]]
            bad.append(f"Steel dropped rows/contacts ({', '.join(fam) or cap[:90]})")
    if m.get("invalid_contact_frames"):
        bad.append(f"{m['invalid_contacts']} contacts with non-finite geometry in {m['invalid_contact_frames']} frames from t={m.get('invalid_contact_first_t')}s")
    if m.get("overflow_frames"):
        if not any("dropped" in b for b in bad):
            bad.append(f"constraint_overflow set on {m['overflow_frames']} frames from t={m.get('overflow_first_t')}s")
    if m.get("max_lin_speed", 0) > 3 * m.get("lin_speed_bound", 1e9):
        bad.append(f"body speed {m['max_lin_speed']:.0f} m/s at t={m.get('max_lin_speed_t')}s (bound {m['lin_speed_bound']:.0f})")
    elif m.get("max_lin_speed", 0) > m.get("lin_speed_bound", 1e9):
        sus.append(f"body speed {m['max_lin_speed']:.1f} m/s at t={m.get('max_lin_speed_t')}s (bound {m['lin_speed_bound']:.0f})")
    if m.get("max_ang_speed", 0) > 3 * m.get("ang_speed_bound", 1e9):
        bad.append(f"angular speed {m['max_ang_speed']:.0f} rad/s (bound {m['ang_speed_bound']:.0f})")
    elif m.get("max_ang_speed", 0) > m.get("ang_speed_bound", 1e9):
        sus.append(f"angular speed {m['max_ang_speed']:.0f} rad/s at t={m.get('max_ang_speed_t')}s")
    if m.get("bodies_below_ground"):
        bad.append(f"{m['bodies_below_ground']} bodies fell through the ground (t={m.get('below_ground_t')}s)")
    if m.get("bodies_inside_static"):
        bad.append(f"{m['bodies_inside_static']} bodies sank into static geometry (t={m.get('inside_static_t')}s, {m.get('inside_static_depth_max_m', 0) * 1000:.0f} mm deep)")
    if m.get("bodies_out_of_bounds"):
        sus.append(f"{m['bodies_out_of_bounds']} bodies left the scene bounds (t={m.get('out_of_bounds_t')}s)")
    scale = m.get("scene_scale_m", 1.0)
    pen_thr = min(max(0.01 * scale, 0.003), 0.02)
    if m.get("penetration_sustained_m", 0) > 2 * pen_thr:
        bad.append(f"sustained penetration {m['penetration_sustained_m'] * 1000:.1f} mm")
    elif m.get("penetration_sustained_m", 0) > pen_thr:
        sus.append(f"sustained penetration {m['penetration_sustained_m'] * 1000:.1f} mm")
    elif m.get("penetration_p95_m", 0) > 3 * pen_thr:
        sus.append(f"p95 penetration {m['penetration_p95_m'] * 1000:.0f} mm (peak {m['penetration_peak_m'] * 1000:.0f} mm at t={m.get('penetration_peak_t')}s)")
    passive = not m.get("kinematic_bodies_move") and not m.get("has_joint_drives")
    if passive and m.get("energy_gain_ratio", 0) > 1.0:
        sus.append(f"energy rose {m['energy_gain_ratio']:.1f}x the mgL scale in a passive scene (t={m.get('energy_gain_t')}s)")
    if m.get("jittering_bodies_last_s", 0):
        sus.append(
            f"{m['jittering_bodies_last_s']} resting bodies jitter in the last second "
            f"(path {m.get('rest_path_max_m', 0) * 1000:.0f} mm, {m.get('rest_angpath_max_rad', 0):.2f} rad)"
        )
    jv = m.get("joint_limit_violation_max", 0)
    if jv > 0.2:
        bad.append(f"joint {m.get('joint_limit_violation_joint')} past its limit by {jv:.2f} at t={m.get('joint_limit_violation_t')}s")
    elif jv > 0.05:
        sus.append(f"joint {m.get('joint_limit_violation_joint')} past its limit by {jv:.3f} at t={m.get('joint_limit_violation_t')}s")
    if res.get("stop_reason"):
        sus.append(f"stopped early: {res['stop_reason']}")
    if check.get("result") == "FAILED":
        bad.append(f"shelf check failed: {check.get('note', '')[:140]}")
    elif check.get("result") is None:
        sus.append("shelf check did not report")
    if bad:
        return "unstable", bad + sus
    if sus:
        return "suspect", sus
    return "ok", []


def grade_changes(prev: dict, cur: dict) -> dict:
    """Per-scene grade diff of cur against an earlier results.json: improved, regressed, new, removed."""
    before = {s["scene"]: s for s in prev["scenes"]}
    after = {s["scene"]: s for s in cur["scenes"]}
    prev_skipped = {s["scene"] for s in prev.get("skipped", [])}
    out = {"previous_commit": prev["commit"], "improved": [], "regressed": [], "new": [], "removed": []}
    for name, s in after.items():
        b = before.get(name)
        row = {"scene": name, "before": b["grade"] if b else ("skipped" if name in prev_skipped else None), "after": s["grade"],
               "reason": (s["reasons"] or [""])[0][:200]}  # fmt: skip
        if b is None:
            out["new"].append(row)
        elif RANK[s["grade"]] < RANK[b["grade"]]:
            out["improved"].append({**row, "was": (b["reasons"] or [""])[0][:200]})
        elif RANK[s["grade"]] > RANK[b["grade"]]:
            out["regressed"].append(row)
    out["removed"] = [{"scene": n, "before": b["grade"]} for n, b in before.items() if n not in after]
    return out


def parse_check(path: Path) -> dict:
    """The check's result line, overruled by the process's exit code (render_one.sh appends ``CHECK_RC n``)."""
    if not path.exists():
        return {"result": None}
    txt = path.read_text(errors="replace")
    rc = last_rc(txt, "CHECK_RC")
    out = {"result": None, "tail": txt.strip().splitlines()[-1:] if txt.strip() else []}
    for ln in txt.splitlines():
        m = re.match(r"^(\S+)\s+(\S+)\s+(works|FAILED)\s+([\d.]+) s\s*(.*)$", ln)
        if m:
            out = {"result": m.group(3), "seconds": float(m.group(4)), "note": m.group(5).strip()}
            break
    if rc is None:
        return {"result": None, "note": "the check did not finish (no CHECK_RC)"}
    if rc != 0:
        out["result"] = "FAILED"
        out["note"] = (out.get("note") or "") + f" (check exited {rc})"
    return out


def last_rc(text: str, tag: str) -> int | None:
    """The last ``TAG n`` exit code render_one.sh wrote, None if it wrote none."""
    codes = re.findall(rf"^{tag} (\d+)", text, re.M)
    return int(codes[-1]) if codes else None


def render_problems(d: Path, res: dict | None) -> list[str]:
    """Reasons this scene's outputs are not one finished, successful render of the current attempt."""
    if res is None:
        return []  # grade() reports the missing metrics
    out = []
    attempt = (d / "attempt").read_text().strip() if (d / "attempt").exists() else None
    if attempt is None or res.get("attempt") != attempt:
        out.append("metrics.json does not belong to the latest render attempt")
    log = (d / "render.log").read_text(errors="replace") if (d / "render.log").exists() else ""
    rc = last_rc(log, "RENDER_RC")
    if rc != 0:
        out.append(f"the render process exited {rc}" if rc is not None else "the render did not finish (no RENDER_RC)")
    clip = d / "clip_raw.mp4"
    if res.get("loop_ran") and not (clip.exists() and clip.stat().st_size > 0):
        out.append("no clip was written")
    return out


def encode(src: Path, dst: Path) -> None:
    """Copy the raw clip, re-encoding at a higher crf until it fits MAX_BYTES."""
    shutil.copy(src, dst)
    crf = 30
    while dst.stat().st_size > MAX_BYTES and crf <= 42:
        tmp = dst.with_suffix(".tmp.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)], check=True)  # fmt: skip
        tmp.replace(dst)
        crf += 4


def decodes(path: Path) -> None:
    """Raise unless ffmpeg decodes every frame of ``path`` without an error."""
    proc = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True,
                          timeout=300)  # fmt: skip
    if proc.returncode != 0 or proc.stderr.strip():
        raise subprocess.SubprocessError(proc.stderr.strip() or f"ffmpeg exited {proc.returncode}")


def summary_numbers(res: dict | None) -> dict:
    if not res:
        return {}
    m = res.get("metrics", {})
    keep = ("max_lin_speed", "lin_speed_bound", "max_ang_speed", "penetration_peak_m", "penetration_sustained_m",
            "penetration_p95_m", "energy_gain_ratio", "jittering_bodies_last_s", "bodies_sank_below_start", "joint_limit_violation_max", "bodies_out_of_bounds",
            "bodies_below_ground", "bodies_inside_static", "overflow_frames", "nonfinite_frame", "contacts_max", "bodies", "particles",
            "scene_scale_m")  # fmt: skip
    out = {k: m.get(k) for k in keep if k in m}
    out.update({k: res.get(k) for k in ("step_ms_mean", "step_ms_p95", "fps", "frames_done", "frames_planned", "wall_s")})
    steel = m.get("steel", [])
    if steel:
        wm = steel[0].get("row_watermarks", {})
        out["row_high_water"] = {k.replace("_high_water", ""): v for k, v in wm.items() if k.endswith("_high_water") and "raw" not in k and "dropped" not in k and "excess" not in k}
        caps = {k: v for k, v in steel[0].get("ctor_kwargs", {}).items() if k.endswith("_max")}
        if caps:
            out["capacities"] = caps
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--previous", type=Path, help="an earlier run dir: record per-scene grade changes against it")
    a = ap.parse_args()
    local = a.run
    raw = local / "out"
    media = local / "media"
    media.mkdir(parents=True, exist_ok=True)
    plan = json.loads((local / "plan.json").read_text())
    review_path = local / "visual_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    rows = []
    newton = None
    for e in plan["render"]:
        d = raw / e["scene"]
        res = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None
        check = parse_check(d / "check.log")
        auto, reasons = grade(e, res, check)
        problems = render_problems(d, res)
        for old in (f"{e['scene']}.mp4", f"{e['scene']}.jpg", f"{e['scene']}_sheet.jpg"):
            (media / old).unlink(missing_ok=True)  # a gallery never shows an earlier run's media
        files = {}
        if not problems and (d / "clip_raw.mp4").exists():
            dst = media / f"{e['scene']}.mp4"
            try:
                encode(d / "clip_raw.mp4", dst)
                decodes(dst)
                files["video"] = f"media/{e['scene']}.mp4"
            except (OSError, subprocess.SubprocessError) as exc:
                dst.unlink(missing_ok=True)
                problems.append(f"the clip is not usable video: {str(exc)[-200:]}")
        if problems:
            auto, reasons = "failed", problems + reasons
        g = auto
        v = review.get(e["scene"])
        if v:
            vg = v.get("grade")
            # the visual verdict wins, except that it never lowers an automatic unstable or failed
            if vg and (g not in ("unstable", "failed") or RANK[vg] > RANK[g]):
                g = vg
            if v.get("note"):
                reasons = [f"visual: {v['note']}"] + reasons
        for name, key in (("poster.jpg", "poster"), ("sheet.jpg", "sheet")):
            if (d / name).exists():
                tgt = media / f"{e['scene']}{'' if key == 'poster' else '_sheet'}.jpg"
                shutil.copy(d / name, tgt)
                files[key] = f"media/{tgt.name}"
        if res and res.get("newton_commit"):
            newton = res["newton_commit"]
        rows.append({
            "scene": e["scene"], "title": e["title"], "short": e["short"], "path": e["path"], "category": e["category"],
            "preset": e["preset"], "kind": e["kind"], "preset_status": e["status"], "preset_checked": e["checked"],
            "settings": e["settings"], "check_args": e["check_args"],
            "sim_seconds": e["sim_seconds"], "video_speed": e["speed"],
            "grade": g, "auto_grade": auto, "visual": v, "reasons": reasons,
            "render_status": "ok" if res and res.get("loop_ran") and not res.get("error") and not problems else ("error" if res and res.get("loop_ran") else "failed"),
            "render_error": (res or {}).get("error") or (res or {}).get("build_error"),
            "check": check, "metrics": summary_numbers(res), "files": files,
            "gpu": (res or {}).get("gpu"),
        })  # fmt: skip
    counts = {k: sum(r["grade"] == k for r in rows) for k in RANK}
    out = {
        "commit": plan["commit"], "dirty": plan.get("dirty", False), "newton_pin": plan["newton_pin"], "newton_commit": newton,
        "date": __import__("datetime").date.today().isoformat(), "counts": counts,
        "rendered": sum(r["render_status"] == "ok" for r in rows), "render_failed": sum(r["render_status"] != "ok" for r in rows),
        "skipped": plan["skipped"], "excluded": plan.get("excluded", []), "scenes": rows,
    }  # fmt: skip
    if a.previous:
        out["changes"] = grade_changes(json.loads((a.previous / "results.json").read_text()), out)
    (local / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("counts", "rendered", "render_failed")}), f"skipped={len(plan['skipped'])} excluded={len(out['excluded'])}")
    if "changes" in out:
        print("changes vs", out["changes"]["previous_commit"][:7], {k: len(v) for k, v in out["changes"].items() if isinstance(v, list)})
    total = sum(p.stat().st_size for p in media.iterdir())
    print(f"media {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
