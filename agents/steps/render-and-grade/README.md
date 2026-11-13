# Render and grade

> **Does:** renders examples with their Steel presets on one or more GPUs, records stability metrics, and grades each scene ok, suspect, unstable or failed.
> **Use in:** validate-a-solver-change, tune-a-scene, add-an-example, debug-instability.
> **Passes when:** every planned scene has a grade other than `failed`, or failed twice and is reported as broken.

**Inputs.** The examples tree at a commit and its environment; CUDA GPUs; headless EGL (`libEGL.so.1`, from the
NVIDIA driver); `ffmpeg` with libx264 (on `PATH` or `FFMPEG=...`); optionally `STEEL_WEBGPU_ASSETS` for the exported
scenes' meshes. **Tools.** `examples/scripts/render_audit/` (`plan.py`, `run_all.sh`, `render_one.sh`,
`render_scene.py`, `collect.py`, `build_gallery.py`, `steel_scope.py`).

From the repository root:

```bash
PY=examples/.venv/bin/python                       # Python 3.11+, the examples' environment
T=examples/scripts/render_audit
RUN=examples/outputs/render-audit/$(git rev-parse --short HEAD)
$PY $T/plan.py examples "$RUN/plan.json"           # every example, or name scenes after the path
$T/run_all.sh "$RUN" 0 1 2 3                        # one worker per listed GPU
$T/render_one.sh "$RUN" SCENE                       # one scene (also what a cluster job array runs per task)
$PY $T/collect.py "$RUN" --previous OLD_RUN_DIR     # compress, grade, results.json, grade changes
$PY $T/build_gallery.py "$RUN"                      # index.html
```

- **Plan.** Every `examples/shelf/*/*/presets.toml`. A scene renders with its `[steel]` table, or a coupled preset
  whose name contains `steel`. Examples with no Steel preset are triaged by `steel_scope.py`: `excluded` (no rigid
  or articulated physics) or `skipped` ("no Steel preset yet", "needs Steel feature"). `plan.py` warns when the
  examples have uncommitted changes: audit a commit, and do not edit a tree a render imports from.
- **Render the preset as committed.** The `[check]` scene arguments apply; its `--set` overrides do not. The only
  addition is `SolverSteel(row_watermark=True)`. `SHELF_SET`, `SHELF_PRESETS` and `SHELF_SOLVER` are dropped from
  the environment, and a render whose `presets.toml` no longer matches `plan.json` fails.
- **Each attempt** deletes the scene's old outputs and writes a new id to `out/SCENE/attempt`, which `metrics.json`
  must carry. `render_one.sh` and `run_all.sh` exit nonzero when a render or check failed; `run_all.sh` lists those
  scenes in `$RUN/failed.txt`. Render again once each scene whose render failed (`RENDER_RC` not 0); a failed check
  alone is a finding.
- **Collect** counts a scene as rendered only when its render exited 0, its metrics belong to the latest attempt,
  and its clip exists and decodes without errors; a check passes only when its process exited 0.

**Grades.** *failed*: the render did not finish, exited nonzero, left no usable clip, or its metrics are stale.
*unstable*: non-finite state; contacts with non-finite geometry; dropped rows or contacts; a speed above 3x the
scene's bound; a body through the ground; a body's centre of mass inside a colliding static box; sustained
penetration above 2x the threshold; a joint more than 0.2 past its limit; an exception while stepping; a failed
check. *suspect*: lesser versions of those, bodies leaving the bounds, energy rising above 1x mgL in a passive
scene, resting bodies that jitter, an early stop. Thresholds scale with the scene's extent; the joint-limit check
covers one-DOF revolute and prismatic joints.

**Outputs.** `$RUN/{plan.json,out/,media/,results.json,index.html,failed.txt}`. Grades are a screen: follow with
visual-review.
