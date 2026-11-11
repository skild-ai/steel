# Working in Steel examples

Read this before editing an example or a preset.

This project lives in `steel/examples/`. All paths below are relative to `examples/`,
and commands run from that directory. The parent repository does not contain the Newton solver.

## Layout

```
common/presets.py      loads presets.toml and builds solvers
common/modes.py        the play / bench / tune loops (and the tuner's recorder)
common/live.py         how a changed setting takes effect without a rebuild (live / model / swap tiers)
common/panel.py        the tuner's side bar: Scene and Tuner tabs, controls, themes
common/ranges.py       which control a setting gets and its range, from common/ranges.toml
common/looks.py        palettes and lighting for the OpenGL viewer, from common/looks.toml (`--look`)
common/gallery.py      the example browser: cards with previews and preset chips (`shelf play` with no example)
common/thumbs.py       the browser's previews, one frame per example, cached per look (`shelf thumbs`)
common/steel_scenes.py the base for scenes ported from steel-webgpu's tools/scenes.py (builder in, shelf loop around)
common/exported_scene.py  steel-webgpu's exported hero-montage scenes rebuilt from scene.json + scene.bin
common/loop_linkage.py the base for the closed-loop linkage examples (tree + loop closures, scripted drive, reference check)
common/loop_mechanisms.py  the closed-loop mechanisms (bodies, joints, cuts, trajectories)
common/cli.py          `python -m shelf list | play | bench | tune | check`
shelf/<NN_section>/<example>/
    example.py         a Newton example (Example class + __main__ block), reading its preset
    presets.toml       one table per solver
PRESETS.md             generated status matrix; never edit by hand
```

Use the repo's environment: `uv sync`, then `uv run python -m shelf ...` (or `.venv/bin/python -m shelf ...`).
If `VIRTUAL_ENV` points at another project, unset it first.

## Presets

`presets.toml` has a top-level `example`, `source`, `default` (the preset used when no `--solver` is given),
a `[check]` table with the arguments `check` passes, and one table per solver:
`xpbd vbd mujoco featherstone semi_implicit feather_pgs steel kamino implicit_mpm style3d`.

```toml
[mujoco]
status = "works"          # works | unstable | unsupported | untested (written by `check`)
checked = "..."           # written by `check`: date, Newton commit, GPU
note = "..."              # optional; `check` writes failures here
reason = "..."            # required when status = "unsupported"
sim = { fps = 60, substeps = 4 }
collision = { ... }       # collision pipeline settings the example reads
solver = { ... }          # solver constructor kwargs, or Config fields (Kamino, implicit MPM)
model = { ... }           # per-solver model/builder tweaks the example applies
check = { args = [...] }  # optional per-preset override of [check].args
```

and a `[gallery]` table for the example browser:

```toml
[gallery]
short = "..."             # one line: what the scene does, not which API it shows
frame = 60                # optional: the frame the preview is rendered at, when the first second is still the scene as built
title = "..."             # optional: the card's name instead of the folder name
```

Rules:

- Every `presets.toml` has a `[gallery]` table; the browser warns about a file without one. After changing
  `frame`, render the preview again with `python -m shelf thumbs <example> --force`.
- A scene ported from steel-webgpu keeps its builder as written there (`common/steel_scenes.py`) or its exported
  `scene.json` + `scene.bin` (`common/exported_scene.py`); its `[steel]` preset carries the browser's settings
  and says where this Newton's Steel needed a different one. A browser compare toggle is a `[steel_off]`
  table with `kind = "steel"`.
- Every table stands alone. There are no shared defaults; repeat a value in each solver that needs it.
- A table with no settings means the example has no code path for that solver yet. Leave it `untested`.
- Only `check` sets `status = "works"`. Never write it by hand.
- `unsupported` is for solvers that cannot simulate the scene at all, and always carries a `reason`.
- An example that builds no physics solver (viewer, replay, IK-only, camera demo) uses `default = "none"`
  and a `[none]` table (with `sim` if it has timing); the ten solver tables are `unsupported` with
  `reason = "the example builds no physics solver"`.
- A second configuration of the same solver (e.g. Kamino's DVI mode next to its default PADMM) is a
  variant table named `<solver>_<variant>` with `kind = "<solver>"`, e.g. `[kamino_dvi]` with
  `kind = "kamino"`. The example branches on `self.preset.name` only where the code differs.
- Examples that run several solvers side by side (e.g. `basic_multi_solver_overlay`) use one preset named
  like a coupled one, with an `entries` table per solver.
- Coupled examples name presets by joining solvers with `_` (`mujoco_vbd`, `steel_vbd`). The preset's
  `solver` table holds the coupling settings; each part has its own table under `entries`:

  ```toml
  [mujoco_vbd]
  sim = { fps = 60, substeps = 8 }
  solver = { coupling_mode = "lagged", proxy_iterations = 1, mass_scale = 1.0 }
  entries.mujoco.solver = { use_mujoco_contacts = false, njmax = 200 }
  entries.vbd.solver = { iterations = 10 }
  ```

## Converting an example to read its presets

`shelf/00_basics/basic_pendulum` is the reference. Keep the diff small and the physics identical.

1. Add `from common import presets` and make `self.preset = presets.load(__file__)` the first line of
   `Example.__init__`. Use `self.preset.name` wherever the example branched on its solver choice.
2. Read timing from the preset: `self.fps = self.preset.fps`, `self.sim_substeps = self.preset.substeps`.
   If the example derived substeps (e.g. from a target dt), store the resulting integer.
3. Build the solver with `self.preset.make_solver(self.model, **runtime_kwargs)`. Every literal tuning
   value (iterations, tolerances, modes, flags) moves to the `solver` table. Only values computed at run
   time (arrays, callbacks, objects) stay in code as `runtime_kwargs`. For coupled entries use
   `self.preset.entry("mujoco").make_solver(view, ...)`.
4. Solver-dependent model or collision values move to `model` / `collision` and are read with
   `self.preset.model.need("soft_contact_kd")`, `self.preset.collision.need(...)`, or
   `self.preset.apply_model(target)` for plain attributes.
5. Delete the example's own `--solver` argument (the runner owns `--solver`) and any argument that only
   tunes a solver (`--iterations`, `--substeps`, solver modes): those values now live in presets. Keep
   arguments that change the scene (world count, which robot, which scene).
   If solver settings are consumed by code inside the installed `newton` package (e.g. the Steel
   tabletop demos read `args.substeps` and `args.iterations`), fill those `args` attributes from the
   preset before handing `args` over, and say so in a comment.
6. Fill one table per solver the example already supports, with exactly the values the original code
   used, and set `default` to the original default. Leave every other solver `untested` with no settings.
7. Do not change scenes, tests (`test_final`, `test_post_step`) or anything else. Never loosen a test.
8. Verify with `python -m shelf check <example> --every`. Every filled table must reach `works`. If one
   cannot, leave the `unstable` result and its note, and report why.

## Live tuning and `build_solver()`

`python -m shelf tune` applies a changed setting with the cheapest *tier* that makes it take effect
(`common/live.py`): **live** sets an attribute the solver reads every step, **model** writes a Model
array and calls `notify_model_changed`, **swap** builds a new solver on the running model and states
(the scene keeps its state), **rebuild** constructs the example again. The panel shows each setting's
tier. After every tier the example's CUDA graph is captured again by calling `capture()`; a graph stored
under another name (`_graph`, `sand_graph`, ...) that `capture()` does not re-create is dropped, and that
part of the example then steps eagerly. So keep `self.graph` and `capture()` as Newton writes them, and
if `build_solver()` re-creates objects a second graph recorded, re-capture that graph inside it too.

The swap tier needs to build the solver the way `__init__` did. When `__init__` builds it with exactly
`self.solver = self.preset.make_solver(self.model)` and does nothing else with the instance, the tuner
swaps it by itself. Otherwise the example defines

```python
def build_solver(self):
    """Build the solver and everything bound to its instance; the tuner calls this again to swap it."""
```

and `__init__` calls `self.build_solver()` where the construction was. Everything that depends on the
solver *instance* goes inside: extra `make_solver` arguments, `self.solver.observables(...)`, pipelines
and `contacts` sized by `get_max_contact_count()`, `set_rigid_history_update`, `rebuild_bvh`,
`update_contacts` / `prepare_contacts`, coupled sub-solvers (`self.preset.entry(...).make_solver`), and
solvers under other names (`self.robot_solver`, `self.cloth_solver`). What depends only on the model
(states, control, `viewer.set_model`, `capture()`) stays in `__init__`. `build_solver()` must be safe
to call again at any time: it re-creates, never accumulates, and reads `self.preset` (the tuner
replaces it before calling). See `shelf/07_articulations/robot_g1/example.py`.

Verify with `python scripts/live_check.py <example> --solver <preset>` for every preset marked
`works` (it prints `LIVE OK ... tiers=...`), then `python -m shelf check <example> --dry-run`.

## Checking

```
python -m shelf check <example>            # tables marked works/unstable
python -m shelf check <example> --every    # every table that has settings
python -m shelf check --all                # whole shelf; rewrites PRESETS.md
python -m shelf play <example> --solver kamino --set solver.iterations=16  # try a change without editing
python -m shelf tune <example>             # edit settings live; Save as writes a new <name>.toml
python -m shelf check <example> --presets <name>.toml --every   # check a preset file saved by tune
python -m shelf bench <example> --world-count 1024   # throughput, step time, GPU memory
python -m shelf play <example> --look newton         # Newton's own look instead of the default `steel`
python -m shelf play                                 # the example browser (also `shelf tune`); renders missing previews in the background
python -m shelf thumbs --missing                     # render the browser's missing previews now (--force: all again)
```

A look never changes the simulation. Do not hard-code look colors in an example to match a look; add or
edit a look in `common/looks.toml` instead.

`check` runs the example headless (`--viewer null --test`) with the `[check]` arguments, fails it when a Steel
solver dropped contacts or rows (`SolverSteel.check_constraint_capacity()` after the run), writes `status`,
`checked` and any failure `note` into `presets.toml`, and regenerates `PRESETS.md`.
