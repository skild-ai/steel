# Tune a scene

> **Goal:** a Steel preset for one scene that is correct over the whole task first and as fast as it stays correct second.
> **Use when:** asked to make a Steel scene stable or faster, to pick Steel settings or a preset, or to check whether a setting is safe to ship.
> **Done when:** the settings you ship pass `shelf check`, cover the whole task with zero dropped rows, and their frames look right.

**Inputs.** An example (`uv run python -m shelf list` from `examples/`) or a scene of your own; where to start
(`preset` or `defaults`); the world count you will use. Read the [tuning guide](../../../examples/docs/steel_tuning/guide.md)
sections 1-3 first.

## Steps

1. [run-trial](../../steps/run-trial/README.md) at the current preset, to see where you start.
   - A critical flag names a scene bug (start state written to body arrays only, shapes overlapping at rest,
     non-finite contact geometry) → fix the example's code; settings cannot. If you cannot tell what is wrong →
     [debug-instability](../debug-instability/README.md).
2. [autotune-settings](../../steps/autotune-settings/README.md) from `preset` or `defaults`.
   - `final_healthy: false` → [debug-instability](../debug-instability/README.md) with the last trial.
3. Render the current preset as the base: [render-and-grade](../../steps/render-and-grade/README.md) for this
   scene alone, from a clean checkout. Keep its run dir.
4. Assemble the candidate on its own branch or worktree (the base checkout stays untouched): copy `final_settings`
   into the example's `[steel]` table, then apply [size-capacities](../../steps/size-capacities/README.md) at the
   world count you will ship, so the capacities you measure are the last change to the table. Commit it.
5. In the candidate checkout: [check-example](../../steps/check-example/README.md), then
   [render-and-grade](../../steps/render-and-grade/README.md) for the scene with `collect.py --previous BASE_RUN_DIR`,
   then [visual-review](../../steps/visual-review/README.md), comparing its sheet with the base's. A cheaper setting
   can change the motion without failing a metric.
   - The motion changed in a way the task does not allow → go back to the last kept setting, or
     [tune-one-knob](../../steps/tune-one-knob/README.md) from it, and repeat steps 4-5.
6. Optional: [benchmark-throughput](../../steps/benchmark-throughput/README.md), base preset against the candidate
   preset, at the shipping world count.

## Before shipping

- The example's own check passes at the settings you ship, never at cheaper test settings.
- The run covers the whole task, through the load (impact, grasp, pour, settling).
- Zero overflowing world-steps and `check_constraint_capacity()` passes, at the world count you will use: rare
  failures show only at scale (one world in 64, or a few in 256).
- The settings reached the solver: `trial.json` `facts.solver_kwargs`. Examples that build their own solver ignore
  preset keys, and the advisor flags it.

## Rules of thumb (details in the guide)

- Never turn off physics the scene relies on (limits, velocity limits, friction anchors that hold a grasp, torsion,
  collision, self-collision) to win time.
- Capacities: 1.3x the requested high-water mark of a run that did not overflow.
- Heavy impactor (mass ratio above about 1000:1): 2x substeps, 2x iterations, `baumgarte` 0.5. Resting stack
  (100-1000:1): momentum, then a sweep budget spent as iterations in the fewest substeps.
- Anything held or carried: friction anchors on, warm start off.
- Fewer substeps is the usual throughput lever; a time-step-limited scene (ropes, fast chains) refuses it.

## Deliverables

The edited `presets.toml` (with a comment per setting saying why), `tune.json`, the gallery for the scene, and a
short report: settings, trials to the first healthy one and in total, frame time against where you started, every
fix and cost change kept or rejected, what you looked at.
