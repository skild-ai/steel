# Size capacities

> **Does:** sets Steel's constraint-row capacities so no row is ever dropped, with headroom, from measured demand.
> **Use in:** tune-a-scene, debug-instability, port-from-physx.
> **Passes when:** every row family reports zero overflowing world-steps and `check_constraint_capacity()` passes on full episodes at the target world count.

**Inputs.** A scene that runs its whole task; the world count you will use; several seeds.
**Tools.** `shelf trial` for examples; in your own code, `SolverSteel(..., row_watermark=True)`,
`solver.constraint_row_watermarks()` and `solver.check_constraint_capacity()`.

1. Start generous: raise `articulation_constraint_max`, `free_body_constraint_max` (and
   `propagation_constraint_max` under the propagation response) well above what you expect.
2. Run the complete task, through every load (impact, grasp, pour, settling), with `row_watermark=True`, over
   several seeds:
   ```bash
   cd examples && uv run python -m shelf trial EXAMPLE --out outputs/EXAMPLE_cap
   ```
   Read `facts.episode.row_watermarks` in `trial.json`: per family the requested high-water mark
   (`*_raw_high_water`), `*_overflow_world_steps` and dropped contact rows; and the contact buffer's high-water mark
   against its size.
3. Set each family to 1.3x its requested high-water mark from a run that did not overflow. A family that
   overflowed under-reports its demand: double it and measure again.
4. Re-run at the world count you will ship. The busiest world sets the requirement and it varies between runs, so
   repeat near the boundary.

**Rules.** The families and the contact buffer are independent; raising one never repairs another. Any overflowing
world-step fails the run however the video looks. Smaller capacities save memory but rarely change speed.

**Outputs.** The capacities, the high-water marks they came from, and the runs that confirm zero overflow.
