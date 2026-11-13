# Run a trial

> **Does:** runs one example's whole episode with Steel at given settings and lists what is wrong, in fix order.
> **Use in:** tune-a-scene, debug-instability, add-an-example.
> **Passes when:** `trial.json` has `status: "ok"`, no `advice.actions` and no critical flag, and the check passed.

**Inputs.** An example name (`uv run python -m shelf list`); settings, or nothing for the example's `[steel]` preset.
**Tool.** `shelf trial` (`examples/common/advisor/`). Run from `examples/` in its environment (`uv sync` first).

```bash
cd examples
uv run python -m shelf trial EXAMPLE --out outputs/EXAMPLE_t0                 # the [steel] preset
uv run python -m shelf trial EXAMPLE --settings '{"sim": {"fps": 60, "substeps": 4}, "solver": {}}' --out outputs/EXAMPLE_t1
uv run python -m shelf advise outputs/EXAMPLE_t0/trial.json                    # the rules again, offline
```

The trial runs the example for 10 s (or its check's length, up to 20 s) with `row_watermark=True`, records the
episode facts (row watermarks, start state against forward kinematics, contact-island mass ratios, penetration,
loop gaps, grasp slip, warm-start bursts, joint-limit excess, a frozen state, the options the solver actually got),
runs the example's own `shelf check` under the same settings and writes `trial.json`. Add `--no-check` to skip the
check and `--seconds` to change the length. `shelf trial` has no time limit of its own.

**Read the advice in order.** Fixes come in the tuning guide's section 3 order
([guide](../../../examples/docs/steel_tuning/guide.md)); an earlier one makes later numbers meaningless:

1. construction errors with a known fix (loops under propagation, torsion without the device path, momentum outside
   the serial sweep, a renamed option, a setting the example reads);
2. dropped rows (`*_capacity`): any overflowing world-step fails the run, however it looks;
3. scene bugs no setting fixes (critical flags): a start state written to `body_q`/`body_qd` only, shapes that
   overlap at rest, bodies that tunnel into static geometry, contacts with non-finite geometry;
4. physics accuracy: authored limits left off, open loops, mass-ratio islands that sink or topple, grasp slip,
   warm-start bursts or unmatched carried contacts;
5. a failed check with no other cause: substeps, then iterations.

`level` says how sure a rule is; `metric` names the number its fix must lower; `evidence` holds the measurements.

**Outputs.** `outputs/EXAMPLE_tN/{trial.json,presets.toml}`. **Fails when** the status is not `ok`, the episode is
truncated, a capacity check fails, a critical flag remains, or the check fails; `trial.json` says which.

**Your own scene.** Use `EpisodeProbe` and `recommend_stability` from `common.advisor` in your own loop (guide
section 9).
