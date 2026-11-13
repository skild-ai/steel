# Autotune settings

> **Does:** repeats trials until no fix is left, then keeps only cheaper settings that stay healthy.
> **Use in:** tune-a-scene, add-an-example.
> **Passes when:** `tune.json` has `final_healthy: true`.

**Inputs.** An example; a start point (`preset`, `defaults`, or a JSON settings file); trial budgets.
**Tool.** `shelf autotune` (`examples/common/advisor/loop.py`).

```bash
cd examples
uv run python -m shelf autotune EXAMPLE --start defaults --out outputs/EXAMPLE_tune     # or --start preset / FILE.json
uv run python -m shelf autotune EXAMPLE --start preset --max-trials 6 --cost-trials 5 --timeout 2400 --out outputs/EXAMPLE_tune
```

- **Phase 1 (fixes).** Applies the advice's fixes until none is left. When every fix of a round names a metric,
  none drops by 25% and the trial is not healthy, the round is reverted and those fixes are not tried again.
- **Healthy** means: a finite, complete episode; finite contact geometry; a passing capacity check; the example's
  check passing; a trial process that exited 0; no fix left (an ignored fix counts as left).
- **Phase 2 (cost).** Uses the rest of the shared budget (`--max-trials` + `--cost-trials` trials in all) on colored
  propagation (piles of 20+ free bodies), half the substeps, half the substeps at twice the iterations, a 0.1 limit
  gap, point friction (nothing held) and half the iterations. A candidate is kept only if it is healthy, at least 3%
  faster, and its penetration, speed, sinking, bursts, joint-limit, loop-gap, frozen-state and grasp-slip
  (translation and rotation) numbers stay near the first healthy settings', with none of them going unmeasured.
- Each trial runs in its own process group with a limit of twice `--timeout`; on timeout the whole group is ended.
  Re-running into the same `--out` is safe: each trial's old `trial.json` is removed first, and a result counts only
  if it carries this run's attempt id and its process exited 0.

**Outputs.** `outputs/EXAMPLE_tune/tune.json` (every trial, kept and rejected changes, `final_settings`,
`first_healthy_trial`, `final_healthy`) and `trialN/` per trial.

**Fails when** `final_healthy` is false. The last trial's flags say why; `shelf advise` re-reads any trial.

**Limits.** There is no reference trajectory: a cheaper setting that changes the motion without failing a metric or
the check is kept, so look at the frames (visual-review). The numbers are heuristics for one to a few hundred
worlds. It tunes at the example's own world count; throughput at thousands of worlds is a separate measurement.

**Many scenes.** One `shelf autotune` per GPU job, a private warm `WARP_CACHE_PATH` per job, all from a frozen copy
of the repository (never edit a tree a queued job imports). Check each job's terminal state before reading its
`tune.json`: a submitted job is not a finished one.
