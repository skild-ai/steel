# Benchmark throughput

> **Does:** measures simulation and application throughput and memory for two configurations (two backends, or two settings) under one timing contract, and tunes Steel in a fixed order.
> **Use in:** port-from-physx, validate-a-solver-change, tune-a-scene (at the shipping world count).
> **Passes when:** both configurations ran the same workload under the same contract, every plotted point passed the behavior gates and showed no overflow or dropped work, and invalid points are marked.

**Inputs.** Two configurations that already pass the same behavior gates; the workload; GPUs. **Tools.**
`shelf bench EXAMPLE --world-count N` for the examples (throughput, step time, GPU memory); for an application, its
own step loop with the timing contract below.

```bash
cd examples
uv run python -m shelf bench EXAMPLE --solver steel --presets "$PWD/outputs/EXAMPLE_ref/presets.toml" --world-count 1024
uv run python -m shelf bench EXAMPLE --solver steel --presets "$PWD/outputs/EXAMPLE_cand/presets.toml" --world-count 1024
uv run python -m shelf bench EXAMPLE --solver mujoco --world-count 1024          # another solver's preset, for reference
```

Always name the solver and the exact preset file: without `--solver`, `bench` times the file's default preset, which
may be another solver. A trial's `presets.toml` sets `row_watermark = true`, a diagnostic counter that costs time: for timing, copy it,
delete that key, and bench the copy.

## Three workload levels


Keep each as its own named workload with its own raw results. A simulator microbenchmark never replaces an
end-to-end number.

1. **Environment stepping without cameras.** Start from the same recorded initial state and replay the same actions
   (or the same policy) on both configurations. Disable cameras, rendering, visual-only geometry, planners, judges,
   encoders and dataset writes. Prepare actions before timing and time only repeated environment steps. This is
   environment-loop throughput: it still includes action application, physics substeps, state readback, resets
   inside the environment and non-camera observations.
2. **Application loop without cameras** (when attribution matters): the real planner or policy, judge, episode
   lifecycle and resets, with camera modalities and storage off.
3. **Full application:** planner or policy, observations, judge, cameras, encoder, episode buffering, resets.
   Disable dataset writes unless storage is what you measure, and record the storage mode either way.

Start with trajectory disturbances and observation noise off. Run the task's real randomization and disturbances as
a separately labeled curve after nominal parity passes: independently sampled disturbances are not a paired
comparison.

## Timing contract

One timing implementation for both configurations within a workload:

```text
construct environment and load data
reset
run warm-up steps            (includes CUDA graph capture when graphs are on)
synchronize the device
reset peak-memory counters
start = monotonic clock
run a fixed number of control steps
synchronize the device
elapsed = monotonic clock - start

control steps/s = worlds * measured control steps / elapsed
physics steps/s (inside the environment loop) = control steps/s * decimation
```

The second number counts physics advances inside the full environment loop; it is not physics-only throughput.
Label it that way in tables and plots. Use the same step count on both configurations, large enough to average out noise,
and record control and physics time steps, decimation, warm-up and measured counts, and whether resets happened
inside the window. Keep scene construction, asset loading, kernel compilation, graph capture and teardown out of
the timed window and record startup separately. Warm the kernel cache serially before a parallel campaign, or give
each concurrent job its own cache, and record the cache state.

## Memory

Record total device memory in use (free/total from the driver), framework peak allocated and reserved after
resetting its counters, host resident memory, and the GPU model. Device memory covers every allocator (Warp,
PhysX, renderer); an endpoint sample is not a time-series peak, so say which you plot. Cameras can dominate device
memory: never compare a camera-free point with a rendered one without labeling both.

## Tuning Steel in stages

Change one family of settings at a time, with the task seeds, workload, revision, time step and all other options
fixed.

1. **Behavior contract first.** From the deterministic traces, choose task gates: success and termination reasons;
   trajectory, action and reward deltas; penetration and friction behavior; joint-limit and velocity-limit
   violations; object drift, grasp retention or ball speed; finite state and actions. Throughput never makes up for
   a failed gate.
2. **Capacity.** Start with generous row capacities and `row_watermark=True`. Run the complete contact-rich behavior
   over several seeds and record each family's high-water mark (articulation, free-body, propagation rows) and its
   overflow counters, plus the contact buffer's high-water mark. The families are independent: raising one does not
   repair an overflow in another. Size each from its per-world tail with headroom (the steel-tuning guide uses
   1.3x of a run that did not overflow; see size-capacities), then A/B generous against selected capacities. Smaller capacities often
   save memory without changing throughput. Repeat near a boundary: the busiest world's demand varies between runs
   even when the average does not.
3. **Iterations and substeps.** Step iterations down by powers of two from a conservative value, then try fewer
   substeps, with capacity and graph state fixed and the behavior gates beside every number. Fewer iterations can
   make failing episodes end sooner and look faster: reject those points. Zero overflow with a behavior regression
   points to convergence, not capacity. Any other change (warm start, regularization, velocity iterations) is its
   own one-factor experiment.
4. **CUDA graphs.** Compare graphs off and on after capacity and iterations are chosen, with diagnostics such as
   watermarks off. Confirm capture in the log. Graphs mostly cut launch overhead, so the gain can be large in
   camera-free stepping and small in the full loop; report both.
5. **World count.** Sweep both configurations over powers of two (or the task's usual counts) until memory, capacity or
   construction fails. Scan every point's full log for overflow, dropped work, out-of-memory, non-finite values and
   tracebacks; a process that exits 0 with a plausible number can still have dropped contacts. Mark invalid points
   in the raw data and leave them out of plotted lines.

## Repeated contact-rich outcomes

Fixed seeds and actions do not guarantee bitwise-identical GPU rollouts. Separate repeatable inputs, deterministic
contact ordering and whole-solver determinism, and record the flags you used. When a small difference would change
a conclusion, repeat the matched full workload within each configuration, keeping world layout and batch size (a
one-world rerun is a different experiment), and report per-item transitions next to aggregate counts. A single-run
difference is an observation, not a regression; repeats that reverse it show variability, not equivalence.

## Reading the gap between workloads

At shared world counts, report the throughput ratio for environment stepping and for the full application, the
full-application/stepping ratio within each configuration, and the memory change between workloads. The gap shows how
much solver speed survives the surrounding pipeline; call it a rendering cost only if an intermediate camera-free
application loop holds everything else constant.

## Result metadata

Keep raw JSON and full logs. Each result names: workload and timed scope; backend, solver settings and source
commits; task, asset or motion, seed, randomization and disturbance state; world count, step counts, time step,
substeps and decimation; elapsed time and throughput; memory; capacity telemetry and any failure reason; host, GPU
and completion state. The report gives the exact launch commands, throughput and memory curves, behavior gates,
invalid points and a recommendation limited to the tested workload.
