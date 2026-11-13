# Tune one knob

> **Does:** changes a single setting, holding everything else fixed, and keeps it only if every gate still passes.
> **Use in:** debug-instability, port-from-physx, tune-a-scene (by hand, past what autotune tries).
> **Passes when:** the changed setting passes the same behavior gates as the reference and, for a speed change, is faster beyond noise.

**Inputs.** Reference settings that pass your gates; one setting to change; the gates (the example's check, the
advisor's health, task metrics).
**Tools.** `shelf trial --settings`, `shelf play --set section.key=value` (look at it), `shelf bench` (speed),
`shelf check --presets FILE`.

```bash
cd examples
# the reference and the candidate, each a full trial; each trial writes the preset it ran (one [steel] table)
uv run python -m shelf trial EXAMPLE --settings "$(cat ref.json)" --out outputs/EXAMPLE_ref
uv run python -m shelf trial EXAMPLE --settings "$(cat cand.json)" --out outputs/EXAMPLE_cand
# look at and time exactly those presets: always name the solver and the preset file (absolute path)
uv run python -m shelf play EXAMPLE --solver steel --presets "$PWD/outputs/EXAMPLE_cand/presets.toml"
uv run python -m shelf bench EXAMPLE --solver steel --presets "$PWD/outputs/EXAMPLE_ref/presets.toml" --world-count 1024
uv run python -m shelf bench EXAMPLE --solver steel --presets "$PWD/outputs/EXAMPLE_cand/presets.toml" --world-count 1024
```

`ref.json` and `cand.json` are settings dicts (`{"sim": {...}, "solver": {...}}`) that differ in one key. Without
`--solver`, `play` and `bench` use the file's default preset, which may be another solver. A trial's `presets.toml` sets `row_watermark = true`, a diagnostic counter that costs time: for timing, copy it,
delete that key, and bench the copy.

1. Write down the reference and the one change. Keep seeds, world count, time step and every other option fixed.
2. Run the whole task with the change and compare against the reference: the same gates, the same length.
3. For speed, compare at the same world count on the same GPU, and repeat when the ratio is under about 1.05x.
4. Keep the change only if every gate passes. A faster setting that fails a gate is a diagnostic, not a candidate.

**Never** turn off physics the scene relies on (joint limits, velocity limits, friction anchors that hold a grasp,
torsion, collision, self-collision) to win time; report such runs as diagnostics. The tuning guide
([guide](../../../examples/docs/steel_tuning/guide.md), sections 4 and 6) lists what each setting costs and fixes.

**Outputs.** One row per change: setting, gates, frame time against the reference, kept or rejected.
