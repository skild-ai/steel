# Check an example

> **Does:** runs an example's own test headless with a preset and records the result in `presets.toml`.
> **Use in:** tune-a-scene, add-an-example, validate-a-solver-change.
> **Passes when:** `shelf check` reports `works` for every preset you changed (status written to `presets.toml`).

**Inputs.** An example and the preset tables to check.
**Tool.** `shelf check` (`examples/common/cli.py`).

```bash
cd examples
uv run python -m shelf check EXAMPLE                     # tables marked works/unstable
uv run python -m shelf check EXAMPLE --every             # every table that has settings
uv run python -m shelf check EXAMPLE --presets FILE.toml --every   # a preset file saved elsewhere (tune, a trial)
uv run python -m shelf check --all --dry-run             # whole collection, results not written
```

`check` runs the example with `--viewer null --test` and the `[check]` arguments, fails it when a Steel solver
dropped contacts or rows (`check_constraint_capacity()` after the run), writes `status`, `checked` and any failure
`note` into `presets.toml` (unless `--dry-run`), and regenerates `PRESETS.md`.

**To ship settings,** copy them into the example's `[steel]` table, then run `check`. Only `check` sets
`status = "works"`; never write it by hand. Check the settings you ship, not cheaper test settings.

**Fails when** the status is `unstable` (the note says why) or the run timed out.
