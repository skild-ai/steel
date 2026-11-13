# Add an example

> **Goal:** a new example in `examples/shelf/` with a checked Steel preset and a preview, following the collection's conventions.
> **Use when:** adding a scene to the collection, or giving an existing example a Steel preset it does not have yet.
> **Done when:** `shelf check EXAMPLE --every` reports `works` for every filled table, the example renders and looks right, and `PRESETS.md` is regenerated.

**Inputs.** The scene (a Newton example script, or a new one). Read [examples/AGENTS.md](../../../examples/AGENTS.md)
for the `presets.toml` format, the conversion steps and the rules (never write `status = "works"` by hand; never
loosen an example's test).

## Steps

1. Put the example under `examples/shelf/<NN_section>/<example>/` as `example.py` plus `presets.toml`, converted as
   `examples/AGENTS.md` describes (settings out of code into presets, `self.preset = presets.load(__file__)`). Fill
   the tables of solvers the code already supports with the values it used; leave others `untested`.
2. For Steel: [run-trial](../../steps/run-trial/README.md) at the new `[steel]` table, then
   [autotune-settings](../../steps/autotune-settings/README.md) from `defaults` if the table is new. Copy the result
   into `[steel]` with a comment per setting saying why.
3. [check-example](../../steps/check-example/README.md) with `--every`.
4. [render-and-grade](../../steps/render-and-grade/README.md) for this example and
   [visual-review](../../steps/visual-review/README.md).
5. Add the `[gallery]` table and render the preview: `uv run python -m shelf thumbs EXAMPLE --force`.

## Deliverables

The example folder, its checked `presets.toml`, the regenerated `PRESETS.md`, and the gallery for it.
