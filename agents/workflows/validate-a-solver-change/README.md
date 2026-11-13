# Validate a solver change

> **Goal:** a verdict on whether a Newton or Steel change (a version bump, a solver patch, a changed default) breaks or slows any example.
> **Use when:** before updating the pinned Newton revision, before merging a solver change, or when asked which presets look wrong across the collection.
> **Done when:** both revisions were rendered and checked on the same examples, every grade change is explained, and speed changes on the flagged scenes are measured.

**Inputs.** A base and a candidate revision (two checkouts of this repository, or one checkout with two Newton pins);
GPUs. Run each revision from its own frozen checkout; never edit a tree a job imports from.

## Steps

1. In the base checkout: [render-and-grade](../../steps/render-and-grade/README.md) for every example, then
   [visual-review](../../steps/visual-review/README.md). Keep the run dir.
2. In the candidate checkout: [render-and-grade](../../steps/render-and-grade/README.md) with
   `collect.py --previous BASE_RUN_DIR`, then [visual-review](../../steps/visual-review/README.md). `results.json`
   lists regressed, improved, new and removed scenes.
3. [check-example](../../steps/check-example/README.md) `--all --dry-run` on the candidate.
4. For each regressed scene → [debug-instability](../debug-instability/README.md) on the candidate, with the base
   render as the reference. For each scene a new setting could now speed up → [tune-a-scene](../tune-a-scene/README.md)
   (later, as its own change).
5. [benchmark-throughput](../../steps/benchmark-throughput/README.md) on a few representative scenes and every
   flagged one, base against candidate, same GPU, same world counts.

## Decision

- Any scene going from ok to unstable or failed without an explained, accepted cause blocks the change.
- A slowdown beyond noise on representative scenes needs a reason in the change's description.

## Deliverables

Both galleries, the grade changes with a one-line cause each, the check results, the throughput table, and the
verdict.
