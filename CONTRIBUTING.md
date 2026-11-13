# Contributing

This repository holds Steel's documentation site, its tutorials and its examples. The solver itself lives
in [Newton](https://github.com/newton-physics/newton); see [Solver changes](#solver-changes). For the site
and the tutorials, see [Documentation site and tutorials](#documentation-site-and-tutorials).

## Setup

The examples are a [uv](https://docs.astral.sh/uv/) project in `examples/`. Run every command below from there.

```bash
git clone https://github.com/skild-ai/steel.git
cd steel/examples
uv sync
uv run python -m shelf list
```

`uv sync` installs the Newton revision pinned in `pyproject.toml`, from
[skild-ai/newton](https://github.com/skild-ai/newton), a Newton development fork, until the changes the
examples need are in a Newton release. A CUDA GPU is recommended; most examples also run on CPU.
If `VIRTUAL_ENV` points at another project, unset it first.

## Running an example

```bash
uv run python -m shelf play                                  # the example browser
uv run python -m shelf play basic_pendulum                   # one example, default preset
uv run python -m shelf play basic_pendulum --solver kamino   # another solver's preset
uv run python -m shelf tune basic_pendulum                   # change solver settings live
uv run python -m shelf bench robot_cartpole --world-count 4096
```

The [examples README](examples/README.md) covers every mode and option.

## Adding or changing a preset

Each example folder has an `example.py` and a `presets.toml` with one table per solver. The rules are in
[examples/AGENTS.md](examples/AGENTS.md#presets); the short version:

- Every table stands alone. Repeat a value in each solver that needs it.
- Never write `status = "works"` by hand. Only `shelf check` sets it.
- A solver that cannot simulate the scene is `unsupported` and gives a `reason`.
- To try values, use `shelf tune` and **Save as**, then copy the table into `presets.toml`.

Then check the example on a GPU:

```bash
uv run python -m shelf check <example> --every
```

This writes the result and a `checked` stamp (date, Newton commit, GPU) into `presets.toml` and regenerates
`PRESETS.md`. Commit both. Never edit `PRESETS.md` by hand.

## Checks

CI runs on GitHub-hosted runners with no GPU. For the site it builds `site/`, checks every internal link
and anchor, loads every page in headless Chromium to check that each figure renders and no table is clipped,
and runs `tutorials/tools/check.py`. Each push to `main` deploys the site to
[skild-ai.github.io/steel](https://skild-ai.github.io/steel/). For the examples it checks that `uv.lock` matches `pyproject.toml`, runs ruff for
syntax errors and undefined names, runs the license audit on the built wheel and sdist, and lists the examples.
You can run the same checks locally:

```bash
uv lock --check
uvx ruff check --select E9,F63,F7,F82 .
uv run python -m unittest discover -s tests -p 'test_distribution_licenses.py'
uv build -o dist && uv run python scripts/check_distribution_licenses.py dist
```

Physics is checked by hand. For a change to a preset, an example, `common/` or the Newton pin, run the GPU
check in [the examples README](examples/README.md#checking-on-a-gpu) and paste its result table into the pull
request.

## Assets and attribution

Everything that ships in the wheel or sdist is declared in
[`third_party_inventory.json`](examples/third_party_inventory.json). The audit fails on undeclared scene data,
model files, binary files and asset downloads.

- Keep existing copyright and SPDX headers. An example ported from Newton keeps Newton's header; one ported from
  elsewhere names its direct source in a header line and in the `source` of its `presets.toml`. New code you
  write carries the header in [Contribution terms](#contribution-terms).
- Add only files whose license allows redistribution, and say where each came from. A new bundled file needs
  an inventory entry with its path, SHA-256, license and notice, and the complete notice text under
  `examples/licenses/`. Describe it in [THIRD_PARTY.md](examples/THIRD_PARTY.md).
- Prefer downloading models at run time, pinned to a revision, over committing them. A new or changed download
  call is caught by the audit; after reviewing its license, record it with
  `python scripts/check_distribution_licenses.py --update-call-sites` and add its notice under
  `examples/licenses/runtime-assets/`.
- Evidence in the inventory and THIRD_PARTY.md must be something any reader can open: public commits, files
  and URLs.

## Contribution terms

Contributions are accepted under the [Apache License, Version 2.0](LICENSE), the license of the project: what
you submit is licensed inbound under the same terms it is distributed outbound, as Apache-2.0 section 5 sets
out. There is no CLA and no sign-off. You keep the copyright in your contribution. A new file you write starts
with this header (in the comment style of its language):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
```

The project's authors are listed in [AUTHORS](AUTHORS).

## Documentation site and tutorials

The site in `site/` is an [Astro](https://astro.build) + [Starlight](https://starlight.astro.build)
project. Its pages are Markdown or MDX, its interactive demos are components, and its tutorial pages are
generated from `tutorials/`. Node 22.23.4 or newer:

```bash
cd site
npm ci
npm run dev       # http://localhost:4321, reloads on edits
npm run build     # the static site in site/dist, then a check of every internal link and anchor
```

### A docs page

Add a `.md` or `.mdx` file under `site/src/content/docs/<group>/` with a `title`, a one-sentence
`description` and `sidebar.order` in its frontmatter; the folder (`start`, `concepts`, `guides`,
`reference`) is its sidebar group. Link to other pages with root-relative paths such as
`/concepts/friction/#friction-patches-and-anchors`; the build fails on a broken link or anchor. Math is TeX
between `$...$` or `$$...$$`. In MDX, the components in `site/src/components/` are available: `Note`,
`CardGrid` with `InfoCard`, `Figure` for a static numbered figure, `Viz` for an interactive one, and
`Term` for a term with a concept bubble (`<Term name="warp">warp</Term>`; definitions in
`site/src/viz/terms.js`).

### An interactive figure

An interactive figure is one script in `site/src/viz/<name>.js` that calls
`SteelViz.register("<name>", (root, lib) => { ... })` and draws only inside `root`. `lib` (from
`site/src/viz/_lib.js`) provides `frame`, `lineChart`, `stepper`, `tabs`, `stage3d` (a three.js orbit
stage), `css` for token colors, `onTheme` to redraw when the theme changes, and `data`, the measured series
in `series.json`. Place it with `<Viz name="<name>" n={14} title="Bold first sentence.">Caption.</Viz>`.
Larger demos with their own controls can be Astro components in `site/src/components/demos/` instead,
importing `three` and `site/src/lib/stage3d.ts`. Say what a figure simplifies.

### A tutorial

A tutorial is three files in `tutorials/`: the script `NN_name.py`, its page text `NN_name.toml`, and its
recording `recordings/<id>.json`; its id goes into a group in `tutorials/index.toml`. Record on a GPU with
`python tutorials/tools/record_all.py --only <id>` and check with `python tutorials/tools/check.py --poses`.
The site shows the committed script next to the replay, so a script changed after recording fails the
check until it is recorded again. The digest skips the file's SPDX header, so adding
or changing that header does not. [tutorials/README.md](tutorials/README.md) lists every field.

### An example's page

The site builds a gallery card and a page at `/examples/<name>/` for every folder in `examples/shelf/`, from its
`presets.toml` (`[gallery]` `short` and `title`), `example.py` and README. Until it has media, both show a
placeholder. Record the Steel preset with `shelf play <name>` and **Record** (sim time), then run
`python scripts/example_media.py <name> <recording.mp4> --preset steel` from `site/`.

### Style

- Colors and fonts come only from the Steel style tokens in `site/src/styles/steel-tokens.css`, which is
  kept verbatim from the shared Steel style; update it by replacing the whole file. Components use
  `var(--token)` in CSS and `css('--token')` in scripts, never a literal color.
- Green (`--hot`, `--hot-fill`, `--series-2`) marks the one thing a figure or section is about; comparisons
  use `--series-1`, `--series-3` and `--series-4`.
- Write plainly. State measured numbers with the scene, commit and GPU they come from, and keep claims to
  what the code at the documented commit does.
- Check new pages at phone width (390 px): no sideways scrolling of the page, controls large enough to tap.

### What is public

Everything in this repository is public. Link only to public repositories, commits and pages; do not add
internal hostnames, cluster or job-scheduler details, absolute paths from your machine, or private notes.
Recordings carry the Newton commit and GPU name they were made with, nothing else about the machine.

## Pull requests

Branch from `main` and open a pull request into `main`. Say what changed and include the checks you ran. Keep
one change per pull request; a preset retune and an unrelated example fix are two.

## Solver changes

Changes to Steel itself go to Newton as a pull request against the Newton repository, not here. When a change
an example needs is merged there, update the pin in `examples/pyproject.toml`, run `uv lock`, and run the GPU
check on the whole shelf.
