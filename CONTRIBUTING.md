# Contributing

This repository holds Steel's documentation and its examples. The solver itself lives in
[Newton](https://github.com/newton-physics/newton); see [Solver changes](#solver-changes).

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

CI runs on GitHub-hosted runners with no GPU. It checks that `uv.lock` matches `pyproject.toml`, runs ruff for
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
  write carries the header of its copyright holder, e.g. `Copyright (c) 2026 Skild AI` for Skild AI work.
- Add only files whose license allows redistribution, and say where each came from. A new bundled file needs
  an inventory entry with its path, SHA-256, license and notice, and the complete notice text under
  `examples/licenses/`. Describe it in [THIRD_PARTY.md](examples/THIRD_PARTY.md).
- Prefer downloading models at run time, pinned to a revision, over committing them. A new or changed download
  call is caught by the audit; after reviewing its license, record it with
  `python scripts/check_distribution_licenses.py --update-call-sites` and add its notice under
  `examples/licenses/runtime-assets/`.
- Evidence in the inventory and THIRD_PARTY.md must be something any reader can open: public commits, files
  and URLs.

## Pull requests

Branch from `main` and open a pull request into `main`. Say what changed and include the checks you ran. Keep
one change per pull request; a preset retune and an unrelated example fix are two.

## Solver changes

Changes to Steel itself go to Newton as a pull request against the Newton repository, not here. When a change
an example needs is merged there, update the pin in `examples/pyproject.toml`, run `uv lock`, and run the GPU
check on the whole shelf.
