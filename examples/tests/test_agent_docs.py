# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""The agent workflows and steps (../agents): summaries, links and generated adapters stay consistent."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "agents"
if not AGENTS.exists():
    pytest.skip("agents/ lives at the repository root (repository checkout only)", allow_module_level=True)

DOCS = [ROOT / "AGENTS.md", *sorted(AGENTS.rglob("*.md")), *sorted((ROOT / ".claude").rglob("SKILL.md"))]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc.read_text()):
        if "://" not in target:
            assert (doc.parent / target).exists(), f"{doc.relative_to(ROOT)} links to missing {target}"


@pytest.mark.parametrize("readme", sorted(AGENTS.glob("*/*/README.md")), ids=lambda p: p.parent.name)
def test_every_readme_opens_with_its_summary(readme):
    keys = (
        ("Goal", "Use when", "Done when")
        if readme.parent.parent.name == "workflows"
        else ("Does", "Use in", "Passes when")
    )
    lines = readme.read_text().splitlines()
    assert re.fullmatch(r"[a-z]+(-[a-z]+)+", readme.parent.name), "names are verb-noun, lowercase-hyphen"
    assert [ln.split(":**")[0] for ln in lines[2:5]] == [f"> **{k}" for k in keys]


def test_steps_never_call_workflows():
    for readme in AGENTS.glob("steps/*/README.md"):
        assert "workflows/" not in readme.read_text(), f"{readme.parent.name} links a workflow"


def test_adapters_are_generated_and_current():
    subprocess.run([sys.executable, str(AGENTS / "make_adapters.py"), "--check"], check=True)


def test_use_in_lists_the_workflows_that_link_each_step():
    links = {
        w.parent.name: set(re.findall(r"\]\(\.\./\.\./steps/([a-z0-9-]+)/README\.md\)", w.read_text()))
        for w in AGENTS.glob("workflows/*/README.md")
    }
    for readme in AGENTS.glob("steps/*/README.md"):
        use_in = re.search(r"^> \*\*Use in:\*\* (.+)$", readme.read_text(), re.M).group(1)
        declared = set(re.findall(r"[a-z]+(?:-[a-z]+)+", use_in)) & set(links)
        assert declared == {w for w, steps in links.items() if readme.parent.name in steps}, readme.parent.name
