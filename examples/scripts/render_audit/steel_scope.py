# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Which examples fall outside a Steel render audit, and why the rest have no Steel preset yet.

plan.py reads this for every example it cannot render:
  excluded  no rigid or articulated physics for Steel to do; left out of the gallery, kept in results.json
  preset    Steel should run it and a preset is feasible today ("no Steel preset yet")
  feature   needs a Steel feature that does not exist yet ("needs Steel feature: ...")
An example not listed keeps plan.py's generic skip reason. Re-triage after a Newton update changes what Steel supports.
"""

# Every shipped example has a Steel or Steel-coupled preset today, so all three tables are empty.
EXCLUDED: dict[str, str] = {}

PRESET: dict[str, str] = {}

FEATURE: dict[str, str] = {}


def classify(scene: str) -> tuple[str, str] | None:
    """(class, text) for a triaged scene: ("excluded", reason), ("preset", plan) or ("feature", missing feature)."""
    for kind, table in (("excluded", EXCLUDED), ("preset", PRESET), ("feature", FEATURE)):
        if scene in table:
            return kind, table[scene]
    return None
