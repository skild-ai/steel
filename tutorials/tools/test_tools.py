"""CPU-only tests of the tutorial tools: timing-group routing and cost-table provenance.

usage: python -m unittest discover -s tutorials/tools -p 'test_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_all  # noqa: E402
from timing_groups import runners  # noqa: E402

TUTORIALS = Path(__file__).resolve().parent.parent


class TimingGroups(unittest.TestCase):
    def test_every_timing_group_has_a_runner(self):
        owner = runners()
        for path in TUTORIALS.glob("*.toml"):
            if path.name in ("index.toml", "ruff.toml"):
                continue
            group = tomllib.loads(path.read_text()).get("timing", {}).get("group")
            if group:
                self.assertIn(group, owner, path.name)

    def test_groups_are_routed_to_their_runner(self):
        calls = []

        def fake_run(cmd, env, check, timeout):
            calls.append((Path(cmd[1]).name, cmd[3:], env.get("CUDA_VISIBLE_DEVICES")))
            rows = [{"group": g, "label": "x", "worlds": 1} for g in cmd[3:]]
            Path(cmd[2]).write_text(json.dumps({"gpu": "GPU", "rows": rows}))

        selected = [
            {"id": "a", "timing": {"group": "patch"}},
            {"id": "b", "timing": {"group": "row_schedule"}},
            {"id": "c", "timing": {"group": "patch"}},
        ]
        with mock.patch.object(record_all.subprocess, "run", fake_run):
            result = record_all.measure_timing(selected, {"NEWTON_COMMIT": "abc"}, "cuda:3", 10)
        self.assertEqual(
            sorted(calls),
            [("timing.py", ["patch"], "3"), ("timing_rows.py", ["row_schedule"], "3")],
        )
        self.assertEqual(result["a"]["rows"], result["c"]["rows"])
        self.assertEqual(result["b"]["runner"], "timing_rows.py")
        self.assertEqual(result["a"]["newton_commit"], "abc")


class Provenance(unittest.TestCase):
    def record(self, script, newton, previous):
        """Run record_all.main on a one-tutorial folder whose fake recorder writes script/newton."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recordings").mkdir()
            (root / "index.toml").write_text('[[group]]\nname = "G"\ntutorials = ["t"]\n')
            (root / "01_t.toml").write_text('id = "t"\n')
            out = root / "recordings" / "t.json"
            out.write_text(json.dumps(previous))

            def fake_run(cmd, cwd, env, check, timeout):
                Path(cmd[3]).write_text(
                    json.dumps({"script_sha256": script, "newton_commit": newton})
                )

            with (
                mock.patch.object(record_all, "TUTORIALS", root),
                mock.patch.object(record_all.subprocess, "run", fake_run),
                mock.patch.object(sys, "argv", ["record_all.py"]),
                mock.patch.dict(record_all.os.environ, {"NEWTON_COMMIT": newton}),
            ):
                record_all.main()
            return json.loads(out.read_text())

    OLD = {
        "script_sha256": "s1",
        "newton_commit": "n1",
        "timing": {"gpu": "OLD", "rows": [{"ms": 99}]},
    }

    def test_unchanged_recording_keeps_its_cost_table(self):
        self.assertEqual(self.record("s1", "n1", self.OLD)["timing"], self.OLD["timing"])

    def test_changed_script_drops_the_cost_table(self):
        self.assertNotIn("timing", self.record("s2", "n1", self.OLD))

    def test_changed_newton_drops_the_cost_table(self):
        self.assertNotIn("timing", self.record("s1", "n2", self.OLD))


if __name__ == "__main__":
    unittest.main()
