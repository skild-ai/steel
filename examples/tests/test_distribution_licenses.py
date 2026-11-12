"""Regression checks for the license audit: call-site signatures and source checks."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[1] / 'scripts/check_distribution_licenses.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def signature(source):
    return audit.call_site_signatures({'example.py': audit.asset_calls(source)})


class AssetCallSignatures(unittest.TestCase):
    def test_unrelated_lines_do_not_change_signature(self):
        source = "newton.utils.download_asset('franka_emika_panda')"
        self.assertEqual(signature(source), signature('\n# unrelated edit\nx = 1\n\n' + source))

    def test_call_order_does_not_change_signature(self):
        a = "newton.utils.download_asset('a')\n"
        b = "newton.utils.download_asset('b')\n"
        self.assertEqual(signature(a + b), signature(b + a))

    def test_changed_asset_is_detected(self):
        self.assertNotEqual(signature("newton.utils.download_asset('a')"),
                            signature("newton.utils.download_asset('b')"))

    def test_changed_revision_is_detected(self):
        self.assertNotEqual(signature("newton.utils.download_asset('a', ref='old')"),
                            signature("newton.utils.download_asset('a', ref='new')"))

    def test_additional_call_is_detected(self):
        call = "newton.utils.download_asset('a')\n"
        self.assertNotEqual(signature(call), signature(call + call))

    def test_direct_imported_helper_is_tracked(self):
        self.assertNotEqual(signature("get_kamino_basics_asset('boxes_fourbar.usda')"),
                            signature("get_kamino_basics_asset('cartpole.usda')"))

    def test_directory_asset_filename_is_tracked(self):
        self.assertNotEqual(signature("os.path.join(warp.examples.get_asset_directory(), 'square_cloth.usd')"),
                            signature("os.path.join(warp.examples.get_asset_directory(), 'other.usd')"))



class SourceChecks(unittest.TestCase):
    def test_tree_passes(self):
        audit.check_sources()

    def test_changed_scene_hash_fails(self):
        inventory = copy.deepcopy(audit.INVENTORY)
        inventory['bundled_scenes'][0]['sha256'] = '0' * 64
        with mock.patch.object(audit, 'INVENTORY', inventory):
            self.assertRaises(audit.AuditError, audit.check_sources)

    def test_require_is_not_an_assert(self):
        self.assertRaises(audit.AuditError, audit.require, False, 'message')


if __name__ == '__main__':
    unittest.main()
