import copy
import importlib.util
import json
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate_rust_capabilities.py"
SPEC = importlib.util.spec_from_file_location("validate_rust_capabilities", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CapabilityManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(MODULE.MANIFEST.read_text(encoding="utf-8"))

    def test_repository_manifest_is_valid(self):
        self.assertEqual(MODULE.validate(self.data), [])

    def test_supported_claim_without_complete_evidence_fails(self):
        data = copy.deepcopy(self.data)
        data["capabilities"][1]["status"] = "supported"
        self.assertTrue(any("missing" in error for error in MODULE.validate(data)))

    def test_stale_evidence_path_fails(self):
        data = copy.deepcopy(self.data)
        item = data["capabilities"][0]
        item["status"] = "supported"
        item["evidence"] = {key: ["missing/evidence.file"] for key in data["policy"]["supported_requires"]}
        self.assertTrue(any("does not exist" in error for error in MODULE.validate(data)))

    def test_duplicate_capability_ids_fail(self):
        data = copy.deepcopy(self.data)
        data["capabilities"].append(copy.deepcopy(data["capabilities"][0]))
        self.assertTrue(any("duplicate capability id" in error for error in MODULE.validate(data)))

    def test_self_hosting_requires_reproducibility_gate(self):
        data = copy.deepcopy(self.data)
        data["policy"]["self_hosting_requires_stage_0_to_3_reproducibility"] = False
        self.assertTrue(any("self-hosting must require" in error for error in MODULE.validate(data)))


if __name__ == "__main__":
    unittest.main()
