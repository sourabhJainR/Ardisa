import importlib.util
import pathlib
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate_self_hosting_evidence.py"
SPEC = importlib.util.spec_from_file_location("self_hosting_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelfHostingEvidenceTests(unittest.TestCase):
    def test_host_replay_is_not_promoted_to_self_hosting(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / "self-hosting.txt"
            evidence.write_text(
                "probe_stage2_stage3_byte_identical=true\n"
                "probe_replay_mode=host_compiler_deterministic_rebuild\n"
                "compiler_self_hosting=false\n",
                encoding="utf-8",
            )
            with patch.object(MODULE, "EVIDENCE", evidence):
                self.assertEqual(MODULE.validate(), [])

    def test_incomplete_self_hosting_claim_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / "self-hosting.txt"
            evidence.write_text(
                "compiler_self_hosting=true\n"
                "self_hosting_ready=true\n"
                "stage2_stage3_byte_identical=true\n"
                "native_compiler_only=true\n"
                "host_fallback_detected=false\n",
                encoding="utf-8",
            )
            with patch.object(MODULE, "EVIDENCE", evidence):
                errors = MODULE.validate()
            self.assertTrue(any("missing evidence keys" in error for error in errors), errors)

    def test_missing_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / "missing.txt"
            with patch.object(MODULE, "EVIDENCE", evidence):
                errors = MODULE.validate()
            self.assertTrue(any("missing self-hosting evidence" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
