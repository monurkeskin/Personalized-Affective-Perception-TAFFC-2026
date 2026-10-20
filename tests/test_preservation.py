"""Maintenance must preserve the study inputs and original numerical source."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "verification/protected-sources.json"


class PreservationTests(unittest.TestCase):
    def test_original_numerical_source_data_and_legacy_constraints_are_unchanged(self):
        reference = json.loads(MANIFEST.read_text())
        for name, digest in reference["sha256"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_facechannel_model_methods_are_unchanged(self):
        reference = json.loads(MANIFEST.read_text())
        path = ROOT / reference["facechannel_class"]["path"]
        model = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "FaceChannelV1")
        for node in model.body:
            if isinstance(node, ast.FunctionDef) and node.name != "__init__":
                with self.subTest(method=node.name):
                    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                    self.assertEqual(digest, reference["facechannel_class"]["methods"][node.name])


if __name__ == "__main__":
    unittest.main()
