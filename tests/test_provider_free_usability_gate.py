import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "provider_free_usability_gate.sh"


class ProviderFreeUsabilityGateTests(unittest.TestCase):
    def test_existing_output_directory_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "existing"
            output.mkdir()
            sentinel = output / "operator-owned.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            before = {
                path.name: (path.stat().st_mode, path.read_bytes())
                for path in output.iterdir()
            }

            result = subprocess.run(
                [os.fspath(GATE), os.fspath(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("refusing to overwrite existing output directory", result.stderr)
            after = {
                path.name: (path.stat().st_mode, path.read_bytes())
                for path in output.iterdir()
            }
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
