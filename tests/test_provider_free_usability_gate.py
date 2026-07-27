import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "provider_free_usability_gate.sh"


class ProviderFreeUsabilityGateTests(unittest.TestCase):
    def test_new_output_directory_is_private_despite_permissive_caller_umask(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "private-output"

            result = subprocess.run(
                [
                    "/bin/sh",
                    "-c",
                    'umask 000; exec "$1" "$2"',
                    "provider-free-usability-test",
                    os.fspath(GATE),
                    os.fspath(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_dir())
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            for artifact in output.iterdir():
                if artifact.is_file():
                    self.assertEqual(
                        artifact.stat().st_mode & 0o777,
                        0o600,
                        artifact.name,
                    )

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

    def test_dangling_symlink_output_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing_target = root / "operator-selected-target"
            output = root / "output-alias"
            output.symlink_to(missing_target, target_is_directory=True)
            before_link = os.readlink(output)

            result = subprocess.run(
                [os.fspath(GATE), os.fspath(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("refusing to overwrite existing output directory", result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertEqual(os.readlink(output), before_link)
            self.assertFalse(missing_target.exists())

    def test_symlinked_output_parent_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target_parent = root / "operator-selected-parent"
            target_parent.mkdir()
            parent_alias = root / "parent-alias"
            parent_alias.symlink_to(target_parent, target_is_directory=True)
            output = parent_alias / "new-output"

            result = subprocess.run(
                [os.fspath(GATE), os.fspath(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("refusing symlinked output parent", result.stderr)
            self.assertTrue(parent_alias.is_symlink())
            self.assertFalse(output.exists())
            self.assertEqual(list(target_parent.iterdir()), [])

    def test_missing_output_parent_is_rejected_without_creating_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing_parent = root / "must-not-be-created"
            output = missing_parent / "new-output"

            result = subprocess.run(
                [os.fspath(GATE), os.fspath(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn(
                "refusing missing or non-directory output parent",
                result.stderr,
            )
            self.assertFalse(missing_parent.exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
