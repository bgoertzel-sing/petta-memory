"""Unit tests for petta_runtime.make_petta_parse_checker and _load_petta_runtime.

The petta_runtime module provides the glue between the dependency-free store and
an optional external PeTTa runtime: it builds a parse-check callback that the
store invokes before appending a cluster.  These tests use lightweight mock/stand-in
runtimes (no real PeTTa required) so they run in any environment.
"""
from __future__ import annotations

import io
import contextlib
import sys
import types
import unittest
from pathlib import Path
import tempfile

from petta_memory.petta_runtime import (
    make_petta_parse_checker,
    _load_petta_runtime,
)
from petta_memory.store import MemoryCluster, MediumMemoryStore, ValidationError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# A minimal valid cluster that satisfies the store's own validation rules.
_VALID_CLUSTER_TEXT = """
(MemoryCluster mc-test)
(SchemaVersion mc-test medium-memory-v1)
(ClusterType mc-test episode-record)
(ClusterOpenedAt mc-test "2026-09-04 16:00 UTC")
(ClusterSource mc-test src-test)
(Contains mc-test e-test)
(ClusterStatus mc-test active)
(ObservedEvent e-test)
(EpistemicRole e-test observed-event)
(HasStatus e-test recorded)
"""


def _make_cluster() -> MemoryCluster:
    """Build a MemoryCluster with the minimum required atoms."""
    return MemoryCluster(
        cluster_id="mc-test",
        atoms=(
            "(MemoryCluster mc-test)",
            "(SchemaVersion mc-test medium-memory-v1)",
            "(ClusterType mc-test episode-record)",
            '(ClusterOpenedAt mc-test "2026-09-04 16:00 UTC")',
            "(ClusterSource mc-test src-test)",
            "(Contains mc-test e-test)",
            "(ClusterStatus mc-test active)",
            "(ObservedEvent e-test)",
            "(EpistemicRole e-test observed-event)",
            "(HasStatus e-test recorded)",
        ),
    )


class _AcceptingRuntime:
    """Minimal runtime stub that accepts everything."""
    def process_metta_string(self, program: str):
        self.last_program = program
        return None


class _RejectingRuntime:
    """Minimal runtime stub that always raises."""
    def process_metta_string(self, program: str):
        raise RuntimeError("syntax error in program")


class _PrintingRuntime:
    """Runtime that prints to stdout/stderr — used to verify suppression."""
    def process_metta_string(self, program: str):
        print("stdout noise")
        print("stderr noise", file=sys.stderr)
        self.last_program = program


# ---------------------------------------------------------------------------
# make_petta_parse_checker
# ---------------------------------------------------------------------------

class TestMakePeTTaParseChecker(unittest.TestCase):

    def test_returns_callable(self):
        checker = make_petta_parse_checker(_AcceptingRuntime())
        self.assertTrue(callable(checker))

    def test_accepting_runtime_does_not_raise(self):
        checker = make_petta_parse_checker(_AcceptingRuntime())
        checker(_make_cluster())

    def test_rejecting_runtime_raises_validation_error(self):
        checker = make_petta_parse_checker(_RejectingRuntime())
        with self.assertRaises(ValidationError) as ctx:
            checker(_make_cluster())
        self.assertIn("mc-test", str(ctx.exception))

    def test_validation_error_is_re_raised_unchanged(self):
        """If the runtime itself raises ValidationError, it passes through."""
        class _VErrorRuntime:
            def process_metta_string(self, program: str):
                raise ValidationError("direct validation error")
        checker = make_petta_parse_checker(_VErrorRuntime())
        with self.assertRaises(ValidationError) as ctx:
            checker(_make_cluster())
        self.assertIn("direct validation error", str(ctx.exception))

    def test_checker_passes_cluster_text_to_runtime(self):
        runtime = _AcceptingRuntime()
        checker = make_petta_parse_checker(runtime)
        cluster = _make_cluster()
        checker(cluster)
        self.assertEqual(runtime.last_program, cluster.text)

    def test_output_suppression_on_by_default(self):
        runtime = _PrintingRuntime()
        checker = make_petta_parse_checker(runtime)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            checker(_make_cluster())
        self.assertEqual(captured.getvalue(), "")

    def test_output_suppression_off(self):
        runtime = _PrintingRuntime()
        checker = make_petta_parse_checker(runtime, suppress_output=False)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            checker(_make_cluster())
        self.assertIn("stdout noise", captured.getvalue())

    def test_petta_kwarg_builds_checker_from_runtime_instance(self):
        """Explicit petta= argument is used directly (no import)."""
        runtime = _AcceptingRuntime()
        checker = make_petta_parse_checker(petta=runtime)
        cluster = _make_cluster()
        checker(cluster)
        self.assertEqual(runtime.last_program, cluster.text)

    def test_cluster_text_ends_with_newline(self):
        """Verify the checker sends cluster.text which ends with newline."""
        runtime = _AcceptingRuntime()
        checker = make_petta_parse_checker(runtime)
        cluster = _make_cluster()
        checker(cluster)
        self.assertTrue(runtime.last_program.endswith("\n"))

    def test_rejecting_runtime_error_message_contains_exception_text(self):
        """The ValidationError message includes the original exception."""
        class _MsgRuntime:
            def process_metta_string(self, program: str):
                raise ValueError("specific parse error")
        checker = make_petta_parse_checker(_MsgRuntime())
        with self.assertRaises(ValidationError) as ctx:
            checker(_make_cluster())
        self.assertIn("specific parse error", str(ctx.exception))


# ---------------------------------------------------------------------------
# _load_petta_runtime
# ---------------------------------------------------------------------------

class TestLoadPeTTaRuntime(unittest.TestCase):

    def _install_fake_petta(self, PeTTa_cls):
        """Install a fake 'petta' module with the given PeTTa class."""
        fake_mod = types.ModuleType("petta")
        fake_mod.PeTTa = PeTTa_cls
        self._saved = sys.modules.get("petta")
        sys.modules["petta"] = fake_mod

    def _restore_petta(self):
        if hasattr(self, "_saved"):
            if self._saved is not None:
                sys.modules["petta"] = self._saved
            else:
                sys.modules.pop("petta", None)

    def test_load_raises_validation_error_when_petta_not_importable(self):
        """Simulate petta not being installed by setting sys.modules['petta']=None."""
        saved = sys.modules.get("petta")
        sys.modules["petta"] = None  # triggers ImportError on `import petta`
        try:
            with self.assertRaises(ValidationError) as ctx:
                _load_petta_runtime()
            self.assertIn("not importable", str(ctx.exception))
        finally:
            if saved is not None:
                sys.modules["petta"] = saved
            else:
                sys.modules.pop("petta", None)

    def test_load_with_petta_path_kwarg(self):
        """When petta_path is given and import succeeds, kwargs include it."""
        class FakePeTTa:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def process_metta_string(self, program: str):
                pass

        self._install_fake_petta(FakePeTTa)
        try:
            runtime = _load_petta_runtime(petta_path="/tmp/fake")
            self.assertIsInstance(runtime, FakePeTTa)
            self.assertEqual(runtime.kwargs.get("petta_path"), "/tmp/fake")
            self.assertFalse(runtime.kwargs.get("verbose"))
        finally:
            self._restore_petta()

    def test_load_falls_back_on_typeerror(self):
        """If PeTTa(**kwargs) raises TypeError, fall back to positional."""
        class FallbackPeTTa:
            def __init__(self, petta_path=None):
                self.petta_path = petta_path
            def process_metta_string(self, program: str):
                pass

        self._install_fake_petta(FallbackPeTTa)
        try:
            runtime = _load_petta_runtime(petta_path="/tmp/fallback")
            self.assertIsInstance(runtime, FallbackPeTTa)
            self.assertEqual(runtime.petta_path, "/tmp/fallback")
        finally:
            self._restore_petta()

    def test_load_falls_back_on_typeerror_no_path(self):
        """TypeError fallback without petta_path calls PeTTa() with no args."""
        class FallbackPeTTa:
            def __init__(self):
                self.called_no_args = True
            def process_metta_string(self, program: str):
                pass

        self._install_fake_petta(FallbackPeTTa)
        try:
            runtime = _load_petta_runtime()
            self.assertIsInstance(runtime, FallbackPeTTa)
            self.assertTrue(runtime.called_no_args)
        finally:
            self._restore_petta()

    def test_load_raises_validation_error_on_init_failure(self):
        """If PeTTa() raises a non-TypeError, propagate as ValidationError."""
        class BrokenPeTTa:
            def __init__(self, **kwargs):
                raise OSError("init failed")
            def process_metta_string(self, program: str):
                pass

        self._install_fake_petta(BrokenPeTTa)
        try:
            with self.assertRaises(ValidationError) as ctx:
                _load_petta_runtime()
            self.assertIn("could not be initialized", str(ctx.exception))
        finally:
            self._restore_petta()

    def test_load_default_no_petta_path(self):
        """Without petta_path, PeTTa(verbose=False) is called."""
        class DefaultPeTTa:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def process_metta_string(self, program: str):
                pass

        self._install_fake_petta(DefaultPeTTa)
        try:
            runtime = _load_petta_runtime()
            self.assertIsInstance(runtime, DefaultPeTTa)
            self.assertFalse(runtime.kwargs.get("verbose"))
            self.assertNotIn("petta_path", runtime.kwargs)
        finally:
            self._restore_petta()

    def test_load_with_petta_path_fallback_positional(self):
        """In fallback mode with path, PeTTa(petta_path=...) is called."""
        class PosPeTTa:
            def __init__(self, petta_path=None):
                self.petta_path = petta_path
            def process_metta_string(self, program: str):
                pass

        self._install_fake_petta(PosPeTTa)
        try:
            runtime = _load_petta_runtime(petta_path="/custom/path")
            self.assertIsInstance(runtime, PosPeTTa)
            self.assertEqual(runtime.petta_path, "/custom/path")
        finally:
            self._restore_petta()


# ---------------------------------------------------------------------------
# Protocol compliance & integration
# ---------------------------------------------------------------------------

class TestProtocolCompliance(unittest.TestCase):

    def test_accepting_runtime_satisfies_protocol_duck_type(self):
        runtime = _AcceptingRuntime()
        self.assertTrue(hasattr(runtime, "process_metta_string"))

    def test_rejecting_runtime_satisfies_protocol_duck_type(self):
        runtime = _RejectingRuntime()
        self.assertTrue(hasattr(runtime, "process_metta_string"))

    def test_checker_integration_with_store(self):
        """End-to-end: store with parse_checker accepts valid cluster."""
        runtime = _AcceptingRuntime()
        checker = make_petta_parse_checker(runtime)
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta",
                                      parse_checker=checker)
            cluster = store.append_cluster(_VALID_CLUSTER_TEXT)
            self.assertEqual(cluster.cluster_id, "mc-test")
            self.assertEqual(runtime.last_program, cluster.text)
            # File should now contain the cluster
            self.assertTrue(store.path.exists())

    def test_checker_rejection_prevents_append(self):
        """End-to-end: rejecting runtime prevents the append."""
        checker = make_petta_parse_checker(_RejectingRuntime())
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta",
                                      parse_checker=checker)
            with self.assertRaises(ValidationError):
                store.append_cluster(_VALID_CLUSTER_TEXT)
            # File should not exist (rejection before write)
            self.assertFalse(store.path.exists())
if __name__ == "__main__":
    unittest.main()
