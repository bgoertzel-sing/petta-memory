import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory import MediumMemoryStore, make_petta_parse_checker

ROOT = Path(__file__).resolve().parents[5]
PROJECT = ROOT / "projects" / "petta-memory"
PETTACHAINER = PROJECT / "repos" / "PeTTaChainer"
PETTA = PROJECT / "repos" / "PeTTa"
SWI_PREFIX = ROOT / "projects" / "omegaclaw" / "local" / "swipl-9.3.36"
VENVS = sorted((PETTACHAINER / ".venv" / "lib").glob("python*/site-packages"))

PROMOTED_CLUSTER = """
(MemoryCluster mc-pettachainer-smoke)
(SchemaVersion mc-pettachainer-smoke medium-memory-v1)
(ClusterType mc-pettachainer-smoke belief-promotion)
(ClusterOpenedAt mc-pettachainer-smoke "2026-07-01 21:00 PDT")
(ClusterSource mc-pettachainer-smoke src-test)
(Contains mc-pettachainer-smoke pe-pettachainer-smoke)
(Contains mc-pettachainer-smoke b-pettachainer-smoke)
(ClusterStatus mc-pettachainer-smoke active)
(PromotionEvent pe-pettachainer-smoke)
(PromotesFrom pe-pettachainer-smoke qc-pettachainer-smoke)
(PromotesTo pe-pettachainer-smoke b-pettachainer-smoke)
(PromotionRule pe-pettachainer-smoke explicit-pettachainer-smoke)
(PromotionTrust pe-pettachainer-smoke 0.91)
(PromotionDomain pe-pettachainer-smoke memory-architecture)
(DerivedBelief b-pettachainer-smoke)
(BeliefContent b-pettachainer-smoke (Requires MediumPeTTaMemory PLNReadyViews))
(TruthValue b-pettachainer-smoke (stv 0.88 0.77))
(EvidenceFor b-pettachainer-smoke qc-pettachainer-smoke)
"""


def _configure_local_pettachainer_runtime():
    if not (PETTACHAINER / "pettachainer").exists() or not (PETTA / "python").exists() or not SWI_PREFIX.exists() or not VENVS:
        raise unittest.SkipTest("local PeTTaChainer/SWI runtime checkout is not available")
    os.environ.setdefault("SWIPL_HOME", str(SWI_PREFIX))
    os.environ.setdefault("SWI_HOME_DIR", str(SWI_PREFIX / "lib" / "swipl"))
    os.environ["PATH"] = f"{SWI_PREFIX / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{SWI_PREFIX / 'lib' / 'swipl' / 'lib' / 'x86_64-linux'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for path in [str(VENVS[-1]), str(PETTACHAINER), str(PETTA / "python")]:
        if path not in sys.path:
            sys.path.insert(0, path)


class PeTTaChainerSmokeTests(unittest.TestCase):
    def test_petta_parse_checker_accepts_canonical_cluster(self):
        _configure_local_pettachainer_runtime()
        from petta import PeTTa

        with tempfile.TemporaryDirectory() as td:
            checker = make_petta_parse_checker(PeTTa(verbose=False, petta_path=str(PETTA)))
            store = MediumMemoryStore(Path(td) / "medium_memory.metta", parse_checker=checker)
            cluster = store.append_cluster(PROMOTED_CLUSTER)

        self.assertEqual(cluster.cluster_id, "mc-pettachainer-smoke")

    def test_exported_promoted_belief_is_pettachainer_checkable(self):
        _configure_local_pettachainer_runtime()
        from pettachainer import check_stmt

        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(PROMOTED_CLUSTER)
            statement = store.pettachainer_evidence_view().strip()

        self.assertEqual(
            statement,
            "(: b-pettachainer-smoke (Requires MediumPeTTaMemory PLNReadyViews) (STV 0.88 0.77))",
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = check_stmt(statement)
        self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
