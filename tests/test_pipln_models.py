import math
import unittest
from dataclasses import FrozenInstanceError

from petta_memory.pipln_models import (
    EvidenceBasis,
    EvidencePacket,
    EvidenceToken,
    canonical_local_chart_projection,
    deterministic_stamp_map,
)


class PiPlnModelTests(unittest.TestCase):
    def basis(self, basis_id, token_id):
        return EvidenceBasis(basis_id, (token_id,), (), "UNKNOWN", f"cid:{basis_id}")

    def test_token_and_packet_are_immutable_and_packet_digest_is_stable(self):
        token = EvidenceToken("t1", "sensor", "s1", "c1", "2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z")
        packet = EvidencePacket("p1", "(S x)", "c1", 2, 1, (token.id,), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        self.assertEqual(len(packet.provenance_digest), 64)
        with self.assertRaises(FrozenInstanceError):
            packet.status = "RETRACTED"

    def test_packet_rejects_unsorted_duplicate_tokens_and_nonfinite_counts(self):
        common = dict(id="p", statement="(S x)", context_id="c", negative_delta=0, source_reliability=1,
                      temporal_relevance=1, status="ACTIVE", assumption_fingerprint="a",
                      ontology_fingerprint="o", created_by="OBSERVATION")
        with self.assertRaises(ValueError):
            EvidencePacket(positive_delta=1, token_ids=("b", "a"), **common)
        with self.assertRaises(ValueError):
            EvidencePacket(positive_delta=math.inf, token_ids=("a",), **common)

    def test_deterministic_stamp_map_is_permutation_invariant(self):
        a, b = self.basis("basis-a", "t1"), self.basis("basis-b", "t2")
        forward = deterministic_stamp_map("episode-1", [a, b])
        reverse = deterministic_stamp_map("episode-1", [b, a])
        self.assertEqual(forward, reverse)
        self.assertEqual([(e.basis_id, e.stamp_int) for e in forward], [("basis-a", 0), ("basis-b", 1)])

    def test_deterministic_stamp_map_rejects_duplicate_basis_ids(self):
        with self.assertRaises(ValueError):
            deterministic_stamp_map("episode-1", [self.basis("same", "t1"), self.basis("same", "t2")])

    def test_projection_matches_canonical_equations(self):
        projection = canonical_local_chart_projection(3, 1, prior_strength=0.5, prior_weight=2)
        self.assertAlmostEqual(projection.strength, 4 / 6)
        self.assertAlmostEqual(projection.confidence, 4 / 6)
        self.assertEqual(projection.evidence_mass, 4)
        self.assertEqual(projection.conflict_balance, 0.5)
        self.assertEqual(projection.signed_tendency, 0.5)
        self.assertEqual((projection.beta_alpha, projection.beta_beta), (4, 2))

    def test_balanced_conflict_preserves_mass_distinction(self):
        small = canonical_local_chart_projection(1, 1, prior_strength=0.5, prior_weight=2)
        large = canonical_local_chart_projection(1000, 1000, prior_strength=0.5, prior_weight=2)
        self.assertEqual(small.strength, large.strength)
        self.assertEqual(small.conflict_balance, large.conflict_balance)
        self.assertLess(small.confidence, large.confidence)
        self.assertEqual((small.evidence_mass, large.evidence_mass), (2, 2000))

    def test_ignorance_and_balanced_conflict_are_distinct(self):
        ignorant = canonical_local_chart_projection(0, 0, prior_strength=0.5, prior_weight=2)
        conflict = canonical_local_chart_projection(1, 1, prior_strength=0.5, prior_weight=2)
        self.assertEqual(ignorant.strength, conflict.strength)
        self.assertEqual(ignorant.conflict_balance, 0)
        self.assertEqual(conflict.conflict_balance, 1)
        self.assertEqual(ignorant.confidence, 0)

    def test_projection_rejects_invalid_parameters(self):
        for value in (-1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                canonical_local_chart_projection(value, 0, prior_strength=0.5, prior_weight=2)
        with self.assertRaises(ValueError):
            canonical_local_chart_projection(1, 0, prior_strength=1.1, prior_weight=2)
        with self.assertRaises(ValueError):
            canonical_local_chart_projection(1, 0, prior_strength=0.5, prior_weight=0)


if __name__ == "__main__":
    unittest.main()
