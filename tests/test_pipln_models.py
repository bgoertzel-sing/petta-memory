import math
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

from petta_memory.pipln_models import (
    EvidenceBasis,
    EvidenceCapsule,
    EvidenceContribution,
    EvidencePacket,
    EvidenceSnapshot,
    EvidenceSnapshotRepository,
    EvidenceToken,
    EpisodeBudget,
    ChartPolicy,
    PiContext,
    build_pi_chart,
    build_evidence_snapshot,
    build_episode_manifest,
    canonical_local_chart_projection,
    canonical_projection_from_beta,
    compile_episode_inputs,
    compiled_episode_inputs_document,
    cycle_local_chart_prior,
    deterministic_stamp_map,
    evidence_basis_from_packet,
    evidence_snapshot_document,
    episode_manifest_document,
    merge_evidence_capsules,
    read_compiled_episode_inputs,
    read_evidence_snapshot,
    read_episode_manifest,
    read_validated_kernel_result,
    validate_exact_kernel_replay,
    validate_kernel_result,
    validated_kernel_result_document,
    write_compiled_episode_inputs,
    write_evidence_snapshot,
    write_episode_manifest,
    write_validated_kernel_result,
)


class PiPlnModelTests(unittest.TestCase):
    def basis(self, basis_id, token_id):
        return EvidenceBasis(basis_id, (token_id,), (), "UNKNOWN", f"cid:{basis_id}")

    def snapshot(self, packet_ids=("p1", "p2"), *, snapshot_id="snapshot-1", context_id="ctx", positive_delta=1):
        packets = [
            EvidencePacket(packet_id, "(S x)", context_id, positive_delta, 0, (f"t-{packet_id}",),
                           1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
            for packet_id in packet_ids
        ]
        return build_evidence_snapshot(
            snapshot_id=snapshot_id, packets=packets, context_id=context_id,
            assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now",
        )

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

    def test_packet_basis_builder_closes_provenance_and_collects_causal_groups(self):
        tokens = [
            EvidenceToken("t2", "sensor", "s2", "c1", "2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z", causal_group_id="cg-b"),
            EvidenceToken("t1", "sensor", "s1", "c1", "2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z", causal_group_id="cg-a"),
        ]
        packet = EvidencePacket("p1", "(S x)", "c1", 2, 0, ("t1", "t2"), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        basis = evidence_basis_from_packet(packet, reversed(tokens), independence_status="COUPLED", justification_cid="cid:review-1")
        self.assertEqual(basis.member_token_ids, ("t1", "t2"))
        self.assertEqual(basis.causal_group_ids, ("cg-a", "cg-b"))
        self.assertEqual(basis.independence_status, "COUPLED")
        self.assertEqual(basis, evidence_basis_from_packet(packet, tokens, independence_status="COUPLED", justification_cid="cid:review-1"))

    def test_packet_basis_builder_rejects_incomplete_extra_or_duplicate_metadata(self):
        token1 = EvidenceToken("t1", "sensor", "s1", "c1", "2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z")
        token2 = EvidenceToken("t2", "sensor", "s2", "c1", "2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z")
        packet = EvidencePacket("p1", "(S x)", "c1", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        for tokens, message in (([], "missing"), ([token1, token2], "extra"), ([token1, token1], "duplicate")):
            with self.assertRaisesRegex(ValueError, message):
                evidence_basis_from_packet(packet, tokens, independence_status="UNKNOWN", justification_cid="cid:review")

    def test_deterministic_stamp_map_is_permutation_invariant(self):
        a, b = self.basis("basis-a", "t1"), self.basis("basis-b", "t2")
        forward = deterministic_stamp_map("episode-1", [a, b])
        reverse = deterministic_stamp_map("episode-1", [b, a])
        self.assertEqual(forward, reverse)
        self.assertEqual([(e.basis_id, e.stamp_int) for e in forward], [("basis-a", 0), ("basis-b", 1)])

    def test_deterministic_stamp_map_rejects_duplicate_basis_ids(self):
        with self.assertRaises(ValueError):
            deterministic_stamp_map("episode-1", [self.basis("same", "t1"), self.basis("same", "t2")])

    def test_evidence_snapshot_is_order_invariant_and_version_closed(self):
        def packet(packet_id, token_id, **changes):
            values = dict(id=packet_id, statement="(S x)", context_id="ctx", positive_delta=1,
                          negative_delta=0, token_ids=(token_id,), source_reliability=1,
                          temporal_relevance=1, status="ACTIVE", assumption_fingerprint="a1",
                          ontology_fingerprint="o1", created_by="OBSERVATION")
            values.update(changes)
            return EvidencePacket(**values)
        p1, p2 = packet("p1", "t1"), packet("p2", "t2")
        kwargs = dict(snapshot_id="snap", context_id="ctx", assumption_fingerprint="a1",
                      ontology_fingerprint="o1", created_at="2026-07-12T16:00:00Z")
        forward = build_evidence_snapshot(packets=[p1, p2], **kwargs)
        reverse = build_evidence_snapshot(packets=[p2, p1], **kwargs)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.packet_ids, ("p1", "p2"))
        self.assertEqual(len(forward.snapshot_fingerprint), 64)
        changed_evidence = build_evidence_snapshot(
            packets=[packet("p1", "t1", positive_delta=2), p2], **kwargs
        )
        self.assertNotEqual(forward.snapshot_fingerprint, changed_evidence.snapshot_fingerprint)
        with self.assertRaisesRegex(ValueError, "not ACTIVE"):
            build_evidence_snapshot(packets=[packet("p3", "t3", status="RETRACTED")], **kwargs)
        with self.assertRaisesRegex(ValueError, "ontology mismatch"):
            build_evidence_snapshot(packets=[packet("p3", "t3", ontology_fingerprint="o2")], **kwargs)

    def test_evidence_snapshot_rejects_duplicate_packet_ids(self):
        packet = EvidencePacket("p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        with self.assertRaisesRegex(ValueError, "unique"):
            build_evidence_snapshot(snapshot_id="snap", packets=[packet, packet], context_id="ctx",
                                    assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")

    def test_evidence_snapshot_rejects_empty_selection_and_invalid_fingerprint(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_evidence_snapshot(snapshot_id="snap", packets=[], context_id="ctx",
                                    assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            EvidenceSnapshot("snap", ("p1",), "ctx", "a1", "o1", "now", "not-a-digest",
                             (("p1", "0" * 64),))

    def test_evidence_snapshot_persistence_is_immutable_and_roundtrips(self):
        packet = EvidencePacket("p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        snapshot = build_evidence_snapshot(snapshot_id="snap", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "snapshot.json"
            write_evidence_snapshot(path, snapshot)
            self.assertEqual(read_evidence_snapshot(path), snapshot)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_evidence_snapshot(path, snapshot)

    def test_evidence_snapshot_persistence_rejects_tampering_and_schema_drift(self):
        packet = EvidencePacket("p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        snapshot = build_evidence_snapshot(snapshot_id="snap", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            document = evidence_snapshot_document(snapshot)
            document["payload"]["context_id"] = "tampered"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_evidence_snapshot(path)
            document["schema"] = "unknown"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                read_evidence_snapshot(path)

    def test_evidence_snapshot_persistence_rejects_checksummed_invalid_payload(self):
        packet = EvidencePacket("p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        snapshot = build_evidence_snapshot(snapshot_id="snap", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            document = evidence_snapshot_document(snapshot)
            document["payload"]["snapshot_fingerprint"] = "forged"
            document["document_digest"] = __import__("hashlib").sha256(
                json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                read_evidence_snapshot(path)

            document = evidence_snapshot_document(snapshot)
            document["payload"]["packet_content_digests"][0][1] = "0" * 64
            document["document_digest"] = sha256(
                json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match packet content digests"):
                read_evidence_snapshot(path)

    def test_snapshot_repository_adds_discovers_and_gets_content_addressed_documents(self):
        packet = EvidencePacket("p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        snapshot = build_evidence_snapshot(snapshot_id="snap", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        with tempfile.TemporaryDirectory() as directory:
            repository = EvidenceSnapshotRepository(Path(directory) / "snapshots")
            path = repository.add(snapshot)
            self.assertEqual(path.name, f"{snapshot.snapshot_fingerprint}.json")
            self.assertEqual(repository.snapshots(), (snapshot,))
            self.assertEqual(repository.get("snap"), snapshot)
            with self.assertRaises(FileExistsError):
                repository.add(snapshot)
            with self.assertRaises(KeyError):
                repository.get("missing")

    def test_snapshot_repository_fails_closed_on_filename_drift_and_duplicate_ids(self):
        packet = EvidencePacket("p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        first = build_evidence_snapshot(snapshot_id="snap", packets=[packet], context_id="ctx",
                                        assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="one")
        second = build_evidence_snapshot(snapshot_id="snap", packets=[packet], context_id="ctx",
                                         assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="two")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evidence_snapshot(root / "wrong.json", first)
            with self.assertRaisesRegex(ValueError, "filename fingerprint"):
                EvidenceSnapshotRepository(root).snapshots()
            (root / "wrong.json").unlink()
            write_evidence_snapshot(root / f"{first.snapshot_fingerprint}.json", first)
            # created_at is audit metadata, so use a changed packet to force another content address.
            changed_packet = EvidencePacket("p1", "(S x)", "ctx", 2, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
            second = build_evidence_snapshot(snapshot_id="snap", packets=[changed_packet], context_id="ctx",
                                             assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="two")
            write_evidence_snapshot(root / f"{second.snapshot_fingerprint}.json", second)
            with self.assertRaisesRegex(ValueError, "duplicate snapshot id"):
                EvidenceSnapshotRepository(root).snapshots()

    def test_chart_fingerprint_is_packet_order_invariant_and_policy_sensitive(self):
        context = PiContext("ctx", "lang-v1", "world", "(Guard x)", "guard-v1", "query", "assumptions-v1",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor-v1", "projection-v1", "kernel-projection-v1", "rules-v1", "kernel-v1", "translator-v1")
        kwargs = dict(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                      prior_provenance="review:1", policy=policy, evidence_snapshot=self.snapshot(),
                      adequacy_certificate_id="adequacy-1")
        forward = build_pi_chart(selected_packet_ids=["p2", "p1"], **kwargs)
        reverse = build_pi_chart(selected_packet_ids=["p1", "p2"], **kwargs)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.selected_packet_ids, ("p1", "p2"))
        changed = build_pi_chart(selected_packet_ids=["p1", "p2"], **{**kwargs, "prior_weight_k": 3})
        self.assertNotEqual(forward.chart_fingerprint, changed.chart_fingerprint)

    def test_chart_fingerprint_covers_complete_selection_and_policy_identity(self):
        context = PiContext("ctx", "lang-v1", "world", "(Guard x)", "guard-v1", "query", "assumptions-v1",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor-v1", "projection-v1", "kernel-projection-v1", "rules-v1", "kernel-v1", "translator-v1")
        kwargs = dict(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                      prior_provenance="review:1", policy=policy, selected_packet_ids=["p1"],
                      evidence_snapshot=self.snapshot(), adequacy_certificate_id="adequacy-1")
        original = build_pi_chart(**kwargs)
        changed_selection = build_pi_chart(**{**kwargs, "selected_packet_ids": ["p2"]})
        changed_certificate = build_pi_chart(**{**kwargs, "adequacy_certificate_id": "adequacy-2"})
        changed_kernel_projection = build_pi_chart(**{
            **kwargs,
            "policy": ChartPolicy("factor-v1", "projection-v1", "kernel-projection-v2", "rules-v1", "kernel-v1", "translator-v1"),
        })
        for changed in (changed_selection, changed_certificate, changed_kernel_projection):
            self.assertNotEqual(original.chart_fingerprint, changed.chart_fingerprint)

        changed_snapshot = self.snapshot(("p1", "p2"), positive_delta=2)
        changed_snapshot_chart = build_pi_chart(**{**kwargs, "evidence_snapshot": changed_snapshot})
        self.assertNotEqual(original.chart_fingerprint, changed_snapshot_chart.chart_fingerprint)
        self.assertEqual(changed_snapshot_chart.evidence_snapshot_fingerprint, changed_snapshot.snapshot_fingerprint)

    def test_chart_closes_selected_packets_and_context_against_snapshot(self):
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor-v1", "projection-v1", "kernel-projection-v1", "rules-v1", "kernel-v1", "translator-v1")
        kwargs = dict(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                      prior_provenance="review:1", policy=policy, adequacy_certificate_id="adequacy-1")
        with self.assertRaisesRegex(ValueError, "absent from evidence snapshot"):
            build_pi_chart(selected_packet_ids=["p3"], evidence_snapshot=self.snapshot(("p1",)), **kwargs)
        with self.assertRaisesRegex(ValueError, "context"):
            build_pi_chart(selected_packet_ids=["p1"], evidence_snapshot=self.snapshot(("p1",), context_id="other"), **kwargs)

    def test_chart_rejects_empty_duplicate_or_blank_packet_selection(self):
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor-v1", "projection-v1", "kernel-projection-v1", "rules-v1", "kernel-v1", "translator-v1")
        kwargs = dict(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                      prior_provenance="review:1", policy=policy, evidence_snapshot=self.snapshot(),
                      adequacy_certificate_id="adequacy-1")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_pi_chart(selected_packet_ids=[], **kwargs)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_pi_chart(selected_packet_ids=["p1", "p1"], **kwargs)
        with self.assertRaisesRegex(ValueError, "selected_packet_id"):
            build_pi_chart(selected_packet_ids=[""], **kwargs)

    def test_episode_input_compiler_is_deterministic_and_preserves_provenance(self):
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor-v1", "pipl-local-chart-v1", "prior-aware-no-revision-v1",
                             "deduction-only-v1", "kernel-v1", "translator-v1")
        tokens = [
            EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted"),
            EvidenceToken("t2", "sensor", "s2", "ctx", "observed", "minted"),
        ]
        packets = [
            EvidencePacket("p2", "(S b)", "ctx", 1, 1, ("t2",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p1", "(S a)", "ctx", 3, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
        ]
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=packets, context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        chart = build_pi_chart(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                               prior_provenance="review:1", policy=policy, selected_packet_ids=["p2", "p1"],
                               evidence_snapshot=snapshot, adequacy_certificate_id="adequacy:1")
        bases = [
            evidence_basis_from_packet(packet, [tokens[index]], independence_status="PROVEN_DISJOINT",
                                       justification_cid=f"review:{packet.id}")
            for index, packet in enumerate(reversed(packets))
        ]
        compiled = compile_episode_inputs(episode_id="episode-1", chart=chart, evidence_snapshot=snapshot,
                                          packets=reversed(packets), bases=reversed(bases))
        repeated = compile_episode_inputs(episode_id="episode-1", chart=chart, evidence_snapshot=snapshot,
                                          packets=packets, bases=bases)
        self.assertEqual(compiled, repeated)
        self.assertEqual([item.meta.canonical_term for item in compiled.sentences], ["(S a)", "(S b)"])
        self.assertEqual([entry.stamp_int for entry in compiled.stamp_map], [0, 1])
        self.assertEqual(compiled.sentences[0].projection.positive_count, 3)
        self.assertIn("(Sentence ((S a) (stv", compiled.sentences[0].atom)
        self.assertEqual(compiled.sentences[0].meta.context_id, chart.context_id)
        self.assertEqual(compiled.evidence_snapshot_fingerprint, snapshot.snapshot_fingerprint)

    def test_episode_input_compiler_fails_closed_on_snapshot_packet_or_basis_drift(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor", "projection", "kernel-projection", "rules", "kernel", "translator")
        chart = build_pi_chart(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                               prior_provenance="review", policy=policy, selected_packet_ids=["p1"],
                               evidence_snapshot=snapshot, adequacy_certificate_id="adequacy")
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        other_snapshot = self.snapshot(("p1",), snapshot_id="other")
        with self.assertRaisesRegex(ValueError, "chart does not match"):
            compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=other_snapshot,
                                   packets=[packet], bases=[basis])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                   packets=[], bases=[basis])
        with self.assertRaisesRegex(ValueError, "missing exact packet basis"):
            compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                   packets=[packet], bases=[])

    def test_episode_input_compiler_rejects_packet_content_drift_under_snapshot_identity(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        for changed in (
            EvidencePacket("p1", "(S changed)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p1", "(S a)", "ctx", 2, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 0.5, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "REVIEWED_EXPORT"),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "packet content does not match"):
                compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                       packets=[changed], bases=[basis])

    def test_episode_input_compiler_canonicalizes_terms_and_rejects_control_forms(self):
        def compile_statement(statement):
            packet = EvidencePacket("p1", statement, "ctx", 1, 0, ("t1",), 1, 1,
                                    "ACTIVE", "a1", "o1", "OBSERVATION")
            token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
            snapshot = build_evidence_snapshot(
                snapshot_id="snapshot", packets=[packet], context_id="ctx",
                assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now",
            )
            context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                                "ontology", "ontology-v1", "weak-v1", "relevance-v1")
            chart = build_pi_chart(
                chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
                selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
            )
            basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                               justification_cid="review:basis")
            return compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])

        compiled = compile_statement(" (S   (Nested x)) ; harmless comment\n ")
        self.assertEqual(compiled.sentences[0].meta.canonical_term, "(S (Nested x))")
        self.assertIn("(Sentence ((S (Nested x))", compiled.sentences[0].atom)
        for statement in ("(import! &self PLN)", "(Claim (eval dangerous))", "(let $x value body)",
                          "(Claim (if condition then else))"):
            with self.subTest(statement=statement), self.assertRaisesRegex(ValueError, "executable/control form"):
                compile_statement(statement)
        with self.assertRaisesRegex(ValueError, "one valid S-expression"):
            compile_statement("(S x) (S y)")

    def test_episode_input_compiler_enforces_sentence_and_character_budgets(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        kwargs = dict(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                      packets=[packet], bases=[basis])
        with self.assertRaisesRegex(ValueError, "max_sentences"):
            compile_episode_inputs(**kwargs, max_sentences=True)
        with self.assertRaisesRegex(ValueError, "max_atom_chars"):
            compile_episode_inputs(**kwargs, max_atom_chars=10)

    def test_compiled_episode_inputs_persistence_round_trips_and_is_create_once(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 2, 1, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        policy = ChartPolicy("factor", "pipl-local-chart-v1", "kernel-projection", "rules", "kernel", "translator")
        chart = build_pi_chart(chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
                               prior_provenance="review", policy=policy, selected_packet_ids=["p1"],
                               evidence_snapshot=snapshot, adequacy_certificate_id="adequacy")
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])
        document = compiled_episode_inputs_document(compiled)
        self.assertEqual(document["schema"], "petta-memory-pipln-compiled-episode-inputs-v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compiled.json"
            write_compiled_episode_inputs(path, compiled)
            self.assertEqual(read_compiled_episode_inputs(path), compiled)
            with self.assertRaises(FileExistsError):
                write_compiled_episode_inputs(path, compiled)

    def test_kernel_result_validator_closes_numeric_output_to_episode_provenance(self):
        packets = [
            EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p2", "(S b)", "ctx", 1, 0, ("t2",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
        ]
        tokens = [
            EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted"),
            EvidenceToken("t2", "sensor", "s2", "ctx", "observed", "minted"),
        ]
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=packets, context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1", "p2"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        bases = [
            evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                       justification_cid=f"review:{packet.id}")
            for packet, token in zip(packets, tokens)
        ]
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=packets, bases=bases)

        result = validate_kernel_result(
            "((stv 0.91 0.75) (0 1))", query_term=" (Derived   x) ; comment", compiled=compiled,
        )
        self.assertEqual(result.query_term, "(Derived x)")
        self.assertEqual(result.stamp_ints, (0, 1))
        self.assertEqual(
            result.evidence_basis_ids,
            tuple(entry.basis_id for entry in compiled.stamp_map),
        )
        self.assertEqual(len(result.result_digest), 64)

    def test_validated_kernel_result_persistence_is_create_once_and_replay_closed(self):
        packets = [
            EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p2", "(S b)", "ctx", 1, 0, ("t2",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
        ]
        tokens = [
            EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted"),
            EvidenceToken("t2", "sensor", "s2", "ctx", "observed", "minted"),
        ]
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=packets, context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1", "p2"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        bases = [
            evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                       justification_cid=f"review:{packet.id}")
            for packet, token in zip(packets, tokens)
        ]
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=packets, bases=bases)
        result = validate_kernel_result("((stv 0.91 0.75) (0 1))", query_term="(Derived x)", compiled=compiled)
        document = validated_kernel_result_document(result)
        self.assertEqual(document["schema"], "petta-memory-pipln-validated-kernel-result-v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_validated_kernel_result(path, result)
            self.assertEqual(read_validated_kernel_result(path, compiled=compiled), result)
            with self.assertRaises(FileExistsError):
                write_validated_kernel_result(path, result)

            drifted_compiled = compile_episode_inputs(episode_id="episode-2", chart=chart, evidence_snapshot=snapshot,
                                                      packets=packets, bases=bases)
            with self.assertRaisesRegex(ValueError, "compiled episode"):
                read_validated_kernel_result(path, compiled=drifted_compiled)

    def test_validated_kernel_result_loader_rejects_checksum_and_semantic_drift(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])
        result = validate_kernel_result("((stv 0.5 0.75) (0))", query_term="(Derived x)", compiled=compiled)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            document = validated_kernel_result_document(result)
            document["payload"]["strength"] = 0.9
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_validated_kernel_result(path, compiled=compiled)

            document = validated_kernel_result_document(result)
            document["payload"]["query_term"] = "(eval dangerous)"
            encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            document["document_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable/control form"):
                read_validated_kernel_result(path, compiled=compiled)

            document = validated_kernel_result_document(result)
            document["payload"]["stamp_ints"] = [1]
            document["payload"]["evidence_basis_ids"] = ["basis-missing"]
            encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            document["document_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown episode stamps"):
                read_validated_kernel_result(path, compiled=compiled)

            document = validated_kernel_result_document(result)
            document["unexpected"] = "sidecar"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                read_validated_kernel_result(path, compiled=compiled)

            document = validated_kernel_result_document(result)
            document["payload"]["stamp_ints"] = [True]
            encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            document["document_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical non-negative integers"):
                read_validated_kernel_result(path, compiled=compiled)

    def test_kernel_result_validator_rejects_malformed_unbounded_or_unclosed_output(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])
        invalid = (
            ("((stv nan 0.5) (0))", "finite"),
            ("((stv 1.1 0.5) (0))", "finite"),
            ("((stv 0.5 0.5) (1))", "unknown episode stamps"),
            ("((stv 0.5 0.5) (0 0))", "unique"),
            ("((stv 0.5 0.5) (00))", "canonical"),
            ("((stv 0.5 0.5) (0)) (injected)", "one valid"),
        )
        for atom, message in invalid:
            with self.subTest(atom=atom), self.assertRaisesRegex(ValueError, message):
                validate_kernel_result(atom, query_term="(Q x)", compiled=compiled)
        with self.assertRaisesRegex(ValueError, "max_result_chars"):
            validate_kernel_result("((stv 0.5 0.5) (0))", query_term="(Q x)", compiled=compiled,
                                   max_result_chars=5)
        with self.assertRaisesRegex(ValueError, "executable/control form"):
            validate_kernel_result("((stv 0.5 0.5) (0))", query_term="(eval dangerous)", compiled=compiled)

    def test_exact_kernel_replay_requires_semantic_and_provenance_identity(self):
        packets = [
            EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
            EvidencePacket("p2", "(S b)", "ctx", 1, 0, ("t2",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION"),
        ]
        tokens = [
            EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted"),
            EvidenceToken("t2", "sensor", "s2", "ctx", "observed", "minted"),
        ]
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=packets, context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1", "p2"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        bases = [
            evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                       justification_cid=f"review:{packet.id}")
            for packet, token in zip(packets, tokens)
        ]
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=packets, bases=bases)
        expected = validate_kernel_result("((stv 0.5 0.75) (0 1))", query_term="(Derived x)", compiled=compiled)

        replayed = validate_exact_kernel_replay(
            "( ( stv 0.50 0.750 ) ( 0 1 ) )", expected=expected, compiled=compiled
        )
        self.assertEqual(replayed, expected)
        for changed in ("((stv 0.6 0.75) (0 1))", "((stv 0.5 0.75) (0))"):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "exactly match"):
                validate_exact_kernel_replay(changed, expected=expected, compiled=compiled)

        other_compiled = compile_episode_inputs(episode_id="other", chart=chart, evidence_snapshot=snapshot,
                                                packets=packets, bases=bases)
        with self.assertRaisesRegex(ValueError, "compiled episode"):
            validate_exact_kernel_replay("((stv 0.5 0.75) (0 1))", expected=expected, compiled=other_compiled)

    def test_episode_manifest_closes_program_result_and_runtime_audit_identity(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel-projection", "rules", "kernel-v1", "translator-v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])
        result = validate_kernel_result("((stv 0.75 0.5) (0))", query_term="(Q a)", compiled=compiled)
        program = "\n".join(("!(import! &self pln:core)", compiled.sentences[0].atom, "!(PLN.Derive (Q a))"))
        digest = sha256(b"audit object").hexdigest()
        manifest = build_episode_manifest(
            compiled=compiled, chart=chart, evidence_snapshot=snapshot, result=result,
            complete_program=program, kernel_name="patham9-pln", kernel_capabilities_cid=digest,
            controller_envelope_cid=digest, seed=7,
            budget=EpisodeBudget(max_steps=3, max_runtime_ms=1000, max_output_chars=4096),
            started_at="2026-07-14T12:00:00Z", finished_at="2026-07-14T12:00:01Z",
            return_code=0, stdout="result", stderr="", parent_episode_ids=("parent",),
        )
        self.assertEqual(manifest.result_cid, result.result_digest)
        self.assertEqual(manifest.projection_policy_ids, ("kernel-projection", "projection"))
        self.assertEqual(episode_manifest_document(manifest)["schema"], "petta-memory-pipln-episode-manifest-v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_episode_manifest(path, manifest)
            self.assertEqual(read_episode_manifest(path), manifest)
            with self.assertRaises(FileExistsError):
                write_episode_manifest(path, manifest)

    def test_episode_manifest_rejects_incomplete_program_and_tampered_artifact(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT", justification_cid="review")
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])
        result = validate_kernel_result("((stv 0.5 0.5) (0))", query_term="(Q a)", compiled=compiled)
        digest = sha256(b"audit object").hexdigest()
        kwargs = dict(
            compiled=compiled, chart=chart, evidence_snapshot=snapshot, result=result,
            kernel_name="patham9", kernel_capabilities_cid=digest, controller_envelope_cid=digest,
            seed=0, budget=EpisodeBudget(1, 100, 100), started_at="2026-07-14T12:00:00Z",
            finished_at="2026-07-14T12:00:01Z", return_code=0, stdout="", stderr="",
        )
        with self.assertRaisesRegex(ValueError, "every compiled sentence"):
            build_episode_manifest(complete_program="(unrelated)", **kwargs)
        manifest = build_episode_manifest(complete_program=compiled.sentences[0].atom + "\n(Q a)", **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            document = episode_manifest_document(manifest)
            document["payload"]["return_code"] = 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_episode_manifest(path)
            document["document_digest"] = sha256(json.dumps(
                document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest digest"):
                read_episode_manifest(path)

    def test_compiled_episode_inputs_loader_rejects_checksum_and_semantic_drift(self):
        packet = EvidencePacket("p1", "(S a)", "ctx", 1, 0, ("t1",), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(snapshot_id="snapshot", packets=[packet], context_id="ctx",
                                           assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now")
        context = PiContext("ctx", "lang", "world", "guard", "guard-v1", "query", "assumptions",
                            "ontology", "ontology-v1", "weak-v1", "relevance-v1")
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review", policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot, adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(packet, [token], independence_status="PROVEN_DISJOINT",
                                           justification_cid="review:basis")
        compiled = compile_episode_inputs(episode_id="episode", chart=chart, evidence_snapshot=snapshot,
                                          packets=[packet], bases=[basis])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compiled.json"
            document = compiled_episode_inputs_document(compiled)
            document["payload"]["episode_id"] = "changed"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_compiled_episode_inputs(path)

            document = compiled_episode_inputs_document(compiled)
            document["payload"]["sentences"][0]["atom"] = "(Sentence tampered)"
            encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            document["document_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "typed metadata"):
                read_compiled_episode_inputs(path)

            document = compiled_episode_inputs_document(compiled)
            sentence = document["payload"]["sentences"][0]
            sentence["meta"]["canonical_term"] = "(Changed term)"
            sentence["meta"]["sentence_digest"] = sha256(
                json.dumps({"atom": sentence["atom"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            document["document_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "typed metadata"):
                read_compiled_episode_inputs(path)

            document = compiled_episode_inputs_document(compiled)
            sentence = document["payload"]["sentences"][0]
            sentence["meta"]["canonical_term"] = "(Claim (eval dangerous))"
            sentence["atom"] = sentence["atom"].replace("(S a)", "(Claim (eval dangerous))")
            sentence["meta"]["sentence_digest"] = sha256(
                json.dumps({"atom": sentence["atom"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            document["document_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable/control form"):
                read_compiled_episode_inputs(path)

    def test_context_and_chart_fail_closed_on_noncanonical_identity_inputs(self):
        common = dict(id="ctx", language_fragment_id="lang", universe_id="world", guard="g", guard_version="gv",
                      task_class="query", assumption_package_id="a", ontology_id="o", ontology_version="ov",
                      weakness_policy_id="w", relevance_policy_id="r")
        with self.assertRaisesRegex(ValueError, "parent_context_ids"):
            PiContext(parent_context_ids=("z", "a"), **common)
        with self.assertRaisesRegex(ValueError, "own parent"):
            PiContext(parent_context_ids=("ctx",), **common)

    def test_exact_capsule_merge_deduplicates_shared_basis(self):
        shared = EvidenceContribution("basis-b", positive_weight=2)
        left = EvidenceCapsule((EvidenceContribution("basis-a", positive_weight=1), shared))
        right = EvidenceCapsule((shared, EvidenceContribution("basis-c", negative_weight=3)))
        merged = merge_evidence_capsules(left, right)
        self.assertEqual(merged.basis_ids, ("basis-a", "basis-b", "basis-c"))
        self.assertEqual((merged.positive_count, merged.negative_count), (3, 3))

    def test_exact_capsule_merge_rejects_conflicting_shared_basis(self):
        left = EvidenceCapsule((EvidenceContribution("basis-a", positive_weight=1),))
        right = EvidenceCapsule((EvidenceContribution("basis-a", positive_weight=2),))
        with self.assertRaisesRegex(ValueError, "conflicting contribution"):
            merge_evidence_capsules(left, right)

    def test_reviewed_merge_is_commutative_and_associative_for_disjoint_bases(self):
        capsules = [
            EvidenceCapsule((EvidenceContribution(f"basis-{name}", positive_weight=1),))
            for name in ("a", "b", "c")
        ]
        bases = [
            EvidenceBasis(f"basis-{name}", (f"token-{name}",), (), "PROVEN_DISJOINT", f"cid:basis-{name}")
            for name in ("a", "b", "c")
        ]
        self.assertEqual(
            merge_evidence_capsules(capsules[0], capsules[1], bases=bases),
            merge_evidence_capsules(capsules[1], capsules[0], bases=bases),
        )
        self.assertEqual(
            merge_evidence_capsules(merge_evidence_capsules(capsules[0], capsules[1], bases=bases), capsules[2], bases=bases),
            merge_evidence_capsules(capsules[0], merge_evidence_capsules(capsules[1], capsules[2], bases=bases), bases=bases),
        )

    def test_reviewed_merge_fails_closed_on_partial_or_unknown_overlap(self):
        left = EvidenceCapsule((EvidenceContribution("basis-a", positive_weight=1),))
        right = EvidenceCapsule((EvidenceContribution("basis-b", negative_weight=1),))
        partial = [
            EvidenceBasis("basis-a", ("shared",), (), "PROVEN_DISJOINT", "cid:basis-a"),
            EvidenceBasis("basis-b", ("shared",), (), "PROVEN_DISJOINT", "cid:basis-b"),
        ]
        with self.assertRaisesRegex(ValueError, "partial overlap"):
            merge_evidence_capsules(left, right, bases=partial)
        unknown = [
            EvidenceBasis("basis-a", ("token-a",), (), "PROVEN_DISJOINT", "cid:basis-a"),
            EvidenceBasis("basis-b", ("token-b",), (), "UNKNOWN", "cid-basis-b"),
        ]
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            merge_evidence_capsules(left, right, bases=unknown)

    def test_reviewed_merge_requires_complete_unique_basis_metadata(self):
        left = EvidenceCapsule((EvidenceContribution("basis-a", positive_weight=1),))
        right = EvidenceCapsule((EvidenceContribution("basis-b", negative_weight=1),))
        with self.assertRaisesRegex(ValueError, "missing basis metadata"):
            merge_evidence_capsules(left, right, bases=[self.basis("basis-a", "token-a")])
        duplicate = [self.basis("basis-a", "token-a"), self.basis("basis-a", "token-a")]
        with self.assertRaisesRegex(ValueError, "duplicate basis metadata"):
            merge_evidence_capsules(left, left, bases=duplicate)

    def test_capsule_rejects_unsorted_duplicate_or_empty_contributions(self):
        a = EvidenceContribution("basis-a", positive_weight=1)
        b = EvidenceContribution("basis-b", negative_weight=1)
        for contributions in ((), (b, a), (a, a)):
            with self.assertRaises(ValueError):
                EvidenceCapsule(contributions)
        with self.assertRaises(ValueError):
            EvidenceContribution("basis-empty")

    def test_projection_matches_canonical_equations(self):
        projection = canonical_local_chart_projection(3, 1, prior_strength=0.5, prior_weight=2)
        self.assertAlmostEqual(projection.strength, 4 / 6)
        self.assertAlmostEqual(projection.confidence, 4 / 6)
        self.assertEqual(projection.evidence_mass, 4)
        self.assertEqual(projection.conflict_balance, 0.5)
        self.assertEqual(projection.signed_tendency, 0.5)
        self.assertEqual((projection.beta_alpha, projection.beta_beta), (4, 2))

    def test_beta_roundtrip_recovers_empirical_counts(self):
        original = canonical_local_chart_projection(3.25, 1.5, prior_strength=0.4, prior_weight=2.5)
        recovered = canonical_projection_from_beta(
            original.beta_alpha, original.beta_beta, prior_strength=0.4, prior_weight=2.5
        )
        self.assertAlmostEqual(recovered.positive_count, 3.25)
        self.assertAlmostEqual(recovered.negative_count, 1.5)
        self.assertAlmostEqual(recovered.strength, original.strength)
        self.assertAlmostEqual(recovered.confidence, original.confidence)

    def test_prior_cycle_preserves_evidence_and_is_reversible(self):
        original = canonical_local_chart_projection(7, 2, prior_strength=0.5, prior_weight=2)
        changed = cycle_local_chart_prior(
            original, old_prior_strength=0.5, old_prior_weight=2,
            new_prior_strength=0.8, new_prior_weight=5,
        )
        self.assertEqual((changed.positive_count, changed.negative_count), (7, 2))
        self.assertNotEqual(changed.strength, original.strength)
        restored = cycle_local_chart_prior(
            changed, old_prior_strength=0.8, old_prior_weight=5,
            new_prior_strength=0.5, new_prior_weight=2,
        )
        self.assertAlmostEqual(restored.strength, original.strength)
        self.assertAlmostEqual(restored.confidence, original.confidence)
        self.assertAlmostEqual(restored.beta_alpha, original.beta_alpha)
        self.assertAlmostEqual(restored.beta_beta, original.beta_beta)

    def test_beta_roundtrip_rejects_prior_mass_mismatch(self):
        with self.assertRaisesRegex(ValueError, "less mass"):
            canonical_projection_from_beta(0.1, 0.1, prior_strength=0.5, prior_weight=2)

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
