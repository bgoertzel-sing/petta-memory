import math
import json
import os
import tempfile
import sys
import time
import unittest
from unittest import mock
from dataclasses import FrozenInstanceError, replace
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
    EpisodeManifest,
    KernelProcessCapture,
    Phase0ReferenceArtifact,
    ChartPolicy,
    PiContext,
    build_pi_chart,
    build_captured_episode_manifest,
    build_pettachainer_episode_contract,
    build_evidence_snapshot,
    build_episode_manifest,
    assemble_legacy_kernel_query_program,
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
    run_kernel_subprocess,
    validate_exact_kernel_replay,
    validate_kernel_capture_result,
    validate_kernel_result,
    validate_phase0_reference_artifact,
    validate_phase0_reference_replay,
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

    def assert_persistence_rejects_foreign_ownership(
        self, *, parent, artifact, reader, writer, value
    ):
        parent_metadata = os.stat(parent)
        foreign_parent_fields = list(parent_metadata)
        foreign_parent_fields[4] = parent_metadata.st_uid + 1
        foreign_parent = os.stat_result(foreign_parent_fields)
        with mock.patch("petta_memory.pipln_models.os.fstat", side_effect=[foreign_parent]):
            with self.assertRaisesRegex(ValueError, "owned by the current user"):
                reader(artifact)

        artifact_metadata = os.stat(artifact)
        foreign_artifact_fields = list(artifact_metadata)
        foreign_artifact_fields[4] = artifact_metadata.st_uid + 1
        foreign_artifact = os.stat_result(foreign_artifact_fields)
        with mock.patch(
            "petta_memory.pipln_models.os.fstat",
            side_effect=[parent_metadata, foreign_artifact],
        ):
            with self.assertRaisesRegex(ValueError, "owned by the current user"):
                reader(artifact)

        destination = Path(parent) / "foreign-parent.json"
        with mock.patch("petta_memory.pipln_models.os.fstat", side_effect=[foreign_parent]):
            with self.assertRaisesRegex(ValueError, "owned by the current user"):
                writer(destination, value)
        self.assertFalse(destination.exists())

    def test_phase0_reference_artifact_closes_frozen_content_and_boundaries(self):
        source = b"(Sentence (Reference fact) (stv 1 1) (0))\n"
        semantic_result = "((stv 0.5 0.75) (0))"
        output = f"[{semantic_result}]\n[((Passed: #t))]\n".encode()
        source_digest = sha256(source).hexdigest()
        output_digest = sha256(output).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "Smokes.metta"
            output_path = root / "smokes_output.txt"
            manifest_path = root / "reference_manifest.json"
            source_path.write_bytes(source)
            output_path.write_bytes(output)
            manifest = {
                "schema": "petta-memory-phase0-reference-artifact-v1",
                "determinism": {
                    "run1_sha256": output_digest,
                    "run2_sha256": output_digest,
                    "identical": True,
                },
                "example": {"name": "Smokes", "source_sha256": source_digest},
                "runtime": {"metta_binary_sha256": "a" * 64},
                "repositories": {"patham9-pln": {"commit": "b" * 40}},
                "result": {
                    "passed": True,
                    "semantic_result": semantic_result,
                    "query_target": "(Evaluation (Predicate cancerous) (List (Concept Edward)))",
                    "passed_marker": "#t",
                    "output_sha256": output_digest,
                    "output_bytes": len(output),
                    "output_file": output_path.name,
                },
                "boundaries": {
                    "no_memory_write": True,
                    "no_inferred_belief_promotion": True,
                    "no_live_omegaclaw_goalchainer_integration": True,
                    "no_pettachainer_compileadd": True,
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            admitted = validate_phase0_reference_artifact(manifest_path, source_path=source_path)
            self.assertEqual(admitted.semantic_result, semantic_result)
            self.assertEqual(admitted.output_sha256, output_digest)
            self.assertEqual(admitted.runtime_executable_sha256, "a" * 64)

            output_path.write_bytes(output + b"tampered")
            with self.assertRaisesRegex(ValueError, "output checksum"):
                validate_phase0_reference_artifact(manifest_path, source_path=source_path)
            output_path.write_bytes(output)
            manifest["boundaries"]["no_memory_write"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no_memory_write"):
                validate_phase0_reference_artifact(manifest_path, source_path=source_path)

    def test_phase0_reference_manifest_rejects_duplicate_json_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "reference_manifest.json"
            source_path = root / "Smokes.metta"
            source_path.write_text("(Reference)\n", encoding="utf-8")
            manifest_path.write_text(
                '{"schema":"petta-memory-phase0-reference-artifact-v1",'
                '"schema":"petta-memory-phase0-reference-artifact-v1"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate JSON object member: schema"):
                validate_phase0_reference_artifact(
                    manifest_path, source_path=source_path,
                )

    def test_phase0_reference_replay_requires_exact_successful_capture(self):
        semantic_result = "((stv 0.5 0.75) (0))"
        stdout = f"[{semantic_result}]\n[((Passed: #t))]\n"
        reference = Phase0ReferenceArtifact(
            example_name="Reference",
            source_sha256="a" * 64,
            runtime_executable_sha256="b" * 64,
            kernel_commit="c" * 40,
            semantic_result=semantic_result,
            query_target="(Reference fact)",
            output_sha256=sha256(stdout.encode()).hexdigest(),
            output_bytes=len(stdout.encode()),
        )
        capture = KernelProcessCapture(("/runtime", "Reference.metta"), 0, stdout, "")
        self.assertIs(validate_phase0_reference_replay(reference, capture), capture)

        for bad_capture, message in (
            (KernelProcessCapture(capture.argv, 1, stdout, ""), "exit successfully"),
            (KernelProcessCapture(capture.argv, 0, stdout, "warning"), "stderr"),
            (KernelProcessCapture(capture.argv, 0, stdout + "x", ""), "byte count"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_phase0_reference_replay(reference, bad_capture)

    def test_token_and_packet_are_immutable_and_packet_digest_is_stable(self):
        token = EvidenceToken("t1", "sensor", "s1", "c1", "2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z")
        packet = EvidencePacket("p1", "(S x)", "c1", 2, 1, (token.id,), 1, 1, "ACTIVE", "a1", "o1", "OBSERVATION")
        self.assertEqual(len(packet.provenance_digest), 64)
        with self.assertRaises(FrozenInstanceError):
            packet.status = "RETRACTED"

    def test_kernel_subprocess_is_shell_free_and_captures_bounded_output(self):
        capture = run_kernel_subprocess(
            "(PLN.Query)",
            argv=(sys.executable, "-c", "import sys; data=sys.stdin.read(); print(data); print('audit', file=sys.stderr)"),
            timeout_ms=1000,
            max_capture_bytes=100,
        )
        self.assertIsInstance(capture, KernelProcessCapture)
        self.assertEqual(capture.return_code, 0)
        self.assertEqual(capture.stdout, "(PLN.Query)\n")
        self.assertEqual(capture.stderr, "audit\n")
        self.assertEqual(capture.program_cid, sha256(json.dumps(
            {"complete_program": "(PLN.Query)"}, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest())

    def test_kernel_subprocess_fails_closed_on_timeout_or_capture_overflow(self):
        with self.assertRaisesRegex(ValueError, "timeout"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", "import time; time.sleep(1)"), timeout_ms=10,
            )
        with self.assertRaisesRegex(ValueError, "timeout"):
            run_kernel_subprocess(
                "x" * 200_000,
                argv=(sys.executable, "-c", "import time; time.sleep(1)"),
                timeout_ms=10,
            )
        with self.assertRaisesRegex(ValueError, "stdout"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", "print('x' * 20)"),
                timeout_ms=1000, max_capture_bytes=10,
            )

        marker = Path(tempfile.gettempdir()) / "petta-memory-overflow-child-finished"
        marker.unlink(missing_ok=True)
        script = (
            "import sys, time; from pathlib import Path; "
            "sys.stdout.write('x' * 100000); sys.stdout.flush(); "
            "time.sleep(0.2); Path(sys.argv[1]).touch()"
        )
        with self.assertRaisesRegex(ValueError, "stdout"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", script, str(marker)),
                timeout_ms=1000, max_capture_bytes=10,
            )
        self.assertFalse(marker.exists())

    def test_kernel_subprocess_requires_complete_program_delivery(self):
        with self.assertRaisesRegex(ValueError, "complete program delivery"):
            run_kernel_subprocess(
                "x" * 1_000_000,
                argv=(
                    sys.executable,
                    "-c",
                    "import os, sys; os.close(0); print('not-admitted'); sys.stdout.flush()",
                ),
                timeout_ms=1000,
                max_program_bytes=1_000_000,
            )

    def test_kernel_subprocess_closes_inherited_descendant_pipes(self):
        script = (
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)']); "
            "print('parent complete')"
        )
        started = time.monotonic()
        capture = run_kernel_subprocess(
            "program", argv=(sys.executable, "-c", script), timeout_ms=1000,
        )
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(capture.stdout, "parent complete\n")

    def test_kernel_subprocess_bounds_program_by_encoded_bytes_before_launch(self):
        marker = Path(tempfile.gettempdir()) / "petta-memory-runner-must-not-launch"
        marker.unlink(missing_ok=True)
        with self.assertRaisesRegex(ValueError, "max_program_bytes"):
            run_kernel_subprocess(
                "é",  # two UTF-8 bytes despite being one character
                argv=(sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
                timeout_ms=1000,
                max_program_bytes=1,
            )
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", "pass"),
                timeout_ms=1000, max_program_bytes=0,
            )

    def test_kernel_subprocess_bounds_argv_by_encoded_bytes_before_launch(self):
        marker = Path(tempfile.gettempdir()) / "petta-memory-argv-must-not-launch"
        marker.unlink(missing_ok=True)
        launch = f"from pathlib import Path; Path({str(marker)!r}).touch()"
        with self.assertRaisesRegex(ValueError, "max_argv_bytes"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch, "é"),
                timeout_ms=1000,
                max_argv_bytes=sum(map(len, (sys.executable.encode(), b"-c", launch.encode()))) + 1,
            )
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "NUL"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "bad\0arg"), timeout_ms=1000,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", "pass"),
                timeout_ms=1000, max_argv_bytes=0,
            )

        framing_marker = Path(tempfile.gettempdir()) / "petta-memory-argv-framing-must-not-launch"
        framing_marker.unlink(missing_ok=True)
        framing_launch = f"from pathlib import Path; Path({str(framing_marker)!r}).touch()"
        payload_bytes = sum(len(item.encode("utf-8")) for item in (sys.executable, "-c", framing_launch))
        with self.assertRaisesRegex(ValueError, "max_argv_bytes"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", framing_launch), timeout_ms=1000,
                max_argv_bytes=payload_bytes,
            )
        self.assertFalse(framing_marker.exists())

    def test_kernel_subprocess_bounds_cwd_by_encoded_bytes_before_launch(self):
        marker = Path(tempfile.gettempdir()) / "petta-memory-cwd-must-not-launch"
        marker.unlink(missing_ok=True)
        launch = f"from pathlib import Path; Path({str(marker)!r}).touch()"
        with self.assertRaisesRegex(ValueError, "max_cwd_bytes"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                cwd="é", max_cwd_bytes=1,
            )
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "NUL"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                cwd="bad\0cwd",
            )
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", "pass"), timeout_ms=1000,
                max_cwd_bytes=0,
            )
        with self.assertRaisesRegex(ValueError, "max_cwd_bytes"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                cwd=".", max_cwd_bytes=1,
            )
        self.assertFalse(marker.exists())

    def test_kernel_subprocess_bounds_explicit_env_before_launch(self):
        marker = Path(tempfile.gettempdir()) / "petta-memory-env-must-not-launch"
        marker.unlink(missing_ok=True)
        launch = f"from pathlib import Path; Path({str(marker)!r}).touch()"
        with self.assertRaisesRegex(ValueError, "max_env_bytes"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                env={"MODE": "é"}, max_env_bytes=5,
            )
        self.assertFalse(marker.exists())
        for env in ({"BAD\0KEY": "x"}, {"BAD=KEY": "x"}, {"KEY": "bad\0value"}):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "environment strings"):
                run_kernel_subprocess(
                    "program", argv=(sys.executable, "-c", launch), timeout_ms=1000, env=env,
                )
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", "pass"), timeout_ms=1000,
                max_env_bytes=0,
            )
        with self.assertRaisesRegex(ValueError, "max_env_bytes"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                env={"A": ""}, max_env_bytes=1,
            )
        self.assertFalse(marker.exists())
        capture = run_kernel_subprocess(
            "program",
            argv=(sys.executable, "-c", "import os; print(os.environ['KERNEL_MODE'])"),
            timeout_ms=1000, env={"KERNEL_MODE": "isolated"},
        )
        self.assertEqual(capture.stdout, "isolated\n")

    def test_kernel_subprocess_can_pin_executable_digest_before_launch(self):
        executable_digest = sha256(Path(sys.executable).read_bytes()).hexdigest()
        capture = run_kernel_subprocess(
            "program", argv=(sys.executable, "-c", "print('pinned')"), timeout_ms=1000,
            expected_executable_sha256=executable_digest,
        )
        self.assertEqual(capture.stdout, "pinned\n")
        self.assertEqual(capture.argv[0], str(Path(sys.executable).resolve(strict=True)))

        marker = Path(tempfile.gettempdir()) / "petta-memory-digest-must-not-launch"
        marker.unlink(missing_ok=True)
        launch = f"from pathlib import Path; Path({str(marker)!r}).touch()"
        with self.assertRaisesRegex(ValueError, "does not match"):
            run_kernel_subprocess(
                "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                expected_executable_sha256="0" * 64,
            )
        self.assertFalse(marker.exists())
        for digest in ("A" * 64, "0" * 63, True):
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                run_kernel_subprocess(
                    "program", argv=(sys.executable, "-c", launch), timeout_ms=1000,
                    expected_executable_sha256=digest,
                )
        with self.assertRaisesRegex(ValueError, "absolute regular file"):
            run_kernel_subprocess(
                "program", argv=("python3", "-c", launch), timeout_ms=1000,
                expected_executable_sha256="0" * 64,
            )
        self.assertFalse(marker.exists())

        with tempfile.TemporaryDirectory() as directory:
            resolved_executable = Path(directory) / "a-deliberately-long-resolved-kernel-executable"
            resolved_executable.write_bytes(b"not launched")
            short_link = Path(directory) / "p"
            short_link.symlink_to(resolved_executable)
            unresolved_argv_bytes = sum(
                len(item.encode("utf-8")) + 1 for item in (str(short_link), "-c", launch)
            )
            self.assertGreater(
                len(str(resolved_executable).encode("utf-8")),
                len(str(short_link).encode("utf-8")),
            )
            with self.assertRaisesRegex(ValueError, "resolved argv"):
                run_kernel_subprocess(
                    "program", argv=(str(short_link), "-c", launch), timeout_ms=1000,
                    max_argv_bytes=unresolved_argv_bytes,
                    expected_executable_sha256=sha256(b"not launched").hexdigest(),
                )
        self.assertFalse(marker.exists())

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
            path.parent.chmod(0o770)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                read_evidence_snapshot(path)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                write_evidence_snapshot(path.parent / "second.json", snapshot)
            self.assertFalse((path.parent / "second.json").exists())
            safe_parent = path.parent / "safe"
            safe_parent.mkdir(mode=0o700)
            linked_parent = path.parent / "linked"
            linked_parent.symlink_to(safe_parent, target_is_directory=True)
            with self.assertRaises(OSError):
                write_evidence_snapshot(linked_parent / "redirected.json", snapshot)
            self.assertFalse((safe_parent / "redirected.json").exists())
            safe_artifact = safe_parent / "snapshot.json"
            write_evidence_snapshot(safe_artifact, snapshot)
            self.assert_persistence_rejects_foreign_ownership(
                parent=safe_parent, artifact=safe_artifact,
                reader=read_evidence_snapshot, writer=write_evidence_snapshot,
                value=snapshot,
            )
            with self.assertRaises(OSError):
                read_evidence_snapshot(linked_parent / safe_artifact.name)
            hard_link = safe_parent / "snapshot-alias.json"
            os.link(safe_artifact, hard_link)
            with self.assertRaisesRegex(ValueError, "exactly one filesystem link"):
                read_evidence_snapshot(safe_artifact)

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

    def test_evidence_snapshot_admission_rejects_late_parent_metadata_change(self):
        packet = EvidencePacket(
            "p1", "(S x)", "ctx", 1, 0, ("t1",), 1, 1,
            "ACTIVE", "a1", "o1", "OBSERVATION",
        )
        snapshot = build_evidence_snapshot(
            snapshot_id="snap", packets=[packet], context_id="ctx",
            assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="now",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            write_evidence_snapshot(path, snapshot)
            parent_metadata = os.stat(directory)
            artifact_metadata = os.stat(path)
            changed_parent_fields = list(parent_metadata)
            changed_parent_fields[1] = parent_metadata.st_ino + 1
            changed_parent = os.stat_result(changed_parent_fields)
            with mock.patch(
                "petta_memory.pipln_models.os.fstat",
                side_effect=[
                    parent_metadata,
                    artifact_metadata,
                    artifact_metadata,
                    changed_parent,
                ],
            ):
                with self.assertRaisesRegex(ValueError, "parent changed during admission"):
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

        contract = build_pettachainer_episode_contract(compiled=compiled, query_term="(S a)")
        self.assertEqual(contract.query_atom, "(: $prf (S a) $tv)")
        self.assertEqual(len(contract.statements), 2)
        first = contract.statements[0]
        self.assertEqual(first.proof_id, f"pm-{compiled.sentences[0].meta.sentence_digest}")
        self.assertEqual(first.stamp_ints, compiled.sentences[0].meta.stamp_ints)
        self.assertEqual(first.evidence_basis_ids, compiled.sentences[0].meta.evidence_basis_ids)
        self.assertEqual(
            first.atom,
            f"(: {first.proof_id} (S a) (STV {compiled.sentences[0].projection.strength} "
            f"{compiled.sentences[0].projection.confidence}))",
        )
        self.assertNotIn("Sentence", first.atom)
        self.assertNotIn("compileadd", first.atom)
        with self.assertRaisesRegex(ValueError, "does not match typed content"):
            replace(first, atom="(: forged (S a) (STV 1.0 1.0))")
        with self.assertRaisesRegex(ValueError, "does not match typed content"):
            replace(contract, query_atom="(: fixed-proof (S a) (STV 1.0 1.0))")

        repeated_contract = build_pettachainer_episode_contract(compiled=repeated, query_term="(S a)")
        self.assertEqual(contract, repeated_contract)

        with self.assertRaisesRegex(ValueError, "already be canonical"):
            build_pettachainer_episode_contract(compiled=compiled, query_term="(S   a)")
        with self.assertRaisesRegex(ValueError, "executable/control form"):
            build_pettachainer_episode_contract(compiled=compiled, query_term="(PLN.Query (S a))")
        with self.assertRaisesRegex(ValueError, "max_atom_chars"):
            build_pettachainer_episode_contract(compiled=compiled, query_term="(S a)", max_atom_chars=10)

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
        for statement in (
            "(import! &self PLN)", "(Claim (eval dangerous))", "(let $x value body)",
            "(Claim (if condition then else))", "(PLN.Init ())",
            "(PLN.Config MaxSteps 10000)", "(Evidence (PLN.Query (S a) (Q a)))",
            "(Evidence (PLN.Derive (S a)))",
        ):
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
            Path(directory).chmod(0o770)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                read_compiled_episode_inputs(path)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                write_compiled_episode_inputs(Path(directory) / "second.json", compiled)
            self.assertFalse((Path(directory) / "second.json").exists())
            safe_parent = Path(directory) / "safe"
            safe_parent.mkdir(mode=0o700)
            linked_parent = Path(directory) / "linked"
            linked_parent.symlink_to(safe_parent, target_is_directory=True)
            with self.assertRaises(OSError):
                write_compiled_episode_inputs(linked_parent / "redirected.json", compiled)
            self.assertFalse((safe_parent / "redirected.json").exists())
            safe_artifact = safe_parent / "compiled.json"
            write_compiled_episode_inputs(safe_artifact, compiled)
            self.assert_persistence_rejects_foreign_ownership(
                parent=safe_parent, artifact=safe_artifact,
                reader=read_compiled_episode_inputs,
                writer=write_compiled_episode_inputs, value=compiled,
            )
            with self.assertRaises(OSError):
                read_compiled_episode_inputs(linked_parent / safe_artifact.name)
            hard_link = safe_parent / "compiled-alias.json"
            os.link(safe_artifact, hard_link)
            with self.assertRaisesRegex(ValueError, "exactly one filesystem link"):
                read_compiled_episode_inputs(safe_artifact)

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
            parent = os.stat(directory)
            foreign_parent_fields = list(parent)
            foreign_parent_fields[4] = parent.st_uid + 1
            foreign_parent = os.stat_result(foreign_parent_fields)
            with mock.patch("petta_memory.pipln_models.os.fstat", side_effect=[foreign_parent]):
                with self.assertRaisesRegex(ValueError, "owned by the current user"):
                    read_validated_kernel_result(path, compiled=compiled)
            artifact = os.stat(path)
            foreign_artifact_fields = list(artifact)
            foreign_artifact_fields[4] = artifact.st_uid + 1
            foreign_artifact = os.stat_result(foreign_artifact_fields)
            with mock.patch(
                "petta_memory.pipln_models.os.fstat",
                side_effect=[parent, foreign_artifact],
            ):
                with self.assertRaisesRegex(ValueError, "owned by the current user"):
                    read_validated_kernel_result(path, compiled=compiled)
            with mock.patch("petta_memory.pipln_models.os.fstat", side_effect=[foreign_parent]):
                with self.assertRaisesRegex(ValueError, "owned by the current user"):
                    write_validated_kernel_result(Path(directory) / "foreign-parent.json", result)
            self.assertFalse((Path(directory) / "foreign-parent.json").exists())
            with self.assertRaises(FileExistsError):
                write_validated_kernel_result(path, result)
            Path(directory).chmod(0o770)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                read_validated_kernel_result(path, compiled=compiled)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                write_validated_kernel_result(Path(directory) / "second.json", result)
            self.assertFalse((Path(directory) / "second.json").exists())
            safe_parent = Path(directory) / "safe"
            safe_parent.mkdir(mode=0o700)
            linked_parent = Path(directory) / "linked"
            linked_parent.symlink_to(safe_parent, target_is_directory=True)
            with self.assertRaises(OSError):
                write_validated_kernel_result(linked_parent / "redirected.json", result)
            self.assertFalse((safe_parent / "redirected.json").exists())
            safe_artifact = safe_parent / "result.json"
            write_validated_kernel_result(safe_artifact, result)
            with self.assertRaises(OSError):
                read_validated_kernel_result(linked_parent / safe_artifact.name, compiled=compiled)
            hard_link = safe_parent / "result-alias.json"
            os.link(safe_artifact, hard_link)
            with self.assertRaisesRegex(ValueError, "exactly one filesystem link"):
                read_validated_kernel_result(safe_artifact, compiled=compiled)
            Path(directory).chmod(0o700)

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

    def test_kernel_capture_result_closes_process_output_and_episode_provenance(self):
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
        result_atom = "((stv 0.75 0.5) (0))"
        capture = KernelProcessCapture(("/kernel",), 0, f"trace\n{result_atom}\n", "")
        result = validate_kernel_capture_result(
            capture, result_atom=result_atom, query_term="(Q a)", compiled=compiled,
        )
        self.assertEqual(result.stamp_ints, (0,))
        self.assertEqual(result.evidence_basis_ids, (basis.basis_id,))

        invalid = (
            (KernelProcessCapture(("/kernel",), 1, result_atom, ""), result_atom, "exit successfully"),
            (KernelProcessCapture(("/kernel",), 0, result_atom, "warning"), result_atom, "unexpected stderr"),
            (KernelProcessCapture(("/kernel",), 0, "other output", ""), result_atom, "not present verbatim"),
        )
        for bad_capture, bad_atom, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_kernel_capture_result(
                    bad_capture, result_atom=bad_atom, query_term="(Q a)", compiled=compiled,
                )

    def test_captured_episode_manifest_binds_one_successful_capture_end_to_end(self):
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
        query = "(Q a)"
        result_atom = "((stv 0.75 0.5) (0))"
        program = assemble_legacy_kernel_query_program(
            compiled=compiled, query_term=query, max_steps=3,
            task_queue_size=5, belief_queue_size=8,
        )
        capture = KernelProcessCapture(
            ("/kernel",), 0, f"trace\n{result_atom}\n", "",
            sha256(json.dumps(
                {"complete_program": program}, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")).hexdigest(),
        )
        digest = sha256(b"audit object").hexdigest()
        kwargs = dict(
            result_atom=result_atom, query_term=query, compiled=compiled, chart=chart,
            evidence_snapshot=snapshot, complete_program=program, kernel_name="patham9",
            kernel_capabilities_cid=digest, controller_envelope_cid=digest, seed=0,
            budget=EpisodeBudget(3, 1000, 4096), started_at="2026-07-15T23:00:00Z",
            finished_at="2026-07-15T23:00:01Z",
        )

        manifest = build_captured_episode_manifest(capture=capture, **kwargs)
        self.assertEqual(manifest.return_code, capture.return_code)
        self.assertEqual(manifest.stdout_cid, sha256(json.dumps(
            {"stdout": capture.stdout}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest())
        self.assertEqual(manifest.stderr_cid, sha256(json.dumps(
            {"stderr": capture.stderr}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest())

        invalid = (
            (KernelProcessCapture(("/kernel",), 1, capture.stdout, ""), "exit successfully"),
            (KernelProcessCapture(("/kernel",), 0, capture.stdout, "warning"), "unexpected stderr"),
            (KernelProcessCapture(("/kernel",), 0, "unrelated", ""), "not present verbatim"),
            (KernelProcessCapture(("/kernel",), 0, capture.stdout, "", "0" * 64), "program does not match"),
        )
        for bad_capture, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_captured_episode_manifest(capture=bad_capture, **kwargs)

    def test_phase1_clean_room_reload_preserves_query_and_provenance_boundaries(self):
        """Round-trip one frozen state without adding a new archive schema."""
        packet = EvidencePacket(
            "p1", "(S archived)", "ctx", 1, 0, ("t1",),
            1, 1, "ACTIVE", "a1", "o1", "OBSERVATION",
        )
        token = EvidenceToken("t1", "sensor", "s1", "ctx", "observed", "minted")
        snapshot = build_evidence_snapshot(
            snapshot_id="snapshot", packets=[packet], context_id="ctx",
            assumption_fingerprint="a1", ontology_fingerprint="o1", created_at="frozen",
        )
        context = PiContext(
            "ctx", "lang", "world", "guard", "guard-v1", "query",
            "assumptions", "ontology", "ontology-v1", "weak-v1", "relevance-v1",
        )
        chart = build_pi_chart(
            chart_id="chart", context=context, prior_strength_p0=0.5, prior_weight_k=2,
            prior_provenance="review",
            policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
            selected_packet_ids=["p1"], evidence_snapshot=snapshot,
            adequacy_certificate_id="adequacy",
        )
        basis = evidence_basis_from_packet(
            packet, [token], independence_status="PROVEN_DISJOINT",
            justification_cid="review:basis",
        )
        compiled = compile_episode_inputs(
            episode_id="capture-run", chart=chart, evidence_snapshot=snapshot,
            packets=[packet], bases=[basis],
        )
        result_atom = "((stv 0.75 0.5) (0))"
        result = validate_kernel_result(result_atom, query_term="(Q archived)", compiled=compiled)
        digest = sha256(b"phase1-audit").hexdigest()
        program = f"{compiled.sentences[0].atom}\n!(PLN.Query (Q archived))"
        manifest = build_episode_manifest(
            compiled=compiled, chart=chart, evidence_snapshot=snapshot, result=result,
            complete_program=program,
            kernel_name="patham9-pln",
            kernel_capabilities_cid=digest, controller_envelope_cid=digest, seed=0,
            budget=EpisodeBudget(3, 1000, 4096),
            started_at="2026-07-22T18:00:00Z", finished_at="2026-07-22T18:00:01Z",
            return_code=0, stdout=result_atom, stderr="",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = []
            for cycle in (1, 2):
                clean_room = root / f"reload-{cycle}"
                clean_room.mkdir(mode=0o700)
                compiled_path = clean_room / "compiled.json"
                result_path = clean_room / "result.json"
                manifest_path = clean_room / "manifest.json"
                write_compiled_episode_inputs(compiled_path, compiled)
                write_validated_kernel_result(result_path, result)
                write_episode_manifest(manifest_path, manifest)

                reference_source = b"(Sentence (Reference archived) (stv 1 1) (0))\n"
                reference_result = "((stv 0.5 0.75) (0))"
                reference_output = f"[{reference_result}]\n[((Passed: #t))]\n".encode()
                reference_source_path = clean_room / "Reference.metta"
                reference_output_path = clean_room / "reference-output.txt"
                reference_manifest_path = clean_room / "reference-manifest.json"
                reference_source_path.write_bytes(reference_source)
                reference_output_path.write_bytes(reference_output)
                reference_manifest_path.write_text(json.dumps({
                    "schema": "petta-memory-phase0-reference-artifact-v1",
                    "determinism": {
                        "run1_sha256": sha256(reference_output).hexdigest(),
                        "run2_sha256": sha256(reference_output).hexdigest(),
                        "identical": True,
                    },
                    "example": {
                        "name": "Reference",
                        "source_sha256": sha256(reference_source).hexdigest(),
                    },
                    "runtime": {"metta_binary_sha256": "a" * 64},
                    "repositories": {"patham9-pln": {"commit": "b" * 40}},
                    "result": {
                        "passed": True,
                        "semantic_result": reference_result,
                        "query_target": "(Reference archived)",
                        "passed_marker": "#t",
                        "output_sha256": sha256(reference_output).hexdigest(),
                        "output_bytes": len(reference_output),
                        "output_file": reference_output_path.name,
                    },
                    "boundaries": {
                        "no_memory_write": True,
                        "no_inferred_belief_promotion": True,
                        "no_live_omegaclaw_goalchainer_integration": True,
                        "no_pettachainer_compileadd": True,
                    },
                }), encoding="utf-8")
                expected_artifacts = {
                    "Reference.metta",
                    "compiled.json",
                    "manifest.json",
                    "reference-manifest.json",
                    "reference-output.txt",
                    "result.json",
                }
                self.assertEqual(
                    {path.name for path in clean_room.iterdir()},
                    expected_artifacts,
                )

                loaded_compiled = read_compiled_episode_inputs(compiled_path)
                loaded_result = read_validated_kernel_result(result_path, compiled=loaded_compiled)
                archived_capture = KernelProcessCapture(
                    ("/frozen/kernel",), 0, result_atom, "",
                    sha256(json.dumps(
                        {"complete_program": program}, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False,
                    ).encode("utf-8")).hexdigest(),
                )
                loaded_manifest = read_episode_manifest(
                    manifest_path, compiled=loaded_compiled, result=loaded_result,
                    complete_program=program, capture=archived_capture,
                )
                loaded_reference = validate_phase0_reference_artifact(
                    reference_manifest_path, source_path=reference_source_path,
                )
                replayed = validate_exact_kernel_replay(
                    result_atom, expected=loaded_result, compiled=loaded_compiled,
                )
                self.assertEqual(replayed, result)
                self.assertEqual(loaded_compiled.episode_id, "capture-run")
                self.assertEqual(loaded_result.episode_id, "capture-run")
                self.assertEqual(loaded_manifest.episode_id, "capture-run")
                self.assertEqual(loaded_reference.query_target, "(Reference archived)")
                self.assertNotEqual(loaded_reference.query_target, replayed.query_term)
                self.assertEqual(loaded_manifest.result_cid, replayed.result_digest)
                self.assertEqual(
                    {path.name for path in clean_room.iterdir()},
                    expected_artifacts,
                )
                with self.assertRaisesRegex(
                        ValueError, "does not match kernel process capture"):
                    read_episode_manifest(
                        manifest_path,
                        capture=replace(archived_capture, stdout=result_atom + "\n"),
                    )
                with self.assertRaisesRegex(
                        ValueError, "does not match captured program"):
                    read_episode_manifest(
                        manifest_path,
                        capture=replace(archived_capture, program_cid="0" * 64),
                    )
                identities.append((loaded_compiled.sentences[0].meta.sentence_digest,
                                   replayed.result_digest,
                                   loaded_manifest.manifest_digest,
                                   loaded_reference.source_sha256,
                                   loaded_reference.output_sha256))

                with self.assertRaisesRegex(ValueError, "compiled episode inputs document schema"):
                    read_compiled_episode_inputs(manifest_path)

                stale_descriptor = compiled_episode_inputs_document(compiled)
                stale_descriptor["payload"]["episode_id"] = "stale-capture-run"
                stale_descriptor["document_digest"] = sha256(json.dumps(
                    stale_descriptor["payload"], sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")).hexdigest()
                stale_path = clean_room / "stale-compiled.json"
                stale_path.write_text(json.dumps(stale_descriptor), encoding="utf-8")
                expected_artifacts.add("stale-compiled.json")
                with self.assertRaisesRegex(ValueError, "stamp map episode mismatch"):
                    read_compiled_episode_inputs(stale_path)

                reference_source_path.write_bytes(reference_source + b"; stale\n")
                with self.assertRaisesRegex(ValueError, "source checksum"):
                    validate_phase0_reference_artifact(
                        reference_manifest_path, source_path=reference_source_path,
                    )
                reference_source_path.write_bytes(reference_source)
                reference_output_path.write_bytes(reference_output + b"; stale\n")
                with self.assertRaisesRegex(ValueError, "output (byte count|checksum)"):
                    validate_phase0_reference_artifact(
                        reference_manifest_path, source_path=reference_source_path,
                    )
                self.assertEqual(
                    {path.name for path in clean_room.iterdir()},
                    expected_artifacts,
                )

            self.assertEqual(identities[0], identities[1])

            collision = compile_episode_inputs(
                episode_id="reload-new-assertion", chart=chart, evidence_snapshot=snapshot,
                packets=[packet], bases=[basis],
            )
            with self.assertRaisesRegex(ValueError, "compiled episode"):
                read_validated_kernel_result(root / "reload-1" / "result.json", compiled=collision)
            with self.assertRaisesRegex(ValueError, "manifest does not match compiled episode"):
                read_episode_manifest(
                    root / "reload-1" / "manifest.json", compiled=collision,
                )

            newly_asserted_packet = replace(
                packet, id="p2", statement="(S asserted-after-reload)",
            )
            newly_asserted_snapshot = build_evidence_snapshot(
                snapshot_id="snapshot-after-reload", packets=[newly_asserted_packet],
                context_id="ctx", assumption_fingerprint="a1",
                ontology_fingerprint="o1", created_at="after-reload",
            )
            newly_asserted_chart = build_pi_chart(
                chart_id="chart-after-reload", context=context,
                prior_strength_p0=0.5, prior_weight_k=2,
                prior_provenance="new-assertion",
                policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
                selected_packet_ids=["p2"], evidence_snapshot=newly_asserted_snapshot,
                adequacy_certificate_id="adequacy-after-reload",
            )
            newly_asserted_basis = evidence_basis_from_packet(
                newly_asserted_packet, [token],
                independence_status="PROVEN_DISJOINT",
                justification_cid="review:new-assertion",
            )
            newly_asserted = compile_episode_inputs(
                episode_id="reload-new-assertion", chart=newly_asserted_chart,
                evidence_snapshot=newly_asserted_snapshot,
                packets=[newly_asserted_packet], bases=[newly_asserted_basis],
            )
            self.assertNotEqual(
                newly_asserted.sentences[0].meta.sentence_digest,
                compiled.sentences[0].meta.sentence_digest,
            )
            self.assertNotEqual(
                newly_asserted.chart_fingerprint, compiled.chart_fingerprint,
            )
            with self.assertRaisesRegex(
                    ValueError, "kernel result does not match compiled episode"):
                validate_exact_kernel_replay(
                    result_atom, expected=result, compiled=newly_asserted,
                )
            with self.assertRaisesRegex(
                    ValueError, "manifest does not match compiled episode"):
                read_episode_manifest(
                    root / "reload-1" / "manifest.json",
                    compiled=newly_asserted,
                )

            mismatched_chart = replace(
                compiled,
                sentences=tuple(
                    replace(sentence, meta=replace(sentence.meta, chart_id="other-chart"))
                    for sentence in compiled.sentences
                ),
            )
            with self.assertRaisesRegex(ValueError, "manifest does not match compiled episode"):
                read_episode_manifest(
                    root / "reload-1" / "manifest.json", compiled=mismatched_chart,
                )

            with self.assertRaisesRegex(ValueError, "manifest does not match complete program"):
                read_episode_manifest(
                    root / "reload-1" / "manifest.json",
                    compiled=compiled,
                    result=result,
                    complete_program=program + "\n; cross-run drift",
                )

            collision_result = validate_kernel_result(
                result_atom, query_term="(Q archived)", compiled=collision,
            )
            with self.assertRaisesRegex(ValueError, "manifest does not match validated result"):
                read_episode_manifest(
                    root / "reload-1" / "manifest.json", result=collision_result,
                )

            colliding_packet = replace(packet, statement="(S cross-run)")
            colliding_snapshot = build_evidence_snapshot(
                snapshot_id=snapshot.id, packets=[colliding_packet], context_id="ctx",
                assumption_fingerprint="a1", ontology_fingerprint="o1",
                created_at="cross-run",
            )
            colliding_chart = build_pi_chart(
                chart_id=chart.id, context=context, prior_strength_p0=0.5,
                prior_weight_k=2, prior_provenance="review",
                policy=ChartPolicy("factor", "projection", "kernel", "rules", "v1", "v1"),
                selected_packet_ids=["p1"], evidence_snapshot=colliding_snapshot,
                adequacy_certificate_id="adequacy",
            )
            colliding_basis = evidence_basis_from_packet(
                colliding_packet, [token], independence_status="PROVEN_DISJOINT",
                justification_cid="review:basis",
            )
            same_named_cross_run = compile_episode_inputs(
                episode_id=compiled.episode_id, chart=colliding_chart,
                evidence_snapshot=colliding_snapshot, packets=[colliding_packet],
                bases=[colliding_basis],
            )
            self.assertEqual(same_named_cross_run.episode_id, compiled.episode_id)
            self.assertNotEqual(
                same_named_cross_run.chart_fingerprint, compiled.chart_fingerprint,
            )
            with self.assertRaisesRegex(
                    ValueError, "kernel result does not match compiled episode"):
                read_validated_kernel_result(
                    root / "reload-1" / "result.json",
                    compiled=same_named_cross_run,
                )
            with self.assertRaisesRegex(
                    ValueError, "complete program does not match compiled episode"):
                read_episode_manifest(
                    root / "reload-1" / "manifest.json",
                    compiled=same_named_cross_run,
                    result=result,
                    complete_program=program,
                )

            duplicate_anchor_repository = root / "duplicate-anchors"
            duplicate_anchor_repository.mkdir(mode=0o700)
            write_evidence_snapshot(
                duplicate_anchor_repository / f"{snapshot.snapshot_fingerprint}.json",
                snapshot,
            )
            write_evidence_snapshot(
                duplicate_anchor_repository
                / f"{colliding_snapshot.snapshot_fingerprint}.json",
                colliding_snapshot,
            )
            with self.assertRaisesRegex(ValueError, "duplicate snapshot id"):
                EvidenceSnapshotRepository(duplicate_anchor_repository).get(snapshot.id)

            forged_result = replace(
                result,
                evidence_basis_ids=("forged-basis",),
                result_digest=sha256(json.dumps({
                    "episode_id": result.episode_id,
                    "chart_fingerprint": result.chart_fingerprint,
                    "query_term": result.query_term,
                    "strength": float(result.strength),
                    "confidence": float(result.confidence),
                    "stamp_ints": result.stamp_ints,
                    "evidence_basis_ids": ("forged-basis",),
                }, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False).encode("utf-8")).hexdigest(),
            )
            forged_manifest_values = {
                field: getattr(manifest, field)
                for field in EpisodeManifest.__dataclass_fields__
                if field != "manifest_digest"
            }
            forged_manifest_values["result_cid"] = forged_result.result_digest
            forged_manifest = EpisodeManifest(
                **forged_manifest_values,
                manifest_digest=sha256(json.dumps({
                    **{
                        key: value for key, value in forged_manifest_values.items()
                        if key != "budget"
                    },
                    "parent_episode_ids": list(manifest.parent_episode_ids),
                    "projection_policy_ids": list(manifest.projection_policy_ids),
                    "budget": {
                        "max_steps": manifest.budget.max_steps,
                        "max_runtime_ms": manifest.budget.max_runtime_ms,
                        "max_output_chars": manifest.budget.max_output_chars,
                    },
                }, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False).encode("utf-8")).hexdigest(),
            )
            forged_path = root / "reload-1" / "forged-manifest.json"
            write_episode_manifest(forged_path, forged_manifest)
            with self.assertRaisesRegex(
                    ValueError, "stamps do not match compiled episode evidence bases"):
                read_episode_manifest(
                    forged_path, compiled=compiled, result=forged_result,
                    complete_program=program,
                )

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
            Path(directory).chmod(0o770)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                read_episode_manifest(path)
            with self.assertRaisesRegex(ValueError, "group- or world-writable"):
                write_episode_manifest(Path(directory) / "second.json", manifest)
            self.assertFalse((Path(directory) / "second.json").exists())
            safe_parent = Path(directory) / "safe"
            safe_parent.mkdir(mode=0o700)
            linked_parent = Path(directory) / "linked"
            linked_parent.symlink_to(safe_parent, target_is_directory=True)
            with self.assertRaises(OSError):
                write_episode_manifest(linked_parent / "redirected.json", manifest)
            self.assertFalse((safe_parent / "redirected.json").exists())
            safe_artifact = safe_parent / "manifest.json"
            write_episode_manifest(safe_artifact, manifest)
            self.assert_persistence_rejects_foreign_ownership(
                parent=safe_parent, artifact=safe_artifact,
                reader=read_episode_manifest, writer=write_episode_manifest,
                value=manifest,
            )
            with self.assertRaises(OSError):
                read_episode_manifest(linked_parent / safe_artifact.name)
            hard_link = safe_parent / "manifest-alias.json"
            os.link(safe_artifact, hard_link)
            with self.assertRaisesRegex(ValueError, "exactly one filesystem link"):
                read_episode_manifest(safe_artifact)

    def test_legacy_kernel_program_assembly_is_fixed_bounded_and_deterministic(self):
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
        kwargs = dict(compiled=compiled, query_term="(Q a)", max_steps=3,
                      task_queue_size=5, belief_queue_size=8)
        program = assemble_legacy_kernel_query_program(**kwargs)
        self.assertEqual(program, assemble_legacy_kernel_query_program(**kwargs))
        self.assertTrue(program.startswith("!(import! &self PLN)\n!(PLN.Init ())\n"))
        self.assertTrue(program.endswith("    (Q a)\n    3 5 8)\n"))
        for sentence in compiled.sentences:
            self.assertEqual(program.count(sentence.atom), 1)

        checked_programs = []
        checked = assemble_legacy_kernel_query_program(
            **kwargs, parse_check=checked_programs.append
        )
        self.assertEqual(checked_programs, [checked])

        def reject_program(_: str) -> None:
            raise ValueError("local parser rejected assembled program")

        with self.assertRaisesRegex(ValueError, "local parser rejected"):
            assemble_legacy_kernel_query_program(**kwargs, parse_check=reject_program)

        invalid = (
            ({**kwargs, "query_term": "(eval dangerous)"}, "executable/control"),
            ({**kwargs, "query_term": "(PLN.Query (S a) (Q a))"}, "executable/control"),
            ({**kwargs, "query_term": "(Q (PLN.Derive (S a)))"}, "executable/control"),
            ({**kwargs, "query_term": "(Q   a)"}, "already be canonical"),
            ({**kwargs, "max_steps": 0}, "positive integer"),
            ({**kwargs, "max_steps": 10_001}, "bounded limit"),
            ({**kwargs, "task_queue_size": 100_001}, "bounded limit"),
            ({**kwargs, "belief_queue_size": 100_001}, "bounded limit"),
            ({**kwargs, "max_program_chars": 5}, "exceeds max_program_chars"),
            ({**kwargs, "parse_check": "not-callable"}, "must be callable"),
        )
        for arguments, message in invalid:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, message):
                assemble_legacy_kernel_query_program(**arguments)

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
