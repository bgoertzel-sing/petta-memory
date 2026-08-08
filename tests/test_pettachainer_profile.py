import hashlib
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import call, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import petta_memory.pettachainer_profile as profile
import petta_memory.pipln_models as pipln_models
from petta_memory.pipln_models import (
    EpisodeBudget,
    PeTTaChainerEpisodeContract,
    PeTTaChainerInputStatement,
    build_pettachainer_episode_manifest,
    build_pettachainer_rule_attribution,
    read_pettachainer_rule_attribution,
    read_pettachainer_episode_manifest,
    read_pettachainer_derived_result_capture,
    write_pettachainer_derived_result_capture,
    write_pettachainer_episode_manifest,
    write_pettachainer_rule_attribution,
)
from petta_memory.pettachainer_profile import _run_isolated_stage, build_profile_store, build_promoted_cluster


def _slow_profile_stage() -> dict[str, object]:
    time.sleep(5)
    return {"result": "too late"}


def _echo_profile_stage(value: str) -> dict[str, object]:
    print("noisy runtime output")
    return {"result": value}


class PeTTaChainerProfileWorkloadTests(unittest.TestCase):
    def test_json_artifact_metadata_size_limit_rejects_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(b" " * 11)

            with patch.object(
                pipln_models.os, "fdopen",
                side_effect=AssertionError("oversized artifact must not be read"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "exceeds 10 byte limit",
                ):
                    pipln_models._load_unambiguous_json(path, max_bytes=10)

    def test_json_artifact_stream_close_does_not_mask_read_failure(self):
        class FailingStream:
            def read(self, _size):
                raise OSError("primary artifact read failure")

            def close(self):
                raise OSError("secondary stream close failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}', encoding="utf-8")

            with patch.object(pipln_models.os, "fdopen", return_value=FailingStream()):
                with self.assertRaisesRegex(
                    OSError, "primary artifact read failure",
                ) as raised:
                    pipln_models._load_unambiguous_json(path)
            self.assertIn(
                "JSON artifact stream close failed: secondary stream close failure",
                raised.exception.__notes__,
            )

    def test_json_artifact_stream_close_failure_propagates_after_read(self):
        class CloseFailingStream:
            def __init__(self, descriptor):
                self.descriptor = descriptor

            def read(self, size):
                return os.read(self.descriptor, size)

            def fileno(self):
                return self.descriptor

            def close(self):
                os.close(self.descriptor)
                raise OSError("artifact stream close failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}', encoding="utf-8")

            with patch.object(
                pipln_models.os, "fdopen", side_effect=lambda descriptor, _mode: CloseFailingStream(descriptor),
            ):
                with self.assertRaisesRegex(
                    OSError, "artifact stream close failure",
                ):
                    pipln_models._load_unambiguous_json(path)

    def test_json_artifact_admission_rejects_hard_link_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            alias = Path(directory) / "artifact-alias.json"
            path.write_text('{"value":1}', encoding="utf-8")
            os.link(path, alias)

            with self.assertRaisesRegex(
                ValueError, "must have exactly one filesystem link",
            ):
                pipln_models._load_unambiguous_json(path)

    def test_json_artifact_admission_rejects_group_writable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}', encoding="utf-8")
            path.chmod(0o620)

            with self.assertRaisesRegex(
                ValueError, "must not be group- or world-writable",
            ):
                pipln_models._load_unambiguous_json(path)

    def test_json_artifact_rejection_preserves_both_descriptor_close_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            path.chmod(0o620)
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError(f"close failure for descriptor {descriptor}")

            with patch.object(
                pipln_models.os, "close", side_effect=close_then_fail,
            ):
                with self.assertRaisesRegex(
                    ValueError, "must not be group- or world-writable",
                ) as caught:
                    pipln_models._load_unambiguous_json(path)

            notes = getattr(caught.exception, "__notes__", ())
            self.assertEqual(len(notes), 2)
            self.assertTrue(notes[0].startswith("JSON artifact descriptor close failed:"))
            self.assertTrue(notes[1].startswith("JSON artifact parent descriptor close failed:"))

    def test_json_artifact_success_preserves_both_descriptor_close_failures(self):
        class CloseFailingStream:
            def __init__(self, descriptor):
                self.descriptor = descriptor

            def read(self, size):
                return os.read(self.descriptor, size)

            def fileno(self):
                return self.descriptor

            def close(self):
                real_close(self.descriptor)
                raise OSError("artifact stream close failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            real_close = os.close

            def parent_close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("parent descriptor close failure")

            with patch.object(pipln_models.os, "fdopen", side_effect=lambda descriptor, _mode: CloseFailingStream(descriptor)), patch.object(
                pipln_models.os, "close", side_effect=parent_close_then_fail,
            ):
                with self.assertRaisesRegex(
                    OSError, "artifact stream close failure",
                ) as caught:
                    pipln_models._load_unambiguous_json(path)

            notes = getattr(caught.exception, "__notes__", ())
            self.assertEqual(len(notes), 1)
            self.assertTrue(notes[0].startswith(
                "JSON artifact parent descriptor close failed:",
            ))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires O_NOFOLLOW")
    def test_json_artifact_admission_rejects_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            actual.mkdir()
            (actual / "artifact.json").write_text('{"value":1}\n', encoding="utf-8")
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)

            with self.assertRaises(OSError):
                pipln_models._load_unambiguous_json(linked / "artifact.json")

    def test_json_artifact_admission_rejects_broadly_writable_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "shared"
            parent.mkdir()
            path = parent / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            parent.chmod(0o770)

            with self.assertRaisesRegex(
                ValueError, "parent must not be group- or world-writable",
            ):
                pipln_models._load_unambiguous_json(path)

    def test_json_artifact_parent_rejection_preserves_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "shared"
            parent.mkdir()
            path = parent / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            parent.chmod(0o770)
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with patch.object(pipln_models.os, "close", side_effect=close_then_fail):
                with self.assertRaisesRegex(
                    ValueError, "parent must not be group- or world-writable",
                ) as caught:
                    pipln_models._load_unambiguous_json(path)

            self.assertIn(
                "JSON artifact parent descriptor close failed: secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    def test_json_artifact_parent_metadata_failure_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            real_close = os.close
            closed_descriptors = []

            def record_close(descriptor: int) -> None:
                closed_descriptors.append(descriptor)
                real_close(descriptor)

            with (
                patch.object(
                    pipln_models.os, "fstat",
                    side_effect=OSError("parent metadata unavailable"),
                ),
                patch.object(pipln_models.os, "close", side_effect=record_close),
            ):
                with self.assertRaisesRegex(OSError, "parent metadata unavailable"):
                    pipln_models._load_unambiguous_json(path)

            self.assertEqual(len(closed_descriptors), 1)

    def test_json_artifact_parent_metadata_failure_preserves_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with (
                patch.object(
                    pipln_models.os, "fstat",
                    side_effect=OSError("parent metadata unavailable"),
                ),
                patch.object(
                    pipln_models.os, "close", side_effect=close_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "parent metadata unavailable",
                ) as caught:
                    pipln_models._load_unambiguous_json(path)

            self.assertIn(
                "JSON artifact parent descriptor close failed: "
                "secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    def test_json_artifact_open_rejection_preserves_parent_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with patch.object(pipln_models.os, "close", side_effect=close_then_fail):
                with self.assertRaises(FileNotFoundError) as caught:
                    pipln_models._load_unambiguous_json(path)

            self.assertIn(
                "JSON artifact parent descriptor close failed: secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    def test_json_artifact_admission_rejects_parent_permission_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            parent = os.stat(path.parent)
            artifact = os.stat(path)
            changed_fields = list(parent)
            changed_fields[0] = parent.st_mode | 0o020
            changed = os.stat_result(changed_fields)

            with patch.object(
                pipln_models.os, "fstat",
                side_effect=[parent, artifact, artifact, changed],
            ):
                with self.assertRaisesRegex(
                    ValueError, "parent changed during admission",
                ):
                    pipln_models._load_unambiguous_json(path)

    def test_json_artifact_admission_rejects_parent_ownership_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            parent = os.stat(path.parent)
            artifact = os.stat(path)
            changed_fields = list(parent)
            changed_fields[4] = parent.st_uid + 1
            changed = os.stat_result(changed_fields)

            with patch.object(
                pipln_models.os, "fstat",
                side_effect=[parent, artifact, artifact, changed],
            ):
                with self.assertRaisesRegex(
                    ValueError, "parent changed during admission",
                ):
                    pipln_models._load_unambiguous_json(path)

    def test_json_artifact_parent_drift_preserves_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            parent = os.stat(path.parent)
            artifact = os.stat(path)
            changed_fields = list(parent)
            changed_fields[0] = parent.st_mode | 0o020
            changed = os.stat_result(changed_fields)
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with (
                patch.object(
                    pipln_models.os, "fstat",
                    side_effect=[parent, artifact, artifact, changed],
                ),
                patch.object(
                    pipln_models.os, "close", side_effect=close_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "parent changed during admission",
                ) as caught:
                    pipln_models._load_unambiguous_json(path)

            self.assertIn(
                "JSON artifact parent descriptor close failed: "
                "secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    def test_json_artifact_success_propagates_parent_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("parent close failure")

            with patch.object(
                pipln_models.os, "close", side_effect=close_then_fail,
            ):
                with self.assertRaisesRegex(OSError, "parent close failure"):
                    pipln_models._load_unambiguous_json(path)

    def test_json_artifact_final_parent_metadata_failure_preserves_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}\n', encoding="utf-8")
            parent = os.stat(path.parent)
            artifact = os.stat(path)
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with (
                patch.object(
                    pipln_models.os, "fstat",
                    side_effect=[
                        parent,
                        artifact,
                        artifact,
                        OSError("final parent metadata unavailable"),
                    ],
                ),
                patch.object(
                    pipln_models.os, "close", side_effect=close_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "final parent metadata unavailable",
                ) as caught:
                    pipln_models._load_unambiguous_json(path)

            self.assertIn(
                "JSON artifact parent descriptor close failed: "
                "secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires O_NOFOLLOW")
    def test_create_once_publication_rejects_symlinked_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual_parent = root / "actual"
            actual_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(actual_parent, target_is_directory=True)

            with self.assertRaises(OSError):
                pipln_models._write_create_once_durable(
                    linked_parent / "artifact.json", '{"value":1}\n',
                )

            self.assertFalse((actual_parent / "artifact.json").exists())

    def test_create_once_publication_rejects_group_writable_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "shared"
            parent.mkdir()
            parent.chmod(0o770)
            destination = parent / "artifact.json"

            with self.assertRaisesRegex(
                ValueError, "parent must not be group- or world-writable",
            ):
                pipln_models._write_create_once_durable(
                    destination, '{"value":1}\n',
                )

            self.assertFalse(destination.exists())

    def test_create_once_parent_metadata_failure_preserves_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with (
                patch.object(
                    pipln_models.os, "fstat",
                    side_effect=OSError("parent metadata unavailable"),
                ),
                patch.object(
                    pipln_models.os, "close", side_effect=close_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "parent metadata unavailable",
                ) as caught:
                    pipln_models._write_create_once_durable(
                        destination, '{"value":1}\n',
                    )

            self.assertFalse(destination.exists())
            self.assertIn(
                "parent directory descriptor close failed: "
                "secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    def test_create_once_artifact_creation_failure_preserves_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            real_open = os.open
            real_close = os.close

            def fail_artifact_open(path, flags, *args, **kwargs):
                if kwargs.get("dir_fd") is not None:
                    raise OSError("primary artifact creation failure")
                return real_open(path, flags, *args, **kwargs)

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("secondary parent close failure")

            with (
                patch.object(pipln_models.os, "open", side_effect=fail_artifact_open),
                patch.object(
                    pipln_models.os, "close", side_effect=close_then_fail,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "primary artifact creation failure",
                ) as caught:
                    pipln_models._write_create_once_durable(
                        destination, '{"value":1}\n',
                    )

            self.assertFalse(destination.exists())
            self.assertIn(
                "parent directory descriptor close failed: "
                "secondary parent close failure",
                getattr(caught.exception, "__notes__", ()),
            )

    def test_create_once_publication_rejects_parent_permission_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            initial = os.stat(directory)
            changed_fields = list(initial)
            changed_fields[0] = initial.st_mode | 0o020
            changed = os.stat_result(changed_fields)
            with patch.object(
                pipln_models.os, "fstat", side_effect=[initial, changed],
            ):
                with self.assertRaisesRegex(
                    ValueError, "parent changed during publication",
                ):
                    pipln_models._write_create_once_durable(
                        destination, '{"value":1}\n',
                    )

            # The completed, file-synced artifact remains create-once even
            # though its publication state was rejected before directory sync.
            self.assertTrue(destination.exists())

    def test_create_once_publication_rejects_parent_ownership_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.json"
            initial = os.stat(directory)
            changed_fields = list(initial)
            changed_fields[4] = initial.st_uid + 1
            changed = os.stat_result(changed_fields)
            with patch.object(
                pipln_models.os, "fstat", side_effect=[initial, changed],
            ):
                with self.assertRaisesRegex(
                    ValueError, "parent changed during publication",
                ):
                    pipln_models._write_create_once_durable(
                        destination, '{"value":1}\n',
                    )

            self.assertTrue(destination.exists())

    def test_json_artifact_admission_rejects_metadata_byte_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}', encoding="utf-8")
            actual = os.stat(path)
            inconsistent_fields = list(actual)
            inconsistent_fields[6] = actual.st_size + 1
            inconsistent = os.stat_result(inconsistent_fields)

            with patch.object(
                pipln_models.os, "fstat", return_value=inconsistent,
            ):
                with self.assertRaisesRegex(
                    ValueError, "byte count does not match file metadata",
                ):
                    pipln_models._load_unambiguous_json(path)

    def test_json_artifact_admission_rejects_concurrent_metadata_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}', encoding="utf-8")
            initial = os.stat(path)
            changed_fields = list(initial)
            changed_fields[6] = initial.st_size + 1
            changed = os.stat_result(changed_fields)
            parent = os.stat(path.parent)

            with patch.object(
                pipln_models.os, "fstat", side_effect=[parent, initial, changed],
            ):
                with self.assertRaisesRegex(ValueError, "changed during admission"):
                    pipln_models._load_unambiguous_json(path)

    def test_json_artifact_admission_rejects_concurrent_mode_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text('{"value":1}', encoding="utf-8")
            initial = os.stat(path)
            changed_fields = list(initial)
            changed_fields[0] = initial.st_mode ^ 0o020
            changed = os.stat_result(changed_fields)
            parent = os.stat(path.parent)

            with patch.object(
                pipln_models.os, "fstat", side_effect=[parent, initial, changed],
            ):
                with self.assertRaisesRegex(ValueError, "changed during admission"):
                    pipln_models._load_unambiguous_json(path)

    @staticmethod
    def episode_contract():
        digest = "a" * 64
        statement = PeTTaChainerInputStatement(
            atom=f"(: pm-{digest} (S a) (STV 0.75 0.8))",
            proof_id=f"pm-{digest}",
            sentence_digest=digest,
            canonical_term="(S a)",
            strength=0.75,
            confidence=0.8,
            stamp_ints=(0,),
            evidence_basis_ids=("basis-1",),
        )
        return PeTTaChainerEpisodeContract(
            episode_id="episode-probe",
            chart_fingerprint="b" * 64,
            statements=(statement,),
            query_term="(S a)",
            query_atom="(: $prf (S a) $tv)",
        )

    @staticmethod
    def rule_episode_contract():
        fact_digest = "a" * 64
        rule_digest = "c" * 64
        fact = PeTTaChainerInputStatement(
            atom=f"(: pm-{fact_digest} (S a) (STV 0.8 0.6))",
            proof_id=f"pm-{fact_digest}", sentence_digest=fact_digest,
            canonical_term="(S a)", strength=0.8, confidence=0.6,
            stamp_ints=(0,), evidence_basis_ids=("basis-fact",),
        )
        rule_term = "(Implication (Premises (S $x)) (Conclusions (T $x)))"
        rule = PeTTaChainerInputStatement(
            atom=f"(: pm-{rule_digest} {rule_term} (STV 0.9 0.8))",
            proof_id=f"pm-{rule_digest}", sentence_digest=rule_digest,
            canonical_term=rule_term, strength=0.9, confidence=0.8,
            stamp_ints=(1,), evidence_basis_ids=("basis-rule",),
        )
        return PeTTaChainerEpisodeContract(
            episode_id="episode-rule-probe", chart_fingerprint="b" * 64,
            statements=(rule, fact), query_term="(T a)",
            query_atom="(: $prf (T a) $tv)",
        )

    def test_input_statement_rejects_boolean_stamp(self):
        statement = self.episode_contract().statements[0]
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            replace(statement, stamp_ints=(True,))

    def test_input_statement_rejects_non_string_evidence_basis(self):
        statement = self.episode_contract().statements[0]
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            replace(statement, evidence_basis_ids=(1,))

    def test_input_statement_bounds_reconstructed_term_before_parsing(self):
        statement = self.episode_contract().statements[0]
        oversized = "x" * (pipln_models.DEFAULT_MAX_COMPILED_ATOM_CHARS + 1)
        with patch.object(
            pipln_models, "_canonical_kernel_term",
            side_effect=AssertionError("oversized term must not be parsed"),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds max_atom_chars"):
                replace(statement, canonical_term=oversized)

    def test_input_statement_rejects_non_string_term_before_parsing(self):
        statement = self.episode_contract().statements[0]
        with patch.object(
            pipln_models, "_canonical_kernel_term",
            side_effect=AssertionError("non-string term must not be parsed"),
        ):
            with self.assertRaisesRegex(ValueError, "must be strings"):
                replace(statement, canonical_term=None)

    def test_input_statement_requires_evidence_basis_for_every_stamp(self):
        statement = self.episode_contract().statements[0]
        with self.assertRaisesRegex(ValueError, "close every stamp"):
            replace(statement, stamp_ints=(0, 1))

    def test_input_statement_rejects_mutable_provenance_collections(self):
        statement = self.episode_contract().statements[0]
        with self.assertRaisesRegex(ValueError, "stamps must be .* tuple"):
            replace(statement, stamp_ints=list(statement.stamp_ints))
        with self.assertRaisesRegex(ValueError, "evidence bases must be .* tuple"):
            replace(
                statement,
                evidence_basis_ids=list(statement.evidence_basis_ids),
            )

    def test_episode_contract_rejects_mutable_statement_collection(self):
        contract = self.episode_contract()
        with self.assertRaisesRegex(ValueError, "at least one statement"):
            replace(contract, statements=list(contract.statements))

    def test_episode_contract_bounds_statements_before_uniqueness_scan(self):
        contract = self.episode_contract()
        statement = contract.statements[0]
        oversized_count = (
            pipln_models.DEFAULT_MAX_COMPILED_ATOM_CHARS // len(statement.atom) + 1
        )
        with self.assertRaisesRegex(ValueError, "exceed.*max_atom_chars"):
            replace(contract, statements=(statement,) * oversized_count)

    def test_episode_contract_bounds_query_before_provenance_scans(self):
        contract = self.episode_contract()
        with self.assertRaisesRegex(ValueError, "exceed.*max_atom_chars"):
            replace(
                contract,
                statements=contract.statements * 2,
                query_atom="x" * pipln_models.DEFAULT_MAX_COMPILED_ATOM_CHARS,
            )

    def test_episode_contract_requires_contiguous_compiler_stamps(self):
        contract = self.episode_contract()
        statement = replace(
            contract.statements[0],
            stamp_ints=(1,),
        )
        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            replace(contract, statements=(statement,))

    def test_episode_contract_probe_runs_runtime_only_after_exact_validation(self):
        events = [
            {"label": "validate_episode_contract", "status": "ok", "statement_results": [1.0], "query_result": 1.0},
            {"label": "compileadd_and_query_episode_contract", "status": "timeout", "timeout_sec": 2.0},
        ]
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=events) as isolated,
        ):
            result = profile.probe_pettachainer_episode_contract(
                self.episode_contract(), project_root=Path("/unused"), stage_timeout_sec=2.0,
            )

        self.assertTrue(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])
        self.assertEqual(isolated.call_count, 2)
        self.assertTrue(result["boundaries"]["no_result_claim_unless_runtime_admitted"])

    def test_episode_contract_probe_fails_closed_before_runtime_on_validator_drift(self):
        validation = {
            "label": "validate_episode_contract", "status": "ok",
            "statement_results": [True], "query_result": 1.0,
        }
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=validation) as isolated,
        ):
            result = profile.probe_pettachainer_episode_contract(
                self.episode_contract(), project_root=Path("/unused"),
            )

        self.assertFalse(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])
        self.assertEqual(isolated.call_count, 1)

    def test_episode_contract_probe_does_not_admit_empty_query_result(self):
        events = [
            {"status": "ok", "statement_results": [1.0], "query_result": 1.0},
            {
                "status": "ok",
                "stages": [
                    {"status": "ok", "result": "added"},
                    {"status": "ok", "result": []},
                ],
            },
        ]
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=events),
        ):
            result = profile.probe_pettachainer_episode_contract(
                self.episode_contract(), project_root=Path("/unused"),
            )

        self.assertTrue(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])

    def test_build_promoted_cluster_exports_statement_and_packet(self):
        with tempfile.TemporaryDirectory() as td:
            store = build_profile_store(Path(td) / "medium_memory.metta", 2)
            statements = store.pettachainer_evidence_view().splitlines()
            packets = store.pettachainer_evidence_packet_view().splitlines()

        self.assertEqual(len(statements), 2)
        self.assertEqual(len(packets), 2)
        self.assertIn("(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))", statements)
        self.assertIn(
            "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) "
            "((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
            packets,
        )

    def test_negative_profile_workload_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                build_profile_store(Path(td) / "medium_memory.metta", -1)

    def test_generated_profile_cluster_uses_explicit_counts(self):
        cluster = build_promoted_cluster(4, support=10.0, opposition=2.0)
        self.assertIn("(EvidenceSupportCount b-profile-004 14.0)", cluster)
        self.assertIn("(EvidenceOppositionCount b-profile-004 3.0)", cluster)
        self.assertIn("(PromotionDomain pe-profile-004 omegaclaw-memory)", cluster)

    def test_isolated_stage_captures_output_and_result(self):
        event = _run_isolated_stage("echo", _echo_profile_stage, ("ok",), stage_timeout_sec=2.0)

        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["label"], "echo")
        self.assertEqual(event["result"], "ok")
        self.assertGreater(event["stdout_chars"], 0)
        self.assertEqual(event["stdout_bytes"], event["stdout_chars"])
        self.assertEqual(
            event["stdout_sha256"],
            hashlib.sha256(b"noisy runtime output\n").hexdigest(),
        )
        self.assertEqual(event["stderr_bytes"], 0)
        self.assertEqual(event["stderr_sha256"], hashlib.sha256(b"").hexdigest())

    def test_captured_stream_provenance_rejects_malformed_or_missing_fields(self):
        valid = {
            "stdout_bytes": 1,
            "stdout_sha256": "a" * 64,
            "stderr_bytes": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        self.assertTrue(profile._captured_stream_provenance_complete(valid))
        for key, value in (
            ("stdout_bytes", True),
            ("stderr_bytes", -1),
            ("stdout_sha256", "A" * 64),
            ("stderr_sha256", "0" * 63),
        ):
            malformed = dict(valid)
            malformed[key] = value
            self.assertFalse(profile._captured_stream_provenance_complete(malformed))

    def test_isolated_stage_timeout_returns_bounded_event(self):
        event = _run_isolated_stage("slow", _slow_profile_stage, (), stage_timeout_sec=0.05)

        self.assertEqual(event["status"], "timeout")
        self.assertEqual(event["label"], "slow")
        self.assertEqual(event["timeout_sec"], 0.05)

    def test_compileadd_probe_call_text_distinguishes_direct_from_eval_control(self):
        statement = "(: p (S x) (STV 1 0.9))"

        direct = profile._compileadd_probe_call_text(statement, "kb", "materialize_stmt_lambdas", "direct")
        eval_control = profile._compileadd_probe_call_text(statement, "kb", "materialize_stmt_lambdas", "eval")

        self.assertEqual(direct, "!(materialize-stmt-lambdas (: p (S x) (STV 1 0.9)))")
        self.assertEqual(eval_control, "!(eval (materialize-stmt-lambdas (: p (S x) (STV 1 0.9))))")

    def test_materialize_identity_match_allows_numeric_rendering_changes(self):
        self.assertTrue(profile._materialize_identity_matches("(STV 0.70 0.55)", ["(STV 0.7 0.55)"]))
        self.assertTrue(
            profile._materialize_identity_matches(
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
                ["(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.7 0.55))"],
            )
        )
        self.assertFalse(profile._materialize_identity_matches("(STV 0.70 0.55)", ["(STV 0.6 0.55)"]))

    def test_materialize_output_summary_bounds_duplicate_fanout(self):
        statement = "(: p (S x) (STV 1 0.9))"
        outputs = [statement] * 20

        summary = profile._summarize_materialize_outputs(
            statement,
            outputs,
            max_output_items=3,
        )

        self.assertTrue(summary["identity_output_present"])
        self.assertEqual(summary["output_count"], 20)
        self.assertEqual(summary["unique_output_count"], 1)
        self.assertEqual(summary["output_items"], [statement] * 3)
        self.assertTrue(summary["output_truncated"])

    def test_materialize_output_summary_rejects_nonpositive_bound(self):
        with self.assertRaises(ValueError):
            profile._summarize_materialize_outputs("(S x)", [], max_output_items=0)

    def test_inspect_pettachainer_add_api_reports_no_precompiled_api(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            package = repo / "pettachainer"
            metta_dir = package / "metta"
            metta_dir.mkdir(parents=True)
            (package / "pettachainer.py").write_text(
                "class PeTTaChainer:\n"
                "    def add_atom(self, atom):\n"
                "        return self.handler.process_metta_string(f'!(compileadd {atom})')\n"
                "    def add_atoms_no_check(self, atoms):\n"
                "        return self.handler.process_metta_string('!(superpose ((compileadd kb a)))')\n"
                "    def query(self, atom):\n"
                "        return []\n",
                encoding="utf-8",
            )
            (metta_dir / "petta_chainer.metta").write_text(
                "(= (compileadd $kb $stmt)\n"
                "  (let* (($stmt1 (materialize-stmt-lambdas $stmt))\n"
                "         ($atoms (collapse (mm2compile $kb $stmt1)))\n"
                "         ($_index (index-source-implication $kb $stmt1))\n"
                "         ($_ (maybe-process-on-add $kb $stmt1)))\n"
                "    $atoms))\n",
                encoding="utf-8",
            )

            summary = profile.inspect_pettachainer_add_api(repo)

        self.assertEqual(summary["public_add_methods"], ["add_atom", "add_atoms_no_check"])
        self.assertFalse(summary["exposes_precompiled_add_api"])
        self.assertEqual(summary["compileadd_definitions"], ["compileadd"])
        self.assertEqual(summary["add_method_compile_calls"]["add_atom"], ["compileadd"])
        self.assertIn("no public precompiled-add API found", summary["recommended_boundary"])

    def test_inspect_materialize_stmt_lambdas_for_statement_marks_lambda_free_identity(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            metta_dir = repo / "pettachainer" / "metta"
            metta_dir.mkdir(parents=True)
            (metta_dir / "petta_chainer.metta").write_text(
                "(= (materialize-stmt-lambdas $term)\n"
                "   (if (is-var $term) $term\n"
                "      (if (is-expr $term)\n"
                "         (if (== (car-atom $term) |->) (eval $term)\n"
                "            (cons (materialize-stmt-lambdas (car-atom $term))\n"
                "               (map-flat materialize-stmt-lambdas (cdr-atom $term))))\n"
                "         $term)))\n",
                encoding="utf-8",
            )

            summary = profile.inspect_materialize_stmt_lambdas_for_statement(
                repo,
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
            )

        self.assertTrue(summary["materialize_expected_identity"])
        self.assertEqual(summary["statement_stats"]["lambda_form_count"], 0)
        self.assertGreater(summary["statement_stats"]["expression_nodes"], 1)
        self.assertIn("eval", summary["definition"]["calls"])
        self.assertIn("runtime success", " ".join(summary["gates"]))
        self.assertEqual(
            summary["next_probe"]["kind"],
            "non-live materialize identity runtime gate",
        )

    def test_run_materialize_identity_gate_uses_source_check_and_isolated_stage(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_inspection(repo, checked_statement):
            captured["repo"] = repo
            captured["checked_statement"] = checked_statement
            return {
                "materialize_expected_identity": True,
                "statement_stats": {"lambda_form_count": 0},
            }

        def fake_isolated_stage(label, target, args, *, stage_timeout_sec):
            captured["label"] = label
            captured["target"] = target
            captured["args"] = args
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "label": label,
                "status": "ok",
                "identity_output_present": True,
                "expected_statement": statement,
            }

        with (
            patch.object(profile, "inspect_materialize_stmt_lambdas_for_statement", side_effect=fake_inspection),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_isolated_stage),
        ):
            result = profile.run_materialize_identity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=4.0,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(captured["repo"], Path("/project/repos/PeTTaChainer"))
        self.assertEqual(captured["checked_statement"], statement)
        self.assertEqual(captured["label"], "materialize_stmt_lambdas_identity")
        self.assertEqual(captured["target"], profile._materialize_identity_stage)
        self.assertEqual(captured["args"], (statement,))
        self.assertEqual(captured["stage_timeout_sec"], 4.0)
        self.assertIn("no mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_run_materialize_identity_gate_skips_lambda_statements(self):
        with patch.object(
            profile,
            "inspect_materialize_stmt_lambdas_for_statement",
            return_value={"materialize_expected_identity": False},
        ), patch.object(profile, "_configure_local_runtime") as configure:
            result = profile.run_materialize_identity_gate(
                "(: p (|-> x x) (STV 1 0.9))",
                project_root=Path("/project"),
                stage_timeout_sec=4.0,
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()
        self.assertIn("statement contains |-> lambda forms", result["reason"])

    def test_run_materialize_identity_ladder_stops_at_first_blocked_rung(self):
        rungs = ["(Requires MemoryTarget0 PLNReadyViews)", "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"]
        seen = []

        def fake_inspection(repo, statement):
            return {"repo_path": str(repo), "statement": statement, "materialize_expected_identity": True}

        def fake_event(statement, *, stage_timeout_sec):
            seen.append((statement, stage_timeout_sec))
            if statement == rungs[0]:
                return {"label": "materialize_stmt_lambdas_identity", "status": "ok", "identity_output_present": True}
            return {"label": "materialize_stmt_lambdas_identity", "status": "timeout", "timeout_sec": stage_timeout_sec}

        with (
            patch.object(profile, "inspect_materialize_stmt_lambdas_for_statement", side_effect=fake_inspection),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_materialize_identity_event", side_effect=fake_event),
        ):
            result = profile.run_materialize_identity_ladder_gate(
                rungs,
                project_root=Path("/project"),
                stage_timeout_sec=2.5,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["first_blocked_rung"], 1)
        self.assertEqual(result["rung_count_executed"], 2)
        self.assertEqual(seen, [(rungs[0], 2.5), (rungs[1], 2.5)])
        self.assertIn("Stop at the first blocked rung", " ".join(result["gates"]))

    def test_materialize_identity_ladder_skips_if_any_rung_has_lambda(self):
        with patch.object(
            profile,
            "inspect_materialize_stmt_lambdas_for_statement",
            side_effect=[
                {"statement": "(Requires MemoryTarget0 PLNReadyViews)", "materialize_expected_identity": True},
                {"statement": "(|-> x x)", "materialize_expected_identity": False},
            ],
        ), patch.object(profile, "_configure_local_runtime") as configure:
            result = profile.run_materialize_identity_ladder_gate(
                ["(Requires MemoryTarget0 PLNReadyViews)", "(|-> x x)"],
                project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()
        self.assertEqual(result["skipped_statements"], ["(|-> x x)"])

    def test_materialize_identity_ladder_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            profile.run_materialize_identity_ladder_gate([], project_root=Path("/project"))

    def test_materialize_proof_shape_rungs_include_prefixes_before_full_proof(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_identity_proof_shape_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(Requires MemoryTarget0 PLNReadyViews)",
                "(STV 0.70 0.55)",
                "(: b-profile-000)",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews))",
                "(: b-profile-000 ProofShapeSentinel (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
                "(: b-profile-000 ProofShapeSentinel (STV 0.70 0.55))",
                statement,
            ],
        )

    def test_materialize_nested_type_proof_rungs_rebuild_type_under_sentinel_tv(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_nested_type_proof_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(: b-profile-000 Requires (STV 1.0 1.0))",
                "(: b-profile-000 (Requires) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 PLNReadyViews) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_nested_type_ladder_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 3,
                "rung_count_executed": 4,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_nested_type_ladder_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas nested-type proof ladder gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["nested_type_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_nested_type_rungs_reject_atom_type(self):
        with self.assertRaises(ValueError):
            profile.materialize_nested_type_proof_rungs("(: p PlainType (STV 1 1))")

    def test_materialize_nested_type_arity_matrix_rungs_test_sentinel_arity_before_original(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_nested_type_arity_matrix_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(: b-profile-000 (Requires) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 PLNReadyViews) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_nested_type_arity_matrix_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 2,
                "rung_count_executed": 3,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_nested_type_arity_matrix_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas nested-type arity matrix gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["nested_type_arity_matrix_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_nested_type_context_matrix_rungs_move_sentinel_type_through_contexts(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_nested_type_context_matrix_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(Requires TypeArgSentinel0 TypeArgSentinel1)",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_nested_type_context_matrix_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 3,
                "rung_count_executed": 4,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_nested_type_context_matrix_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas nested-type context matrix gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["nested_type_context_matrix_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_generic_four_field_context_arity_rungs_order_arity_before_tokens(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_generic_four_field_context_arity_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope b-profile-000 (Requires) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires MemoryTarget0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 PLNReadyViews) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_generic_four_field_context_arity_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 2,
                "rung_count_executed": 3,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_generic_four_field_context_arity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas generic four-field context arity gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["generic_four_field_context_arity_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_nested_position_rungs_move_sentinel_type_by_slot(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_nested_position_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadA PayloadB)",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope PayloadA PayloadB (Requires TypeArgSentinel0 TypeArgSentinel1))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_nested_position_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 0,
                "rung_count_executed": 1,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_nested_position_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field nested-position gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_nested_position_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_neighbor_shape_rungs_add_neighbors_stepwise(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_neighbor_shape_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (TruthValuePayload 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_neighbor_shape_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 4,
                "rung_count_executed": 5,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_neighbor_shape_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field neighbor-shape gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_neighbor_shape_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_right_payload_arity_rungs_vary_generic_before_stv(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_right_payload_arity_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_right_payload_arity_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 3,
                "rung_count_executed": 4,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_right_payload_arity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field right-payload arity gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_right_payload_arity_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_adjacent_nested_arity_rungs_vary_left_nested_type(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_adjacent_nested_arity_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope PayloadA (Requires) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires) (STV 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0) (STV 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_adjacent_nested_arity_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 2,
                "rung_count_executed": 3,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_adjacent_nested_arity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field adjacent-nested arity gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_adjacent_nested_arity_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_nested_type_arity_and_context_matrix_rungs_reject_atom_type(self):
        with self.assertRaises(ValueError):
            profile.materialize_nested_type_arity_matrix_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_nested_type_context_matrix_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_generic_four_field_context_arity_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_nested_position_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_neighbor_shape_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_right_payload_arity_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_adjacent_nested_arity_rungs("(: p PlainType (STV 1 1))")

    def test_materialize_proof_shape_ladder_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 4,
                "rung_count_executed": 5,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_proof_shape_ladder_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas proof-shape ladder gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["proof_shape_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_proof_shape_rungs_reject_non_proof_atom(self):
        with self.assertRaises(ValueError):
            profile.materialize_identity_proof_shape_rungs("(Requires MemoryTarget0 PLNReadyViews)")

    def test_inspect_compileadd_bottleneck_sources_records_target_definitions(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            metta_dir = repo / "pettachainer" / "metta"
            chainer_dir = metta_dir / "chainer"
            chainer_dir.mkdir(parents=True)
            (metta_dir / "petta_chainer.metta").write_text(
                "!(import! &self chainer/compile)\n"
                "!(import! &self chainer/mining)\n"
                "(= (materialize-stmt-lambdas $term)\n"
                "   (if (is-var $term) $term (materialize-stmt-lambdas (car-atom $term))))\n"
                "(= (compileadd $kb $stmt)\n"
                "   (let* (($stmt1 (materialize-stmt-lambdas $stmt))\n"
                "          ($atoms (collapse (mm2compile $kb $stmt1))))\n"
                "      $atoms))\n"
                "(= (compileadd-mine $kb $stmt) (compileadd $kb $stmt))\n",
                encoding="utf-8",
            )
            (chainer_dir / "compile.metta").write_text(
                "(= (index-source-implication $kb $stmt) ())\n"
                "(= (compile $kb $stmt) (((() |- ($stmt)) ())))\n"
                "(= (mm2compile $kb $stmt)\n"
                "   (progn (remove-all-atoms ctx) (superpose ((mm2stmt (compile $kb $stmt)) (get-atoms ctx)))))\n",
                encoding="utf-8",
            )
            (chainer_dir / "mining.metta").write_text(
                "(= (maybe-process-on-add $kb $stmt) ())\n",
                encoding="utf-8",
            )

            summary = profile.inspect_compileadd_bottleneck_sources(repo)

        self.assertIn("chainer/compile", summary["root_imports"])
        self.assertEqual(
            summary["definitions"]["materialize-stmt-lambdas"]["file"],
            "pettachainer/metta/petta_chainer.metta",
        )
        self.assertTrue(summary["definitions"]["materialize-stmt-lambdas"]["recursive"])
        self.assertIn("compile", summary["definitions"]["mm2compile"]["calls"])
        self.assertEqual(
            [target["symbol"] for target in summary["next_instrumentation_targets"]],
            ["materialize-stmt-lambdas", "mm2compile", "compile_"],
        )
        self.assertIn("no compileadd/query/runtime execution", " ".join(summary["gates"]))

    def test_inspect_compile_dispatch_for_statement_maps_promoted_belief_to_fact_branch(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            chainer_dir = repo / "pettachainer" / "metta" / "chainer"
            chainer_dir.mkdir(parents=True)
            (chainer_dir / "compile.metta").write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv)))\n"
                "   (if (is-var $Type) (empty)\n"
                "      (if (= $Type (Implication (cons Premises $premises) (cons Conclusions $conclusions)))\n"
                "         (compile-implication-forward-rules $kb $prf $premises $conclusions)\n"
                "         (if (bidirectional-implication-type? $Type)\n"
                "            (compile_ $kb (: (bi-forward $prf) (Implication $left $right) $tv))\n"
                "            (let $fact-kb (compile-fact-kb $kb)\n"
                "               (superpose ((() |- ((: $fact-kb $prf $Type $tv)))\n"
                "                           (compile-outputs (: $fact-kb $prf $Type $tv)))))))))\n",
                encoding="utf-8",
            )
            (chainer_dir / "logic_config.metta").write_text(
                "!(set-bidirectional-implication-form BiImplication)\n",
                encoding="utf-8",
            )

            summary = profile.inspect_compile_dispatch_for_statement(
                repo,
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
            )

        self.assertEqual(summary["parsed_statement"]["type_head"], "Requires")
        self.assertEqual(summary["selected_compile_branch"], "fact-assertion")
        self.assertIn("concrete non-Implication", summary["reason"])
        self.assertIn("compile-outputs", summary["compile_definition"]["calls"])
        self.assertEqual(summary["configured_bidirectional_heads"], ["BiImplication"])
        self.assertIn("no PeTTaChainer runtime", " ".join(summary["gates"]))

    def test_inspect_compile_dispatch_for_statement_rejects_non_proof_atom(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            chainer_dir = repo / "pettachainer" / "metta" / "chainer"
            chainer_dir.mkdir(parents=True)
            (chainer_dir / "compile.metta").write_text("(= (compile_ $kb $stmt) ())\n", encoding="utf-8")
            (chainer_dir / "logic_config.metta").write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                profile.inspect_compile_dispatch_for_statement(repo, "(Requires MemoryTarget0 PLNReadyViews)")

    def test_run_compile_dispatch_gate_isolates_fact_branch(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(
                label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec,
            )
            return {"label": label, "status": "ok", "output_count": 1, "unique_output_count": 1}

        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_dispatch_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_dispatch")
        self.assertEqual(captured["target"], profile._compile_dispatch_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertTrue(result["boundaries"]["no_mm2compile_or_compileadd"])

    def test_run_compile_dispatch_gate_skips_non_fact_branch(self):
        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "implication-rule"},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_dispatch_gate(
                "(: p (Implication (cons Premises ()) (cons Conclusions ())) (STV 1 0.9))",
                project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_dispatch_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_compile_dispatch_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), max_output_items=0,
            )

    def test_inspect_compile_wrapper_shape_closes_exact_delegation(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text("(= (compile $kb $stmt) (compile_ $kb $stmt))\n", encoding="utf-8")

            result = profile.inspect_compile_wrapper_shape(repo)

        self.assertTrue(result["shape_confirmed"])
        self.assertIn("delegates directly", result["interpretation"])

    def test_inspect_compile_wrapper_shape_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text("(= (compile $kb $stmt) (collapse (compile_ $kb $stmt)))\n", encoding="utf-8")

            result = profile.inspect_compile_wrapper_shape(repo)

        self.assertFalse(result["shape_confirmed"])
        self.assertIn("do not compare", result["interpretation"])

    def test_run_compile_wrapper_direct_gate_compares_two_rungs(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            rung = {"output_count": 256, "unique_output_count": 1}
            return {"status": "ok", "public_compile": rung, "direct_compile_": rung}

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_wrapper_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_wrapper_direct_gate(
                statement, project_root=Path("/project"), stage_timeout_sec=3.0, max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_wrapper_direct")
        self.assertEqual(captured["target"], profile._compile_wrapper_direct_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertTrue(result["boundaries"]["no_mm2compile_or_compileadd"])

    def test_run_compile_wrapper_direct_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_wrapper_shape", return_value={"shape_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_wrapper_direct_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_inspect_compile_fact_branch_shape_closes_copied_source(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv)))\n"
                "   (let $fact-kb (compile-fact-kb $kb)\n"
                "      (superpose ((() |- ((: $fact-kb $prf $Type $tv)))\n"
                "                  (compile-outputs (: $fact-kb $prf $Type $tv))))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_compile_fact_branch_shape(repo)

        self.assertTrue(result["shape_confirmed"])
        self.assertIn("superposes", result["interpretation"])

    def test_inspect_compile_fact_branch_shape_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv))) (compile-fact-kb $kb))\n",
                encoding="utf-8",
            )

            result = profile.inspect_compile_fact_branch_shape(repo)

        self.assertFalse(result["shape_confirmed"])
        self.assertIn("do not run", result["interpretation"])

    def test_run_compile_fact_branch_component_gate_isolates_components(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {"status": "ok", "components_admitted": True}

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_fact_branch_component_gate(
                statement, project_root=Path("/project"), stage_timeout_sec=3.0, max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_branch_components")
        self.assertEqual(captured["target"], profile._compile_fact_branch_component_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_compile_fact_branch_component_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_fact_branch_component_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_fact_literal_kb_gate_measures_three_rungs(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            rung = {"output_count": 32, "unique_output_count": 1}
            return {
                "status": "ok",
                "base_clause_superpose": rung,
                "literal_kb_with_empty_arm": rung,
                "literal_kb_fact_branch": rung,
            }

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_fact_literal_kb_gate(
                statement, project_root=Path("/project"), stage_timeout_sec=3.0, max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_literal_kb_ladder")
        self.assertEqual(captured["target"], profile._compile_fact_literal_kb_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertTrue(result["boundaries"]["literal_kb_substitution_only"])
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_compile_fact_literal_kb_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_fact_literal_kb_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_fact_literal_kb_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_compile_fact_literal_kb_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), stage_timeout_sec=0,
            )

    def test_inspect_compile_fact_dispatch_ladder_shape_closes_nested_gates(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv)))\n"
                " (if (is-var $Type) (empty)\n"
                "  (if (= $Type (Implication (cons Premises $premises) (cons Conclusions $conclusions))) (empty)\n"
                "   (if (bidirectional-implication-type? $Type) (empty)\n"
                "    (let $fact-kb (compile-fact-kb $kb)\n"
                "     (superpose ((() |- ((: $fact-kb $prf $Type $tv)))\n"
                "                 (compile-outputs (: $fact-kb $prf $Type $tv)))))))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_compile_fact_dispatch_ladder_shape(repo)

        self.assertTrue(result["shape_confirmed"])
        self.assertTrue(all(result["matched_shapes"].values()))

    def test_run_compile_fact_dispatch_ladder_gate_measures_four_rungs(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            rung = {"output_count": 2, "unique_output_count": 1}
            return {
                "status": "ok",
                "literal_fact_branch": rung,
                "with_bidirectional_gate": rung,
                "with_implication_gate": rung,
                "with_variable_type_gate": rung,
            }

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_dispatch_ladder_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_fact_dispatch_ladder_gate(
                statement, project_root=Path("/project"), stage_timeout_sec=3.0, max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_dispatch_ladder")
        self.assertEqual(captured["target"], profile._compile_fact_dispatch_ladder_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_compile_fact_dispatch_ladder_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_dispatch_ladder_shape", return_value={"shape_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_fact_dispatch_ladder_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_inspect_compile_import_multiplicity_closes_two_paths(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            metta = repo / "pettachainer" / "metta"
            (metta / "context").mkdir(parents=True)
            (metta / "petta_chainer.metta").write_text(
                "!(import! &self chainer/compile)\n!(import! &self context/context_from_kb)\n",
                encoding="utf-8",
            )
            (metta / "context" / "context_from_kb.metta").write_text(
                "!(import! &self context/context_generation)\n", encoding="utf-8",
            )
            (metta / "context" / "context_generation.metta").write_text(
                "!(import! &self chainer/compile)\n", encoding="utf-8",
            )

            result = profile.inspect_compile_import_multiplicity(repo)

        self.assertTrue(result["two_compile_import_paths_confirmed"])
        self.assertTrue(all(result["matched_imports"].values()))

    def test_inspect_duplicate_compile_import_repair_requires_exact_one_line_removal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = root / "baseline"
            candidate = root / "candidate"
            relative_sources = {
                "pettachainer/metta/petta_chainer.metta": "!(import! &self chainer/compile)\nroot\n",
                "pettachainer/metta/context/context_from_kb.metta": "context\n",
                "pettachainer/metta/context/context_generation.metta": "before\n!(import! &self chainer/compile)\nafter\n",
                "pettachainer/metta/chainer/compile.metta": "(= (compile_ $kb $stmt) $stmt)\n",
            }
            for relative, text in relative_sources.items():
                for repo in (baseline, candidate):
                    path = repo / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
            generation = candidate / "pettachainer/metta/context/context_generation.metta"
            generation.write_text("before\nafter\n", encoding="utf-8")

            result = profile.inspect_duplicate_compile_import_repair(baseline, candidate)

        self.assertTrue(result["exact_targeted_import_removal"])
        self.assertEqual(result["target_import_occurrences"], 1)
        self.assertNotEqual(
            result["baseline_hashes"]["pettachainer/metta/context/context_generation.metta"],
            result["candidate_hashes"]["pettachainer/metta/context/context_generation.metta"],
        )

    def test_inspect_duplicate_compile_import_repair_rejects_adjacent_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = root / "baseline"
            candidate = root / "candidate"
            for repo in (baseline, candidate):
                for relative, text in {
                    "pettachainer/metta/petta_chainer.metta": "root\n",
                    "pettachainer/metta/context/context_from_kb.metta": "context\n",
                    "pettachainer/metta/context/context_generation.metta": "!(import! &self chainer/compile)\nbody\n",
                    "pettachainer/metta/chainer/compile.metta": "compiler\n",
                }.items():
                    path = repo / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
            (candidate / "pettachainer/metta/context/context_generation.metta").write_text(
                "changed-body\n", encoding="utf-8",
            )

            result = profile.inspect_duplicate_compile_import_repair(baseline, candidate)

        self.assertFalse(result["exact_targeted_import_removal"])

    def test_run_duplicate_compile_import_repair_gate_admits_one_output_candidate(self):
        inspection = {"exact_targeted_import_removal": True}
        events = (
            {"status": "ok", "output_count": 128, "unique_output_count": 1, "normalized_output_items": ["fact"]},
            {"status": "ok", "output_count": 1, "unique_output_count": 1, "normalized_output_items": ["fact"]},
        )
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value=inspection),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=events) as run_stage,
        ):
            result = profile.run_duplicate_compile_import_repair_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(run_stage.call_count, 2)
        self.assertTrue(result["boundaries"]["no_mm2compile_or_compileadd_or_query"])

    def test_run_repaired_compile_wrapper_direct_gate_remeasures_exact_candidate(self):
        events = {
            "status": "ok",
            "public_compile": {"output_count": 2, "unique_output_count": 1},
            "direct_compile_": {"output_count": 1, "unique_output_count": 1},
        }
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, timeout=stage_timeout_sec)
            return events

        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_wrapper_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_repaired_compile_wrapper_direct_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "repaired_compile_wrapper_direct")
        self.assertEqual(captured["target"], profile._compile_wrapper_direct_from_repo_stage)
        self.assertEqual(captured["args"], ("(: p (S x) (STV 1 0.9))", "/candidate", 4))
        self.assertTrue(result["boundaries"]["no_mm2compile_or_compileadd_or_query"])

    def test_run_repaired_compile_wrapper_direct_gate_skips_inexact_candidate(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": False}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_wrapper_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_repaired_compile_wrapper_direct_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["runtime_event"])
        configure.assert_not_called()

    def test_run_repaired_fact_conversion_collection_gate_measures_two_rungs(self):
        events = (
            {"status": "ok", "expected_fact_present": True, "converted_count": 1},
            {"status": "ok", "expected_fact_present": True, "output_count": 1},
        )
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2stmt_fact_case_overlap", return_value={"overlap_confirmed": True}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=events) as run_stage,
        ):
            result = profile.run_repaired_fact_conversion_collection_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(run_stage.call_count, 2)
        self.assertIs(run_stage.call_args_list[0].args[1], profile._mm2stmt_deduplicated_fact_from_repo_stage)
        self.assertIs(run_stage.call_args_list[1].args[1], profile._mm2compile_deduplicated_fact_from_repo_stage)
        self.assertTrue(result["boundaries"]["no_compile_or_compileadd_or_query"])

    def test_run_repaired_fact_conversion_collection_gate_stops_after_blocked_conversion(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2stmt_fact_case_overlap", return_value={"overlap_confirmed": True}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value={"status": "timeout"}) as run_stage,
        ):
            result = profile.run_repaired_fact_conversion_collection_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(run_stage.call_count, 1)
        self.assertEqual(len(result["runtime_events"]), 1)

    def test_run_repaired_full_mm2compile_gate_runs_real_entry_point(self):
        inspection = {
            "exact_targeted_import_removal": True,
        }
        event = {
            "status": "ok",
            "expected_fact_present": True,
            "output_count": 1,
        }
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value=inspection),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=event) as run_stage,
        ):
            result = profile.run_repaired_full_mm2compile_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertIs(run_stage.call_args.args[1], profile._full_mm2compile_from_repo_stage)
        self.assertEqual(run_stage.call_args.args[2], (
            "(: p (S x) (STV 1 0.9))", "/candidate", 4,
        ))
        self.assertTrue(result["boundaries"]["full_compile_conversion_collection"])
        self.assertTrue(result["boundaries"]["no_compileadd_or_query_or_result_admission"])

    def test_run_repaired_full_mm2compile_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": False}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_repaired_full_mm2compile_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["runtime_event"])
        configure.assert_not_called()

    def test_run_repaired_compileadd_add_only_gate_checks_exact_stored_fact(self):
        event = {
            "status": "ok",
            "expected_external_present": True,
            "expected_internal_stored": True,
            "stored_match_count": 1,
        }
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=event) as run_stage,
        ):
            result = profile.run_repaired_compileadd_add_only_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertIs(run_stage.call_args.args[1], profile._compileadd_add_only_from_repo_stage)
        self.assertEqual(run_stage.call_args.args[2], (
            "(: p (S x) (STV 1 0.9))", "/candidate", 4,
        ))
        self.assertTrue(result["boundaries"]["single_compileadd_only"])
        self.assertTrue(result["boundaries"]["no_query_or_result_admission"])

    def test_run_repaired_compileadd_add_only_gate_blocks_unstored_output(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value={
                "status": "ok",
                "expected_external_present": True,
                "expected_internal_stored": False,
                "stored_match_count": 0,
            }),
        ):
            result = profile.run_repaired_compileadd_add_only_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "blocked")

    def test_run_repaired_compileadd_add_only_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": False}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_repaired_compileadd_add_only_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["runtime_event"])
        configure.assert_not_called()

    def test_run_repaired_compileadd_exact_fact_query_gate_requires_exact_answer(self):
        event = {
            "status": "ok",
            "stdout_bytes": 1,
            "stdout_sha256": "a" * 64,
            "stderr_bytes": 0,
            "stderr_sha256": "b" * 64,
            "expected_internal_stored": True,
            "exact_answer_present": True,
            "exact_answer_only": True,
            "query_answer_count": 1,
        }
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=event) as run_stage,
        ):
            result = profile.run_repaired_compileadd_exact_fact_query_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                stage_timeout_sec=3.0,
                query_steps=2,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertIs(run_stage.call_args.args[1], profile._compileadd_exact_fact_query_from_repo_stage)
        self.assertEqual(run_stage.call_args.args[2], (
            "(: p (S x) (STV 1 0.9))", "/candidate", 2, 4,
        ))
        self.assertTrue(result["boundaries"]["single_compileadd_then_exact_fact_query"])
        self.assertTrue(result["boundaries"]["content_addressed_process_streams"])
        self.assertTrue(result["boundaries"]["no_inferred_result_promotion_or_memory_write"])

    def test_repaired_episode_contract_gate_binds_typed_contract_to_exact_recall(self):
        validation = {
            "status": "ok",
            "statement_results": [1.0],
            "query_result": 1.0,
            "stdout_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        runtime = {"status": "completed"}
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=validation),
            patch.object(profile, "run_repaired_compileadd_exact_fact_query_gate", return_value=runtime) as run_gate,
        ):
            result = profile.run_repaired_pettachainer_episode_contract_gate(
                self.episode_contract(),
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
                query_steps=2,
            )

        self.assertTrue(result["validators_admitted"])
        self.assertTrue(result["runtime_admitted"])
        self.assertEqual(result["result_classification"], "stored-fact-retrieval")
        self.assertEqual(result["statement_digest"], "a" * 64)
        self.assertEqual(run_gate.call_args.args[0], self.episode_contract().statements[0].atom)
        self.assertTrue(result["boundaries"]["not_a_derived_pln_result"])
        self.assertTrue(result["boundaries"]["no_episode_manifest"])

    def test_repaired_episode_contract_gate_stops_on_validator_drift(self):
        validation = {
            "status": "ok",
            "statement_results": [True],
            "query_result": 1.0,
            "stdout_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=validation),
            patch.object(profile, "run_repaired_compileadd_exact_fact_query_gate") as run_gate,
        ):
            result = profile.run_repaired_pettachainer_episode_contract_gate(
                self.episode_contract(),
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertFalse(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])
        self.assertIsNone(result["runtime_gate"])
        run_gate.assert_not_called()

    def test_repaired_episode_contract_gate_rejects_nonexact_query(self):
        contract = self.episode_contract()
        mismatched = PeTTaChainerEpisodeContract(
            episode_id=contract.episode_id,
            chart_fingerprint=contract.chart_fingerprint,
            statements=contract.statements,
            query_term="(S b)",
            query_atom="(: $prf (S b) $tv)",
        )
        with self.assertRaisesRegex(ValueError, "exact stored-fact query"):
            profile.run_repaired_pettachainer_episode_contract_gate(
                mismatched,
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

    def test_repaired_rule_contract_gate_binds_compiler_provenance(self):
        contract = self.rule_episode_contract()
        with patch.object(
            profile, "run_repaired_pettachainer_one_rule_derivation_gate",
            return_value={"status": "completed"},
        ) as run_gate:
            result = profile.run_repaired_pettachainer_rule_episode_contract_gate(
                contract, project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"), query_steps=7,
            )

        fact = contract.statements[1]
        rule = contract.statements[0]
        self.assertTrue(result["runtime_admitted"])
        self.assertEqual(result["result_classification"], "compiler-bound-one-rule-derived-result")
        self.assertEqual(result["fact_statement_digest"], fact.sentence_digest)
        self.assertEqual(result["rule_statement_digest"], rule.sentence_digest)
        self.assertEqual(result["fact_evidence_basis_ids"], ["basis-fact"])
        self.assertEqual(result["rule_stamps"], [1])
        self.assertEqual(
            result["expected_derived_proof"],
            f"(rule-proof {rule.proof_id} {fact.proof_id})",
        )
        self.assertEqual(run_gate.call_args.args[:3], (fact.atom, rule.atom, "(T a)"))
        self.assertEqual(run_gate.call_args.kwargs["query_steps"], 7)
        self.assertTrue(result["boundaries"]["no_episode_manifest"])

    def test_repaired_rule_contract_capture_closes_result_and_stream_provenance(self):
        contract = self.rule_episode_contract()
        fact, rule = contract.statements[1], contract.statements[0]
        proof = f"(rule-proof {rule.proof_id} {fact.proof_id})"
        validation = {
            "label": "validate_repaired_one_rule_derivation", "seconds": 0.1,
            "stdout_bytes": 0, "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0, "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        runtime = {
            "label": "repaired_one_rule_derivation", "seconds": 0.4,
            "stdout_bytes": 120, "stdout_sha256": "d" * 64,
            "stderr_bytes": 12, "stderr_sha256": "e" * 64,
            "query_unique_answer_count": 1,
            "query_answer_items": [f"(: {proof} (T a) (STV 0.7600000000000001 0.52))"],
            "expected_truth_value": [0.7600000000000001, 0.52],
        }
        gate = {
            "schema": "petta-memory-repaired-pettachainer-rule-contract-gate-v1",
            "episode_id": contract.episode_id,
            "chart_fingerprint": contract.chart_fingerprint,
            "query_term": contract.query_term,
            "fact_statement_digest": fact.sentence_digest,
            "rule_statement_digest": rule.sentence_digest,
            "fact_proof_id": fact.proof_id,
            "rule_proof_id": rule.proof_id,
            "fact_stamps": list(fact.stamp_ints),
            "rule_stamps": list(rule.stamp_ints),
            "fact_evidence_basis_ids": list(fact.evidence_basis_ids),
            "rule_evidence_basis_ids": list(rule.evidence_basis_ids),
            "runtime_admitted": True,
            "result_classification": "compiler-bound-one-rule-derived-result",
            "runtime_gate": {
                "status": "completed", "validation_event": validation,
                "runtime_event": runtime,
            },
        }

        capture = profile.build_repaired_pettachainer_rule_episode_capture(contract, gate)

        self.assertEqual(capture.derived_proof, proof)
        self.assertEqual(capture.fact_evidence_basis_ids, ("basis-fact",))
        self.assertEqual(capture.runtime_capture.stdout_bytes, 120)
        self.assertRegex(capture.runtime_capture.capture_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(capture.result_digest, r"^[0-9a-f]{64}$")
        builder_values = {
            "episode_id": capture.episode_id,
            "chart_fingerprint": capture.chart_fingerprint,
            "fact": fact,
            "rule": rule,
            "query_term": capture.query_term,
            "derived_atom": capture.derived_atom,
            "derived_proof": capture.derived_proof,
            "strength": capture.strength,
            "confidence": capture.confidence,
            "validator_capture": capture.validator_capture,
            "runtime_capture": capture.runtime_capture,
        }
        for field, message in (
            ("fact", "fact must be an immutable PeTTaChainer input statement"),
            ("rule", "rule must be an immutable PeTTaChainer input statement"),
            ("validator_capture", "validator_capture must be a typed PeTTaChainer stage capture"),
            ("runtime_capture", "runtime_capture must be a typed PeTTaChainer stage capture"),
        ):
            with self.subTest(malformed_builder_dependency=field):
                with self.assertRaisesRegex(ValueError, message):
                    pipln_models.build_pettachainer_derived_result_capture(
                        **(builder_values | {field: None})
                    )
        malformed_capture_values = {
            field: getattr(capture, field)
            for field in capture.__dataclass_fields__
            if field != "result_digest"
        } | {"fact_evidence_basis_ids": ("basis-fact", "basis-z")}
        malformed_capture_payload = {
            field: value
            for field, value in malformed_capture_values.items()
            if field not in {"validator_capture", "runtime_capture"}
        } | {
            "validator_capture_digest": capture.validator_capture.capture_digest,
            "runtime_capture_digest": capture.runtime_capture.capture_digest,
        }
        with self.assertRaisesRegex(ValueError, "fact evidence bases must close every stamp"):
            pipln_models.PeTTaChainerDerivedResultCapture(
                **malformed_capture_values,
                result_digest=pipln_models._canonical_hash(malformed_capture_payload),
            )
        list_capture_values = malformed_capture_values | {
            "fact_evidence_basis_ids": list(capture.fact_evidence_basis_ids),
        }
        list_capture_payload = malformed_capture_payload | {
            "fact_evidence_basis_ids": list(capture.fact_evidence_basis_ids),
        }
        with self.assertRaisesRegex(
            ValueError, "fact evidence bases must be a non-empty sorted unique tuple"
        ):
            pipln_models.PeTTaChainerDerivedResultCapture(
                **list_capture_values,
                result_digest=pipln_models._canonical_hash(list_capture_payload),
            )
        for field, label, message in (
            (
                "validator_capture",
                capture.runtime_capture.label,
                "validator capture has the wrong stage label",
            ),
            (
                "runtime_capture",
                capture.validator_capture.label,
                "runtime capture has the wrong stage label",
            ),
        ):
            original = getattr(capture, field)
            forged_capture = pipln_models.build_pettachainer_stage_capture(
                label=label,
                elapsed_seconds=original.elapsed_seconds,
                stdout_bytes=original.stdout_bytes,
                stdout_sha256=original.stdout_sha256,
                stderr_bytes=original.stderr_bytes,
                stderr_sha256=original.stderr_sha256,
            )
            forged = malformed_capture_values | {
                "fact_evidence_basis_ids": capture.fact_evidence_basis_ids,
                field: forged_capture,
            }
            payload = {
                key: value
                for key, value in forged.items()
                if key not in {"validator_capture", "runtime_capture"}
            } | {
                "validator_capture_digest": forged["validator_capture"].capture_digest,
                "runtime_capture_digest": forged["runtime_capture"].capture_digest,
            }
            with self.assertRaisesRegex(ValueError, message):
                pipln_models.PeTTaChainerDerivedResultCapture(
                    **forged,
                    result_digest=pipln_models._canonical_hash(payload),
                )
        same_sentence_capture = malformed_capture_values | {
            "fact_evidence_basis_ids": capture.fact_evidence_basis_ids,
            "rule_sentence_digest": capture.fact_sentence_digest,
            "rule_proof_id": capture.fact_proof_id,
        }
        same_sentence_payload = {
            key: value
            for key, value in same_sentence_capture.items()
            if key not in {"validator_capture", "runtime_capture"}
        } | {
            "validator_capture_digest": capture.validator_capture.capture_digest,
            "runtime_capture_digest": capture.runtime_capture.capture_digest,
        }
        with self.assertRaisesRegex(ValueError, "fact and rule sentence digests must be distinct"):
            pipln_models.PeTTaChainerDerivedResultCapture(
                **same_sentence_capture,
                result_digest=pipln_models._canonical_hash(same_sentence_payload),
            )
        for field, malformed, message in (
            ("rule_stamp_ints", capture.fact_stamp_ints, "fact and rule stamps must be disjoint"),
            (
                "rule_evidence_basis_ids",
                capture.fact_evidence_basis_ids,
                "fact and rule evidence bases must be disjoint",
            ),
        ):
            forged = malformed_capture_values | {
                "fact_evidence_basis_ids": capture.fact_evidence_basis_ids,
                field: malformed,
            }
            payload = {
                key: value
                for key, value in forged.items()
                if key not in {"validator_capture", "runtime_capture"}
            } | {
                "validator_capture_digest": capture.validator_capture.capture_digest,
                "runtime_capture_digest": capture.runtime_capture.capture_digest,
            }
            with self.assertRaisesRegex(ValueError, message):
                pipln_models.PeTTaChainerDerivedResultCapture(
                    **forged,
                    result_digest=pipln_models._canonical_hash(payload),
                )
        attribution = build_pettachainer_rule_attribution(capture)
        self.assertEqual(attribution.inference_rule, "TotalMP")
        self.assertEqual(attribution.rule_proof_id, rule.proof_id)
        self.assertEqual(attribution.rule_evidence_basis_ids, ("basis-rule",))
        self.assertEqual(attribution.fact_proof_id, fact.proof_id)
        self.assertEqual(attribution.fact_evidence_basis_ids, ("basis-fact",))
        self.assertEqual(attribution.attribution_kind, "compiler-bound-single-rule")
        self.assertFalse(attribution.runtime_trace_decoded)
        self.assertRegex(attribution.attribution_digest, r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(ValueError, "decoded runtime trace"):
            replace(attribution, runtime_trace_decoded=True)
        with self.assertRaisesRegex(ValueError, "rule proof id"):
            replace(attribution, rule_proof_id=fact.proof_id)
        attribution_values = {
            field: getattr(attribution, field)
            for field in attribution.__dataclass_fields__
            if field != "attribution_digest"
        }
        same_sentence_attribution = attribution_values | {
            "rule_sentence_digest": attribution.fact_sentence_digest,
            "rule_proof_id": attribution.fact_proof_id,
        }
        same_sentence_attribution["attribution_digest"] = pipln_models._canonical_hash(
            same_sentence_attribution
        )
        with self.assertRaisesRegex(
            ValueError, "fact and rule attribution sentence digests must be distinct"
        ):
            pipln_models.PeTTaChainerRuleAttribution(**same_sentence_attribution)
        for field, malformed, message in (
            ("rule_stamp_ints", (), "rule attribution stamps"),
            ("rule_stamp_ints", (2, 1), "rule attribution stamps"),
            ("fact_stamp_ints", (True,), "fact attribution stamps"),
            ("rule_evidence_basis_ids", ("basis-rule", "basis-rule"), "rule attribution evidence bases"),
            ("fact_evidence_basis_ids", (" ",), "fact attribution evidence bases"),
            ("fact_evidence_basis_ids", ("basis-fact", "basis-z"), "fact attribution evidence bases must close every stamp"),
            ("rule_stamp_ints", attribution.fact_stamp_ints, "fact and rule attribution stamps must be disjoint"),
            (
                "rule_evidence_basis_ids",
                attribution.fact_evidence_basis_ids,
                "fact and rule attribution evidence bases must be disjoint",
            ),
        ):
            forged = attribution_values | {field: malformed}
            forged["attribution_digest"] = pipln_models._canonical_hash(forged)
            with self.assertRaisesRegex(ValueError, message):
                pipln_models.PeTTaChainerRuleAttribution(**forged)

        manifest_kwargs = {
            "contract": contract,
            "result": capture,
            "attribution": attribution,
            "kernel_name": "PeTTaChainer",
            "kernel_version": "e4db5ca+single-import-repair",
            "kernel_capabilities_cid": "1" * 64,
            "repair_profile_id": "single-import-compile-repair-v1",
            "repaired_source_cid": "2" * 64,
            "controller_envelope_cid": "3" * 64,
            "seed": 0,
            "budget": EpisodeBudget(max_steps=5, max_runtime_ms=5000, max_output_chars=1_000_000),
            "started_at": "2026-07-18T07:00:00-07:00",
            "finished_at": "2026-07-18T07:00:01-07:00",
        }
        manifest = build_pettachainer_episode_manifest(**manifest_kwargs)
        self.assertEqual(manifest.result_cid, capture.result_digest)
        self.assertEqual(manifest.attribution_cid, attribution.attribution_digest)
        self.assertEqual(manifest.runtime_capture_cid, capture.runtime_capture.capture_digest)
        self.assertFalse(manifest.promotion_authorized)
        self.assertEqual(
            build_pettachainer_episode_manifest(**manifest_kwargs).manifest_digest,
            manifest.manifest_digest,
        )
        with self.assertRaisesRegex(
            ValueError, "budget must be an immutable episode budget"
        ):
            build_pettachainer_episode_manifest(**(manifest_kwargs | {"budget": None}))
        with self.assertRaisesRegex(ValueError, "cannot authorize promotion"):
            replace(manifest, promotion_authorized=True)
        with self.assertRaisesRegex(
            ValueError, "typed PeTTaChainer episode manifest",
        ):
            pipln_models.pettachainer_episode_manifest_document(None)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "derived-result.json"
            manifest_path = Path(directory) / "episode-manifest.json"
            attribution_path = Path(directory) / "rule-attribution.json"
            malformed_attribution_parent = Path(directory) / "attribution-must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "typed attribution",
            ):
                write_pettachainer_rule_attribution(
                    malformed_attribution_parent / "attribution.json", None,
                )
            self.assertFalse(malformed_attribution_parent.exists())
            malformed_capture_parent = Path(directory) / "capture-must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "typed PeTTaChainer capture",
            ):
                write_pettachainer_derived_result_capture(
                    malformed_capture_parent / "capture.json", None,
                )
            self.assertFalse(malformed_capture_parent.exists())
            malformed_parent = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "typed PeTTaChainer episode manifest",
            ):
                write_pettachainer_episode_manifest(
                    malformed_parent / "manifest.json", None,
                )
            self.assertFalse(malformed_parent.exists())
            write_pettachainer_rule_attribution(attribution_path, attribution)
            self.assertEqual(
                read_pettachainer_rule_attribution(
                    attribution_path, result=capture,
                ),
                attribution,
            )
            with self.assertRaises(FileExistsError):
                write_pettachainer_rule_attribution(attribution_path, attribution)
            changed_gate = json.loads(json.dumps(gate))
            changed_gate["runtime_gate"]["runtime_event"]["query_answer_items"] = [
                f"(: {proof} (T a) (STV 0.5 0.52))"
            ]
            changed_gate["runtime_gate"]["runtime_event"]["expected_truth_value"] = [
                0.5, 0.52,
            ]
            changed_capture = profile.build_repaired_pettachainer_rule_episode_capture(
                contract, changed_gate,
            )
            with self.assertRaisesRegex(ValueError, "does not match derived result"):
                read_pettachainer_rule_attribution(
                    attribution_path,
                    result=changed_capture,
                )
            original_attribution_text = attribution_path.read_text(encoding="utf-8")
            attribution_document = json.loads(original_attribution_text)
            attribution_payload = attribution_document["payload"]
            attribution_payload["result_digest"] = changed_capture.result_digest
            attribution_digest_payload = {
                key: value
                for key, value in attribution_payload.items()
                if key != "attribution_digest"
            }
            attribution_payload["attribution_digest"] = pipln_models._canonical_hash(
                attribution_digest_payload
            )
            attribution_document["document_digest"] = pipln_models._canonical_hash(
                attribution_payload
            )
            attribution_path.write_text(
                json.dumps(attribution_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match derived result"):
                read_pettachainer_rule_attribution(
                    attribution_path,
                    result=capture,
                )
            attribution_path.write_text(original_attribution_text, encoding="utf-8")
            with (patch("petta_memory.pipln_models.os.fsync", wraps=os.fsync) as fsync,
                  patch("petta_memory.pipln_models.os.open", wraps=os.open) as open_file):
                write_pettachainer_episode_manifest(manifest_path, manifest)
            self.assertEqual(fsync.call_count, 2)
            self.assertEqual(open_file.call_args_list[0].args[0], manifest_path.parent)
            self.assertEqual(open_file.call_args_list[1].args[0], manifest_path.name)
            self.assertIsInstance(open_file.call_args_list[1].kwargs["dir_fd"], int)
            self.assertEqual(
                read_pettachainer_episode_manifest(
                    manifest_path, contract=contract, result=capture,
                    attribution=attribution,
                ),
                manifest,
            )
            missing_manifest_path = manifest_path.with_name("missing-manifest.json")
            malformed_dependencies = (
                ("contract", "immutable PeTTaChainer episode contract"),
                ("result", "typed PeTTaChainer derived capture"),
                ("attribution", "close against the PeTTaChainer derived capture"),
            )
            valid_dependencies = {
                "contract": contract,
                "result": capture,
                "attribution": attribution,
            }
            for field, message in malformed_dependencies:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                    read_pettachainer_episode_manifest(
                        missing_manifest_path,
                        **{**valid_dependencies, field: None},
                    )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=changed_capture,
                    attribution=build_pettachainer_rule_attribution(changed_capture),
                )
            original_manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["attribution_cid"] = "4" * 64
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["result_cid"] = changed_capture.result_digest
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["contract_cid"] = "5" * 64
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["validator_capture_cid"] = "6" * 64
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["runtime_capture_cid"] = "7" * 64
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["episode_id"] = "episode-rehashed-drift"
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["chart_fingerprint"] = "6" * 64
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["promotion_authorized"] = True
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "manifests cannot authorize promotion",
            ):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["result_classification"] = (
                "runtime-trace-derived-result"
            )
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "unsupported PeTTaChainer result classification",
            ):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["finished_at"] = (
                "2026-07-18T06:59:59-07:00"
            )
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "finished_at must not precede started_at",
            ):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["seed"] = -1
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "seed must be a non-negative integer",
            ):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            manifest_document = json.loads(original_manifest_text)
            manifest_document["payload"]["budget"]["max_steps"] = 0
            manifest_payload = manifest_document["payload"]
            digest_payload = {
                key: value
                for key, value in manifest_payload.items()
                if key != "manifest_digest"
            }
            manifest_payload["manifest_digest"] = pipln_models._canonical_hash(
                digest_payload
            )
            manifest_document["document_digest"] = pipln_models._canonical_hash(
                manifest_payload
            )
            manifest_path.write_text(
                json.dumps(manifest_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "max_steps must be a positive integer",
            ):
                read_pettachainer_episode_manifest(
                    manifest_path,
                    contract=contract,
                    result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(original_manifest_text, encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_pettachainer_episode_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(
                ValueError, "derived result reload requires a typed PeTTaChainer contract",
            ):
                read_pettachainer_derived_result_capture(
                    Path(directory) / "absent-derived-result.json", contract=None,
                )
            with patch("petta_memory.pipln_models.os.fsync", wraps=os.fsync) as fsync:
                write_pettachainer_derived_result_capture(path, capture)
            self.assertEqual(fsync.call_count, 2)
            self.assertEqual(
                read_pettachainer_derived_result_capture(path, contract=contract),
                capture,
            )
            with self.assertRaises(FileExistsError):
                write_pettachainer_derived_result_capture(path, capture)

            reload_identities = []
            for cycle in (1, 2):
                clean_room = Path(directory) / f"clean-room-{cycle}"
                clean_room.mkdir(mode=0o700)
                reloaded_capture_path = clean_room / "derived-result.json"
                reloaded_manifest_path = clean_room / "episode-manifest.json"
                reloaded_attribution_path = clean_room / "rule-attribution.json"
                write_pettachainer_derived_result_capture(reloaded_capture_path, capture)
                write_pettachainer_episode_manifest(reloaded_manifest_path, manifest)
                write_pettachainer_rule_attribution(
                    reloaded_attribution_path, attribution,
                )

                reloaded_capture = read_pettachainer_derived_result_capture(
                    reloaded_capture_path, contract=contract,
                )
                reloaded_attribution = read_pettachainer_rule_attribution(
                    reloaded_attribution_path, result=reloaded_capture,
                )
                reloaded_manifest = read_pettachainer_episode_manifest(
                    reloaded_manifest_path, contract=contract, result=reloaded_capture,
                    attribution=reloaded_attribution,
                )
                self.assertEqual(reloaded_capture.derived_atom, capture.derived_atom)
                self.assertEqual(reloaded_capture.query_term, "(T a)")
                self.assertEqual(reloaded_manifest.result_classification,
                                 "compiler-bound-one-rule-derived-result")
                self.assertFalse(reloaded_manifest.promotion_authorized)
                self.assertEqual(reloaded_attribution.inference_rule, "TotalMP")
                self.assertFalse(reloaded_attribution.runtime_trace_decoded)
                reload_identities.append((
                    reloaded_capture.result_digest,
                    reloaded_capture.validator_capture.capture_digest,
                    reloaded_capture.runtime_capture.capture_digest,
                    reloaded_manifest.manifest_digest,
                    reloaded_attribution.attribution_digest,
                ))

                with self.assertRaisesRegex(
                    ValueError, "derived result capture document schema",
                ):
                    read_pettachainer_derived_result_capture(
                        reloaded_manifest_path, contract=contract,
                    )
                with self.assertRaisesRegex(
                    ValueError, "episode manifest document schema",
                ):
                    read_pettachainer_episode_manifest(
                        reloaded_capture_path, contract=contract, result=reloaded_capture,
                        attribution=reloaded_attribution,
                    )
                with self.assertRaisesRegex(
                    ValueError, "rule attribution document schema",
                ):
                    read_pettachainer_rule_attribution(
                        reloaded_manifest_path, result=reloaded_capture,
                    )
                with self.assertRaisesRegex(
                    ValueError, "derived result capture document schema",
                ):
                    read_pettachainer_derived_result_capture(
                        reloaded_attribution_path, contract=contract,
                    )
                with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                    read_pettachainer_episode_manifest(
                        reloaded_manifest_path,
                        contract=self.episode_contract(),
                        result=reloaded_capture,
                        attribution=reloaded_attribution,
                    )

            self.assertEqual(reload_identities[0], reload_identities[1])

            failed_file_sync_path = Path(directory) / "file-sync-failed.json"
            real_fsync = os.fsync
            sync_calls = 0

            def fail_file_sync(descriptor):
                nonlocal sync_calls
                sync_calls += 1
                if sync_calls == 1:
                    raise OSError("simulated file fsync failure")
                return real_fsync(descriptor)

            with patch("petta_memory.pipln_models.os.fsync", side_effect=fail_file_sync):
                with self.assertRaisesRegex(OSError, "file fsync failure"):
                    write_pettachainer_derived_result_capture(
                        failed_file_sync_path, capture,
                    )
            self.assertEqual(sync_calls, 2)
            self.assertFalse(failed_file_sync_path.exists())

            failed_cleanup_sync_path = Path(directory) / "cleanup-sync-failed.json"
            sync_calls = 0

            def fail_file_and_cleanup_sync(descriptor):
                nonlocal sync_calls
                sync_calls += 1
                if sync_calls == 1:
                    raise OSError("primary file fsync failure")
                raise OSError("secondary cleanup fsync failure")

            with patch(
                "petta_memory.pipln_models.os.fsync",
                side_effect=fail_file_and_cleanup_sync,
            ):
                with self.assertRaisesRegex(
                    OSError, "primary file fsync failure",
                ) as raised:
                    write_pettachainer_derived_result_capture(
                        failed_cleanup_sync_path, capture,
                    )
            self.assertEqual(sync_calls, 2)
            self.assertFalse(failed_cleanup_sync_path.exists())
            self.assertIn(
                "partial artifact cleanup sync failed: secondary cleanup fsync failure",
                raised.exception.__notes__,
            )

            failed_unlink_path = Path(directory) / "unlink-failed.json"
            sync_calls = 0

            with (
                patch(
                    "petta_memory.pipln_models.os.fsync",
                    side_effect=fail_file_sync,
                ),
                patch(
                    "petta_memory.pipln_models.os.unlink",
                    side_effect=OSError("secondary unlink failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated file fsync failure",
                ) as raised:
                    write_pettachainer_derived_result_capture(
                        failed_unlink_path, capture,
                    )
            self.assertTrue(failed_unlink_path.exists())
            self.assertIn(
                "partial artifact cleanup failed: secondary unlink failure",
                raised.exception.__notes__,
            )

            failed_close_path = Path(directory) / "close-failed.json"
            real_close = os.close
            sync_calls = 0
            close_calls = 0

            def fail_parent_close(descriptor):
                nonlocal close_calls
                close_calls += 1
                if close_calls == 1:
                    real_close(descriptor)
                    raise OSError("secondary parent close failure")
                return real_close(descriptor)

            with (
                patch(
                    "petta_memory.pipln_models.os.fsync",
                    side_effect=fail_file_sync,
                ),
                patch(
                    "petta_memory.pipln_models.os.close",
                    side_effect=fail_parent_close,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated file fsync failure",
                ) as raised:
                    write_pettachainer_derived_result_capture(
                        failed_close_path, capture,
                    )
            self.assertEqual(close_calls, 1)
            self.assertFalse(failed_close_path.exists())
            self.assertIn(
                "parent directory descriptor close failed: secondary parent close failure",
                raised.exception.__notes__,
            )

            failed_stream_close_path = Path(directory) / "stream-close-failed.json"
            real_fdopen = os.fdopen
            sync_calls = 0

            class CloseFailingStream:
                def __init__(self, handle):
                    self.handle = handle

                def __getattr__(self, name):
                    return getattr(self.handle, name)

                def close(self):
                    self.handle.close()
                    raise OSError("secondary artifact stream close failure")

            def close_failing_fdopen(*args, **kwargs):
                return CloseFailingStream(real_fdopen(*args, **kwargs))

            with (
                patch(
                    "petta_memory.pipln_models.os.fsync",
                    side_effect=fail_file_sync,
                ),
                patch(
                    "petta_memory.pipln_models.os.fdopen",
                    side_effect=close_failing_fdopen,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated file fsync failure",
                ) as raised:
                    write_pettachainer_derived_result_capture(
                        failed_stream_close_path, capture,
                    )
            self.assertFalse(failed_stream_close_path.exists())
            self.assertIn(
                "artifact stream close failed: secondary artifact stream close failure",
                raised.exception.__notes__,
            )

            failed_stream_open_path = Path(directory) / "stream-open-failed.json"
            opened_artifact_descriptor = None
            real_open = os.open
            real_close = os.close

            def capture_artifact_descriptor(*args, **kwargs):
                nonlocal opened_artifact_descriptor
                descriptor = real_open(*args, **kwargs)
                if kwargs.get("dir_fd") is not None:
                    opened_artifact_descriptor = descriptor
                return descriptor

            with (
                patch(
                    "petta_memory.pipln_models.os.open",
                    side_effect=capture_artifact_descriptor,
                ),
                patch(
                    "petta_memory.pipln_models.os.fdopen",
                    side_effect=OSError("primary artifact stream open failure"),
                ),
                patch(
                    "petta_memory.pipln_models.os.close",
                    wraps=real_close,
                ) as close_descriptor,
            ):
                with self.assertRaisesRegex(
                    OSError, "primary artifact stream open failure",
                ):
                    write_pettachainer_derived_result_capture(
                        failed_stream_open_path, capture,
                    )
            self.assertFalse(failed_stream_open_path.exists())
            self.assertIsNotNone(opened_artifact_descriptor)
            self.assertIn(
                call(opened_artifact_descriptor), close_descriptor.call_args_list,
            )

            failed_stream_open_close_path = (
                Path(directory) / "stream-open-close-failed.json"
            )
            close_calls = 0

            def fail_artifact_descriptor_close(descriptor):
                nonlocal close_calls
                close_calls += 1
                if close_calls == 1:
                    real_close(descriptor)
                    raise OSError("secondary artifact descriptor close failure")
                return real_close(descriptor)

            with (
                patch(
                    "petta_memory.pipln_models.os.fdopen",
                    side_effect=OSError("primary artifact stream open failure"),
                ),
                patch(
                    "petta_memory.pipln_models.os.close",
                    side_effect=fail_artifact_descriptor_close,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "primary artifact stream open failure",
                ) as raised:
                    write_pettachainer_derived_result_capture(
                        failed_stream_open_close_path, capture,
                    )
            self.assertFalse(failed_stream_open_close_path.exists())
            self.assertIn(
                "artifact descriptor close failed: secondary artifact descriptor close failure",
                raised.exception.__notes__,
            )

            failed_stream_open_dual_close_path = (
                Path(directory) / "stream-open-dual-close-failed.json"
            )

            def fail_both_descriptor_closes(descriptor):
                real_close(descriptor)
                raise OSError(f"secondary close failure for descriptor {descriptor}")

            with (
                patch(
                    "petta_memory.pipln_models.os.fdopen",
                    side_effect=OSError("primary artifact stream open failure"),
                ),
                patch(
                    "petta_memory.pipln_models.os.close",
                    side_effect=fail_both_descriptor_closes,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "primary artifact stream open failure",
                ) as raised:
                    write_pettachainer_derived_result_capture(
                        failed_stream_open_dual_close_path, capture,
                    )
            self.assertFalse(failed_stream_open_dual_close_path.exists())
            self.assertEqual(len(raised.exception.__notes__), 2)
            self.assertTrue(
                raised.exception.__notes__[0].startswith(
                    "artifact descriptor close failed: secondary close failure"
                )
            )
            self.assertTrue(
                raised.exception.__notes__[1].startswith(
                    "parent directory descriptor close failed: secondary close failure"
                )
            )

            successful_stream_close_path = Path(directory) / "successful-stream-close-failed.json"
            with patch(
                "petta_memory.pipln_models.os.fdopen",
                side_effect=close_failing_fdopen,
            ):
                with self.assertRaisesRegex(
                    OSError, "secondary artifact stream close failure",
                ):
                    write_pettachainer_derived_result_capture(
                        successful_stream_close_path, capture,
                    )
            self.assertTrue(successful_stream_close_path.is_file())
            self.assertEqual(
                read_pettachainer_derived_result_capture(
                    successful_stream_close_path, contract=contract,
                ),
                capture,
            )
            with self.assertRaises(FileExistsError):
                write_pettachainer_derived_result_capture(
                    successful_stream_close_path, capture,
                )

            successful_close_path = Path(directory) / "successful-close-failed.json"
            real_close = os.close

            def close_successful_parent_then_fail(descriptor):
                real_close(descriptor)
                raise OSError("successful publication parent close failure")

            with patch(
                "petta_memory.pipln_models.os.close",
                side_effect=close_successful_parent_then_fail,
            ):
                with self.assertRaisesRegex(
                    OSError, "successful publication parent close failure",
                ):
                    write_pettachainer_derived_result_capture(
                        successful_close_path, capture,
                    )
            self.assertTrue(successful_close_path.is_file())
            self.assertEqual(
                read_pettachainer_derived_result_capture(
                    successful_close_path, contract=contract,
                ),
                capture,
            )
            with self.assertRaises(FileExistsError):
                write_pettachainer_derived_result_capture(
                    successful_close_path, capture,
                )

            failed_sync_path = Path(directory) / "directory-sync-failed.json"
            real_fsync = os.fsync
            sync_calls = 0

            def fail_directory_sync(descriptor):
                nonlocal sync_calls
                sync_calls += 1
                if sync_calls == 2:
                    raise OSError("simulated directory fsync failure")
                return real_fsync(descriptor)

            with patch("petta_memory.pipln_models.os.fsync", side_effect=fail_directory_sync):
                with self.assertRaisesRegex(OSError, "directory fsync failure"):
                    write_pettachainer_derived_result_capture(failed_sync_path, capture)
            self.assertTrue(failed_sync_path.is_file())
            self.assertEqual(
                read_pettachainer_derived_result_capture(
                    failed_sync_path, contract=contract,
                ),
                capture,
            )
            with self.assertRaises(FileExistsError):
                write_pettachainer_derived_result_capture(failed_sync_path, capture)

            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(
                manifest_text.replace(
                    '{\n  "document_digest"',
                    '{\n  "schema": "petta-memory-pettachainer-episode-manifest-v1",\n  "document_digest"',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object member: schema"):
                read_pettachainer_episode_manifest(
                    manifest_path, contract=contract, result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(manifest_text, encoding="utf-8")

            capture_text = path.read_text(encoding="utf-8")
            path.write_text(
                capture_text.replace(
                    '{\n  "document_digest"',
                    '{\n  "schema": "petta-memory-pettachainer-derived-result-capture-v1",\n  "document_digest"',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object member: schema"):
                read_pettachainer_derived_result_capture(path, contract=contract)
            path.write_text(capture_text, encoding="utf-8")

            manifest_path.write_bytes(b" " * 1_000_001)
            with self.assertRaisesRegex(ValueError, "exceeds 1000000 byte limit"):
                read_pettachainer_episode_manifest(
                    manifest_path, contract=contract, result=capture,
                    attribution=attribution,
                )
            manifest_path.write_text(manifest_text, encoding="utf-8")

            path.write_bytes(b" " * 1_000_001)
            with self.assertRaisesRegex(ValueError, "exceeds 1000000 byte limit"):
                read_pettachainer_derived_result_capture(path, contract=contract)
            path.write_text(capture_text, encoding="utf-8")

            manifest_link = Path(directory) / "episode-manifest-link.json"
            manifest_link.symlink_to(manifest_path)
            with self.assertRaises(OSError):
                read_pettachainer_episode_manifest(
                    manifest_link, contract=contract, result=capture,
                    attribution=attribution,
                )
            capture_link = Path(directory) / "derived-result-link.json"
            capture_link.symlink_to(path)
            with self.assertRaises(OSError):
                read_pettachainer_derived_result_capture(
                    capture_link, contract=contract,
                )

            with self.assertRaisesRegex(ValueError, "must be a regular file"):
                read_pettachainer_derived_result_capture(
                    Path(directory), contract=contract,
                )

            real_close = os.close

            def close_then_fail(descriptor):
                real_close(descriptor)
                raise OSError("secondary artifact close failure")

            with patch(
                "petta_memory.pipln_models.os.close", side_effect=close_then_fail,
            ):
                with self.assertRaisesRegex(
                    ValueError, "must be a regular file",
                ) as raised:
                    read_pettachainer_derived_result_capture(
                        Path(directory), contract=contract,
                    )
            self.assertIn(
                "JSON artifact descriptor close failed: secondary artifact close failure",
                raised.exception.__notes__,
            )

            capture_fifo = Path(directory) / "derived-result.fifo"
            os.mkfifo(capture_fifo)
            with self.assertRaisesRegex(ValueError, "must be a regular file"):
                read_pettachainer_derived_result_capture(
                    capture_fifo, contract=contract,
                )

            rule, fact = contract.statements
            drifted_fact = PeTTaChainerInputStatement(
                atom=fact.atom, proof_id=fact.proof_id,
                sentence_digest=fact.sentence_digest,
                canonical_term=fact.canonical_term, strength=fact.strength,
                confidence=fact.confidence, stamp_ints=fact.stamp_ints,
                evidence_basis_ids=("basis-other",),
            )
            drifted_contract = PeTTaChainerEpisodeContract(
                episode_id=contract.episode_id,
                chart_fingerprint=contract.chart_fingerprint,
                statements=(rule, drifted_fact), query_term=contract.query_term,
                query_atom=contract.query_atom,
            )
            with self.assertRaisesRegex(ValueError, "compiler provenance mismatch"):
                read_pettachainer_derived_result_capture(path, contract=drifted_contract)
            with self.assertRaisesRegex(ValueError, "does not close"):
                build_pettachainer_episode_manifest(
                    **(manifest_kwargs | {"contract": drifted_contract}),
                )
            with self.assertRaisesRegex(ValueError, "input provenance mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path, contract=drifted_contract, result=capture,
                    attribution=attribution,
                )

            manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_document["payload"]["seed"] = 1
            manifest_path.write_text(json.dumps(manifest_document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_pettachainer_episode_manifest(
                    manifest_path, contract=contract, result=capture,
                    attribution=attribution,
                )

            document = json.loads(path.read_text(encoding="utf-8"))
            document["payload"]["runtime_capture"]["stdout_bytes"] += 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_pettachainer_derived_result_capture(path, contract=contract)

        runtime["stdout_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            profile.build_repaired_pettachainer_rule_episode_capture(contract, gate)

    def test_repaired_rule_contract_capture_rejects_gate_identity_drift(self):
        contract = self.rule_episode_contract()
        with self.assertRaisesRegex(ValueError, "not admitted for this contract"):
            profile.build_repaired_pettachainer_rule_episode_capture(contract, {
                "schema": "petta-memory-repaired-pettachainer-rule-contract-gate-v1",
                "runtime_admitted": True,
                "result_classification": "compiler-bound-one-rule-derived-result",
                "episode_id": "other-episode",
                "chart_fingerprint": contract.chart_fingerprint,
                "query_term": contract.query_term,
            })

    def test_repaired_rule_contract_gate_rejects_wrong_shape_or_stored_query(self):
        contract = self.rule_episode_contract()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            profile.run_repaired_pettachainer_rule_episode_contract_gate(
                self.episode_contract(), project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )
        other_digest = "d" * 64
        other_fact = PeTTaChainerInputStatement(
            atom=f"(: pm-{other_digest} (S b) (STV 0.7 0.5))",
            proof_id=f"pm-{other_digest}", sentence_digest=other_digest,
            canonical_term="(S b)", strength=0.7, confidence=0.5,
            stamp_ints=(1,), evidence_basis_ids=("basis-other",),
        )
        two_facts = PeTTaChainerEpisodeContract(
            episode_id=contract.episode_id, chart_fingerprint=contract.chart_fingerprint,
            statements=(contract.statements[1], other_fact),
            query_term="(T a)", query_atom="(: $prf (T a) $tv)",
        )
        with self.assertRaisesRegex(ValueError, "one fact and one implication"):
            profile.run_repaired_pettachainer_rule_episode_contract_gate(
                two_facts, project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )
        stored_query = PeTTaChainerEpisodeContract(
            episode_id=contract.episode_id, chart_fingerprint=contract.chart_fingerprint,
            statements=contract.statements, query_term="(S a)",
            query_atom="(: $prf (S a) $tv)",
        )
        with self.assertRaisesRegex(ValueError, "differ from stored inputs"):
            profile.run_repaired_pettachainer_rule_episode_contract_gate(
                stored_query, project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

    def test_repaired_one_rule_derivation_gate_requires_exact_derived_answer(self):
        validation = {
            "status": "ok", "statement_results": [1.0, 1.0], "query_result": 1.0,
            "stdout_bytes": 0, "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0, "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        runtime = {
            "status": "ok", "query_answer_count": 1, "malformed_answer_count": 0,
            "query_target_only": True, "exact_derived_proof_only": True, "typed_stv_only": True,
            "truth_formula_match": True,
            "stdout_bytes": 10, "stdout_sha256": "a" * 64,
            "stderr_bytes": 2, "stderr_sha256": "b" * 64,
        }
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", side_effect=[
                {"selected_compile_branch": "fact-assertion"},
                {"selected_compile_branch": "implication-rule"},
            ]),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "inspect_one_rule_truth_formula_path", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=[validation, runtime]),
        ):
            result = profile.run_repaired_pettachainer_one_rule_derivation_gate(
                "(: fact_a (S a) (STV 0.8 0.6))",
                "(: rule_s_t (Implication (Premises (S $x)) (Conclusions (T $x))) (STV 0.9 0.8))",
                "(T a)", project_root=Path("/project"), candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_classification"], "one-rule-derived-result")
        self.assertTrue(result["boundaries"]["exact_rule_proof_required"])
        self.assertTrue(result["boundaries"]["no_episode_manifest"])

        runtime["exact_derived_proof_only"] = False
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", side_effect=[
                {"selected_compile_branch": "fact-assertion"},
                {"selected_compile_branch": "implication-rule"},
            ]),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "inspect_one_rule_truth_formula_path", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=[validation, runtime]),
        ):
            blocked = profile.run_repaired_pettachainer_one_rule_derivation_gate(
                "(: fact_a (S a) (STV 0.8 0.6))",
                "(: rule_s_t (Implication (Premises (S $x)) (Conclusions (T $x))) (STV 0.9 0.8))",
                "(T a)", project_root=Path("/project"), candidate_repo_path=Path("/candidate"),
            )
        self.assertEqual(blocked["status"], "blocked")

    def test_one_rule_truth_formula_path_matches_pinned_source(self):
        repo = Path(__file__).resolve().parents[2] / "PeTTaChainer"
        result = profile.inspect_one_rule_truth_formula_path(repo)
        self.assertTrue(result["shape_confirmed"])
        self.assertEqual(result["fallback_truth_value"], [0.2, 0.2])
        self.assertEqual(set(result["source_sha256"]), {"compile.metta", "tv_formulas.metta"})

    def test_run_repaired_compileadd_exact_fact_query_gate_blocks_missing_capture_provenance(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value={
                "status": "ok",
                "expected_internal_stored": True,
                "exact_answer_present": True,
                "exact_answer_only": True,
                "query_answer_count": 1,
            }),
        ):
            result = profile.run_repaired_compileadd_exact_fact_query_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "blocked")

    def test_all_materialize_identity_matches_rejects_mixed_answers(self):
        expected = "(: p (S x) (STV 1.0 0.70))"
        self.assertTrue(profile._all_materialize_identity_matches(
            expected, ["(: p (S x) (STV 1 0.7))"],
        ))
        self.assertFalse(profile._all_materialize_identity_matches(
            expected,
            ["(: p (S x) (STV 1 0.7))", "(: other (S x) (STV 1 0.7))"],
        ))
        self.assertFalse(profile._all_materialize_identity_matches(expected, []))

    def test_run_repaired_compileadd_exact_fact_query_gate_blocks_wrong_answer(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value={
                "status": "ok",
                "expected_internal_stored": True,
                "exact_answer_present": False,
                "query_answer_count": 1,
            }),
        ):
            result = profile.run_repaired_compileadd_exact_fact_query_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "blocked")

    def test_run_repaired_compileadd_exact_fact_query_gate_blocks_extra_answer(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": True}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value={
                "status": "ok",
                "expected_internal_stored": True,
                "exact_answer_present": True,
                "exact_answer_only": False,
                "query_answer_count": 2,
            }),
        ):
            result = profile.run_repaired_compileadd_exact_fact_query_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "blocked")

    def test_run_repaired_compileadd_exact_fact_query_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_duplicate_compile_import_repair", return_value={"exact_targeted_import_removal": False}),
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_mm2compile_collection_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_repaired_compileadd_exact_fact_query_gate(
                "(: p (S x) (STV 1 0.9))",
                project_root=Path("/project"),
                candidate_repo_path=Path("/candidate"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["runtime_event"])
        configure.assert_not_called()

    def test_run_compile_single_registration_gate_compares_clone_and_direct(self):
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {
                "status": "ok",
                "single_registration_clone": {"output_count": 64},
                "direct_compile_": {"output_count": 128},
            }

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_dispatch_ladder_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "inspect_compile_import_multiplicity", return_value={"two_compile_import_paths_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_single_registration_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), stage_timeout_sec=3.0,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_single_registration")
        self.assertEqual(captured["target"], profile._compile_single_registration_stage)
        self.assertTrue(result["boundaries"]["no_mm2compile_or_compileadd_or_query"])

    def test_run_compile_single_registration_gate_skips_import_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_dispatch_ladder_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "inspect_compile_import_multiplicity", return_value={"two_compile_import_paths_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_single_registration_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_annotation_dispatch_gate_compares_matching_heads(self):
        captured = {}
        definition = {"snippet": "(= (compile_ $kb (@ $stmt (: $prf $Type $tv))) body)"}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {
                "status": "ok",
                "annotated_head": {"output_count": 64},
                "structural_head": {"output_count": 32},
            }

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_dispatch_ladder_shape", return_value={"shape_confirmed": True, "definition": definition}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_annotation_dispatch_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), stage_timeout_sec=3.0,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_annotation_dispatch")
        self.assertEqual(captured["target"], profile._compile_annotation_dispatch_stage)
        self.assertTrue(result["annotated_head_confirmed"])
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd_or_query"])

    def test_run_compile_annotation_dispatch_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(
                profile,
                "inspect_compile_fact_dispatch_ladder_shape",
                return_value={"shape_confirmed": True, "definition": {"snippet": "(= (compile_ $kb $stmt) body)"}},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_annotation_dispatch_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_annotation_dispatch_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_compile_annotation_dispatch_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), max_output_items=0,
            )

    def test_fact_fanout_repair_plan_closes_all_measured_factors(self):
        result = profile.build_pettachainer_fact_fanout_repair_plan(
            public_compile_count=256,
            direct_compile_count=128,
            single_registration_annotated_count=64,
            single_registration_structural_count=32,
            literal_fact_count=1,
            bidirectional_gate_count=4,
            fact_kb_count=8,
            mm2stmt_count=2,
            mm2compile_collection_count=4,
        )

        self.assertEqual(result["status"], "attribution-closed")
        self.assertEqual(
            result["factors"],
            {
                "public_wrapper": 2,
                "duplicate_registration": 2,
                "annotated_head": 2,
                "bidirectional_classifier": 4,
                "fact_kb": 8,
                "mm2stmt_overlap": 2,
                "mm2compile_collection": 2,
            },
        )
        self.assertEqual(result["closure"]["compiler_product"], 256)
        self.assertEqual(result["closure"]["collection_product"], 4)
        self.assertEqual(result["repair_order"][0]["target"], "duplicate compiler registration")
        self.assertTrue(result["boundaries"]["no_set_collapse_approved"])

    def test_fact_fanout_repair_plan_rejects_stale_or_malformed_counts(self):
        valid = {
            "public_compile_count": 256,
            "direct_compile_count": 128,
            "single_registration_annotated_count": 64,
            "single_registration_structural_count": 32,
            "literal_fact_count": 1,
            "bidirectional_gate_count": 4,
            "fact_kb_count": 8,
            "mm2stmt_count": 2,
            "mm2compile_collection_count": 4,
        }
        for name, value in (("public_compile_count", 255), ("literal_fact_count", True)):
            with self.subTest(name=name):
                drifted = {**valid, name: value}
                with self.assertRaises(ValueError):
                    profile.build_pettachainer_fact_fanout_repair_plan(**drifted)

    def test_fact_compiled_clause_builds_one_canonical_base_fact(self):
        clause = profile._fact_compiled_clause(
            "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))", "kb-test",
        )

        self.assertEqual(
            clause,
            "(() |- ((: (kb-test MAIN Nil) p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))))",
        )
        with self.assertRaises(ValueError):
            profile._fact_compiled_clause("(Requires A B)", "kb-test")
        with self.assertRaises(ValueError):
            profile._fact_compiled_clause("(: p (S x) (STV 1 0.9))", "bad kb")

    def test_inspect_mm2stmt_fact_case_overlap_closes_exact_source_cause(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2stmt $stmt) (case $stmt (((() |- ($ccl)) $ccl) "
                "(($prms |- ($ccl)) (rules ($prms |- $ccl))))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2stmt_fact_case_overlap(repo)

        self.assertTrue(result["fact_arm_present"])
        self.assertTrue(result["general_arm_present"])
        self.assertTrue(result["overlap_confirmed"])
        self.assertIn("matches both", result["interpretation"])
        self.assertTrue(result["boundaries"]["source_inspection_only"])

    def test_inspect_mm2stmt_fact_case_overlap_fails_closed_on_source_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2stmt $stmt) (case $stmt (((() |- ($ccl)) $ccl))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2stmt_fact_case_overlap(repo)

        self.assertTrue(result["fact_arm_present"])
        self.assertFalse(result["general_arm_present"])
        self.assertFalse(result["overlap_confirmed"])
        self.assertIn("do not attribute", result["interpretation"])

    def test_inspect_mm2compile_collection_shape_closes_copied_source(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2compile $kb $stmt)\n"
                "   (progn (remove-all-atoms ctx) "
                "(superpose ((mm2stmt (compile $kb $stmt)) (get-atoms ctx)))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2compile_collection_shape(repo)

        self.assertTrue(result["shape_confirmed"])
        self.assertIn("superposes", result["interpretation"])
        self.assertTrue(result["boundaries"]["source_inspection_only"])

    def test_inspect_mm2compile_collection_shape_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2compile $kb $stmt) (mm2stmt (compile $kb $stmt)))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2compile_collection_shape(repo)

        self.assertFalse(result["shape_confirmed"])
        self.assertIn("do not run", result["interpretation"])

    def test_run_mm2stmt_deduplicated_fact_gate_isolates_one_clause(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {
                "label": label,
                "status": "ok",
                "expected_fact_present": True,
                "converted_count": 2,
                "converted_unique_count": 1,
                "ctx_atom_count": 0,
            }

        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(
                profile,
                "inspect_mm2stmt_fact_case_overlap",
                return_value={"overlap_confirmed": True},
            ),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_mm2stmt_deduplicated_fact_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "mm2stmt_deduplicated_fact")
        self.assertEqual(captured["target"], profile._mm2stmt_deduplicated_fact_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertTrue(result["mm2stmt_inspection"]["overlap_confirmed"])
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_mm2stmt_deduplicated_fact_gate_skips_non_fact_branch(self):
        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "implication-rule"},
            ),
            patch.object(
                profile,
                "inspect_mm2stmt_fact_case_overlap",
                return_value={"overlap_confirmed": True},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_mm2stmt_deduplicated_fact_gate(
                "(: p (Implication (cons Premises ()) (cons Conclusions ())) (STV 1 0.9))",
                project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_mm2stmt_deduplicated_fact_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_mm2stmt_deduplicated_fact_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), stage_timeout_sec=0,
            )

    def test_run_mm2compile_deduplicated_fact_gate_isolates_collection(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {
                "label": label,
                "status": "ok",
                "expected_fact_present": True,
                "output_count": 2,
                "unique_output_count": 1,
            }

        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(
                profile,
                "inspect_mm2compile_collection_shape",
                return_value={"shape_confirmed": True},
            ),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_mm2compile_deduplicated_fact_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "mm2compile_deduplicated_fact_collection")
        self.assertEqual(captured["target"], profile._mm2compile_deduplicated_fact_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertTrue(result["boundaries"]["no_compile_or_compileadd"])

    def test_run_mm2compile_deduplicated_fact_gate_skips_source_drift(self):
        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(
                profile,
                "inspect_mm2compile_collection_shape",
                return_value={"shape_confirmed": False},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_mm2compile_deduplicated_fact_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("source shape", result["reason"])
        configure.assert_not_called()

    def test_run_mm2compile_deduplicated_fact_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_mm2compile_deduplicated_fact_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), max_output_items=0,
            )

    def test_inspect_petta_static_import_source_flags_current_export_as_unsafe(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            lib_dir = repo / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "lib_import.pl").write_text(
                "metta_file_to_prolog(Input, Space, Output) :- true.\n"
                "convert_stream(In, Out, Space) :- read_line_to_string(In, Line), convert_line(Line, Space, Out).\n"
                "convert_line(Line0, Space, Out) :- sub_string(Line0, 1, _, 1, Inner0), replace_all(\"(\", \"[\", Inner0, Inner1).\n"
                "'static-import!'(Space, File, true) :- metta_file_to_prolog(MettaFile, Space, PlFile), qcompile(PlFile), consult(QlfFile).\n"
                ":- multifile '~w'/3.\n",
                encoding="utf-8",
            )

            summary = profile.inspect_petta_static_import_source(
                repo,
                ["(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"],
            )

        self.assertTrue(summary["source_features"]["defines_static_import"])
        self.assertTrue(summary["source_features"]["uses_qcompile"])
        self.assertFalse(summary["sample_atoms_safe_for_current_converter"])
        converted = summary["sample_conversions"][0]["converted_prolog_fact"]
        self.assertEqual(converted, "'gckb'(:,b-profile-000,[Requires,MemoryTarget0,PLNReadyViews],[STV,0.70,0.55]).")
        self.assertIn("MemoryTarget0", " ".join(summary["sample_conversions"][0]["warnings"]))
        self.assertIn("Do not use static-import! directly", summary["recommendation"])
        self.assertIn("no SWI qcompile", " ".join(summary["gates"]))

    def test_design_static_import_microbenchmark_atoms_uses_safe_three_argument_facts(self):
        summary = profile.design_static_import_microbenchmark_atoms(
            [
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
                "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) "
                "((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
            ]
        )

        self.assertTrue(summary["all_records_safe_for_current_converter"])
        self.assertEqual(
            summary["records"][0]["normalized_atom"],
            "(pm_stv_statement b_profile_000 (pm_stv_payload requires_memorytarget0_plnreadyviews 0.70 0.55))",
        )
        self.assertEqual(
            summary["records"][0]["converted_prolog_fact"],
            "'gckb'(pm_stv_statement,b_profile_000,[pm_stv_payload,requires_memorytarget0_plnreadyviews,0.70,0.55]).",
        )
        self.assertEqual(
            summary["records"][1]["normalized_atom"],
            "(pm_evidence_packet requires_memorytarget0_plnreadyviews (pm_ec_payload 3.0 1.0 pe_profile_000))",
        )
        self.assertIn("temporary scratch", " ".join(summary["benchmark_gate"]))

    def test_compileadd_strategy_summary_recommends_precompiled_cache_gate(self):
        sample_profile = {
            "results": [
                {
                    "events": [
                        {"label": "check_stmt_all", "status": "ok"},
                        {"label": "pettachainer_init_only", "status": "ok"},
                        {"label": "compileadd_probe_materialize_direct", "status": "timeout"},
                        {"label": "compileadd_probe_materialize_eval_control", "status": "timeout"},
                        {"label": "compileadd_probe_mm2compile_direct", "status": "timeout"},
                        {"label": "compileadd_probe_mm2compile_eval_control", "status": "timeout"},
                        {"label": "compileadd_probe_index_source_direct", "status": "ok"},
                        {"label": "compileadd_probe_maybe_process_on_add_direct", "status": "ok"},
                        {"label": "proof_runtime_add_only", "status": "timeout"},
                    ]
                }
            ]
        }

        summary = profile.summarize_compileadd_strategy(sample_profile)

        self.assertEqual(summary["recommended_next_add_path"], "precompiled_statement_cache_gate")
        self.assertEqual(
            summary["fast_later_probes"],
            ["compileadd_probe_index_source_direct", "compileadd_probe_maybe_process_on_add_direct"],
        )
        self.assertIn("checked handoff inputs only", " ".join(summary["gates"]))

    def test_contextual_profile_schedules_add_only_bottleneck_stages(self):
        def fake_isolated_stage(label, _target, _args, *, stage_timeout_sec):
            return {"label": label, "status": "ok", "timeout_sec": stage_timeout_sec}

        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(
                profile,
                "_build_export_payload",
                return_value={
                    "statements": ["(: p (S x) (STV 1 0.9))"],
                    "packets": ["(EvidencePacket (S x) (EC 1 0) () pe)"],
                },
            ),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_isolated_stage),
        ):
            result = profile.profile_sizes(
                [1],
                steps=1,
                timeout_sec=1.0,
                project_root=Path("/unused"),
                stage_timeout_sec=2.0,
                include_runtime_add=True,
                include_contextual=True,
            )

        labels = [event["label"] for event in result["results"][0]["events"]]
        self.assertEqual(
            labels,
            [
                "build_store_and_exports",
                "check_stmt_all",
                "pettachainer_init_only",
                "compileadd_probe_materialize_direct",
                "compileadd_probe_materialize_eval_control",
                "compileadd_probe_mm2compile_direct",
                "compileadd_probe_mm2compile_eval_control",
                "compileadd_probe_internalize_direct",
                "compileadd_probe_externalize_direct",
                "compileadd_probe_index_source_direct",
                "compileadd_probe_add_internalized_direct",
                "compileadd_probe_maybe_process_on_add_direct",
                "proof_runtime_add_only",
                "proof_runtime_add_and_query",
                "contextual_packet_add_only",
                "contextual_runtime_add_and_query",
            ],
        )


    def test_static_import_fact_goal_strips_clause_full_stop(self):
        fact = "'pmbench'(pm_stv_statement,b_profile_000,[pm_stv_payload,key,0.70,0.55])."

        self.assertEqual(
            profile._petta_static_import_fact_goal(fact),
            "'pmbench'(pm_stv_statement,b_profile_000,[pm_stv_payload,key,0.70,0.55])",
        )
        with self.assertRaises(ValueError):
            profile._petta_static_import_fact_goal("'pmbench'(a,b,c)")

    def test_run_static_import_microbenchmark_skips_unsafe_atoms(self):
        """Microbenchmark should skip when atoms are not converter-safe."""
        result = profile.run_static_import_microbenchmark(
            ["(unsupported_atom_only)"],
            project_root=Path("/unused"),
            stage_timeout_sec=1.0,
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("not all normalized atoms are safe", result["reason"])
        self.assertFalse(result["design"]["all_records_safe_for_current_converter"])

    def test_run_static_import_microbenchmark_rejects_unsafe_space_name(self):
        sample = ["(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"]

        with self.assertRaises(ValueError):
            profile.run_static_import_microbenchmark(
                sample,
                project_root=Path("/unused"),
                stage_timeout_sec=5.0,
                space="bad-space",
            )

    def test_run_static_import_microbenchmark_uses_isolated_stage(self):
        """Microbenchmark should delegate to _run_isolated_stage with expected args."""
        sample = [
            "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
            "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) "
            "((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
        ]
        captured = {}

        def fake_isolated_stage(label, target, args, *, stage_timeout_sec):
            captured["label"] = label
            captured["target"] = target
            captured["args"] = args
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "label": label,
                "status": "ok",
                "seconds": 0.1,
                "result": "loaded",
                "loaded_fact_count": 2,
                "expected_fact_count": 2,
                "facts_match": True,
            }

        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_isolated_stage),
        ):
            result = profile.run_static_import_microbenchmark(
                sample,
                project_root=Path("/unused"),
                stage_timeout_sec=5.0,
                space="pmbench",
            )

        self.assertEqual(result["source"], "non-live static-import microbenchmark")
        self.assertEqual(captured["label"], "static_import_load_and_query")
        self.assertEqual(captured["stage_timeout_sec"], 5.0)
        self.assertTrue(result["design"]["all_records_safe_for_current_converter"])
        self.assertEqual(result["runtime_event"]["status"], "ok")
        self.assertTrue(result["runtime_event"]["facts_match"])
        # Verify normalized atoms were passed to the stage
        normalized = captured["args"][0]
        self.assertTrue(all("(" in atom and ")" in atom for atom in normalized))
        # Verify expected facts were passed
        expected = captured["args"][1]
        self.assertTrue(all("pmbench" in fact for fact in expected))
        self.assertEqual(captured["args"][2], "pmbench")
        # Gates
        gates = " ".join(result["gates"])
        self.assertIn("no petta-memory journal writes", gates)
        self.assertIn("not PeTTaChainer compileadd/query success", gates)


if __name__ == "__main__":
    unittest.main()
