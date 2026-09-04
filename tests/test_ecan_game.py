"""Tests for the ECAN Game World: simulated scenarios for attention allocation.

These tests verify that the ECAN attention dynamics work correctly under
scripted game scenarios - the same kind of scenarios Ben described as
"a game or simulated situation to enable testing of these functions."
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.ecan_game import (
    ECANGameWorld,
    GameEvent,
    GameTurn,
    GameTurnExpectation,
    GameTurnResult,
)
from petta_memory.simworld import WorldItem

REPO = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

class TestECANGameWorldBasic(unittest.TestCase):
    """Basic game world functionality."""

    def test_game_world_initializes(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        self.assertEqual(len(game.items), 0)
        self.assertEqual(len(game._all_belief_ids), 0)
        self.assertEqual(game.turn_index, 0)

    def test_stv_item_factory(self):
        item = ECANGameWorld.stv_item("b1", "deploy_canary", 0.9, 0.8)
        self.assertEqual(item.belief_id, "b1")
        self.assertEqual(item.slot, "acceptability-belief-evidence")
        self.assertIn("0.9", item.atom)
        self.assertIn("deploy_canary", item.atom)

    def test_ec_item_factory(self):
        item = ECANGameWorld.ec_item("b2", "rollback", 5, 1)
        self.assertEqual(item.belief_id, "b2")
        self.assertEqual(item.slot, "contextual-appraisal-evidence")
        self.assertIn("rollback", item.atom)

    def test_register_belief(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        game._register_belief("b-test")
        self.assertIn("b-test", game._all_belief_ids)
        av = game.bridge.bank.get_av("b-test")
        self.assertEqual(av.sti, 0.0)
        self.assertEqual(av.lti, 0.0)

    def test_stimulate_adds_sti(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        game._register_belief("b-stim")
        game.bridge.run_cycle(stimuli={"b-stim": 10.0})
        av = game.bridge.bank.get_av("b-stim")
        self.assertGreater(av.sti, 0.0, "STI should increase after stimulation")

    def test_evidence_map_builds_from_items(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        item = ECANGameWorld.stv_item("b1", "deploy", 0.9, 0.8)
        game.items.append(item)
        emap = game._build_evidence_map()
        self.assertIn("b1", emap)
        self.assertEqual(len(emap["b1"]), 1)


class TestECANGameWorldAttention(unittest.TestCase):
    """Test attention allocation dynamics in game scenarios."""

    def test_top_belief_has_highest_sti(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        game._register_belief("b-strong")
        game._register_belief("b-weak")
        game.bridge.run_cycle(stimuli={"b-strong": 10.0, "b-weak": 2.0})
        prioritized = game.bridge.get_prioritized_beliefs(limit=5)
        self.assertEqual(prioritized[0][0], "b-strong")
        self.assertGreater(prioritized[0][1], prioritized[1][1])

    def test_forget_candidates_have_low_sti(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        game._register_belief("b-forgotten")
        game._register_belief("b-remembered")
        game.bridge.run_cycle(stimuli={"b-remembered": 20.0})
        game.bridge.run_cycles(20)
        forget = game.bridge.get_forget_candidates()
        self.assertIn("b-forgotten", forget)
        self.assertNotIn("b-remembered", forget)

    def test_attentional_focus_filters_by_threshold(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        game._register_belief("b-af")
        game._register_belief("b-low")
        game.bridge.run_cycle(stimuli={"b-af": 15.0, "b-low": 0.5})
        af = game.bridge.get_attentional_focus_beliefs()
        self.assertIn("b-af", af)

    def test_attention_shifts_with_new_stimulus(self):
        game = ECANGameWorld(goalchainer_repo=REPO)
        game._register_belief("b-old")
        game._register_belief("b-new")
        game.bridge.run_cycle(stimuli={"b-old": 10.0, "b-new": 1.0})
        prioritized = game.bridge.get_prioritized_beliefs()
        self.assertEqual(prioritized[0][0], "b-old")
        game.bridge.run_cycle(stimuli={"b-new": 20.0})
        prioritized = game.bridge.get_prioritized_beliefs()
        self.assertEqual(prioritized[0][0], "b-new")

class TestECANGameWorldScenarios(unittest.TestCase):
    """Full game scenario tests: multi-turn scripted games."""

    def test_incident_response_scenario(self):
        """Simulated incident: evidence arrives, attention shifts, decision changes."""
        game = ECANGameWorld(goalchainer_repo=REPO)

        # Turn 1: Initial evidence for deploy
        result1 = game.play_turn(GameTurn(
            label="initial-evidence",
            events=[GameEvent(
                label="deploy-evidence-arrives",
                add_evidence=[ECANGameWorld.stv_item("b-deploy", "deploy_canary", 0.85, 0.80)],
                stimulate={"b-deploy": 10.0},
            )],
            expectation=GameTurnExpectation(
                expected_top_attention_belief="b-deploy",
                expected_af_contains=["b-deploy"],
                description="deploy_canary should have attention after initial evidence",
            ),
        ))
        self.assertTrue(result1.checks_passed, f"Turn 1 failed: {result1.check_errors}")
        self.assertEqual(result1.top_attention_belief, "b-deploy")

        # Turn 2: Contradicting evidence for rollback
        result2 = game.play_turn(GameTurn(
            label="contradiction-arrives",
            events=[GameEvent(
                label="rollback-evidence-arrives",
                add_evidence=[ECANGameWorld.stv_item("b-rollback", "rollback", 0.75, 0.70)],
                stimulate={"b-rollback": 15.0},
            )],
            expectation=GameTurnExpectation(
                expected_af_contains=["b-rollback"],
                description="rollback should be in AF after strong stimulus",
            ),
        ))
        self.assertTrue(result2.checks_passed, f"Turn 2 failed: {result2.check_errors}")

        # Turn 3: More rollback evidence, attention should shift
        result3 = game.play_turn(GameTurn(
            label="attention-shifts",
            events=[GameEvent(
                label="more-rollback-evidence",
                add_evidence=[ECANGameWorld.ec_item("b-rollback", "rollback", 8, 1)],
                stimulate={"b-rollback": 10.0},
            )],
            expectation=GameTurnExpectation(
                expected_top_attention_belief="b-rollback",
                description="rollback should be top attention after repeated stimulus",
            ),
        ))
        self.assertTrue(result3.checks_passed, f"Turn 3 failed: {result3.check_errors}")
        self.assertEqual(result3.top_attention_belief, "b-rollback")

        # All turns should pass
        self.assertTrue(game.all_passed(), game.summary())

    def test_decay_and_forgetting_scenario(self):
        """Beliefs that lose attention should decay and become forget candidates."""
        game = ECANGameWorld(goalchainer_repo=REPO)

        # Turn 1: Two beliefs get evidence, both in AF
        result1 = game.play_turn(GameTurn(
            label="two-beliefs-active",
            events=[GameEvent(
                label="both-stimulated",
                add_evidence=[
                    ECANGameWorld.stv_item("b-a", "deploy", 0.9, 0.8),
                    ECANGameWorld.stv_item("b-b", "rollback", 0.7, 0.6),
                ],
                stimulate={"b-a": 15.0, "b-b": 10.0},
            )],
            expectation=GameTurnExpectation(
                expected_af_contains=["b-a"],
                description="Both beliefs should be active",
            ),
        ))
        self.assertTrue(result1.checks_passed, f"Turn 1 failed: {result1.check_errors}")

        # Turn 2: Only b-a gets stimulus, b-b decays over many cycles
        result2 = game.play_turn(GameTurn(
            label="b-a-stays-b-b-decays",
            events=[GameEvent(
                label="time-passes",
                stimulate={"b-a": 10.0},
                extra_cycles=25,
            )],
            expectation=GameTurnExpectation(
                expected_top_attention_belief="b-a",
                description="b-a should remain top after continued stimulation",
            ),
        ))
        self.assertTrue(result2.checks_passed, f"Turn 2 failed: {result2.check_errors}")

        # Turn 3: b-b should now be a forget candidate
        result3 = game.play_turn(GameTurn(
            label="b-b-forgotten",
            events=[GameEvent(
                label="more-time-passes",
                stimulate={"b-a": 5.0},
                extra_cycles=30,
            )],
            expectation=GameTurnExpectation(
                expected_forget_candidates=["b-b"],
                description="b-b should be a forget candidate after prolonged decay",
            ),
        ))
        self.assertTrue(result3.checks_passed, f"Turn 3 failed: {result3.check_errors}")

    def test_evidence_removal_scenario(self):
        """Removing evidence for a belief should reduce its relevance."""
        game = ECANGameWorld(goalchainer_repo=REPO)

        # Turn 1: Two beliefs with evidence
        result1 = game.play_turn(GameTurn(
            label="both-active",
            events=[GameEvent(
                label="evidence-added",
                add_evidence=[
                    ECANGameWorld.stv_item("b-keep", "deploy", 0.9, 0.8),
                    ECANGameWorld.stv_item("b-remove", "rollback", 0.8, 0.7),
                ],
                stimulate={"b-keep": 10.0, "b-remove": 10.0},
            )],
            expectation=GameTurnExpectation(
                expected_af_contains=["b-keep", "b-remove"],
                description="Both should be in AF initially",
            ),
        ))
        self.assertTrue(result1.checks_passed, f"Turn 1 failed: {result1.check_errors}")

        # Turn 2: Remove evidence for b-remove
        result2 = game.play_turn(GameTurn(
            label="evidence-removed",
            events=[GameEvent(
                label="remove-rollback-evidence",
                remove_evidence=["b-remove"],
                stimulate={"b-keep": 5.0},
            )],
            expectation=GameTurnExpectation(
                expected_af_contains=["b-keep"],
                description="b-keep should remain in AF",
            ),
        ))
        self.assertTrue(result2.checks_passed, f"Turn 2 failed: {result2.check_errors}")
        # b-remove should no longer have evidence
        self.assertNotIn("b-remove", [it.belief_id for it in game.items])

    def test_multi_belief_priority_scenario(self):
        """Multiple beliefs compete for attention; strongest wins."""
        game = ECANGameWorld(goalchainer_repo=REPO)

        # Add 5 beliefs with varying stimulus
        beliefs = ["b1", "b2", "b3", "b4", "b5"]
        items = [ECANGameWorld.stv_item(b, f"action_{b}", 0.8, 0.7) for b in beliefs]
        stimuli = {"b1": 5.0, "b2": 8.0, "b3": 15.0, "b4": 3.0, "b5": 12.0}

        result = game.play_turn(GameTurn(
            label="five-beliefs-compete",
            events=[GameEvent(
                label="all-arrive",
                add_evidence=items,
                stimulate=stimuli,
            )],
            expectation=GameTurnExpectation(
                expected_top_attention_belief="b3",
                description="b3 has highest stimulus, should be top",
            ),
        ))
        self.assertTrue(result.checks_passed, f"Failed: {result.check_errors}")

        # Verify priority ordering matches stimulus ordering
        prioritized = game.bridge.get_prioritized_beliefs(limit=5)
        top3 = [p[0] for p in prioritized[:3]]
        self.assertIn("b3", top3)
        self.assertIn("b5", top3)

    def test_game_summary_works(self):
        """The summary method should produce human-readable output."""
        game = ECANGameWorld(goalchainer_repo=REPO)
        game.play_turn(GameTurn(
            label="test-turn",
            events=[GameEvent(
                label="test-event",
                add_evidence=[ECANGameWorld.stv_item("b1", "deploy", 0.9, 0.8)],
                stimulate={"b1": 10.0},
            )],
        ))
        summary = game.summary()
        self.assertIn("Turn 1", summary)
        self.assertIn("test-turn", summary)

    def test_all_passed_after_successful_game(self):
        """all_passed should return True when all turns pass."""
        game = ECANGameWorld(goalchainer_repo=REPO)
        game.play_turn(GameTurn(
            label="passing-turn",
            events=[GameEvent(
                label="test",
                add_evidence=[ECANGameWorld.stv_item("b1", "deploy", 0.9, 0.8)],
                stimulate={"b1": 10.0},
            )],
            expectation=GameTurnExpectation(
                expected_top_attention_belief="b1",
            ),
        ))
        self.assertTrue(game.all_passed())

    def test_play_scenarios_returns_bool(self):
        """play_scenarios should return True when all turns pass."""
        game = ECANGameWorld(goalchainer_repo=REPO)
        turns = [
            GameTurn(
                label="turn-1",
                events=[GameEvent(
                    label="e1",
                    add_evidence=[ECANGameWorld.stv_item("b1", "deploy", 0.9, 0.8)],
                    stimulate={"b1": 10.0},
                )],
                expectation=GameTurnExpectation(expected_top_attention_belief="b1"),
            ),
            GameTurn(
                label="turn-2",
                events=[GameEvent(
                    label="e2",
                    add_evidence=[ECANGameWorld.stv_item("b2", "rollback", 0.8, 0.7)],
                    stimulate={"b2": 20.0},
                )],
                expectation=GameTurnExpectation(expected_top_attention_belief="b2"),
            ),
        ]
        result = game.play_scenarios(turns)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
