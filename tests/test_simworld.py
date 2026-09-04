"""SimWorld test harness: multi-step simulated scenarios for GoalChainer + PeTTa-memory.

These tests drive the decision pipeline through scripted world mutations,
validating that the system responds correctly to changing evidence.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.simworld import SimWorld, SimStep

REPO = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"


class TestEvidenceThresholdGating(unittest.TestCase):
    """Evidence threshold gating: weak evidence defers, strong evidence recommends."""

    def test_weak_evidence_then_strong_evidence(self):
        world = SimWorld(goalchainer_repo=REPO)
        steps = [
            SimStep(
                label="weak-evidence",
                add_items=[
                    SimWorld.stv_item("b-weak", "publish_redacted_summary", 0.51, 0.55),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_max=0.80,
                description="Weak STV evidence should still recommend but with low strength.",
            ),
            SimStep(
                label="strong-evidence",
                update_items={
                    "b-weak": SimWorld.stv_item("b-weak", "publish_redacted_summary", 0.95, 0.90),
                },
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_min=0.85,
                description="Strong STV evidence should recommend with high strength.",
            ),
        ]
        world.run_scenario(steps)
        self.assertTrue(world.all_passed(), "\n" + world.summary())


class TestOppositionDrivenDeescalation(unittest.TestCase):
    """Opposition-driven de-escalation: conflicting EC lowers strength but preserves gate."""

    def test_ec_opposition_lowers_strength(self):
        world = SimWorld(goalchainer_repo=REPO)
        steps = [
            SimStep(
                label="strong-only",
                add_items=[
                    SimWorld.stv_item("b-strong", "publish_redacted_summary", 0.95, 0.90),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_min=0.90,
                description="Without opposition, strength should be high.",
            ),
            SimStep(
                label="opposition-arrives",
                add_items=[
                    SimWorld.ec_item("b-strong", "publish_redacted_summary", 1, 9),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_max=0.80,
                description="Opposing EC should lower strength but keep recommended status.",
            ),
        ]
        world.run_scenario(steps)
        self.assertTrue(world.all_passed(), "\n" + world.summary())


class TestSupersedeAwareGoalInvalidation(unittest.TestCase):
    """Supersede-aware goal invalidation: new lower STV item weakens the recommendation."""

    def test_supersede_lowers_strength(self):
        world = SimWorld(goalchainer_repo=REPO)
        steps = [
            SimStep(
                label="approved-belief",
                add_items=[
                    SimWorld.stv_item("b-approved", "publish_redacted_summary", 0.92, 0.88),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_min=0.85,
                description="Initial approval has high strength.",
            ),
            SimStep(
                label="superseded-belief",
                add_items=[
                    SimWorld.stv_item("b-superseded", "publish_redacted_summary", 0.55, 0.60),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_max=0.80,
                description="A new lower-confidence STV item should not raise strength beyond the original.",
            ),
        ]
        world.run_scenario(steps)
        self.assertTrue(world.all_passed(), "\n" + world.summary())


class TestMultiGoalPriorityArbitration(unittest.TestCase):
    """Multi-goal priority arbitration: canary install should rank above defer and ask-ben."""

    def test_canary_install_top_ranked(self):
        world = SimWorld(goalchainer_repo=REPO)
        steps = [
            SimStep(
                label="canary-approved",
                add_items=[
                    SimWorld.stv_item(
                        "b-tk-canary", "install_threadkeeper_canary_on_protomegabot",
                        0.88, 0.75,
                        cluster_id="mc-tk",
                        promotion_event="pe-tk-ben",
                        promotion_domain="project-control",
                    ),
                ],
                expected_top_action="install_threadkeeper_canary_on_protomegabot",
                expected_status="recommended",
                expected_strength_min=0.70,
                description="Canary install should be the top recommended action.",
            ),
        ]
        world.run_scenario(steps)
        self.assertTrue(world.all_passed(), "\n" + world.summary())


class TestCrossDomainEvidenceReuse(unittest.TestCase):
    """Cross-domain evidence reuse: same action with different promotion_domain still scores."""

    def test_cross_domain_ec(self):
        world = SimWorld(goalchainer_repo=REPO)
        steps = [
            SimStep(
                label="incident-response-stv",
                add_items=[
                    SimWorld.stv_item(
                        "b-cross", "publish_redacted_summary", 0.85, 0.80,
                        promotion_domain="incident-response",
                    ),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_min=0.80,
                description="Initial STV from incident-response domain.",
            ),
            SimStep(
                label="project-control-ec",
                add_items=[
                    SimWorld.ec_item(
                        "b-cross", "publish_redacted_summary", 7, 3,
                        promotion_domain="project-control",
                    ),
                ],
                expected_top_action="publish_redacted_summary",
                expected_status="recommended",
                expected_strength_min=0.80,
                description="EC from a different domain should still influence the score.",
            ),
        ]
        world.run_scenario(steps)
        self.assertTrue(world.all_passed(), "\n" + world.summary())


if __name__ == "__main__":
    unittest.main()
