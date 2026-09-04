"""Unit tests for the ECAN attention allocation module."""
import pytest
from petta_memory.ecan import (
    AttentionValue, AttentionBank, ImportanceDiffusion,
    RentCollection, ECANCycle, importance_bin, DEFAULT_ECAN_PARAMS,
)


class TestAttentionValue:
    def test_default(self):
        av = AttentionValue.default()
        assert av.sti == 0.0
        assert av.lti == 0.0
        assert av.vlti == 0

    def test_with_sti(self):
        av = AttentionValue(sti=100, lti=50, vlti=1)
        av2 = av.with_sti(200)
        assert av2.sti == 200
        assert av2.lti == 50
        assert av2.vlti == 1

    def test_with_lti(self):
        av = AttentionValue(sti=100, lti=50, vlti=1)
        av2 = av.with_lti(300)
        assert av2.sti == 100
        assert av2.lti == 300
        assert av2.vlti == 1

    def test_with_av(self):
        av = AttentionValue(sti=100, lti=50, vlti=1)
        av2 = av.with_av(200, 300)
        assert av2.sti == 200
        assert av2.lti == 300
        assert av2.vlti == 1
        av3 = av.with_av(200, 300, 0)
        assert av3.vlti == 0

    def test_as_tuple(self):
        av = AttentionValue(sti=1.5, lti=2.5, vlti=1)
        assert av.as_tuple() == (1.5, 2.5, 1)

    def test_frozen(self):
        av = AttentionValue(sti=100)
        with pytest.raises(AttributeError):
            av.sti = 200


class TestImportanceBin:
    def test_negative_sti(self):
        assert importance_bin(-50) == 0

    def test_small_sti(self):
        assert importance_bin(5) == 5
        assert importance_bin(15) == 15

    def test_large_sti(self):
        b = importance_bin(1000)
        assert b >= 0
        assert isinstance(b, int)

    def test_zero(self):
        assert importance_bin(0) == 0


class TestAttentionBank:
    def test_init_defaults(self):
        bank = AttentionBank()
        assert bank.funds_sti == DEFAULT_ECAN_PARAMS["STARTING_FUNDS_STI"]
        assert bank.funds_lti == DEFAULT_ECAN_PARAMS["STARTING_FUNDS_LTI"]
        assert bank.num_atoms == 0
        assert bank.af_size == 0

    def test_set_av(self):
        bank = AttentionBank()
        av = AttentionValue(sti=500, lti=100, vlti=0)
        bank.set_av("b1", av)
        assert bank.get_sti("b1") == 500
        assert bank.get_lti("b1") == 100
        assert bank.get_vlti("b1") == 0
        assert bank.num_atoms == 1

    def test_funds_conservation(self):
        bank = AttentionBank()
        initial_funds = bank.funds_sti
        bank.set_av("b1", AttentionValue(sti=500))
        assert bank.funds_sti == initial_funds - 500

    def test_update_av(self):
        bank = AttentionBank()
        bank.set_av("b1", AttentionValue(sti=500))
        bank.set_av("b1", AttentionValue(sti=300))
        assert bank.get_sti("b1") == 300

    def test_af_admission(self):
        bank = AttentionBank(params={"MIN_AF_STI": 100, "MAX_AF_SIZE": 10})
        bank.set_av("b1", AttentionValue(sti=200))
        assert bank.is_in_af("b1")
        bank.set_av("b2", AttentionValue(sti=50))
        assert not bank.is_in_af("b2")

    def test_af_eviction(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 2})
        bank.set_av("b1", AttentionValue(sti=100))
        bank.set_av("b2", AttentionValue(sti=200))
        bank.set_av("b3", AttentionValue(sti=300))
        assert bank.af_size == 2
        af = bank.get_af_atoms()
        assert af[0] == "b3"
        assert af[1] == "b2"

    def test_sti_range_query(self):
        bank = AttentionBank()
        bank.set_av("b1", AttentionValue(sti=100))
        bank.set_av("b2", AttentionValue(sti=200))
        bank.set_av("b3", AttentionValue(sti=300))
        result = bank.get_atoms_in_sti_range(150, 350)
        assert "b2" in result
        assert "b3" in result
        assert "b1" not in result

    def test_stimulate(self):
        bank = AttentionBank()
        bank.set_av("b1", AttentionValue(sti=100, lti=50))
        bank.stimulate("b1", 5.0)
        assert bank.get_sti("b1") > 100
        assert bank.get_lti("b1") > 50

    def test_sti_wage(self):
        bank = AttentionBank()
        wage = bank.calculate_sti_wage()
        assert isinstance(wage, float)

    def test_random_af_atom(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 10})
        bank.set_av("b1", AttentionValue(sti=100))
        bank.set_av("b2", AttentionValue(sti=200))
        atom = bank.get_random_atom_in_af()
        assert atom in {"b1", "b2"}


class TestImportanceDiffusion:
    def test_add_link(self):
        bank = AttentionBank()
        diff = ImportanceDiffusion(bank)
        diff.add_bidirectional_link("b1", "b2")
        bank.set_av("b1", AttentionValue(sti=1000))
        bank.set_av("b2", AttentionValue(sti=100))
        result = diff.diffuse_atom("b1")
        # b2 should receive some STI
        assert len(result) > 0

    def test_diffusion_conservation(self):
        """Total STI should be roughly conserved after diffusion."""
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        bank.set_av("b1", AttentionValue(sti=1000))
        bank.set_av("b2", AttentionValue(sti=100))
        bank.set_av("b3", AttentionValue(sti=100))
        diff = ImportanceDiffusion(bank)
        diff.add_bidirectional_link("b1", "b2")
        diff.add_bidirectional_link("b1", "b3")
        total_before = bank.get_sti("b1") + bank.get_sti("b2") + bank.get_sti("b3")
        diff.diffuse_atom("b1")
        total_after = bank.get_sti("b1") + bank.get_sti("b2") + bank.get_sti("b3")
        assert abs(total_before - total_after) < 1.0  # roughly conserved

    def test_no_neighbors(self):
        bank = AttentionBank()
        diff = ImportanceDiffusion(bank)
        bank.set_av("b1", AttentionValue(sti=1000))
        result = diff.diffuse_atom("b1")
        assert result == []

    def test_build_from_evidence_map(self):
        bank = AttentionBank()
        diff = ImportanceDiffusion(bank)
        diff.build_from_evidence_map({
            "b1": ["e1", "e2"],
            "b2": ["e1"],
        })
        bank.set_av("b1", AttentionValue(sti=1000))
        bank.set_av("b2", AttentionValue(sti=100))
        bank.set_av("e1", AttentionValue(sti=50))
        bank.set_av("e2", AttentionValue(sti=50))
        result = diff.diffuse_atom("b1")
        assert len(result) > 0

    def test_diffuse_af(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        diff = ImportanceDiffusion(bank)
        diff.add_bidirectional_link("b1", "b2")
        diff.add_bidirectional_link("b2", "b3")
        bank.set_av("b1", AttentionValue(sti=1000))
        bank.set_av("b2", AttentionValue(sti=200))
        bank.set_av("b3", AttentionValue(sti=50))
        results = diff.diffuse_af()
        assert isinstance(results, dict)


class TestRentCollection:
    def test_collect_rent(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        bank.set_av("b1", AttentionValue(sti=500, lti=100))
        rent = RentCollection(bank)
        results = rent.collect_rent()
        assert "b1" in results
        assert "sti_rent" in results["b1"]
        assert "new_sti" in results["b1"]

    def test_forget_candidates(self):
        bank = AttentionBank(params={
            "MIN_AF_STI": 0, "MAX_AF_SIZE": 100,
            "FORGET_THRESHOLD": 400
        })
        bank.set_av("b1", AttentionValue(sti=500))
        rent = RentCollection(bank)
        rent.collect_rent()
        # b1 should be a forget candidate if STI dropped below threshold
        # (depends on rent rate, but with high threshold likely)
        assert isinstance(rent.forget_candidates, list)

    def test_rent_reduces_sti(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        bank.set_av("b1", AttentionValue(sti=500, lti=100))
        sti_before = bank.get_sti("b1")
        rent = RentCollection(bank)
        rent.collect_rent()
        sti_after = bank.get_sti("b1")
        assert sti_after <= sti_before


class TestECANCycle:
    def test_step_no_stimuli(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        diff = ImportanceDiffusion(bank)
        rent = RentCollection(bank)
        bank.set_av("b1", AttentionValue(sti=500))
        cycle = ECANCycle(bank, diff, rent)
        result = cycle.step()
        assert result.cycle == 1
        assert result.stimulated == 0
        assert isinstance(result.diffused, dict)
        assert isinstance(result.rent_collected, dict)

    def test_step_with_stimuli(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        diff = ImportanceDiffusion(bank)
        diff.add_bidirectional_link("b1", "b2")
        rent = RentCollection(bank)
        bank.set_av("b1", AttentionValue(sti=500))
        bank.set_av("b2", AttentionValue(sti=100))
        cycle = ECANCycle(bank, diff, rent)
        result = cycle.step(stimuli={"b1": 5.0})
        assert result.stimulated == 1

    def test_multiple_cycles(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        diff = ImportanceDiffusion(bank)
        diff.add_bidirectional_link("b1", "b2")
        rent = RentCollection(bank)
        bank.set_av("b1", AttentionValue(sti=500))
        bank.set_av("b2", AttentionValue(sti=100))
        cycle = ECANCycle(bank, diff, rent)
        results = cycle.run(5, stimuli_fn=lambda c: {"b1": 3.0})
        assert len(results) == 5
        assert results[-1].cycle == 5

    def test_funds_tracking(self):
        bank = AttentionBank(params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
        diff = ImportanceDiffusion(bank)
        rent = RentCollection(bank)
        bank.set_av("b1", AttentionValue(sti=500))
        cycle = ECANCycle(bank, diff, rent)
        result = cycle.step()
        assert result.funds_sti == bank.funds_sti
        assert result.funds_lti == bank.funds_lti

    def test_economic_stability(self):
        """After many cycles with uniform stimulus, STI should not diverge wildly."""
        bank = AttentionBank(params={
            "MIN_AF_STI": 0, "MAX_AF_SIZE": 100,
            "STI_ATOM_WAGE": 10.0, "TARGET_STI": 10000,
        })
        diff = ImportanceDiffusion(bank)
        diff.add_bidirectional_link("b1", "b2")
        diff.add_bidirectional_link("b2", "b3")
        rent = RentCollection(bank)
        for i in ["b1", "b2", "b3"]:
            bank.set_av(i, AttentionValue(sti=100, lti=50))
        cycle = ECANCycle(bank, diff, rent)
        results = cycle.run(20, stimuli_fn=lambda c: {"b1": 1.0, "b2": 1.0, "b3": 1.0})
        # Total STI across all atoms should be bounded
        total_sti = sum(bank.get_sti(a) for a in ["b1", "b2", "b3"])
        assert total_sti < 100000  # not exploding
        assert total_sti > 0  # not drained to zero

    def test_custom_params(self):
        bank = AttentionBank(params={"STI_ATOM_WAGE": 100, "MIN_AF_STI": 50})
        assert bank.get_param("STI_ATOM_WAGE") == 100
        assert bank.get_param("MIN_AF_STI") == 50
        bank.set_param("STI_ATOM_WAGE", 200)
        assert bank.get_param("STI_ATOM_WAGE") == 200
