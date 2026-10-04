"""Auswertung: Live-Lauf, Kette-Auswahl, Zellen-Kennzahlen von Hand, Vollständigkeit der vorgerechneten Datei."""

import pytest

import prio_constants as C
import prio_evaluation as E
import prio_formulas as F
from prio_simulation import SimResult


def fake_run(hi, lo, both):
    return SimResult(c=1, rho=0.8, p=0.2, policy="np", cs2=1.0, n=100, n_hi=20, n_lo=80, delay_hi=hi, delay_lo=lo, delay_all=both)


def test_mean_and_sd_by_hand():
    assert E.mean_and_sd([1.0, 3.0]) == (2.0, pytest.approx(2 ** 0.5)) and E.mean_and_sd([5.0]) == (5.0, 0.0)


def test_formulas_collects_all_reference_values():
    f = E.formulas(1, 90, 20, "np", 1.0)
    assert f["waits"] == pytest.approx([F.np_waits(1, 0.9, 0.2, 1.0)[0], F.np_waits(1, 0.9, 0.2, 1.0)[1]]) and f["fifo"] == pytest.approx(9.0)
    assert f["total"] == pytest.approx(9.0)                                                  # Erhaltungssatz
    none = E.formulas(4, 80, 20, "pr", 4.0)
    assert none["waits"] is None and none["total"] is None and none["fifo"] is None


def test_cell_figures_by_hand():
    cell = E.summarize_runs(1, 80, 20, "np", 1.0, 1000, [fake_run(1.0, 5.0, 4.2), fake_run(3.0, 7.0, 6.2)])
    assert E.mean_of(cell, "hi") == 2.0 and E.mean_of(cell, "lo") == 6.0 and E.mean_of(cell, "all") == pytest.approx(5.2) and E.se_of(cell, "hi") == pytest.approx(1.0)
    fifo = E.summarize_runs(1, 80, 20, "fifo", 1.0, 1000, [fake_run(4.0, 4.0, 4.0), fake_run(6.0, 6.0, 6.0)])
    assert E.speedup_vs_fifo(cell, fifo) == pytest.approx(2.5) and E.penalty_vs_fifo(cell, fifo) == pytest.approx(0.2)


def test_chain_report_selects_the_matching_chain():
    nonpre, kind = E.chain_report(1, 50, 20, "np")
    assert kind == "np" and nonpre.delay1 == pytest.approx(F.np_waits(1, 0.5, 0.2, 1.0)[0], rel=1e-4)
    for c, policy in ((1, "pr"), (4, "pr"), (4, "np"), (1, "fifo")):
        chain, kind = E.chain_report(c, 50, 20, policy)
        assert kind == "pr" and chain.c == c


def test_nearest_picks_the_closest_option_and_the_smaller_on_a_tie():
    assert E.nearest(C.STUDY_RHO_PCT, 60) == 50 and E.nearest(C.STUDY_RHO_PCT, 65) == 50 and E.nearest(C.STUDY_RHO_PCT, 66) == 80 and E.nearest(C.STUDY_RHO_PCT, 95) == 90


def test_live_report_structure_and_reference_values():
    r = E.live_report(4, 80, 20, "pr", 1.0, seed=3, customers=8_000)
    assert r["waits"] == pytest.approx(list(F.pr_waits(4, 0.8, 0.2, 1.0))) and r["fifo"] == pytest.approx(F.mmc_wait(4, 0.8))
    assert r["delay_hi"] == r["sim"].delay_hi and r["delay_lo"] == r["sim"].delay_lo and r["sim"].n == 8_000 - int(0.05 * 8_000)


def test_study_cell_run_small_structure():
    cell = E.study_cell_run(1, 80, 20, "np", 1.0, 10_000, 5, 3)
    assert cell["reps"] == 3 and len(cell["hi"]) == len(cell["lo"]) == len(cell["all"]) == 3 and cell["n"] == 10_000
    assert cell["waits"] == pytest.approx(list(F.np_waits(1, 0.8, 0.2, 1.0)))


def test_precomputed_file_is_complete():
    pre = E.load_precomputed()
    assert {(x["c"], x["rho_pct"], x["policy"], x["cs2"]) for x in pre["study"]} == {(c, r, pol, s) for c in C.STUDY_C for r in C.STUDY_RHO_PCT for pol in C.POLICIES for s in C.STUDY_CS2}
    assert {(x["p_pct"], x["policy"]) for x in pre["pstudy"]} == {(p, pol) for p in C.PSTUDY_P_PCT for pol in C.POLICIES}
    assert pre["study_customers"] == C.STUDY_CUSTOMERS and pre["study_reps"] == C.STUDY_REPS
    for x in pre["study"] + pre["pstudy"]:
        assert x["reps"] == C.STUDY_REPS and len(x["hi"]) == C.STUDY_REPS and x["n"] == C.STUDY_CUSTOMERS
    for x in pre["study"]:
        assert x["p_pct"] == C.STUDY_P_PCT
    for x in pre["pstudy"]:
        assert x["c"] == 1 and x["rho_pct"] == C.PSTUDY_RHO_PCT and x["cs2"] == 1.0


def test_lookups_and_missing_cells():
    pre = E.load_precomputed()
    assert E.study_cell(pre, 1, 80, "np", 1.0)["policy"] == "np" and E.pstudy_cell(pre, 50, "pr")["p_pct"] == 50
    with pytest.raises(KeyError):
        E.study_cell(pre, 2, 80, "np", 1.0)
    with pytest.raises(KeyError):
        E.pstudy_cell(pre, 30, "np")
