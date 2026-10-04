"""Simulation: Sampler einzeln, Mini-Instanz von Hand für alle drei Reihenfolgen, Verdrängung (Auswahl des Opfers, Restdauer), Invarianten, Reproduzierbarkeit, Vergleich gegen die Formeln und
gegen die unabhängige Markov-Kette, Erhaltungssatz in der Simulation."""

import pytest

import prio_chain as K
import prio_constants as C
import prio_formulas as F
import prio_simulation as S
from conftest import ScriptedRng


# ---------------------------------------------------------------- Einheit: Sampler

def test_sampler_by_hand():
    assert S.make_sampler(2.5, 0.0, ScriptedRng())() == 2.5
    assert S.make_sampler(2.0, 1.0, ScriptedRng(exp_values=[0.7]))() == 0.7
    assert S.balanced_h2_probability(1.0) == pytest.approx(0.5) and S.balanced_h2_probability(4.0) == pytest.approx(0.5 * (1 + (3 / 5) ** 0.5))


@pytest.mark.parametrize("scv", C.CS2_OPTIONS)
def test_every_sampler_has_mean_one_and_the_requested_scv(scv):
    rng = S.SplitMix64(7)
    draw = S.make_sampler(1.0, scv, rng)
    xs = [draw() for _ in range(400_000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert mean == pytest.approx(1.0, abs=0.012) and var / mean ** 2 == pytest.approx(scv, abs=0.08 if scv > 1 else 0.02)


# ---------------------------------------------------------------- Mini-Instanz von Hand (siehe conftest)

def _run(fixture, policy, p=0.5, c=1):
    gaps, pick, svc = fixture
    return S.simulate(c, 0.5, p, policy, 1.0, 4, seed=0, warm_fraction=0.0, samplers=((lambda: gaps.expovariate(1.0)), (lambda: svc.expovariate(1.0)), pick.uniform))


def test_non_preemptive_mini_instance_by_hand(priority_mini):
    """Eilige 2.3, Standard (0 + 3.5 + 4.2)/3, gesamt 2.5."""
    r = _run(priority_mini, "np")
    assert (r.n, r.n_hi, r.n_lo) == (4, 1, 3)
    assert r.delay_hi == pytest.approx(2.3) and r.delay_lo == pytest.approx((0 + 3.5 + 4.2) / 3) and r.delay_all == pytest.approx(2.5)


def test_fifo_mini_instance_by_hand(priority_mini):
    """FIFO: Lkw 2 startet bei 4.0 (2.5), Lkw 3 (eilig) bei 5.0 (3.3), Lkw 4 bei 6.0 (4.2): Eilige 3.3, Standard (0 + 2.5 + 4.2)/3, gesamt 2.5 wie bei Vorfahrt (gleiche Dauern)."""
    r = _run(priority_mini, "fifo")
    assert r.delay_hi == pytest.approx(3.3) and r.delay_lo == pytest.approx((0 + 2.5 + 4.2) / 3) and r.delay_all == pytest.approx(2.5)


def test_preemptive_mini_instance_by_hand(priority_mini):
    """Lkw 3 verdrängt Lkw 1 bei 1.7 (Rest 2.3): Eilige 0, Lkw 1 endet bei 5.0 (Verzögerung 1.0), Lkw 2 und 4 mit 3.5 und 4.2; Standard 2.9, gesamt 2.175."""
    r = _run(priority_mini, "pr")
    assert r.delay_hi == pytest.approx(0.0) and r.delay_lo == pytest.approx((1.0 + 3.5 + 4.2) / 3) and r.delay_all == pytest.approx(2.175)


def test_priority_does_not_change_the_total_for_equal_service_times_by_hand(priority_mini):
    gaps_fixture = priority_mini
    a = _run(gaps_fixture, "np").delay_all
    assert a == pytest.approx(2.5)


def test_the_most_recently_started_standard_job_is_preempted(monkeypatch):
    """c = 2: zwei Standard-Lkw laufen (A ab 0.1, B ab 0.2); ein Eiliger bei 0.5 (Dauer 0.5) verdrängt B (zuletzt gestartet): B hat 0.3 geleistet, Rest 3.0 − 0.3 = 2.7.
    Der Eilige endet bei 1.0 (Verzögerung 0), B läuft ab 1.0 weiter und endet bei 3.7 (Verzögerung 3.7 − 0.2 − 3.0 = 0.5); A endet bei 3.1 ohne Verzögerung."""
    captured = {}
    original = S.Gate

    def capturing_gate(*a, **k):
        captured["gate"] = original(*a, **k)
        return captured["gate"]

    monkeypatch.setattr(S, "Gate", capturing_gate)
    gaps, pick, svc = ScriptedRng(exp_values=[0.1, 0.1, 0.3, 100.0]), ScriptedRng(uniform_values=[0.9, 0.9, 0.1]), ScriptedRng(exp_values=[3.0, 3.0, 0.5])
    res = S.simulate(2, 0.5, 0.5, "pr", 1.0, 3, seed=0, warm_fraction=0.0, samplers=((lambda: gaps.expovariate(1.0)), (lambda: svc.expovariate(1.0)), pick.uniform))
    done = {ident: (cls, d) for cls, d, ident in captured["gate"].done}
    assert done[0] == (2, pytest.approx(0.0)) and done[1] == (2, pytest.approx(0.5)) and done[2] == (1, pytest.approx(0.0))
    assert res.delay_hi == 0.0 and res.delay_lo == pytest.approx(0.25)


def test_a_class_two_job_never_preempts_and_class_one_never_waits_behind_class_two_in_the_queue():
    """Bei np und pr stehen Eilige vor Standard-Lkw in der Schlange: ein später ankommender Eiliger startet vor einem früheren Standard-Lkw."""
    gaps = ScriptedRng(exp_values=[1.0, 0.1, 0.1, 100.0])
    pick = ScriptedRng(uniform_values=[0.9, 0.9, 0.1])
    svc = ScriptedRng(exp_values=[2.0, 1.0, 1.0])
    res = S.simulate(1, 0.5, 0.5, "np", 1.0, 3, seed=0, warm_fraction=0.0, samplers=((lambda: gaps.expovariate(1.0)), (lambda: svc.expovariate(1.0)), pick.uniform))
    # A bei 1.0 (Ende 3.0), B (Standard) bei 1.1, C (eilig) bei 1.2: bei 3.0 startet C (Verzögerung 1.8), danach B (3.0 → 4.0 vorbei, startet 4.0, Verzögerung 2.9)
    assert res.delay_hi == pytest.approx(1.8) and res.delay_lo == pytest.approx((0.0 + 2.9) / 2)


# ---------------------------------------------------------------- Invarianten und Reproduzierbarkeit

@pytest.mark.parametrize("policy", C.POLICIES)
def test_every_job_is_served_and_counted_once(policy):
    r = S.simulate(2, 0.8, 0.3, policy, 1.0, 20_000, 5)
    assert r.n == 20_000 - int(0.05 * 20_000) and r.n_hi + r.n_lo == r.n and 0.2 < r.n_hi / r.n < 0.4
    assert r.delay_all == pytest.approx((r.n_hi * r.delay_hi + r.n_lo * r.delay_lo) / r.n)


def test_same_seed_same_result_and_different_seed_differs():
    a, b, c = S.simulate(1, 0.8, 0.2, "pr", 1.0, 20_000, 5), S.simulate(1, 0.8, 0.2, "pr", 1.0, 20_000, 5), S.simulate(1, 0.8, 0.2, "pr", 1.0, 20_000, 6)
    assert a == b and a.delay_lo != c.delay_lo


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        S.simulate(1, 0.5, 0.2, "lifo", 1.0, 100, 1)


def test_fifo_treats_both_classes_alike():
    r = S.simulate(1, 0.8, 0.2, "fifo", 1.0, 200_000, 3)
    assert r.delay_hi == pytest.approx(r.delay_lo, rel=0.08)


# ---------------------------------------------------------------- Referenz: Formeln und Markov-Kette

def _band(c, rho, p, policy, cs2, reps=4, n=200_000, seed0=100):
    runs = [S.simulate(c, rho, p, policy, cs2, n, seed0 + 17 * r) for r in range(reps)]
    out = []
    for key in ("delay_hi", "delay_lo"):
        vals = [getattr(r, key) for r in runs]
        m = sum(vals) / reps
        se = (sum((x - m) ** 2 for x in vals) / (reps - 1) / reps) ** 0.5
        out.append((m, se))
    return out


@pytest.mark.parametrize("c,policy,cs2", [(1, "np", 1.0), (1, "pr", 1.0), (1, "np", 4.0), (1, "pr", 4.0), (4, "np", 1.0), (4, "pr", 1.0)])
def test_simulation_matches_the_formulas(c, policy, cs2):
    exact = F.class_waits(policy, c, 0.8, 0.2, cs2)
    for (m, se), e in zip(_band(c, 0.8, 0.2, policy, cs2), exact):
        assert abs(m - e) < max(4 * se, 0.04 * e), (c, policy, cs2, e, m, se)


@pytest.mark.parametrize("policy", ["np", "pr"])
def test_simulation_matches_the_independent_markov_chain(policy):
    chain = K.np1_chain(0.8, 0.2) if policy == "np" else K.pr_chain(1, 0.8, 0.2)
    for (m, se), e in zip(_band(1, 0.8, 0.2, policy, 1.0), (chain.delay1, chain.delay2)):
        assert abs(m - e) < max(4 * se, 0.04 * e), (policy, e, m, se)


def test_conservation_law_in_the_simulation():
    """Nicht unterbrechende Vorfahrt: Gesamtmittel gleich FIFO (gleiche Dauer in beiden Klassen), aus denselben Strömen (nur die Reihenfolge ändert sich)."""
    fifo = S.simulate(1, 0.8, 0.2, "fifo", 1.0, 300_000, 11).delay_all
    nonpre = S.simulate(1, 0.8, 0.2, "np", 1.0, 300_000, 11).delay_all
    assert nonpre == pytest.approx(fifo, rel=0.01)


def test_preemption_lowers_the_total_for_variable_service_in_the_simulation():
    fifo = S.simulate(1, 0.8, 0.2, "fifo", 4.0, 300_000, 13).delay_all
    pre = S.simulate(1, 0.8, 0.2, "pr", 4.0, 300_000, 13).delay_all
    assert pre < fifo
