"""Formeln: Cobham und Preemptive-Resume von Hand, Erhaltungssatz, Sonderfälle (exponentielle Dauer), Fälle ohne Formel, Hilfsfunktionen."""

import pytest

import prio_formulas as F


def test_cobham_by_hand():
    """ρ = 0.5, p = 0.2, exponentiell: W₀ = 0.5, σ₁ = 0.1 → W₁ = 0.5/0.9 = 0.5556, W₂ = 0.5/(0.9·0.5) = 1.1111; FIFO 1.0."""
    w1, w2 = F.np_waits(1, 0.5, 0.2, 1.0)
    assert w1 == pytest.approx(0.5 / 0.9) and w2 == pytest.approx(0.5 / (0.9 * 0.5)) and F.fifo_wait(1, 0.5, 1.0) == pytest.approx(1.0)


def test_preemptive_resume_by_hand():
    """ρ = 0.5, p = 0.2, exponentiell: R₁ = 0.1, T₁ = 1 + 0.1/0.9 → Verzögerung 0.1111; R₂ = 0.5, T₂ = 1/0.9 + 0.5/(0.9·0.5) = 2.2222 → Verzögerung 1.2222."""
    d1, d2 = F.pr_waits(1, 0.5, 0.2, 1.0)
    assert d1 == pytest.approx(0.1 / 0.9) and d2 == pytest.approx(1 / 0.9 + 0.5 / (0.9 * 0.5) - 1.0)


def test_preemptive_class_one_never_notices_the_lower_class():
    """Klasse 1 sieht bei unterbrechender Vorfahrt nur sich selbst: Verzögerung = M/G/1 mit Auslastung p·ρ (exponentiell: ρ₁/(1 − ρ₁))."""
    for rho, p in ((0.5, 0.2), (0.9, 0.2), (0.8, 0.5)):
        d1, _ = F.pr_waits(1, rho, p, 1.0)
        assert d1 == pytest.approx(F.pk_wait(p * rho, 1.0) , rel=1e-9) and d1 == pytest.approx(p * rho / (1 - p * rho))


@pytest.mark.parametrize("c,cs2", [(1, 0.0), (1, 1.0), (1, 4.0), (4, 1.0)])
def test_conservation_law_for_non_preemptive_priority(c, cs2):
    """Erhaltungssatz: p·W₁ + (1 − p)·W₂ = W_FIFO bei gleicher Dauer in beiden Klassen."""
    for rho, p in ((0.5, 0.2), (0.9, 0.2), (0.8, 0.5), (0.95, 0.1)):
        w = F.np_waits(c, rho, p, cs2)
        assert F.overall_wait(w, p) == pytest.approx(F.fifo_wait(c, rho, cs2), rel=1e-9)


def test_preemption_keeps_the_total_for_exponential_service_and_lowers_it_for_variable_service():
    """Bei exponentieller Dauer ist die Zahl im System von der Reihenfolge unabhängig, also auch das Gesamtmittel; bei cs² = 4 senkt Unterbrechen es, bei cs² = 0 erhöht es."""
    for c in (1, 4):
        w = F.pr_waits(c, 0.9, 0.2, 1.0)
        assert F.overall_wait(w, 0.2) == pytest.approx(F.fifo_wait(c, 0.9, 1.0), rel=1e-9)
    assert F.overall_wait(F.pr_waits(1, 0.8, 0.2, 4.0), 0.2) < F.fifo_wait(1, 0.8, 4.0)
    assert F.overall_wait(F.pr_waits(1, 0.8, 0.2, 0.0), 0.2) > F.fifo_wait(1, 0.8, 0.0)


def test_class_one_is_always_faster_and_class_two_slower_than_fifo():
    for policy in ("np", "pr"):
        for c, cs2 in ((1, 0.0), (1, 4.0), (4, 1.0)):
            hi, lo = F.class_waits(policy, c, 0.8, 0.2, cs2)
            fifo = F.fifo_wait(c, 0.8, cs2)
            assert hi < fifo < lo


def test_preemption_helps_class_one_more_than_waiting_priority():
    for rho in (0.5, 0.8, 0.9):
        assert F.pr_waits(1, rho, 0.2, 1.0)[0] < F.np_waits(1, rho, 0.2, 1.0)[0]


def test_class_waits_dispatch_and_fifo_is_symmetric():
    assert F.class_waits("fifo", 1, 0.8, 0.2, 4.0) == (F.pk_wait(0.8, 4.0), F.pk_wait(0.8, 4.0))
    assert F.class_waits("np", 1, 0.8, 0.2, 1.0) == F.np_waits(1, 0.8, 0.2, 1.0) and F.class_waits("pr", 4, 0.8, 0.2, 1.0) == F.pr_waits(4, 0.8, 0.2, 1.0)
    with pytest.raises(ValueError):
        F.class_waits("lifo", 1, 0.8, 0.2, 1.0)


def test_no_formula_for_several_lanes_with_non_exponential_service():
    assert F.np_waits(4, 0.8, 0.2, 4.0) is None and F.pr_waits(4, 0.8, 0.2, 0.0) is None and F.fifo_wait(4, 0.8, 4.0) is None
    assert F.class_waits("np", 4, 0.8, 0.2, 0.0) is None and F.class_waits("fifo", 4, 0.8, 0.2, 4.0) is None


def test_erlang_c_and_mmc_wait_by_hand():
    assert F.erlang_c(2, 1.0) == pytest.approx(1 / 3) and F.mmc_wait(2, 0.5) == pytest.approx(1 / 3) and F.mmc_wait(1, 0.8) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        F.erlang_c(3, 3.0)


def test_speedup_penalty_and_minutes():
    assert F.speedup((1.0, 5.0), 10.0) == pytest.approx(10.0) and F.penalty((1.0, 5.0), 4.0) == pytest.approx(0.25) and F.to_minutes(2.0) == 6.0


def test_invalid_input_is_rejected():
    for bad in (0.0, 1.0, 1.3):
        with pytest.raises(ValueError):
            F.np_waits(1, bad, 0.2, 1.0)
        with pytest.raises(ValueError):
            F.pr_waits(1, bad, 0.2, 1.0)
        with pytest.raises(ValueError):
            F.pk_wait(bad, 1.0)
    for bad_p in (0.0, 1.0):
        with pytest.raises(ValueError):
            F.np_waits(1, 0.5, bad_p, 1.0)


def test_closed_forms_for_gain_and_loss_of_priority():
    """Eine Spur, jede Dauer: nicht unterbrechend Gewinn (1 − pρ)/(1 − ρ), Verlust pρ/(1 − pρ); unterbrechend Gewinn (1 − pρ)/(p(1 − ρ)); alle unabhängig von cs²."""
    for rho, p in ((0.5, 0.2), (0.9, 0.2), (0.8, 0.5)):
        for cs2 in (0.0, 1.0, 4.0):
            fifo = F.fifo_wait(1, rho, cs2)
            hi_np, lo_np = F.np_waits(1, rho, p, cs2)
            assert fifo / hi_np == pytest.approx(F.np_speedup(rho, p)) and lo_np / fifo - 1 == pytest.approx(F.np_penalty(rho, p))
            assert fifo / F.pr_waits(1, rho, p, cs2)[0] == pytest.approx(F.pr_speedup(rho, p))
    assert [round(F.np_speedup(r, 0.2), 1) for r in (0.5, 0.8, 0.9)] == [1.8, 4.2, 8.2]
    assert [round(100 * F.np_penalty(r, 0.2), 1) for r in (0.5, 0.8, 0.9)] == [11.1, 19.0, 22.0]
    assert [round(F.pr_speedup(r, 0.2), 1) for r in (0.5, 0.8, 0.9)] == [9.0, 21.0, 41.0]
    with pytest.raises(ValueError):
        F.np_speedup(1.0, 0.2)
