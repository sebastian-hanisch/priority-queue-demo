"""Markov-Kette: von Hand gelöster Dreizustands-Fall, Gesamtzahl gleich M/M/c, Übereinstimmung mit den Formeln (unabhängige Rechnung), Güte der Lösung."""

import math

import numpy as np
import pytest

import prio_chain as K
import prio_formulas as F


def test_three_state_chain_by_hand():
    """Grenze n₁ + n₂ ≤ 1, eine Spur, ρ = 0.5, p = 0.5: Zustände (0,0), (1,0), (0,1); λ₁ = λ₂ = 0.25, Abgangsrate 1. Gleichgewicht: π₁₀ = π₀₁ = 0.25·π₀₀, also π₀₀ = 2/3 und je 1/6."""
    for chain in (K.pr_chain(1, 0.5, 0.5, n_max=1), K.np1_chain(0.5, 0.5, n_max=1)):
        assert chain.grid[0, 0] == pytest.approx(2 / 3) and chain.grid[1, 0] == pytest.approx(1 / 6) and chain.grid[0, 1] == pytest.approx(1 / 6)
        assert chain.grid.sum() == pytest.approx(1.0)


def test_truncation_by_hand():
    """c = 1, ρ = 0.5: ⌈ln(10⁻⁷)/ln(0.5)⌉ = 24, plus c plus Reserve 5 = 30."""
    assert K.truncation(1, 0.5) == 30 and K.truncation(4, 0.8) == 4 + math.ceil(math.log(1e-7) / math.log(0.8)) + 5


@pytest.mark.parametrize("c,rho,p", [(1, 0.5, 0.2), (1, 0.9, 0.2), (1, 0.8, 0.5), (4, 0.8, 0.2), (4, 0.5, 0.1)])
def test_preemptive_chain_matches_the_formulas(c, rho, p):
    """Die Kette ist eine unabhängige Rechnung (lineares System über alle Zustände); sie bestätigt die geschlossenen Formeln, auch die Herleitung für mehrere Spuren."""
    chain = K.pr_chain(c, rho, p)
    d1, d2 = F.pr_waits(c, rho, p, 1.0)
    assert chain.delay1 == pytest.approx(d1, rel=1e-4, abs=1e-6) and chain.delay2 == pytest.approx(d2, rel=1e-4)


@pytest.mark.parametrize("rho,p", [(0.5, 0.2), (0.8, 0.2), (0.9, 0.1), (0.8, 0.5)])
def test_non_preemptive_chain_matches_cobham(rho, p):
    chain = K.np1_chain(rho, p)
    w1, w2 = F.np_waits(1, rho, p, 1.0)
    assert chain.delay1 == pytest.approx(w1, rel=1e-4) and chain.delay2 == pytest.approx(w2, rel=1e-4)


@pytest.mark.parametrize("builder", [lambda: K.pr_chain(1, 0.8, 0.2), lambda: K.pr_chain(4, 0.8, 0.3), lambda: K.np1_chain(0.8, 0.2)])
def test_total_number_in_the_system_is_the_mmc_birth_death_chain(builder):
    """Die Gesamtzahl n₁ + n₂ ist in jeder Reihenfolge die Zahl im System von M/M/c (gleiche exponentielle Dauer); die Randverteilung der Kette muss π_n von M/M/c sein."""
    chain = builder()
    c, rho = chain.c, chain.rho
    a = c * rho
    w = [1.0]
    for n in range(1, chain.n_max + 1):
        w.append(w[-1] * a / min(n, c))
    pi = np.array(w) / sum(w)
    assert chain.total_distribution() == pytest.approx(pi, abs=1e-6)


def test_solution_quality_and_normalisation():
    chain = K.pr_chain(4, 0.8, 0.2)
    assert chain.residual < 1e-12 and chain.boundary_mass < 1e-6 and chain.grid.sum() == pytest.approx(1.0)
    assert (chain.grid >= -1e-15).all() and chain.marginal(0).sum() == pytest.approx(1.0) and chain.marginal(1).sum() == pytest.approx(1.0)


def test_means_follow_littles_law_by_hand():
    """E[n_k] = λ_k·(Verzögerung + 1): Eilige und Standard-Lkw zusammen im Mittel c·ρ plus die mittlere Schlange."""
    chain = K.pr_chain(1, 0.5, 0.2)
    assert chain.e_n1 == pytest.approx(0.1 * (chain.delay1 + 1)) and chain.e_n2 == pytest.approx(0.4 * (chain.delay2 + 1))
    assert chain.e_n1 + chain.e_n2 == pytest.approx(1.0, rel=1e-4)                                 # M/M/1 mit ρ = 0.5: L = ρ/(1 − ρ) = 1


def test_marginal_axes_are_the_two_classes():
    chain = K.pr_chain(1, 0.8, 0.2)
    m1, m2 = chain.marginal(0), chain.marginal(1)
    assert (np.arange(len(m1)) * m1).sum() == pytest.approx(chain.e_n1) and (np.arange(len(m2)) * m2).sum() == pytest.approx(chain.e_n2)


def test_preemption_changes_the_chain_but_not_the_total_for_exponential_service():
    a, b = K.pr_chain(1, 0.8, 0.2), K.np1_chain(0.8, 0.2)
    assert a.delay1 < b.delay1 and a.delay2 > b.delay2
    assert 0.2 * a.delay1 + 0.8 * a.delay2 == pytest.approx(0.2 * b.delay1 + 0.8 * b.delay2, rel=1e-4)


def test_invalid_input_is_rejected():
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            K.pr_chain(1, bad, 0.2)
        with pytest.raises(ValueError):
            K.np1_chain(0.5, bad)
