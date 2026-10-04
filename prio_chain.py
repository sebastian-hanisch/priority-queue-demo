"""Exakte Markov-Kette des Gates mit zwei Klassen bei exponentieller Dauer (Rate 1 je Spur): die unabhängige Referenz für Formeln und Simulation.

**Unterbrechende Vorfahrt** (beliebig viele Spuren): Zustand (n₁, n₂) = Zahl der Eiligen und der Standard-Lkw im System. Eilige belegen min(n₁, c) Spuren, Standard-Lkw die
übrigen: Abgangsrate Klasse 1 = min(n₁, c), Klasse 2 = min(n₂, c − min(n₁, c)); Ankunftsraten λ₁ = p·c·ρ und λ₂ = (1 − p)·c·ρ. Wegen der Gedächtnislosigkeit der exponentiellen Dauer
ist (n₁, n₂) allein ein Markov-Zustand.

**Nicht unterbrechende Vorfahrt** (eine Spur): zusätzlich die Klasse s des Lkw in Abfertigung (0 = leer). Wird die Spur frei, kommt der nächste Eilige, sonst der nächste Standard-Lkw dran.

Die Zahl im System n₁ + n₂ ist in beiden Fällen die Geburts-Sterbe-Kette von M/M/c; deshalb genügt die Grenze n₁ + n₂ ≤ N mit N aus dem Schwanz von M/M/c.
Das Gleichgewicht ist die Lösung von πQ = 0 mit Σπ = 1 (dünn besetztes lineares System, scipy). Mittlere Verzögerung je Klasse nach Little: E[n_k]/λ_k − 1."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from prio_constants import CHAIN_TAIL


@dataclass
class ChainResult:
    policy: str
    c: int
    rho: float
    p: float
    n_max: int
    grid: np.ndarray            # grid[n1, n2] = Gleichgewichtswahrscheinlichkeit (bei nicht unterbrechend über die Klasse in Abfertigung summiert)
    e_n1: float
    e_n2: float
    delay1: float               # Verzögerung der Eiligen (Abfertigungsdauern)
    delay2: float
    residual: float             # max |πQ| (Güte der Lösung)
    boundary_mass: float        # Wahrscheinlichkeit auf dem Rand n₁ + n₂ = N

    def marginal(self, axis):
        """Verteilung der Zahl der Eiligen (axis 0) bzw. Standard-Lkw (axis 1) im System."""
        return self.grid.sum(axis=1 - axis)

    def total_distribution(self):
        """Verteilung der Gesamtzahl n₁ + n₂ (muss die von M/M/c sein)."""
        out = np.zeros(self.n_max + 1)
        for n1 in range(self.n_max + 1):
            for n2 in range(self.n_max + 1 - n1):
                out[n1 + n2] += self.grid[n1, n2]
        return out


def truncation(c, rho, tail=CHAIN_TAIL):
    """Grenze N für n₁ + n₂: c plus so viele Plätze, dass der Geometrie-Schwanz ρ^k unter `tail` liegt (M/M/c fällt ab c wie ρ^k), plus Reserve."""
    return c + int(math.ceil(math.log(tail) / math.log(rho))) + 5


def _solve(rows, cols, vals, n_states, balance_rows):
    """πQ = 0, Σπ = 1: Q als dünn besetzte Matrix (Zeilen = von, Spalten = nach), Gleichgewichtsgleichungen Qᵀπ = 0 mit einer durch die Normierung ersetzten Gleichung."""
    q = coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
    diag = -np.asarray(q.sum(axis=1)).ravel()
    qt = (q + coo_matrix((diag, (np.arange(n_states), np.arange(n_states))), shape=(n_states, n_states)).tocsr()).T.tolil()
    qt[0, :] = 1.0
    b = np.zeros(n_states)
    b[0] = 1.0
    pi = spsolve(qt.tocsr(), b)
    residual = float(np.max(np.abs((q + coo_matrix((diag, (np.arange(n_states), np.arange(n_states))), shape=(n_states, n_states)).tocsr()).T @ pi))) if balance_rows else 0.0
    return pi, residual


def pr_chain(c, rho, p, n_max=None):
    """Exakte Kette der unterbrechenden Vorfahrt bei exponentieller Dauer (beliebig viele Spuren)."""
    if not (0 < rho < 1 and 0 < p < 1):
        raise ValueError("ρ und p müssen in (0, 1) liegen")
    n_max = n_max or truncation(c, rho)
    lam1, lam2 = p * c * rho, (1 - p) * c * rho
    idx = {}
    for n1 in range(n_max + 1):
        for n2 in range(n_max + 1 - n1):
            idx[(n1, n2)] = len(idx)
    rows, cols, vals = [], [], []
    for (n1, n2), i in idx.items():
        if n1 + n2 < n_max:
            rows += [i, i]
            cols += [idx[(n1 + 1, n2)], idx[(n1, n2 + 1)]]
            vals += [lam1, lam2]
        busy1 = min(n1, c)
        busy2 = min(n2, c - busy1)
        if busy1 > 0:
            rows.append(i)
            cols.append(idx[(n1 - 1, n2)])
            vals.append(float(busy1))
        if busy2 > 0:
            rows.append(i)
            cols.append(idx[(n1, n2 - 1)])
            vals.append(float(busy2))
    pi, residual = _solve(rows, cols, vals, len(idx), True)
    grid = np.zeros((n_max + 1, n_max + 1))
    for (n1, n2), i in idx.items():
        grid[n1, n2] = pi[i]
    return _result("pr", c, rho, p, n_max, grid, lam1, lam2, residual)


def np1_chain(rho, p, n_max=None):
    """Exakte Kette der nicht unterbrechenden Vorfahrt bei einer Spur und exponentieller Dauer; Zustand (n₁, n₂, s), s = Klasse in Abfertigung (0 = leer)."""
    if not (0 < rho < 1 and 0 < p < 1):
        raise ValueError("ρ und p müssen in (0, 1) liegen")
    n_max = n_max or truncation(1, rho)
    lam1, lam2 = p * rho, (1 - p) * rho
    idx = {}
    for n1 in range(n_max + 1):
        for n2 in range(n_max + 1 - n1):
            if n1 + n2 == 0:
                idx[(0, 0, 0)] = len(idx)
                continue
            if n1 >= 1:
                idx[(n1, n2, 1)] = len(idx)
            if n2 >= 1:
                idx[(n1, n2, 2)] = len(idx)
    rows, cols, vals = [], [], []
    for (n1, n2, s), i in idx.items():
        if n1 + n2 < n_max:
            for k, lam in ((1, lam1), (2, lam2)):
                m1, m2 = n1 + (k == 1), n2 + (k == 2)
                rows.append(i)
                cols.append(idx[(m1, m2, k if s == 0 else s)])
                vals.append(lam)
        if s > 0:                                       # Abgang des Lkw in Abfertigung (Rate 1), dann der nächste nach Vorfahrtsregel
            m1, m2 = n1 - (s == 1), n2 - (s == 2)
            nxt = 1 if m1 > 0 else (2 if m2 > 0 else 0)
            rows.append(i)
            cols.append(idx[(m1, m2, nxt)])
            vals.append(1.0)
    pi, residual = _solve(rows, cols, vals, len(idx), True)
    grid = np.zeros((n_max + 1, n_max + 1))
    for (n1, n2, _), i in idx.items():
        grid[n1, n2] += pi[i]
    return _result("np", 1, rho, p, n_max, grid, lam1, lam2, residual)


def _result(policy, c, rho, p, n_max, grid, lam1, lam2, residual):
    n1s = np.arange(n_max + 1)[:, None]
    n2s = np.arange(n_max + 1)[None, :]
    e1, e2 = float((grid * n1s).sum()), float((grid * n2s).sum())
    boundary = sum(grid[n1, n_max - n1] for n1 in range(n_max + 1))
    return ChainResult(policy=policy, c=c, rho=rho, p=p, n_max=n_max, grid=grid, e_n1=e1, e_n2=e2, delay1=e1 / lam1 - 1.0, delay2=e2 / lam2 - 1.0,
                       residual=residual, boundary_mass=float(boundary))
