"""Orakel-Tests (unabhängiger Rechenweg): Cobham und Preemptive-Resume gegen eigene Markov-Ketten mit Phasentyp-Dauer (Erlang-2, Hyperexponential; die Ketten der Demo
gelten nur für exponentielle Dauer), mehrspurig nicht unterbrechend gegen eine Kette mit Zustand (n₁, n₂, Eilige in Abfertigung), und die Simulation gegen eine
Brute-Force-Ereignisrechnung (Heap, Versionsstempel) mit denselben Ankünften, Klassen und Dauern."""

import heapq
from collections import deque

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")

import prio_formulas as F  # noqa: E402
import prio_simulation as S  # noqa: E402


def _solve_chain(start, trans):
    """Gleichgewicht einer Kette, deren Zustände per Breitensuche aus `trans(zustand) -> [(zustand, rate)]` entstehen."""
    idx, states, rows, cols, vals = {start: 0}, [start], [], [], []
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for t, r in trans(s):
            if r > 0:
                if t not in idx:
                    idx[t] = len(states)
                    states.append(t)
                    queue.append(t)
                rows.append(idx[s]); cols.append(idx[t]); vals.append(r)
    n = len(states)
    q = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    q = q - sp.diags(np.asarray(q.sum(axis=1)).ravel())
    a = sp.vstack([sp.csr_matrix(np.ones((1, n))), q.T.tocsr()[1:]]).tocsc()
    b = np.zeros(n)
    b[0] = 1.0
    return states, spla.spsolve(a, b)


def _delays(states, pi, l1, l2):
    """Verzögerung je Klasse nach Little: E[n_k]/λ_k − 1 (mittlere Dauer 1)."""
    e1 = sum(pi[i] * s[0] for i, s in enumerate(states))
    e2 = sum(pi[i] * s[1] for i, s in enumerate(states))
    return e1 / l1 - 1.0, e2 / l2 - 1.0


def _service_phases(kind):
    """(Anfangsverteilung, Phasenübergänge, Austrittsraten) mit Mittel 1: 'e2' = Erlang-2 (cs² = 0.5), 'h2' = Hyperexponential mit gleichen Phasenanteilen am Mittel,
    cs² = 4 (p₁ = (1 + √(3/5))/2 aus 1/(2p₁) + 1/(2p₂) = 1 + cs² gelöst und hier als Zahl eingesetzt)."""
    if kind == "e2":
        return np.array([1.0, 0.0]), np.array([[0.0, 2.0], [0.0, 0.0]]), np.array([0.0, 2.0])
    p1 = (1 + (3 / 5) ** 0.5) / 2
    return np.array([p1, 1 - p1]), np.zeros((2, 2)), np.array([2 * p1, 2 * (1 - p1)])


def _one_lane_priority(policy, rho, p, kind, n_max):
    """Eine Spur, zwei Klassen. np: Zustand (n₁, n₂, Klasse in Abfertigung, Phase); pr (resume): (n₁, n₂, Phase des laufenden Eiligen, Phase des Standard-Kopfs), die Phase
    des unterbrochenen Standard-Lkw bleibt erhalten."""
    alpha, T, t = _service_phases(kind)
    k = len(alpha)
    l1, l2 = p * rho, (1 - p) * rho

    def np_trans(s):
        n1, n2, cl, ph = s
        out = []
        if n1 + n2 < n_max:
            for c_, lam in ((1, l1), (2, l2)):
                m1, m2 = n1 + (c_ == 1), n2 + (c_ == 2)
                if cl == 0:
                    out += [((m1, m2, c_, j), lam * alpha[j]) for j in range(k)]
                else:
                    out.append(((m1, m2, cl, ph), lam))
        if cl:
            out += [((n1, n2, cl, j), T[ph, j]) for j in range(k) if j != ph]
            m1, m2 = n1 - (cl == 1), n2 - (cl == 2)
            nxt = 1 if m1 > 0 else (2 if m2 > 0 else 0)
            out += [((0, 0, 0, -1), t[ph])] if nxt == 0 else [((m1, m2, nxt, j), t[ph] * alpha[j]) for j in range(k)]
        return out

    def pr_trans(s):
        n1, n2, a, b = s
        out = []
        if n1 + n2 < n_max:
            out += [((1, n2, j, b), l1 * alpha[j]) for j in range(k)] if n1 == 0 else [((n1 + 1, n2, a, b), l1)]
            out += [((n1, 1, a, j), l2 * alpha[j]) for j in range(k)] if n2 == 0 else [((n1, n2 + 1, a, b), l2)]
        if n1 > 0:
            out += [((n1, n2, j, b), T[a, j]) for j in range(k) if j != a]
            out += [((0, n2, -1, b), t[a])] if n1 == 1 else [((n1 - 1, n2, j, b), t[a] * alpha[j]) for j in range(k)]
        elif n2 > 0:
            out += [((0, n2, -1, j), T[b, j]) for j in range(k) if j != b]
            out += [((0, 0, -1, -1), t[b])] if n2 == 1 else [((0, n2 - 1, -1, j), t[b] * alpha[j]) for j in range(k)]
        return out

    if policy == "np":
        states, pi = _solve_chain((0, 0, 0, -1), np_trans)
    else:
        states, pi = _solve_chain((0, 0, -1, -1), pr_trans)
    return _delays(states, pi, l1, l2)


def _np_exponential(c, rho, p, n_max):
    """Nicht unterbrechend, c Spuren, exponentielle Dauer: Zustand (n₁, n₂, j), j = Eilige in Abfertigung, k = min(c, n₁ + n₂) − j Standard-Lkw in Abfertigung."""
    l1, l2 = p * c * rho, (1 - p) * c * rho

    def trans(s):
        n1, n2, j = s
        busy = min(c, n1 + n2)
        k = busy - j
        out = []
        if n1 + n2 < n_max:
            out += [((n1 + 1, n2, j + (busy < c)), l1), ((n1, n2 + 1, j), l2)]
        if j > 0:                                       # Abgang eines Eiligen; wartet ein Eiliger, startet er (j bleibt), sonst j − 1
            out.append(((n1 - 1, n2, j if n1 - j > 0 else j - 1), j))
        if k > 0:                                       # Abgang eines Standard-Lkw; wartet ein Eiliger, startet er (j + 1)
            out.append(((n1, n2 - 1, j + 1 if n1 - j > 0 else j), k))
        return out

    states, pi = _solve_chain((0, 0, 0), trans)
    return _delays(states, pi, l1, l2)


def test_oracle_itself_by_hand():
    """M/M/1, ρ = 0.5, p = 0.5: Cobham W₀ = 0.5, W₁ = 0.5/0.75 = 2/3, W₂ = 0.5/(0.75·0.5) = 4/3; preemptive: Eilige 1/3 (M/M/1 mit Last 0.25), Standard 5/3 (Erhaltung)."""
    assert _np_exponential(1, 0.5, 0.5, 40) == pytest.approx((2 / 3, 4 / 3), abs=1e-6)
    assert _one_lane_priority("pr", 0.5, 0.5, "e2", 40)[0] == pytest.approx(0.25, abs=1e-6)       # Erlang-2: E[S²] = 1.5, R₁ = 0.25·1.5/2, R₁/(1 − 0.25) = 0.25


@pytest.mark.parametrize("policy", ["np", "pr"])
@pytest.mark.parametrize("kind,cs2", [("e2", 0.5), ("h2", 4.0)])
def test_one_lane_formulas_against_phase_type_chains(policy, kind, cs2):
    rho, p = 0.5, 0.3
    f = F.np_waits(1, rho, p, cs2) if policy == "np" else F.pr_waits(1, rho, p, cs2)
    assert _one_lane_priority(policy, rho, p, kind, 60) == pytest.approx(f, rel=1e-4)


@pytest.mark.parametrize("c,rho,p", [(2, 0.6, 0.3), (4, 0.7, 0.2)])
def test_non_preemptive_multi_lane_cobham_against_the_class_in_service_chain(c, rho, p):
    assert _np_exponential(c, rho, p, 50) == pytest.approx(F.np_waits(c, rho, p, 1.0), rel=1e-4)


def _brute_force(c, policy, jobs):
    """Verzögerung je Job per Ereignis-Heap: (Zeit, Art, id, Version); Art 0 = Fertig, 1 = Ankunft. Regeln wie in der Demo dokumentiert."""
    n = len(jobs)
    events = [(a, 1, i, 0) for i, (a, _, _) in enumerate(jobs)]
    heapq.heapify(events)
    remaining = [d for (_, _, d) in jobs]
    start, version, delay = [0.0] * n, [0] * n, [None] * n
    running, q1, q2 = set(), deque(), deque()

    def begin(i, now):
        start[i] = now
        version[i] += 1
        running.add(i)
        heapq.heappush(events, (now + remaining[i], 0, i, version[i]))

    while events:
        now, kind, i, ver = heapq.heappop(events)
        if kind == 0:
            if ver != version[i] or i not in running:
                continue
            running.discard(i)
            delay[i] = now - jobs[i][0] - jobs[i][2]
            while len(running) < c and (q1 or q2):
                begin((q1 if q1 or policy == "fifo" else q2).popleft(), now)
        elif len(running) < c:
            begin(i, now)
        elif policy == "pr" and jobs[i][1] == 1 and any(jobs[j][1] == 2 for j in running):
            v = max((j for j in running if jobs[j][1] == 2), key=lambda j: start[j])
            running.discard(v)
            remaining[v] -= now - start[v]
            version[v] += 1
            q2.appendleft(v)
            begin(i, now)
        else:
            (q1 if policy == "fifo" or jobs[i][1] == 1 else q2).append(i)
    return delay


@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("policy", ["fifo", "np", "pr"])
@pytest.mark.parametrize("scv", [0.0, 1.0, 4.0])
def test_simulation_against_brute_force_event_calculation(c, policy, scv):
    rng = np.random.default_rng(10 * c + len(policy) + int(scv))
    n, rho, p = 300, 0.85, 0.4
    gaps = rng.exponential(1.0 / (c * rho), n + 1)
    picks = rng.random(n)
    p1 = 0.5 * (1 + ((scv - 1) / (scv + 1)) ** 0.5) if scv > 1 else 1.0
    svc = np.ones(n) if scv == 0 else (rng.exponential(1.0, n) if scv == 1 else np.where(rng.random(n) < p1, rng.exponential(1 / (2 * p1), n), rng.exponential(1 / (2 * (1 - p1)), n)))
    gi, pi_, si = iter(gaps), iter(picks), iter(svc)
    res = S.simulate(c, rho, p, policy, scv, n, 0, warm_fraction=0.1, samplers=((lambda: next(gi)), (lambda: next(si)), (lambda: next(pi_))))
    t = np.cumsum(gaps[:n])
    cls = np.where(picks < p, 1, 2)
    delay = _brute_force(c, policy, [(t[i], int(cls[i]), float(svc[i])) for i in range(n)])
    warm = int(0.1 * n)
    hi = [delay[i] for i in range(warm, n) if cls[i] == 1]
    lo = [delay[i] for i in range(warm, n) if cls[i] == 2]
    assert (res.n_hi, res.n_lo) == (len(hi), len(lo))
    assert res.delay_hi == pytest.approx(np.mean(hi), abs=1e-9) and res.delay_lo == pytest.approx(np.mean(lo), abs=1e-9)
    assert res.delay_all == pytest.approx(np.mean(hi + lo), abs=1e-9)
