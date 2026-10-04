"""Ereignisdiskrete Simulation eines Gates mit c Spuren und zwei Klassen (Klasse 1 = eilig, Klasse 2 = Standard) unter drei Reihenfolgen: FIFO (alle gleich), Vorfahrt nicht
unterbrechend (`np`) und Vorfahrt unterbrechend mit Wiederaufnahme (`pr`, preemptive-resume). Poisson-Ankünfte, beide Klassen mit derselben Abfertigungsdauer (Mittel 1, Streuung
cs² über Erlang-k / exponentiell / Hyperexponential oder fest).

Verzögerung eines Lkw = Zeit im System − Dauer, also bei Unterbrechung inklusive der Unterbrechungen. Bei Unterbrechung wird der laufende Standard-Lkw verdrängt, der zuletzt
gestartet hat (am wenigsten bisherige Arbeit), und mit seiner Restdauer an den Kopf der Standard-Schlange gestellt.

Aufbau nach Einheiten: `make_sampler`, `start_job`, `handle_arrival`, `handle_completion`, `fill_servers`, `simulate` (Schleife). Zufall nur über übergebene `SplitMix64`-Ströme
(Zwischenankunft, Dauer, Klassenwahl)."""

import math
from collections import deque
from dataclasses import dataclass, field

_MASK = (1 << 64) - 1


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self):
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def expovariate(self, rate):
        """Exponentiell mit Mittel 1/rate (Inversion; 1 − u liegt in (0, 1], der Logarithmus ist endlich)."""
        return -math.log(1.0 - self.uniform()) / rate


def streams(seed):
    """Die drei Zufallsströme eines Laufs: Zwischenankunft, Abfertigungsdauer, Klassenwahl."""
    return SplitMix64(seed), SplitMix64(seed + 99_991), SplitMix64(seed + 7_777)


def balanced_h2_probability(scv):
    """Anteil p₁ der ersten Phase der Hyperexponentialverteilung mit gleichen Phasenanteilen am Mittel (scv ≥ 1)."""
    return 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))


def make_sampler(mean, scv, rng):
    """Funktion ohne Argument, die Werte mit dem Mittel `mean` und dem Variationskoeffizienten² `scv` zieht: fest, Erlang-k, exponentiell oder Hyperexponential."""
    if scv == 0:
        return lambda: mean
    if scv == 1:
        return lambda: rng.expovariate(1.0 / mean)
    if scv < 1:
        k = round(1.0 / scv)
        return lambda: sum(rng.expovariate(k / mean) for _ in range(k))
    p1 = balanced_h2_probability(scv)
    rate1, rate2 = 2.0 * p1 / mean, 2.0 * (1.0 - p1) / mean

    def draw():
        return rng.expovariate(rate1) if rng.uniform() < p1 else rng.expovariate(rate2)
    return draw


@dataclass
class Job:
    ident: int
    cls: int                    # 1 = eilig, 2 = Standard
    arrival: float
    duration: float             # gesamte Dauer (für die Verzögerung)
    remaining: float            # noch zu leistende Abfertigung
    start: float = 0.0          # Beginn des laufenden Abschnitts


@dataclass
class Gate:
    c: int
    policy: str
    queues: dict = field(default_factory=lambda: {1: deque(), 2: deque()})
    running: list = field(default_factory=list)            # laufende Jobs (höchstens c)
    done: list = field(default_factory=list)               # (Klasse, Verzögerung, id) der fertigen Lkw


def queue_for(gate, job):
    """Bei FIFO teilen sich beide Klassen eine Schlange (Schlange 1), sonst hat jede Klasse ihre eigene."""
    return gate.queues[1] if gate.policy == "fifo" else gate.queues[job.cls]


def start_job(gate, job, now):
    job.start = now
    gate.running.append(job)


def fill_servers(gate, now):
    """Freie Spuren bedienen: bei FIFO den ältesten, sonst zuerst Klasse 1, dann Klasse 2."""
    while len(gate.running) < gate.c:
        q = gate.queues[1] if gate.queues[1] else gate.queues[2]
        if not q:
            return
        start_job(gate, q.popleft(), now)


def handle_arrival(gate, job, now):
    """Ein Lkw kommt an: freie Spur → sofort; sonst bei unterbrechender Vorfahrt und Klasse 1 → verdrängt den zuletzt gestarteten Standard-Lkw; sonst in die Schlange."""
    if len(gate.running) < gate.c:
        start_job(gate, job, now)
        return
    if gate.policy == "pr" and job.cls == 1:
        low = [j for j in gate.running if j.cls == 2]
        if low:
            victim = max(low, key=lambda j: j.start)
            gate.running.remove(victim)
            victim.remaining -= now - victim.start
            gate.queues[2].appendleft(victim)
            start_job(gate, job, now)
            return
    queue_for(gate, job).append(job)


def handle_completion(gate, job, now):
    """Ein Lkw ist fertig: Verzögerung = Zeit im System − Dauer; die freie Spur nimmt den nächsten."""
    gate.running.remove(job)
    gate.done.append((job.cls, now - job.arrival - job.duration, job.ident))
    fill_servers(gate, now)


@dataclass
class SimResult:
    c: int
    rho: float
    p: float
    policy: str
    cs2: float
    n: int                      # ausgewertete Lkw
    n_hi: int
    n_lo: int
    delay_hi: float             # mittlere Verzögerung der Eiligen
    delay_lo: float             # mittlere Verzögerung der Standard-Lkw
    delay_all: float            # Mittel über alle

    def delays(self):
        return self.delay_hi, self.delay_lo


def simulate(c, rho, p, policy, cs2, n_customers, seed, warm_fraction=0.05, rngs=None, samplers=None):
    """Ein Lauf über `n_customers` Lkw; die ersten `warm_fraction` davon werden nicht ausgewertet (Start leer). Auslastung ρ je Spur: Ankunftsrate c·ρ, mittlere Dauer 1;
    Anteil Eiliger `p`. `samplers` = (Zwischenankunft, Dauer, Klassenzufall) ersetzt die Fabrik (Mini-Instanzen von Hand). Alle Lkw werden zu Ende bedient."""
    if policy not in ("fifo", "np", "pr"):
        raise ValueError(f"unbekannte Reihenfolge: {policy}")
    if samplers is None:
        ra, rs, rk = rngs if rngs is not None else streams(seed)
        arrive, serve, pick = make_sampler(1.0 / (c * rho), 1.0, ra), make_sampler(1.0, cs2, rs), rk.uniform
    else:
        arrive, serve, pick = samplers
    gate = Gate(c=c, policy=policy)
    warm = int(warm_fraction * n_customers)
    next_arrival, created = arrive(), 0
    while created < n_customers or gate.running or gate.queues[1] or gate.queues[2]:
        t_done, finisher = math.inf, None
        for job in gate.running:
            end = job.start + job.remaining
            if end < t_done:
                t_done, finisher = end, job
        if created < n_customers and next_arrival <= t_done:
            now = next_arrival
            cls = 1 if pick() < p else 2
            d = serve()
            handle_arrival(gate, Job(created, cls, now, d, d), now)
            created += 1
            next_arrival = now + arrive()
        else:
            handle_completion(gate, finisher, t_done)
    hi = [d for cls, d, ident in gate.done if cls == 1 and ident >= warm]
    lo = [d for cls, d, ident in gate.done if cls == 2 and ident >= warm]
    both = hi + lo
    return SimResult(c=c, rho=rho, p=p, policy=policy, cs2=cs2, n=len(both), n_hi=len(hi), n_lo=len(lo), delay_hi=sum(hi) / len(hi) if hi else float("nan"),
                     delay_lo=sum(lo) / len(lo) if lo else float("nan"), delay_all=sum(both) / len(both))
