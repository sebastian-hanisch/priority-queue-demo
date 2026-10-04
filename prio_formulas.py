"""Formeln zu Prioritätsklassen: ein Gate mit c Spuren, Poisson-Ankünfte, Klasse 1 (eilig) mit Anteil p, Klasse 2 (Standard), beide mit derselben
Abfertigungsdauer (Mittel 1, Variationskoeffizient² cs²), Auslastung ρ je Spur. Alle Verzögerungen in Abfertigungsdauern (mal 3 min). Verzögerung = Zeit im
System − Dauer, also bei Unterbrechung inklusive der Unterbrechungen.

**Eine Spur** (M/G/1 mit Prioritäten, exakt für jede Dauer): W0 = ρ(1 + cs²)/2.
 - FIFO: W = W0/(1 − ρ) für beide Klassen (Pollaczek-Khinchine).
 - nicht unterbrechend (Cobham 1954): W₁ = W0/(1 − σ₁), W₂ = W0/((1 − σ₁)(1 − σ₂)), σ₁ = pρ, σ₂ = ρ.
 - unterbrechend (preemptive-resume): T_k = 1/(1 − σ_{k−1}) + R_k/((1 − σ_{k−1})(1 − σ_k)), R_k = Σ_{i≤k} λ_i·E[S²]/2, Verzögerung = T_k − 1.
**Mehrere Spuren, exponentielle Dauer** (exakt): Erlang C und der Cobham-Faktor: nicht unterbrechend W_k = (C/c)/((1 − σ_{k−1})(1 − σ_k)); unterbrechend: Klasse 1 sieht
ein eigenes M/M/c mit Auslastung pρ, das Gesamtmittel ist das von M/M/c (die Zahl im System ändert sich durch Vorfahrt bei gleicher exponentieller Dauer nicht), Klasse 2 folgt aus dem
Erhaltungssatz. Für andere Dauer bei mehreren Spuren gibt es hier keine Formel (None).

**Erhaltungssatz** (Kleinrock): Bei nicht unterbrechender Vorfahrt und gleicher Dauer in beiden Klassen ist das mit den Anteilen gewichtete Mittel der Verzögerungen gleich dem von FIFO."""


def _check(rho, p):
    if not 0 < rho < 1:
        raise ValueError("ρ muss in (0, 1) liegen")
    if not 0 < p < 1:
        raise ValueError("p muss in (0, 1) liegen")


def erlang_b(c, a):
    """Erlang-B-Verlustwahrscheinlichkeit über die stabile Rekursion B_k = a·B_{k−1}/(k + a·B_{k−1})."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    """Erlang C: Wahrscheinlichkeit zu warten bei unendlicher Schlange; nur für c > a (Kopie aus mmc-queue-demo)."""
    rho = a / c
    if rho >= 1:
        raise ValueError("c ≤ a: keine stationäre Verteilung")
    b = erlang_b(c, a)
    return b / (1 - rho * (1 - b))


def mmc_wait(c, rho):
    """Mittlere Wartezeit der M/M/c-Schlange bei mittlerer Dauer 1 und Auslastung ρ je Spur: C(c, cρ)/(c(1 − ρ))."""
    if not 0 < rho < 1:
        raise ValueError("ρ muss in (0, 1) liegen")
    return erlang_c(c, c * rho) / (c * (1.0 - rho))


def pk_wait(rho, cs2):
    """Pollaczek-Khinchine (M/G/1, FIFO): ρ(1 + cs²)/(2(1 − ρ))."""
    if not 0 < rho < 1:
        raise ValueError("ρ muss in (0, 1) liegen")
    return rho * (1.0 + cs2) / (2.0 * (1.0 - rho))


def fifo_wait(c, rho, cs2):
    """Mittlere Verzögerung bei FIFO (beide Klassen gleich): Pollaczek-Khinchine für eine Spur, Erlang C für mehrere bei exponentieller Dauer; sonst None."""
    if c == 1:
        return pk_wait(rho, cs2)
    return mmc_wait(c, rho) if cs2 == 1.0 else None


def np_waits(c, rho, p, cs2):
    """Nicht unterbrechende Vorfahrt: (Klasse 1, Klasse 2) nach Cobham; c = 1 für jede Dauer, c > 1 nur bei exponentieller Dauer (sonst None)."""
    _check(rho, p)
    s1, s2 = p * rho, rho
    if c == 1:
        w0 = rho * (1.0 + cs2) / 2.0
    elif cs2 == 1.0:
        w0 = erlang_c(c, c * rho) / c
    else:
        return None
    return w0 / (1.0 - s1), w0 / ((1.0 - s1) * (1.0 - s2))


def pr_waits(c, rho, p, cs2):
    """Unterbrechende Vorfahrt (preemptive-resume): (Klasse 1, Klasse 2); c = 1 für jede Dauer, c > 1 nur bei exponentieller Dauer (sonst None)."""
    _check(rho, p)
    s1 = p * rho
    if c == 1:
        es2 = 1.0 + cs2
        r1, r2 = rho * p * es2 / 2.0, rho * es2 / 2.0
        t1 = 1.0 + r1 / (1.0 - s1)
        t2 = 1.0 / (1.0 - s1) + r2 / ((1.0 - s1) * (1.0 - rho))
        return t1 - 1.0, t2 - 1.0
    if cs2 != 1.0:
        return None
    d1 = mmc_wait(c, s1)
    total = mmc_wait(c, rho)
    return d1, (total - p * d1) / (1.0 - p)


def class_waits(policy, c, rho, p, cs2):
    """Verzögerung je Klasse (hoch, niedrig) für die gewählte Reihenfolge; bei FIFO für beide Klassen gleich; None, wo es keine Formel gibt."""
    if policy == "fifo":
        w = fifo_wait(c, rho, cs2)
        return None if w is None else (w, w)
    if policy == "np":
        return np_waits(c, rho, p, cs2)
    if policy == "pr":
        return pr_waits(c, rho, p, cs2)
    raise ValueError(f"unbekannte Reihenfolge: {policy}")


def overall_wait(waits, p):
    """Mit den Anteilen gewichtetes Mittel der Verzögerung beider Klassen."""
    return p * waits[0] + (1.0 - p) * waits[1]


def speedup(policy_waits, fifo):
    """Um welchen Faktor die Eiligen schneller sind als bei FIFO: fifo / Verzögerung der Klasse 1."""
    return fifo / policy_waits[0]


def penalty(policy_waits, fifo):
    """Um wie viel länger die Standard-Lkw warten als bei FIFO: Verzögerung der Klasse 2 / fifo − 1."""
    return policy_waits[1] / fifo - 1.0


def np_speedup(rho, p):
    """Gewinn der Eiligen bei nicht unterbrechender Vorfahrt, eine Spur, jede Dauer: W_FIFO/W₁ = (1 − pρ)/(1 − ρ) (W₀ kürzt sich: unabhängig von cs²)."""
    _check(rho, p)
    return (1.0 - p * rho) / (1.0 - rho)


def np_penalty(rho, p):
    """Verlust der Standard-Lkw bei nicht unterbrechender Vorfahrt, eine Spur, jede Dauer: W₂/W_FIFO − 1 = pρ/(1 − pρ) (unabhängig von cs²)."""
    _check(rho, p)
    return p * rho / (1.0 - p * rho)


def pr_speedup(rho, p):
    """Gewinn der Eiligen bei unterbrechender Vorfahrt, eine Spur, jede Dauer: W_FIFO/D₁ = (1 − pρ)/(p(1 − ρ)) (unabhängig von cs²)."""
    _check(rho, p)
    return (1.0 - p * rho) / (p * (1.0 - rho))


def to_minutes(x, mean_service_min=3.0):
    """Zeit in Abfertigungsdauern → Minuten."""
    return x * mean_service_min

