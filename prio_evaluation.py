"""Auswertung: Live-Lauf gegen Formeln und Markov-Kette, Studienzellen (Wiederholungen: Mittel je Klasse, Gesamtmittel, Gewinn und Verlust gegenüber FIFO). Die teure Studie steht
vorgerechnet in `precomputed_sweep.json` (Generator: generate_precomputed.py, Laden: `load_precomputed`)."""

import json
import math
from pathlib import Path

import prio_constants as C
import prio_formulas as F
from prio_chain import np1_chain, pr_chain
from prio_simulation import simulate

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def mean_and_sd(values):
    """Mittelwert und Stichproben-Standardabweichung (0.0 bei nur einem Wert)."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def formulas(c, rho_pct, p_pct, policy, cs2):
    """Formelwerte einer Einstellung in Abfertigungsdauern: Verzögerung je Klasse (None, wo es keine Formel gibt), FIFO als Bezug, Gesamtmittel."""
    rho, p = rho_pct / 100.0, p_pct / 100.0
    waits = F.class_waits(policy, c, rho, p, cs2)
    return {"waits": None if waits is None else list(waits), "fifo": F.fifo_wait(c, rho, cs2),
            "total": None if waits is None else F.overall_wait(waits, p)}


def live_report(c, rho_pct, p_pct, policy, cs2, seed, customers=C.LIVE_CUSTOMERS):
    """Ein Live-Lauf mit den Formelwerten."""
    sim = simulate(c, rho_pct / 100.0, p_pct / 100.0, policy, cs2, customers, seed)
    out = formulas(c, rho_pct, p_pct, policy, cs2)
    out.update({"sim": sim, "delay_hi": sim.delay_hi, "delay_lo": sim.delay_lo})
    return out


def chain_report(c, rho_pct, p_pct, policy):
    """Exakte Markov-Kette (nur exponentielle Dauer): bei nicht unterbrechender Vorfahrt und einer Spur ihre eigene, sonst die der unterbrechenden Vorfahrt (beliebig viele
    Spuren). Liefert (Ergebnis, Name der Kette)."""
    rho, p = rho_pct / 100.0, p_pct / 100.0
    if policy == "np" and c == 1:
        return np1_chain(rho, p), "np"
    return pr_chain(c, rho, p), "pr"


def summarize_runs(c, rho_pct, p_pct, policy, cs2, n, runs):
    out = {"c": c, "rho_pct": rho_pct, "p_pct": p_pct, "policy": policy, "cs2": cs2, "n": n, "reps": len(runs),
           "hi": [r.delay_hi for r in runs], "lo": [r.delay_lo for r in runs], "all": [r.delay_all for r in runs]}
    out.update(formulas(c, rho_pct, p_pct, policy, cs2))
    return out


def study_cell_run(c, rho_pct, p_pct, policy, cs2, n, seed, reps):
    runs = [simulate(c, rho_pct / 100.0, p_pct / 100.0, policy, cs2, n, seed + 1000 * r) for r in range(reps)]
    return summarize_runs(c, rho_pct, p_pct, policy, cs2, n, runs)


def mean_of(cell, key):
    """Mittel über die Wiederholungen: key = 'hi', 'lo' oder 'all'."""
    return sum(cell[key]) / len(cell[key])


def se_of(cell, key):
    """Standardfehler des Mittels über die Wiederholungen."""
    return mean_and_sd(cell[key])[1] / math.sqrt(len(cell[key]))


def speedup_vs_fifo(cell, fifo_cell):
    """Um welchen Faktor die Eiligen schneller sind als bei FIFO (gleiche Einstellung, gemessen): Verzögerung FIFO / Verzögerung Klasse 1."""
    return mean_of(fifo_cell, "hi") / mean_of(cell, "hi")


def penalty_vs_fifo(cell, fifo_cell):
    """Um wie viel länger die Standard-Lkw warten als bei FIFO (gemessen): Verzögerung Klasse 2 / FIFO − 1."""
    return mean_of(cell, "lo") / mean_of(fifo_cell, "lo") - 1.0


def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def study_cell(pre, c, rho_pct, policy, cs2):
    for cell in pre["study"]:
        if cell["c"] == c and cell["rho_pct"] == rho_pct and cell["policy"] == policy and cell["cs2"] == cs2:
            return cell
    raise KeyError((c, rho_pct, policy, cs2))


def pstudy_cell(pre, p_pct, policy):
    for cell in pre["pstudy"]:
        if cell["p_pct"] == p_pct and cell["policy"] == policy:
            return cell
    raise KeyError((p_pct, policy))


def nearest(options, value):
    """Nächster Wert aus `options` (bei Gleichstand der kleinere)."""
    return min(options, key=lambda o: (abs(o - value), o))
