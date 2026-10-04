import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ScriptedRng:
    """Liefert vorgegebene Werte statt Zufall (Mini-Instanzen von Hand gerechnet). `expovariate(rate)` ignoriert die Rate und gibt der
    Reihe nach die Werte zurück, `uniform()` ebenso aus einer eigenen Liste."""

    def __init__(self, exp_values=(), uniform_values=()):
        self.exp_values, self.uniform_values = list(exp_values), list(uniform_values)
        self.n_exp = self.n_uniform = 0

    def expovariate(self, rate):
        v = self.exp_values[self.n_exp]
        self.n_exp += 1
        return v

    def uniform(self):
        v = self.uniform_values[self.n_uniform]
        self.n_uniform += 1
        return v


@pytest.fixture
def priority_mini():
    """Eine Spur, nicht unterbrechende Priorität, vier Lkw. Zwischenankünfte 1.0 / 0.5 / 0.2 / 0.1 (Ankünfte bei 1.0, 1.5, 1.7, 1.8), Klassen-Zufall 0.9 / 0.9 / 0.1 / 0.9
    (Anteil Eiliger p = 0.5: Zufall < 0.5 heißt eilig, also Klassen 2 / 2 / 1 / 2), Dauern 3.0 / 1.0 / 1.0 / 1.0.
    Von Hand: Lkw 1 (Klasse 2) startet bei 1.0 (Ende 4.0); Lkw 2 (Klasse 2, 1.5) und Lkw 4 (Klasse 2, 1.8) warten; Lkw 3 (eilig, 1.7) wartet ebenfalls, kommt bei 4.0
    aber VOR den Standard-Lkw dran (Verzögerung 2.3, Ende 5.0); Lkw 2 startet bei 5.0 (Verzögerung 3.5, Ende 6.0); Lkw 4 startet bei 6.0 (Verzögerung 4.2, Ende 7.0).
    Gesamtmittel der Verzögerung (0 + 3.5 + 2.3 + 4.2)/4 = 2.5; Eilige 2.3, Standard (0 + 3.5 + 4.2)/3 ≈ 2.567.
    FIFO: Lkw 2 startet bei 4.0 (2.5), Lkw 3 bei 5.0 (3.3), Lkw 4 bei 6.0 (4.2): Gesamtmittel ebenfalls 2.5 (gleiche Dauern, Erhaltungssatz).
    Unterbrechend: Lkw 3 verdrängt Lkw 1 bei 1.7 (Rest 2.3), Lkw 3 endet bei 2.7 (Verzögerung 0), Lkw 1 läuft bei 2.7 weiter und endet bei 5.0 (Verzögerung 1.0),
    Lkw 2 und 4 enden bei 6.0 und 7.0 (3.5 und 4.2): Eilige 0, Standard (1.0 + 3.5 + 4.2)/3 = 2.9, gesamt 2.175."""
    return (ScriptedRng(exp_values=[1.0, 0.5, 0.2, 0.1, 100.0]), ScriptedRng(uniform_values=[0.9, 0.9, 0.1, 0.9]), ScriptedRng(exp_values=[3.0, 1.0, 1.0, 1.0]))
