"""Konstanten der Demo zu Prioritätsklassen: Regler, Voreinstellungen, Studien-Achsen. Zeiteinheit der Rechnung ist die mittlere Abfertigungsdauer (= 1);
angezeigt werden Minuten bei 3 min Mittel. Klasse 1 = eilige Lkw (hohe Priorität), Klasse 2 = Standard-Lkw."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


def fmt_signed_pct(x, digits=0):
    """Abweichung als vorzeichenbehaftetes Prozent (0.22 -> "+22 %", -0.44 -> "−44 %")."""
    return f"{x:+.{digits}%}".replace("%", " %").replace("-", "−")


MEAN_SERVICE_MIN = 3.0                                  # mittlere Abfertigungsdauer je Spur (Minuten)

RHO_PCT_MIN, RHO_PCT_MAX, RHO_PCT_STEP, DEFAULT_RHO_PCT = 50, 95, 5, 90        # Auslastung je Spur in Prozent
P_PCT_MIN, P_PCT_MAX, P_PCT_STEP, DEFAULT_P_PCT = 10, 50, 10, 20               # Anteil eiliger Lkw an allen Ankünften in Prozent
C_OPTIONS = (1, 4)                                      # Spuren
DEFAULT_C = 1
POLICIES = ("fifo", "np", "pr")                         # Reihenfolge: gleiche Behandlung, Vorfahrt ohne / mit Unterbrechung
POLICY_LABELS = {"fifo": "gleiche Behandlung (FIFO)", "np": "Vorfahrt, nicht unterbrechend", "pr": "Vorfahrt, unterbrechend"}
DEFAULT_POLICY = "np"
CS2_OPTIONS = (0.0, 1.0, 4.0)                           # Streuung der Dauer (für beide Klassen gleich): fest / exponentiell / streuend
CS2_LABELS = {0.0: "fest (cs² = 0)", 1.0: "exponentiell (cs² = 1)", 4.0: "streuend (cs² = 4)"}
DEFAULT_CS2 = 1.0
SEED_MAX = 999999
DEFAULT_SEED = 35

LIVE_CUSTOMERS = 150_000                                # Kunden je Live-Lauf

# Vorgerechnete Studie (generate_precomputed.py)
STUDY_C = C_OPTIONS
STUDY_RHO_PCT = (50, 80, 90)
STUDY_CS2 = CS2_OPTIONS
STUDY_P_PCT = 20                                        # Hauptstudie: Anteil Eiliger 20 %
STUDY_CUSTOMERS = 1_000_000                             # Kunden je Wiederholung
STUDY_REPS = 6                                          # unabhängige Wiederholungen je Zelle
PSTUDY_P_PCT = (10, 20, 50)                             # Studie über den Anteil Eiliger (eine Spur, ρ = 90 %, exponentiell)
PSTUDY_RHO_PCT = 90

CHAIN_TAIL = 1e-7                                       # Abschneiden der Markov-Kette: Gesamtwahrscheinlichkeit jenseits der Grenze höchstens so groß

PRESET_ORDER = ("Vorfahrt für 20 % (90 % Last)", "Unterbrechende Vorfahrt", "Geringe Last (50 %)", "Vier Spuren")


def _preset(c=DEFAULT_C, rho_pct=DEFAULT_RHO_PCT, p_pct=DEFAULT_P_PCT, policy=DEFAULT_POLICY, cs2=DEFAULT_CS2):
    return {"c": c, "rho_pct": rho_pct, "p_pct": p_pct, "policy": policy, "cs2": cs2, "seed": DEFAULT_SEED}


PRESETS = {
    "Vorfahrt für 20 % (90 % Last)": _preset(),
    "Unterbrechende Vorfahrt": _preset(policy="pr", cs2=4.0, rho_pct=80),
    "Geringe Last (50 %)": _preset(rho_pct=50),
    "Vier Spuren": _preset(c=4, rho_pct=80),
}
# Zahlen aus den exakten Formeln (Cobham, Preemptive-Resume), 3 min mittlere Abfertigung, Anteil Eiliger 20 %; tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Vorfahrt für 20 % (90 % Last)": "Auslastung 90 %, eine Spur: Eilige warten 3.3 min statt 27.0 min bei FIFO (8.2-fach schneller), Standard-Lkw 32.9 min (+22 %); das Mittel über alle bleibt bei 27.0 min.",
    "Unterbrechende Vorfahrt": "Auslastung 80 %, eine Spur, streuende Dauer (cs² = 4): Eilige 1.4 min statt 30.0 min (21-fach schneller), Standard-Lkw 36.3 min (+21 %); das Gesamtmittel sinkt von 30.0 auf 29.3 min.",
    "Geringe Last (50 %)": "Auslastung 50 %, eine Spur: Vorfahrt bringt wenig: Eilige 1.7 min statt 3.0 min (1.8-fach schneller), Standard-Lkw 3.3 min (+11 %).",
    "Vier Spuren": "Auslastung 80 %, vier Spuren, nicht unterbrechend: Eilige 0.5 min, Standard-Lkw 2.7 min, FIFO 2.2 min: 4.2-fach schneller, Standard-Lkw +19 % langsamer.",
}
