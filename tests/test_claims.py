"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Formelwerte exakt, Studien-Zahlen aus der vorgerechneten Datei (je 6 Läufe à 1 000 000 Lkw, Anteil Eiliger 20 %,
wo nicht anders genannt). Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Verzögerungen in
Abfertigungsdauern, wo nicht Minuten (3 min Mittel) dasteht."""

import pytest

import prio_chain as K
import prio_constants as C
import prio_evaluation as E
import prio_formulas as F

PRE = E.load_precomputed()


def cell(c, rho, policy, cs2=1.0):
    return E.study_cell(PRE, c, rho, policy, cs2)


def pct(x, digits=0):
    return round(100 * x) if digits == 0 else round(100 * x, digits)


def test_closed_forms_for_gain_and_loss():
    """README (eine Spur, 20 % Eilige, jede Dauer): nicht unterbrechend sind die Eiligen (1 − pρ)/(1 − ρ) = 1.8 / 4.2 / 8.2-fach schneller als bei FIFO, die Standard-Lkw pρ/(1 − pρ) = 11.1 / 19.0 /
    22.0 % langsamer (Auslastung 50 / 80 / 90 %); unterbrechend (1 − pρ)/(p(1 − ρ)) = 9.0 / 21.0 / 41.0-fach schneller."""
    rhos = (0.5, 0.8, 0.9)
    assert [round(F.np_speedup(r, 0.2), 1) for r in rhos] == [1.8, 4.2, 8.2]
    assert [round(100 * F.np_penalty(r, 0.2), 1) for r in rhos] == [11.1, 19.0, 22.0]
    assert [round(F.pr_speedup(r, 0.2), 1) for r in rhos] == [9.0, 21.0, 41.0]


def test_simulation_matches_every_formula():
    """README: In allen 69 Zellen und Klassen mit Formel (Verzögerung über 0.01 Abfertigungsdauern) weicht die Simulation höchstens 2.3 % ab, im Mittel 0.5 %."""
    devs = []
    for x in PRE["study"]:
        if x["waits"] is None:
            continue
        for key, f in (("hi", x["waits"][0]), ("lo", x["waits"][1])):
            if f > 0.01:
                devs.append(abs(E.mean_of(x, key) / f - 1))
    assert len(devs) == 69 and pct(max(devs), 1) == 2.3 and pct(sum(devs) / len(devs), 1) == 0.5


def test_measured_gain_and_loss_for_one_and_four_lanes():
    """README (exponentielle Dauer, Auslastung 50 / 80 / 90 %), gemessen gegen den FIFO-Lauf derselben Zelle: eine Spur nicht unterbrechend 1.80 / 4.20 / 8.23-fach schneller, Standard
    +11.7 / +19.2 / +20.5 %; unterbrechend 8.93 / 20.98 / 41.34-fach, +22.9 / +24.1 / +26.0 %; vier Spuren nicht unterbrechend 1.79 / 4.20 / 8.32-fach, +11.4 / +19.7 / +19.5 %."""
    def row(c, pol):
        return [(round(E.speedup_vs_fifo(cell(c, r, pol), cell(c, r, "fifo")), 2), pct(E.penalty_vs_fifo(cell(c, r, pol), cell(c, r, "fifo")), 1)) for r in C.STUDY_RHO_PCT]
    assert row(1, "np") == [(1.80, 11.7), (4.20, 19.2), (8.23, 20.5)]
    assert row(1, "pr") == [(8.93, 22.9), (20.98, 24.1), (41.34, 26.0)]
    assert row(4, "np") == [(1.79, 11.4), (4.20, 19.7), (8.32, 19.5)]


def test_urgent_trucks_on_four_lanes_with_preemption_hardly_wait_at_all():
    """README: Vier Spuren, unterbrechende Vorfahrt, exponentielle Dauer: die Eiligen warten in allen drei Auslastungen weniger als 0.01 min."""
    for r in C.STUDY_RHO_PCT:
        assert F.to_minutes(E.mean_of(cell(4, r, "pr"), "hi")) < 0.01


def test_conservation_law_for_non_preemptive_priority():
    """README: Bei nicht unterbrechender Vorfahrt ist das Gesamtmittel nach Formel gleich FIFO (alle Zellen mit Formel); in der Simulation weichen die 18 Zellen höchstens 3.0 % ab."""
    for c, s in ((1, 0.0), (1, 1.0), (1, 4.0), (4, 1.0)):
        for r in C.STUDY_RHO_PCT:
            x = cell(c, r, "np", s)
            assert x["total"] == pytest.approx(x["fifo"], rel=1e-9)
    devs = [abs(E.mean_of(cell(c, r, "np", s), "all") / E.mean_of(cell(c, r, "fifo", s), "all") - 1) for c in C.STUDY_C for r in C.STUDY_RHO_PCT for s in C.STUDY_CS2]
    assert len(devs) == 18 and pct(max(devs), 1) == 3.0


def test_preemption_changes_the_total_only_for_non_exponential_service():
    """README (eine Spur, Auslastung 50 / 80 / 90 %, nach Formel gegenüber FIFO): unterbrechende Vorfahrt Gesamtmittel bei cs² = 0 +8.9 / +3.8 / +2.0 %, bei cs² = 1 genau 0, bei cs² = 4 −5.3 / −2.3 / −1.2 %."""
    def dev(s):
        return [F.overall_wait(F.pr_waits(1, r / 100, 0.2, s), 0.2) / F.fifo_wait(1, r / 100, s) - 1 for r in C.STUDY_RHO_PCT]
    assert [pct(x, 1) for x in dev(0.0)] == [8.9, 3.8, 2.0] and [pct(x, 1) for x in dev(4.0)] == [-5.3, -2.3, -1.2]
    assert all(abs(x) < 1e-9 for x in dev(1.0))


def test_share_of_urgent_trucks():
    """README (eine Spur, Auslastung 90 %, exponentiell, nicht unterbrechend): Anteil Eiliger 10 / 20 / 50 %: Gewinn der Eiligen 9.1 / 8.2 / 5.5-fach, Verlust der Standard-Lkw +9.9 / +22.0 / +81.8 % (Formel);
    simulierte Verzögerung der Eiligen 0.99 / 1.10 / 1.64 (FIFO 9.0), der Standard-Lkw 9.99 / 10.93 / 16.61 Abfertigungsdauern."""
    ps = (0.1, 0.2, 0.5)
    assert [round(F.np_speedup(0.9, p), 1) for p in ps] == [9.1, 8.2, 5.5] and [pct(F.np_penalty(0.9, p), 1) for p in ps] == [9.9, 22.0, 81.8]
    cells = [E.pstudy_cell(PRE, p, "np") for p in C.PSTUDY_P_PCT]
    assert [round(E.mean_of(x, "hi"), 2) for x in cells] == [0.99, 1.10, 1.64] and [round(E.mean_of(x, "lo"), 2) for x in cells] == [9.99, 10.93, 16.61]


def test_markov_chain_agrees_with_the_formulas_to_five_digits():
    """README (eine Spur, Auslastung 90 %, 20 % Eilige, exponentiell): unterbrechend Kette 0.21951 / 11.19511 gegen Formel 0.21951 / 11.19512 Abfertigungsdauern, 12 880 Zustände, Rest-Fehler unter 10⁻¹²;
    nicht unterbrechend Kette 1.09756 / 10.97560 gegen Cobham 1.09756 / 10.97561 (Abweichung unter 2·10⁻⁵, vom Abschneiden am Rand)."""
    ch = K.pr_chain(1, 0.9, 0.2)
    d1, d2 = F.pr_waits(1, 0.9, 0.2, 1.0)
    assert (round(ch.delay1, 5), round(ch.delay2, 5)) == (0.21951, 11.19511) and (round(d1, 5), round(d2, 5)) == (0.21951, 11.19512)
    assert (ch.n_max + 1) * (ch.n_max + 2) // 2 == 12_880 and ch.residual < 1e-12
    chn = K.np1_chain(0.9, 0.2)
    w1, w2 = F.np_waits(1, 0.9, 0.2, 1.0)
    assert (round(chn.delay1, 5), round(chn.delay2, 5)) == (1.09756, 10.9756) and (round(w1, 5), round(w2, 5)) == (1.09756, 10.97561)
    assert abs(chn.delay2 - w2) < 2e-5 and abs(ch.delay2 - d2) < 2e-5


def test_noise_level_of_the_study_quoted_in_readme():
    """README: Der größte relative Standardfehler einer Klassenverzögerung (über 0.05 Abfertigungsdauern) beträgt 2.4 % (Mittel aus 6 Läufen)."""
    rel = [E.se_of(x, k) / E.mean_of(x, k) for x in PRE["study"] + PRE["pstudy"] for k in ("hi", "lo") if E.mean_of(x, k) > 0.05]
    assert pct(max(rel), 1) == 2.4


def test_preset_help_numbers():
    """PRESET_HELP: jede Zahl aus den vier Texten (siehe prio_constants), Minuten bei 3 min Mittel, nach Formel."""
    h = C.PRESET_HELP
    m = lambda x: format(F.to_minutes(x), ".1f")
    w = F.class_waits("np", 1, 0.9, 0.2, 1.0)
    fifo = F.fifo_wait(1, 0.9, 1.0)
    t = h["Vorfahrt für 20 % (90 % Last)"]
    assert all(x in t for x in (m(w[0]), m(fifo), m(w[1]), f"{F.np_speedup(0.9, 0.2):.1f}-fach", f"+{pct(F.np_penalty(0.9, 0.2))} %")) and m(F.overall_wait(w, 0.2)) == "27.0"
    w = F.class_waits("pr", 1, 0.8, 0.2, 4.0)
    fifo = F.fifo_wait(1, 0.8, 4.0)
    t = h["Unterbrechende Vorfahrt"]
    assert all(x in t for x in (m(w[0]), m(fifo), m(w[1]), f"{fifo / w[0]:.0f}-fach", f"+{pct(w[1] / fifo - 1)} %", m(F.overall_wait(w, 0.2))))
    w = F.class_waits("np", 1, 0.5, 0.2, 1.0)
    fifo = F.fifo_wait(1, 0.5, 1.0)
    t = h["Geringe Last (50 %)"]
    assert all(x in t for x in (m(w[0]), m(fifo), m(w[1]), f"{fifo / w[0]:.1f}-fach", f"+{pct(w[1] / fifo - 1)} %"))
    w = F.class_waits("np", 4, 0.8, 0.2, 1.0)
    fifo = F.fifo_wait(4, 0.8, 1.0)
    t = h["Vier Spuren"]
    assert all(x in t for x in (m(w[0]), m(w[1]), m(fifo), f"{fifo / w[0]:.1f}-fach", f"+{pct(w[1] / fifo - 1)} %"))


def test_default_live_run_numbers():
    """README: Standardlauf (eine Spur, 90 %, 20 % Eilige, nicht unterbrechend, exponentiell, 150 000 Lkw, Seed 35): Eilige 3.33 min (Formel 3.29), Standard-Lkw 35.99 min (Formel 32.93, +9 %):
    der Lauf ist für die Standard-Lkw deutlich zu hoch (+33 % statt +22 % gegenüber FIFO)."""
    r = E.live_report(1, 90, 20, "np", 1.0, 35)
    assert (round(F.to_minutes(r["delay_hi"]), 2), round(F.to_minutes(r["delay_lo"]), 2)) == (3.33, 35.99)
    assert (round(F.to_minutes(r["waits"][0]), 2), round(F.to_minutes(r["waits"][1]), 2)) == (3.29, 32.93)
    assert pct(r["delay_lo"] / r["fifo"] - 1) == 33 and pct(r["delay_lo"] / r["waits"][1] - 1) == 9


def test_four_lanes_preemptive_factors_and_the_noisy_comparison_quoted_in_readme():
    """README: Vier Spuren, unterbrechend: gemessener Faktor 405- bis 953-fach (405.66 / 605.67 / 953.13), wegen Wartezeiten nahe null bedeutungslos; eine Spur, cs² = 4, Auslastung 80 %: Gesamtmittel
    unterbrechend gegenüber dem FIFO-LAUF −5.8 %, gegenüber der FIFO-FORMEL −3.8 % und nach Formel nur −2.3 % (pr-Formel gegen FIFO-Formel); der FIFO-Lauf liegt 2.1 % über seiner Formel."""
    sp = [E.speedup_vs_fifo(cell(4, r, "pr"), cell(4, r, "fifo")) for r in C.STUDY_RHO_PCT]
    assert [round(x, 2) for x in sp] == [405.66, 605.67, 953.13]
    pr, fifo = cell(1, 80, "pr", 4.0), cell(1, 80, "fifo", 4.0)
    assert pct(E.mean_of(pr, "all") / E.mean_of(fifo, "all") - 1, 1) == -5.8 and pct(E.mean_of(pr, "all") / fifo["fifo"] - 1, 1) == -3.8
    assert pct(F.overall_wait(F.pr_waits(1, 0.8, 0.2, 4.0), 0.2) / F.fifo_wait(1, 0.8, 4.0) - 1, 1) == -2.3 and pct(E.mean_of(fifo, "all") / fifo["fifo"] - 1, 1) == 2.1
