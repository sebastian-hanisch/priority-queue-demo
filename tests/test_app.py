"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, Fälle ohne Formel, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Markov-Kette, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import prio_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "FIFO (Bezug, Formel)") == "27.00 min"                                  # ρ = 0.9, M/M/1: 9 Abfertigungsdauern à 3 min
    assert _metric(at, "Eilige: wie viel schneller als FIFO").endswith("-fach") and _metric(at, "Standard-Lkw: wie viel länger als FIFO").startswith(("+", "−"))
    assert float(_metric(at, "Verzögerung der Eiligen (Simulation)").split()[0]) < float(_metric(at, "Verzögerung der Standard-Lkw (Simulation)").split()[0])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["c_select"], at.session_state["rho_slider"], at.session_state["p_slider"], at.session_state["policy_select"], at.session_state["cs2_select"]) == (
        p["c"], p["rho_pct"], p["p_pct"], p["policy"], p["cs2"])
    assert at.metric


def test_fifo_has_neither_winners_nor_losers():
    at = _run(policy_select="fifo")
    _ok(at)
    assert _metric(at, "Eilige: wie viel schneller als FIFO") == "–" and _metric(at, "Standard-Lkw: wie viel länger als FIFO") == "–"
    assert any("weder Gewinner noch Verlierer" in i.value for i in at.info)


def test_no_formula_for_several_lanes_with_non_exponential_service_shows_the_simulation_only():
    at = _run(c_select=4, cs2_select=4.0, policy_select="np")
    _ok(at)
    assert _metric(at, "FIFO (Bezug, Formel)") == "keine Formel" and _metric(at, "Eilige: wie viel schneller als FIFO") == "–"
    assert any("nur bei exponentieller Dauer" in i.value for i in at.info)                      # statt des Formel-Diagramms
    assert float(_metric(at, "Verzögerung der Eiligen (Simulation)").split()[0]) > 0


@pytest.mark.parametrize("kw", [dict(c_select=1, rho_slider=95, p_slider=50, policy_select="pr", cs2_select=4.0), dict(c_select=4, rho_slider=50, p_slider=10, policy_select="np", cs2_select=0.0),
                                 dict(c_select=4, rho_slider=95, p_slider=30, policy_select="pr", cs2_select=1.0), dict(c_select=1, rho_slider=50, p_slider=10, policy_select="fifo"),
                                 dict(c_select=1, rho_slider=65, p_slider=40, policy_select="np", cs2_select=0.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_the_markov_chain_agrees_with_the_formula_shown_next_to_it():
    at = _run(c_select=1, rho_slider=90, p_slider=20, policy_select="pr")
    _ok(at)
    table = next(m.value for m in at.markdown if m.value.startswith("| Größe | Markov-Kette | Formel |"))
    assert "| Verzögerung der Eiligen | 0.66 min | 0.66 min |" in table and "| Verzögerung der Standard-Lkw | 33.59 min | 33.59 min |" in table
    assert any("stimmen überein" in i.value for i in at.info)


def test_non_preemptive_chain_is_used_for_one_lane_and_the_preemptive_one_otherwise():
    np_one = _run(c_select=1, policy_select="np")
    assert any("Vorfahrt, nicht unterbrechend" in m.value and "Zustand des Gates ein Paar" in m.value for m in np_one.markdown)
    other = _run(c_select=4, policy_select="np")
    assert any("keine zweidimensionale Kette der gewählten Reihenfolge" in m.value for m in other.markdown)
    fifo = _run(policy_select="fifo")
    assert any("keine zweidimensionale Kette der gewählten Reihenfolge" in m.value for m in fifo.markdown)


def test_dice_button_changes_the_seed_and_the_simulated_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run()
    old_seed, old = at.session_state["seed_input"], _metric(at, "Verzögerung der Eiligen (Simulation)")
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed and _metric(at, "Verzögerung der Eiligen (Simulation)") != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rho"] = "87"
    at.query_params["p"] = "34"
    at.query_params["c"] = "3"
    at.query_params["cs2"] = "0.6"
    at.query_params["pol"] = "pr"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == 85 and at.session_state["p_slider"] == 30 and at.session_state["c_select"] == 4
    assert at.session_state["cs2_select"] == 1.0 and at.session_state["policy_select"] == "pr"


def test_permalink_ignores_garbage_and_unknown_policies():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rho"] = "viel"
    at.query_params["pol"] = "lifo"
    at.query_params["p"] = "nan"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == C.DEFAULT_RHO_PCT and at.session_state["policy_select"] == C.DEFAULT_POLICY and at.session_state["p_slider"] == C.DEFAULT_P_PCT


def test_charts_sections_and_limits_table_are_present():
    at = _run()
    _ok(at)
    assert len(at.get("plotly_chart")) == 6
    headers = [s.value for s in at.subheader]
    for part in ("Wer gewinnt, wer zahlt", "Der Erhaltungssatz", "Wie viele Eilige verträgt", "Die Markov-Kette dahinter", "Wo die Annahmen enden"):
        assert any(part in h for h in headers), part
    table = next(m.value for m in at.markdown if "Wer setzt an" in m.value)
    for name in ("Jackson-Netze", "Erlang A", "Erlang B", "Zeitvariable Ankünfte", "cμ-Regel"):
        assert name in table
    assert "geplant" not in table      # die Linie wird erst vollständig veröffentlicht, kein Status-Zusatz


def test_study_tables_are_complete():
    at = _run()
    gain = next(m.value for m in at.markdown if m.value.startswith("| Auslastung | Eilige | Standard |"))
    for r in C.STUDY_RHO_PCT:
        assert f"| {r} % |" in gain
    cons = next(m.value for m in at.markdown if m.value.startswith("| Streuung der Dauer | FIFO |"))
    for label in C.CS2_LABELS.values():
        assert f"| {label} |" in cons


def test_seed_control_uses_the_portfolio_wording():
    at = _run()
    assert [n.label for n in at.number_input] == ["Zufalls-Seed"]


def test_related_demos_are_linked_and_footer_is_present():
    at = _run()
    text = " ".join(c.value for c in at.caption)
    for name in ("mg1-kingman-demo", "mmc-queue-demo", "mm1-queue-demo", "power-of-d-demo", "truck-appointment-demo"):
        assert name in text
    assert "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in text


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
