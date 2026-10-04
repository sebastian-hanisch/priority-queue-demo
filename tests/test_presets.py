"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten."""

import pytest

import prio_constants as C
import prio_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["c"] in C.C_OPTIONS and preset["policy"] in C.POLICIES and preset["cs2"] in C.CS2_OPTIONS
        assert C.RHO_PCT_MIN <= preset["rho_pct"] <= C.RHO_PCT_MAX and P.snap_rho(preset["rho_pct"]) == preset["rho_pct"]
        assert C.P_PCT_MIN <= preset["p_pct"] <= C.P_PCT_MAX and P.snap_p(preset["p_pct"]) == preset["p_pct"]
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Unterbrechende Vorfahrt"]["policy"] == "pr" and C.PRESETS["Unterbrechende Vorfahrt"]["cs2"] == 4.0
    assert C.PRESETS["Geringe Last (50 %)"]["rho_pct"] == 50 and C.PRESETS["Vier Spuren"]["c"] == 4
    assert C.PRESETS["Vorfahrt für 20 % (90 % Last)"]["rho_pct"] == 90 and C.PRESETS["Vorfahrt für 20 % (90 % Last)"]["p_pct"] == 20


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Vorfahrt für 20 % (90 % Last)"]
    assert (p["c"], p["rho_pct"], p["p_pct"], p["policy"], p["cs2"], p["seed"]) == (C.DEFAULT_C, C.DEFAULT_RHO_PCT, C.DEFAULT_P_PCT, C.DEFAULT_POLICY, C.DEFAULT_CS2, C.DEFAULT_SEED)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("rho_slider") == (50, 95) and P.bounds("p_slider") == (10, 50)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


@pytest.mark.parametrize("key,value,expected", [("c_select", 2, 1), ("c_select", 3, 4), ("c_select", 9, 4), ("cs2_select", 0.4, 0.0), ("cs2_select", 0.6, 1.0),
                                                ("cs2_select", 2.0, 1.0), ("cs2_select", 2.6, 4.0)])
def test_option_regulators_snap_to_the_nearest_option(key, value, expected):
    assert P.snap_to_option(key, value) == expected


@pytest.mark.parametrize("value,expected", [(0, 50), (52, 50), (53, 55), (87, 85), (88, 90), (99, 95)])
def test_utilisation_snaps_to_the_step_inside_the_bounds(value, expected):
    assert P.snap_rho(value) == expected


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (34, 30), (36, 40), (99, 50)])
def test_share_of_urgent_trucks_snaps_to_the_step_inside_the_bounds(value, expected):
    assert P.snap_p(value) == expected


def test_formatters():
    assert C.fmt_int(150000) == "150.000" and C.fmt_pct(0.2) == "20 %" and C.fmt_pct(0.0898, 1) == "9.0 %"
    assert C.fmt_signed_pct(0.22) == "+22 %" and C.fmt_signed_pct(-0.44) == "−44 %" and C.fmt_signed_pct(0.0) == "+0 %"
