"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).
Alle Regler sind immer sichtbar - es gibt keinen ausblendbaren Regler (also auch kein KEPT-Muster)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import prio_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "rho_slider": SettingSpec("rho", int, C.DEFAULT_RHO_PCT, C.RHO_PCT_MIN, C.RHO_PCT_MAX),
    "p_slider": SettingSpec("p", int, C.DEFAULT_P_PCT, C.P_PCT_MIN, C.P_PCT_MAX),
    "c_select": SettingSpec("c", int, C.DEFAULT_C, C.C_OPTIONS[0], C.C_OPTIONS[-1]),
    "policy_select": SettingSpec("pol", str, C.DEFAULT_POLICY),
    "cs2_select": SettingSpec("cs2", float, C.DEFAULT_CS2, C.CS2_OPTIONS[0], C.CS2_OPTIONS[-1]),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"c": "c_select", "rho_pct": "rho_slider", "p_pct": "p_slider", "policy": "policy_select", "cs2": "cs2_select", "seed": "seed_input"}
OPTIONS = {"c_select": C.C_OPTIONS, "cs2_select": C.CS2_OPTIONS}       # Regler, die nur feste Stufen kennen


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap_to_option(state_key, value):
    """Regler mit festen Stufen: ein Permalink-Wert dazwischen rastet auf die nächste Stufe ein (bei Gleichstand auf die kleinere)."""
    return min(OPTIONS[state_key], key=lambda o: (abs(o - value), o))


def _snap_step(value, lo, hi, step):
    snapped = round(value / step) * step
    return int(min(hi, max(lo, snapped)))


def snap_rho(value):
    """Die Auslastung rastet auf das nächste Vielfache der Schrittweite (5 Prozentpunkte) innerhalb der Grenzen ein."""
    return _snap_step(value, C.RHO_PCT_MIN, C.RHO_PCT_MAX, C.RHO_PCT_STEP)


def snap_p(value):
    """Der Anteil Eiliger rastet auf das nächste Vielfache der Schrittweite (10 Prozentpunkte) innerhalb der Grenzen ein."""
    return _snap_step(value, C.P_PCT_MIN, C.P_PCT_MAX, C.P_PCT_STEP)


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if value != value:                     # NaN
                    continue
                if state_key == "policy_select":
                    if value not in C.POLICIES:
                        continue
                else:
                    if spec.lo is not None:
                        value = max(spec.lo, value)
                    if spec.hi is not None:
                        value = min(spec.hi, value)
                    if state_key in OPTIONS:
                        value = snap_to_option(state_key, value)
                    if state_key == "rho_slider":
                        value = snap_rho(value)
                    if state_key == "p_slider":
                        value = snap_p(value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
