"""Prioritätsklassen - wer darf vor? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der Konzepte-Linie "Warteschlangentheorie und Simulation": Eilige Lkw (Klasse 1) bekommen Vorfahrt vor Standard-Lkw (Klasse 2). Die Demo zeigt Formeln
(Cobham für nicht unterbrechende, Preemptive-Resume für unterbrechende Vorfahrt) gegen Simulation, den Erhaltungssatz (Vorfahrt verschiebt Wartezeit, sie spart keine), den
Anteil Eiliger und die exakt gelöste zweidimensionale Markov-Kette als unabhängige Referenz. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import prio_constants as C
import prio_formulas as F
from prio_evaluation import chain_report, live_report, load_precomputed, mean_of, nearest, penalty_vs_fifo, speedup_vs_fifo, study_cell
from prio_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                          sync_query_params)
from prio_visualization import (build_chain_grid, build_class_chart, build_conservation_chart, build_gain_loss_chart, build_marginals_chart, build_p_chart,
                                policy_label)

st.set_page_config(page_title="Prioritätsklassen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _precomputed():
    return load_precomputed()


@st.cache_data(show_spinner=False)
def _live(c, rho_pct, p_pct, policy, cs2, seed):
    return live_report(c, rho_pct, p_pct, policy, cs2, seed)


@st.cache_data(show_spinner=False)
def _chain(c, rho_pct, p_pct, policy):
    return chain_report(c, rho_pct, p_pct, policy)


def _min(x):
    """Zeit in Abfertigungsdauern → Text in Minuten."""
    return f"{F.to_minutes(x):.2f} min"


st.title("🚦 Prioritätsklassen: wer darf vor?")
st.markdown(
    """
Bisher wurden alle Lkw gleich behandelt. Am Terminal gibt es aber **Eilige** (Express, Kühlcontainer, enge Fristen) und **Standard-Lkw**. Gibt man den Eiligen **Vorfahrt**, wird ihre
Wartezeit drastisch kürzer, aber die Wartezeit verschwindet nicht: Sie wandert zu den anderen. Das ist der **Erhaltungssatz**: Bei gleicher Abfertigungsdauer und nicht
unterbrechender Vorfahrt bleibt das Mittel über alle Lkw genau so groß wie bei FIFO. Die Formel von **Cobham** sagt, wie sie sich verteilt; bei exponentieller Dauer lässt sich das Gate
sogar als **zweidimensionale Markov-Kette** exakt lösen, und diese Lösung ist hier die unabhängige Referenz.
"""
)
st.caption(
    "Elftes Stück der Linie „Warteschlangentheorie und Simulation“, aufbauend auf "
    "[mg1-kingman-demo](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/) (Pollaczek-Khinchine für eine Klasse) und "
    "[mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (mehrere Spuren, Erlang C). Jedes Folgestück hebt eine der Annahmen "
    "unter „Wo die Annahmen enden“ auf."
)

with st.expander("So funktioniert die Vorfahrt", expanded=True):
    st.markdown(
        """
- **Klassen:** Klasse 1 (eilig) mit Anteil p aller Ankünfte, Klasse 2 (Standard); Poisson-Ankünfte, **gleiche Abfertigungsdauer** für beide (Mittel 3 min). **Verzögerung** = Zeit im System − Dauer
  (bei Unterbrechung inklusive der Unterbrechungen).
- **FIFO:** Wer zuerst kommt, wird zuerst bedient, unabhängig von der Klasse.
- **Vorfahrt, nicht unterbrechend (Cobham):** Wird eine Spur frei, kommt zuerst ein Eiliger dran; wer schon abgefertigt wird, wird nicht unterbrochen. Eine Spur: W₁ = W₀/(1 − σ₁),
  W₂ = W₀/((1 − σ₁)(1 − σ₂)) mit W₀ = ρ(1 + cs²)/2, σ₁ = p·ρ, σ₂ = ρ.
- **Vorfahrt, unterbrechend (Wiederaufnahme):** Ein Eiliger verdrängt einen laufenden Standard-Lkw (den zuletzt gestarteten), der später mit seiner Restdauer weitermacht.
- **Mehrere Spuren:** exakte Formeln nur bei exponentieller Dauer (Erlang C mit dem Cobham-Faktor); sonst nur Simulation.
- **Markov-Kette:** Bei exponentieller Dauer ist der Zustand (Zahl Eiliger, Zahl Standard-Lkw) im System, bei nicht unterbrechender Vorfahrt zusätzlich die Klasse in Abfertigung.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    rho_pct = st.slider("Auslastung ρ je Spur", *bounds("rho_slider"), step=C.RHO_PCT_STEP, key="rho_slider", format="%d %%",
                        help="Anteil der Zeit, in der eine Spur im Mittel beschäftigt ist (alle Lkw zusammen).")
    p_pct = st.slider("Anteil eiliger Lkw", *bounds("p_slider"), step=C.P_PCT_STEP, key="p_slider", format="%d %%", help="Anteil der Klasse 1 an allen Ankünften.")
    c = st.select_slider("Spuren c", options=C.C_OPTIONS, key="c_select")
    policy = st.selectbox("Reihenfolge", C.POLICIES, key="policy_select", format_func=policy_label,
                          help="FIFO: alle gleich. Vorfahrt: Eilige zuerst, ohne oder mit Unterbrechung des laufenden Standard-Lkw.")
    cs2 = st.select_slider("Streuung der Dauer cs²", options=C.CS2_OPTIONS, key="cs2_select", format_func=lambda v: C.CS2_LABELS[v],
                           help="Variationskoeffizient² der Abfertigungsdauer (für beide Klassen gleich, Mittel 3 min).")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt alle Zufallszahlen der Simulation.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

rho_pct, p_pct, c, seed = int(rho_pct), int(p_pct), int(c), int(seed)
cs2 = float(cs2)
rho, p = rho_pct / 100.0, p_pct / 100.0
sync_query_params({"rho_slider": rho_pct, "p_slider": p_pct, "c_select": c, "policy_select": policy, "cs2_select": cs2, "seed_input": seed})

pre = _precomputed()
study_rho = nearest(C.STUDY_RHO_PCT, rho_pct)
with st.spinner("Simuliere das Gate …"):
    live = _live(c, rho_pct, p_pct, policy, cs2, seed)
sim = live["sim"]

st.markdown("---")
st.markdown("## 🚦 Vorfahrt am Gate")
st.caption(
    f"{c} Spur(en), Auslastung {rho_pct} %, {p_pct} % eilige Lkw, {policy_label(policy)}, Dauer mit cs² = {cs2:g}. Simuliert: {C.fmt_int(C.LIVE_CUSTOMERS)} Lkw (die ersten 5 % nicht ausgewertet)."
)
r1 = st.columns(3)
r1[0].metric("Verzögerung der Eiligen (Simulation)", _min(live["delay_hi"]), help=f"Formel: {_min(live['waits'][0])}." if live["waits"] else "Für diese Kombination gibt es keine Formel.")
r1[1].metric("Verzögerung der Standard-Lkw (Simulation)", _min(live["delay_lo"]), help=f"Formel: {_min(live['waits'][1])}." if live["waits"] else "Für diese Kombination gibt es keine Formel.")
r1[2].metric("FIFO (Bezug, Formel)", _min(live["fifo"]) if live["fifo"] is not None else "keine Formel",
             help="Alle Lkw gleich behandelt: Pollaczek-Khinchine (eine Spur) bzw. Erlang C (mehrere Spuren, exponentielle Dauer)." if live["fifo"] is not None
             else "Mehrere Spuren mit nicht exponentieller Dauer: keine geschlossene Formel.")
r2 = st.columns(3)
if policy != "fifo" and live["fifo"] is not None:
    r2[0].metric("Eilige: wie viel schneller als FIFO", f"{live['fifo'] / live['delay_hi']:.1f}-fach")
    r2[1].metric("Standard-Lkw: wie viel länger als FIFO", C.fmt_signed_pct(live["delay_lo"] / live["fifo"] - 1))
else:
    r2[0].metric("Eilige: wie viel schneller als FIFO", "–", help="Bei FIFO gibt es keinen Unterschied; für mehrere Spuren mit nicht exponentieller Dauer fehlt der FIFO-Bezug."
                 if policy == "fifo" else "Für mehrere Spuren mit nicht exponentieller Dauer fehlt der FIFO-Bezug.")
    r2[1].metric("Standard-Lkw: wie viel länger als FIFO", "–")
r2[2].metric("Mittel über alle Lkw (Simulation)", _min(sim.delay_all),
             help=(f"FIFO nach Formel: {_min(live['fifo'])}. Bei nicht unterbrechender Vorfahrt und gleicher Dauer gleich (Erhaltungssatz)." if live["fifo"] is not None else None))

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Verzögerung je Klasse**")
    st.plotly_chart(build_class_chart(live), width="stretch", key=f"cls_{c}_{rho_pct}_{p_pct}_{policy}_{cs2}_{seed}")
with col_b:
    st.markdown("**Verzögerung über die Auslastung (Formel)**")
    gl = build_gain_loss_chart(c, p, cs2, policy)
    if gl is None:
        st.info("Für mehrere Spuren gibt es Formeln nur bei exponentieller Dauer (cs² = 1); bei dieser Streuung zeigt die Demo nur die Simulation.")
    else:
        st.plotly_chart(gl, width="stretch", key=f"gl_{c}_{p_pct}_{policy}_{cs2}")
if policy != "fifo" and live["fifo"] is not None:
    st.info(
        f"Mit {policy_label(policy)} warten die Eiligen im Mittel **{_min(live['delay_hi'])}** statt {_min(live['fifo'])} bei FIFO "
        f"({live['fifo'] / live['delay_hi']:.1f}-fach schneller); die Standard-Lkw warten **{_min(live['delay_lo'])}**, das sind {C.fmt_signed_pct(live['delay_lo'] / live['fifo'] - 1)} gegenüber FIFO."
    )
st.caption(
    f"Ein Live-Lauf ist kurz ({C.fmt_int(C.LIVE_CUSTOMERS)} Lkw): bei hoher Auslastung streut er um Zehntel des Werts. Die Studien unten mitteln je {C.STUDY_REPS} Läufe à "
    f"{C.fmt_int(C.STUDY_CUSTOMERS)} Lkw (Anteil Eiliger 20 %)."
)

st.markdown("---")
st.subheader("📐 Wer gewinnt, wer zahlt?")
if policy == "fifo":
    st.info("Wählen Sie links eine Vorfahrtsregel; bei FIFO gibt es weder Gewinner noch Verlierer.")
else:
    st.markdown(
        f"Studie (Anteil Eiliger 20 %): {c} Spur(en), Dauer mit cs² = {cs2:g}, {policy_label(policy)}; je Auslastung die gemessene Verzögerung, der Gewinn der Eiligen und der Verlust der Standard-Lkw gegenüber FIFO."
    )
    rows = []
    for r_pct in C.STUDY_RHO_PCT:
        cell, fifo_cell = study_cell(pre, c, r_pct, policy, cs2), study_cell(pre, c, r_pct, "fifo", cs2)
        rows.append(f"| {r_pct} % | {_min(mean_of(cell, 'hi'))} | {_min(mean_of(cell, 'lo'))} | {_min(mean_of(fifo_cell, 'all'))} | {speedup_vs_fifo(cell, fifo_cell):.1f}-fach | "
                    f"{C.fmt_signed_pct(penalty_vs_fifo(cell, fifo_cell))} |")
    st.markdown("| Auslastung | Eilige | Standard | FIFO | Eilige schneller | Standard langsamer |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
    c90, f90 = study_cell(pre, c, 90, policy, cs2), study_cell(pre, c, 90, "fifo", cs2)
    c50, f50 = study_cell(pre, c, 50, policy, cs2), study_cell(pre, c, 50, "fifo", cs2)
    st.info(
        f"Bei Auslastung 90 % sind die Eiligen {speedup_vs_fifo(c90, f90):.1f}-fach schneller, die Standard-Lkw warten {C.fmt_signed_pct(penalty_vs_fifo(c90, f90))} länger; "
        f"bei 50 % sind es {speedup_vs_fifo(c50, f50):.1f}-fach und {C.fmt_signed_pct(penalty_vs_fifo(c50, f50))}. Vorfahrt lohnt sich dort, wo es eng ist."
    )

st.markdown("---")
st.subheader("🔬 Der Erhaltungssatz: Vorfahrt verschiebt, sie spart nicht")
study_c_rho = nearest(C.STUDY_RHO_PCT, rho_pct)
st.markdown(
    f"Mittlere Verzögerung **aller** Lkw ({c} Spur(en), Auslastung {study_c_rho} %, 20 % Eilige) je Reihenfolge und Streuung der Dauer. Bei nicht unterbrechender Vorfahrt ist sie gleich FIFO; bei unterbrechender Vorfahrt nur bei exponentieller Dauer."
)
st.plotly_chart(build_conservation_chart(pre, c, study_c_rho), width="stretch", key=f"cons_{c}_{study_c_rho}")
rows = []
for s2 in C.STUDY_CS2:
    vals = [mean_of(study_cell(pre, c, study_c_rho, pol, s2), "all") for pol in C.POLICIES]
    rows.append(f"| {C.CS2_LABELS[s2]} | {_min(vals[0])} | {_min(vals[1])} ({C.fmt_signed_pct(vals[1] / vals[0] - 1, 1)}) | {_min(vals[2])} ({C.fmt_signed_pct(vals[2] / vals[0] - 1, 1)}) |")
st.markdown("| Streuung der Dauer | FIFO | nicht unterbrechend | unterbrechend |\n|---|---|---|---|\n" + "\n".join(rows))
cv4 = [mean_of(study_cell(pre, c, study_c_rho, pol, 4.0), "all") for pol in C.POLICIES]
st.info(
    f"Bei streuender Dauer (cs² = 4) liegt das Gesamtmittel bei nicht unterbrechender Vorfahrt {C.fmt_signed_pct(cv4[1] / cv4[0] - 1, 1)} neben FIFO (Rauschen), bei unterbrechender "
    f"{C.fmt_signed_pct(cv4[2] / cv4[0] - 1, 1)}: Unterbrechen spart dort wirklich etwas, weil kurze Aufträge lange nicht mehr blockieren."
)

st.markdown("---")
st.subheader("🔬 Wie viele Eilige verträgt die Vorfahrt?")
pol_for_p = policy if policy != "fifo" else "np"
st.markdown(
    f"Eine Spur, Auslastung {C.PSTUDY_RHO_PCT} %, exponentielle Dauer, {policy_label(pol_for_p)}: Verzögerung je Klasse über den Anteil Eiliger nach Formel (Linien) und Simulation (Kreise). "
    "Je mehr Lkw Vorfahrt haben, desto weniger wert ist sie."
)
st.plotly_chart(build_p_chart(pre, pol_for_p), width="stretch", key=f"p_{pol_for_p}")
p10, p50 = F.class_waits(pol_for_p, 1, C.PSTUDY_RHO_PCT / 100, 0.1, 1.0), F.class_waits(pol_for_p, 1, C.PSTUDY_RHO_PCT / 100, 0.5, 1.0)
fifo90 = F.fifo_wait(1, C.PSTUDY_RHO_PCT / 100, 1.0)
st.info(
    f"Bei 10 % Eiligen warten sie {_min(p10[0])} (FIFO {_min(fifo90)}), bei 50 % {_min(p50[0])}; die Standard-Lkw warten {_min(p10[1])} bzw. {_min(p50[1])}. "
    "Wenn die Hälfte Vorfahrt hat, bleibt für die andere Hälfte kaum etwas übrig."
)

st.markdown("---")
st.subheader("🔗 Die Markov-Kette dahinter")
with st.spinner("Löse die Markov-Kette …"):
    chain, chain_kind = _chain(c, rho_pct, p_pct, policy)
chain_policy = chain_kind
formula = F.class_waits(chain_policy, c, rho, p, 1.0)
st.markdown(
    f"Bei **exponentieller** Dauer ist der Zustand des Gates ein Paar (n₁, n₂): so viele Eilige und so viele Standard-Lkw sind im System"
    f"{' (dazu die Klasse in Abfertigung)' if chain_kind == 'np' else ''}. Die Kette wird exakt gelöst (πQ = 0, {C.fmt_int(chain.n_max)} als Grenze für n₁ + n₂, "
    f"Rest-Fehler {chain.residual:.0e}, Wahrscheinlichkeit auf dem Rand {chain.boundary_mass:.0e}). Gezeigt: **{policy_label(chain_kind)}**"
    f"{'' if chain_kind == policy else ' (für diese Einstellung gibt es keine zweidimensionale Kette der gewählten Reihenfolge)'}."
)
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Gleichgewichtswahrscheinlichkeit je Zustand (n₁, n₂)**")
    st.plotly_chart(build_chain_grid(chain), width="stretch", key=f"grid_{c}_{rho_pct}_{p_pct}_{chain_kind}")
with col_d:
    st.markdown("**Randverteilungen**")
    st.plotly_chart(build_marginals_chart(chain), width="stretch", key=f"marg_{c}_{rho_pct}_{p_pct}_{chain_kind}")
st.markdown(
    "| Größe | Markov-Kette | Formel |\n|---|---|---|\n"
    f"| Eilige im System (Mittel) | {chain.e_n1:.3f} | – |\n| Standard-Lkw im System (Mittel) | {chain.e_n2:.3f} | – |\n"
    f"| Verzögerung der Eiligen | {_min(chain.delay1)} | {_min(formula[0]) if formula else 'keine Formel'} |\n"
    f"| Verzögerung der Standard-Lkw | {_min(chain.delay2)} | {_min(formula[1]) if formula else 'keine Formel'} |"
)
st.info(
    f"Die Kette und die Formel stimmen überein: Eilige {_min(chain.delay1)}, Standard {_min(chain.delay2)}. Die Kette ist eine **unabhängige** Rechnung (ein lineares Gleichungssystem über alle Zustände), "
    "die Formel eine geschlossene Herleitung; dass beide dasselbe ergeben, prüft beide."
    if formula else
    "Für diese Einstellung gibt es keine Formel; die Kette liefert den exakten Wert."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Beide Klassen haben dieselbe Abfertigungsdauer** | Bei kürzeren Aufträgen der Eiligen ändert sich W₀ = Σ λᵢ·E[Sᵢ²]/2 und mit ihm alle Wartezeiten; die Formel von Cobham gilt weiter, hier nicht gerechnet. | kein Folgestück |
| **Zwei Klassen** | Die Formeln gelten für beliebig viele Klassen (σ_k kumuliert), die Demo zeigt zwei. | kein Folgestück |
| **Alle Klassen verlieren gleich viel durch Warten** | Bei unterschiedlichen Wartekosten ist die beste Rangfolge die nach Kosten je Dauer (cμ-Regel), nicht die nach Eile. | kein Folgestück |
| **Geduldige Lkw, unbegrenzter Warteraum** | Mit Abwanderung oder begrenztem Aufstellplatz gehen vor allem die Lkw der unteren Klasse verloren. | **[Erlang A](https://sebastianhanisch-erlang-a-demo.streamlit.app/)** und **[Erlang B](https://sebastianhanisch-erlang-b-demo.streamlit.app/)** |
| **Konstante Ankunftsrate** | Bei Wellen wechseln die Anteile der Klassen über den Tag; die Vorfahrt wirkt in der Spitze am stärksten. | **[Zeitvariable Ankünfte](https://sebastianhanisch-time-varying-arrivals-demo.streamlit.app/)** |
| **Ein Gate** | In Netzen läuft ein Lkw durch mehrere Stationen; Vorfahrt an einer Station verändert den Strom zur nächsten. | **[Jackson-Netze](https://sebastianhanisch-jackson-network-demo.streamlit.app/)** |
| **Die Kette gilt nur für exponentielle Dauer** | Bei anderer Streuung ist (n₁, n₂) kein Markov-Zustand mehr; dort bleiben Formel (eine Spur) und Simulation. | kein Folgestück |
"""
)
st.caption(
    "Verwandt im Portfolio: [markov-queue-demo](https://sebastianhanisch-markov-queue-demo.streamlit.app/) (Zusatzstück: die Grundlagen der Kette, die hier zweidimensional wird), [mg1-kingman-demo](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/) (Stück 10: eine Klasse, Pollaczek-Khinchine; hier mit zwei), "
    "[mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (Stück 3: mehrere Spuren, Erlang C), [mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) "
    "(Stück 1: die Geburts-Sterbe-Kette einer Klasse), [power-of-d-demo](https://sebastianhanisch-power-of-d-demo.streamlit.app/) (Stück 7: ein weiteres Gate, dessen mehrdimensionale Kette "
    "als Referenz dient) und die Hafen-Demo [truck-appointment-demo](https://sebastianhanisch-truck-appointment-demo.streamlit.app/) (Terminvergabe für Lkw)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $c$ Spuren, Poisson-Ankünfte mit Rate $c\rho$, Klasse 1 (eilig) mit Anteil $p$, Klasse 2; Abfertigung mit Mittel 1 und $c_s^2$ für beide Klassen; $\sigma_1 = p\rho$, $\sigma_2 = \rho$.
Verzögerung = Zeit im System − Dauer.

**Eine Spur, nicht unterbrechend (Cobham).** $W_0 = \dfrac{\rho(1 + c_s^2)}{2}$, $\;W_1 = \dfrac{W_0}{1 - \sigma_1}$, $\;W_2 = \dfrac{W_0}{(1 - \sigma_1)(1 - \sigma_2)}$.
FIFO: $W = W_0/(1 - \rho)$.

**Eine Spur, unterbrechend (Wiederaufnahme).** $T_k = \dfrac{1}{1 - \sigma_{k-1}} + \dfrac{R_k}{(1 - \sigma_{k-1})(1 - \sigma_k)}$ mit $R_k = \sum_{i \le k} \lambda_i E[S^2]/2$, Verzögerung $T_k - 1$.

**Mehrere Spuren, exponentielle Dauer.** Nicht unterbrechend: $W_k = \dfrac{C(c, c\rho)/c}{(1 - \sigma_{k-1})(1 - \sigma_k)}$. Unterbrechend: Klasse 1 sieht $M/M/c$ mit Auslastung $p\rho$; das Gesamtmittel ist das von $M/M/c$,
Klasse 2 folgt aus $p\,D_1 + (1 - p)\,D_2 = D_{M/M/c}$.

**Erhaltungssatz.** Bei nicht unterbrechender Vorfahrt und gleicher Dauer: $p\,W_1 + (1 - p)\,W_2 = W_{\text{FIFO}}$.

**Markov-Kette (exponentielle Dauer, unterbrechend).** Zustand $(n_1, n_2)$; Raten $(n_1, n_2) \to (n_1 + 1, n_2)$ mit $p\,c\rho$, $\to (n_1, n_2 + 1)$ mit $(1 - p)\,c\rho$, $\to (n_1 - 1, n_2)$ mit $\min(n_1, c)$,
$\to (n_1, n_2 - 1)$ mit $\min(n_2, c - \min(n_1, c))$. Gleichgewicht $\pi Q = 0$, $\sum \pi = 1$; $E[n_k]/\lambda_k - 1$ (Little) ist die Verzögerung. Nicht unterbrechend (eine Spur): zusätzlich die Klasse $s$ in Abfertigung;
bei frei werdender Spur kommt ein Eiliger dran, sonst ein Standard-Lkw.

Implementiert in `prio_formulas.py` (Formeln, Erhaltungssatz), `prio_chain.py` (Markov-Ketten), `prio_simulation.py` (Ereignisse, Vorfahrt, Unterbrechung), `prio_evaluation.py`
(Live-Lauf, Studienzellen), `generate_precomputed.py` (vorgerechnete Studie).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Warteschlangentheorie: M/M/1 bis Surrogat](https://sebastianhanisch.net/konzepte-warteschlangentheorie.html)."
)
