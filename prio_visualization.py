"""Plotly-Abbildungen der Prioritäts-Demo: Verzögerung je Klasse (Simulation gegen Formel gegen FIFO), Gewinn und Verlust über die Auslastung, Erhaltungssatz, Anteil Eiliger,
Zustandsgitter und Randverteilungen der Markov-Kette. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import prio_constants as C
import prio_evaluation as E
import prio_formulas as F

HI_COLOR = "#e45756"
LO_COLOR = "#4c78a8"
FIFO_COLOR = "#9d9d9d"
SIM_COLOR = "#f58518"
POLICY_COLORS = {"fifo": "#9d9d9d", "np": "#4c78a8", "pr": "#e45756"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y),
                      plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def policy_label(policy):
    return C.POLICY_LABELS[policy]


def build_class_chart(report):
    """Verzögerung in Minuten je Klasse: FIFO (Bezug, Formel), die gewählte Reihenfolge nach Formel (wo vorhanden) und nach Simulation."""
    classes = ["Eilige (Klasse 1)", "Standard (Klasse 2)"]
    fig = go.Figure()
    if report["fifo"] is not None:
        fig.add_trace(go.Bar(x=classes, y=[F.to_minutes(report["fifo"])] * 2, name="FIFO (Formel)", marker_color=FIFO_COLOR))
    if report["waits"] is not None:
        fig.add_trace(go.Bar(x=classes, y=[F.to_minutes(w) for w in report["waits"]], name="gewählte Reihenfolge (Formel)", marker_color=[HI_COLOR, LO_COLOR]))
    fig.add_trace(go.Bar(x=classes, y=[F.to_minutes(report["delay_hi"]), F.to_minutes(report["delay_lo"])], name="Simulation", marker_color=SIM_COLOR))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="mittlere Verzögerung in Minuten", rangemode="tozero")
    return _base(fig, 340)


def build_gain_loss_chart(c, p, cs2, policy):
    """Verzögerung in Minuten über die Auslastung nach Formel: FIFO (grau) und die beiden Klassen der gewählten Reihenfolge; nur wo es eine Formel gibt (eine Spur: jede
    Dauer; mehrere Spuren: exponentielle Dauer). Gibt None zurück, wenn es keine gibt."""
    rhos = [r / 100 for r in range(30, 96)]
    if F.class_waits(policy, c, 0.5, p, cs2) is None or F.fifo_wait(c, 0.5, cs2) is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[100 * r for r in rhos], y=[F.to_minutes(F.fifo_wait(c, r, cs2)) for r in rhos], mode="lines",
                             line=dict(color=FIFO_COLOR, width=2.5, dash="dot"), name="FIFO (beide Klassen)"))
    if policy != "fifo":
        fig.add_trace(go.Scatter(x=[100 * r for r in rhos], y=[F.to_minutes(F.class_waits(policy, c, r, p, cs2)[0]) for r in rhos], mode="lines",
                                 line=dict(color=HI_COLOR, width=2.5), name="Eilige"))
        fig.add_trace(go.Scatter(x=[100 * r for r in rhos], y=[F.to_minutes(F.class_waits(policy, c, r, p, cs2)[1]) for r in rhos], mode="lines",
                                 line=dict(color=LO_COLOR, width=2.5), name="Standard"))
    fig.update_xaxes(title_text="Auslastung der Spur", ticksuffix=" %")
    fig.update_yaxes(title_text="mittlere Verzögerung in Minuten (logarithmisch)", type="log")
    return _base(fig, 340)


def build_conservation_chart(pre, c, rho_pct):
    """Gesamtmittel der Verzögerung (alle Lkw, Anteile gewichtet) je Reihenfolge und Streuung der Dauer, aus der Studie: bei nicht unterbrechender Vorfahrt gleich FIFO."""
    fig = go.Figure()
    labels = [C.CS2_LABELS[x] for x in C.STUDY_CS2]
    for policy in C.POLICIES:
        ys = [F.to_minutes(E.mean_of(E.study_cell(pre, c, rho_pct, policy, cs2), "all")) for cs2 in C.STUDY_CS2]
        fig.add_trace(go.Bar(x=labels, y=ys, name=policy_label(policy), marker_color=POLICY_COLORS[policy]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="mittlere Verzögerung aller Lkw in Minuten", rangemode="tozero")
    return _base(fig, 340)


def build_p_chart(pre, policy):
    """Eine Spur, Auslastung 90 %, exponentielle Dauer: Verzögerung je Klasse über den Anteil Eiliger nach Formel (Linien) und Studie (Punkte); FIFO grau."""
    ps = [x / 100 for x in range(5, 61)]
    fig = go.Figure()
    rho = C.PSTUDY_RHO_PCT / 100
    fig.add_trace(go.Scatter(x=[100 * x for x in ps], y=[F.to_minutes(F.fifo_wait(1, rho, 1.0))] * len(ps), mode="lines", line=dict(color=FIFO_COLOR, width=2.5, dash="dot"),
                             name="FIFO (beide Klassen)"))
    fig.add_trace(go.Scatter(x=[100 * x for x in ps], y=[F.to_minutes(F.class_waits(policy, 1, rho, x, 1.0)[0]) for x in ps], mode="lines", line=dict(color=HI_COLOR, width=2.5),
                             name="Eilige"))
    fig.add_trace(go.Scatter(x=[100 * x for x in ps], y=[F.to_minutes(F.class_waits(policy, 1, rho, x, 1.0)[1]) for x in ps], mode="lines", line=dict(color=LO_COLOR, width=2.5),
                             name="Standard"))
    cells = [E.pstudy_cell(pre, p, policy) for p in C.PSTUDY_P_PCT]
    for key, color in (("hi", HI_COLOR), ("lo", LO_COLOR)):
        fig.add_trace(go.Scatter(x=list(C.PSTUDY_P_PCT), y=[F.to_minutes(E.mean_of(c_, key)) for c_ in cells], mode="markers",
                                 marker=dict(color=color, size=9, symbol="circle-open", line=dict(width=2)), name="Simulation", showlegend=False))
    fig.update_xaxes(title_text="Anteil Eiliger an allen Lkw", ticksuffix=" %")
    fig.update_yaxes(title_text="mittlere Verzögerung in Minuten (logarithmisch)", type="log")
    return _base(fig, 340)


def build_chain_grid(chain, show=None):
    """Zustandsgitter der Markov-Kette: Gleichgewichtswahrscheinlichkeit (Zehnerlogarithmus) für n₁ Eilige (Zeilen) und n₂ Standard-Lkw (Spalten) im System."""
    show = show or min(chain.n_max, 25)
    block = chain.grid[:show + 1, :show + 1]
    z = np.log10(np.where(block > 0, block, np.nan))            # Rundungswerte der Lösung unter null (≈ −1e-18) und exakte Nullen bleiben leer
    fig = go.Figure(go.Heatmap(z=z, x=list(range(show + 1)), y=list(range(show + 1)), colorscale="Viridis", colorbar=dict(title="log₁₀ P")))
    fig.update_xaxes(title_text="Standard-Lkw im System n₂")
    fig.update_yaxes(title_text="Eilige im System n₁", autorange="reversed")
    return _base(fig, 340, top=10)


def build_marginals_chart(chain, show=None):
    """Randverteilungen: Wahrscheinlichkeit für n Eilige bzw. n Standard-Lkw im System (logarithmisch)."""
    show = show or min(chain.n_max, 40)
    fig = go.Figure()
    for axis, name, color in ((0, "Eilige", HI_COLOR), (1, "Standard", LO_COLOR)):
        m = chain.marginal(axis)[:show + 1]
        fig.add_trace(go.Scatter(x=list(range(show + 1)), y=[v if v > 0 else None for v in m], mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    fig.update_xaxes(title_text="Lkw der Klasse im System")
    fig.update_yaxes(title_text="Wahrscheinlichkeit (logarithmisch)", type="log", range=[-12, 0.2], tickvals=[1e-12, 1e-9, 1e-6, 1e-3, 1],
                     ticktext=["10⁻¹²", "10⁻⁹", "10⁻⁶", "0.001", "1"])
    return _base(fig, 340)
