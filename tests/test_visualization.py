"""Abbildungen: gesperrte Achsen, Zahl der Linien, Beschriftungen."""

import prio_chain as K
import prio_constants as C
import prio_evaluation as E
import prio_formulas as F
import prio_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    rep = E.live_report(1, 80, 20, "np", 1.0, seed=1, customers=5_000)
    chain = K.pr_chain(1, 0.8, 0.2)
    figs = [V.build_class_chart(rep), V.build_gain_loss_chart(1, 0.2, 1.0, "np"), V.build_conservation_chart(PRE, 1, 80), V.build_p_chart(PRE, "np"),
            V.build_chain_grid(chain), V.build_marginals_chart(chain)]
    assert all(_locked(f) for f in figs)


def test_policy_labels():
    assert [V.policy_label(p) for p in C.POLICIES] == ["gleiche Behandlung (FIFO)", "Vorfahrt, nicht unterbrechend", "Vorfahrt, unterbrechend"]


def test_class_chart_shows_the_formula_bars_only_where_a_formula_exists():
    with_formula = E.live_report(1, 80, 20, "np", 1.0, seed=1, customers=4_000)
    without = E.live_report(4, 80, 20, "np", 4.0, seed=1, customers=4_000)
    assert [t.name for t in V.build_class_chart(with_formula).data] == ["FIFO (Formel)", "gewählte Reihenfolge (Formel)", "Simulation"]
    assert [t.name for t in V.build_class_chart(without).data] == ["Simulation"]


def test_gain_loss_chart_exists_only_where_a_formula_exists_and_has_a_line_per_class():
    fig = V.build_gain_loss_chart(1, 0.2, 4.0, "pr")
    assert [t.name for t in fig.data] == ["FIFO (beide Klassen)", "Eilige", "Standard"]
    assert V.build_gain_loss_chart(4, 0.2, 4.0, "np") is None and V.build_gain_loss_chart(4, 0.2, 0.0, "pr") is None
    assert len(V.build_gain_loss_chart(4, 0.2, 1.0, "np").data) == 3 and len(V.build_gain_loss_chart(1, 0.2, 1.0, "fifo").data) == 1
    hi, lo, fifo = fig.data[1].y[40], fig.data[2].y[40], fig.data[0].y[40]
    assert hi < fifo < lo


def test_conservation_chart_has_one_bar_group_per_policy():
    fig = V.build_conservation_chart(PRE, 1, 80)
    assert [t.name for t in fig.data] == [V.policy_label(p) for p in C.POLICIES] and all(len(t.x) == len(C.STUDY_CS2) for t in fig.data)


def test_p_chart_has_formula_lines_and_study_points():
    fig = V.build_p_chart(PRE, "np")
    assert len(fig.data) == 5 and list(fig.data[3].x) == list(C.PSTUDY_P_PCT)
    assert abs(fig.data[0].y[0] - F.to_minutes(F.fifo_wait(1, 0.9, 1.0))) < 1e-9


def test_chain_grid_is_log_scaled_and_marginals_cover_both_classes():
    chain = K.pr_chain(1, 0.8, 0.2)
    grid = V.build_chain_grid(chain, show=10)
    assert grid.data[0].z.shape == (11, 11)
    assert [t.name for t in V.build_marginals_chart(chain, show=10).data] == ["Eilige", "Standard"]
