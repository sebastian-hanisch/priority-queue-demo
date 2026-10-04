"""Rechnet die teure Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt
`precomputed_sweep.json`.

  study   Spuren (1, 4) × Auslastung (50, 80, 90 %) × Reihenfolge (FIFO, nicht unterbrechend, unterbrechend) × Streuung der Dauer (0, 1, 4), Anteil Eiliger 20 %:
          je 6 Läufe à 1 000 000 Lkw; mittlere Verzögerung je Klasse und insgesamt, dazu die Formelwerte
  pstudy  eine Spur, Auslastung 90 %, exponentielle Dauer, Anteil Eiliger 10 / 20 / 50 %, drei Reihenfolgen: je 6 Läufe à 1 000 000 Lkw"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import prio_constants as C
from prio_evaluation import PRECOMPUTED_PATH, summarize_runs
from prio_simulation import simulate


def _task(args):
    key, c, rho_pct, p_pct, policy, cs2, seed = args
    return key, simulate(c, rho_pct / 100.0, p_pct / 100.0, policy, cs2, C.STUDY_CUSTOMERS, seed)


def main(workers):
    t0 = time.time()
    jobs, idx = [], 0
    for c in C.STUDY_C:
        for rho_pct in C.STUDY_RHO_PCT:
            for policy in C.POLICIES:
                for cs2 in C.STUDY_CS2:
                    for r in range(C.STUDY_REPS):
                        jobs.append((("study", c, rho_pct, C.STUDY_P_PCT, policy, cs2), c, rho_pct, C.STUDY_P_PCT, policy, cs2, 10_000 + 97 * idx + 1000 * r))
                    idx += 1
    for p_pct in C.PSTUDY_P_PCT:
        for policy in C.POLICIES:
            for r in range(C.STUDY_REPS):
                jobs.append((("pstudy", 1, C.PSTUDY_RHO_PCT, p_pct, policy, 1.0), 1, C.PSTUDY_RHO_PCT, p_pct, policy, 1.0, 400_000 + 97 * idx + 1000 * r))
            idx += 1
    jobs.sort(key=lambda j: -j[1])                       # Mehrspur-Läufe zuerst
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_task, jobs, chunksize=1))
    groups = {}
    for key, res in out:
        groups.setdefault(key, []).append(res)
    study, pstudy = [], []
    for (which, c, rho_pct, p_pct, policy, cs2), runs in groups.items():
        (study if which == "study" else pstudy).append(summarize_runs(c, rho_pct, p_pct, policy, cs2, C.STUDY_CUSTOMERS, runs))
    study.sort(key=lambda x: (x["c"], x["rho_pct"], C.POLICIES.index(x["policy"]), x["cs2"]))
    pstudy.sort(key=lambda x: (x["p_pct"], C.POLICIES.index(x["policy"])))
    Path(PRECOMPUTED_PATH).write_text(json.dumps({"study_customers": C.STUDY_CUSTOMERS, "study_reps": C.STUDY_REPS, "study": study, "pstudy": pstudy}), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(6, os.cpu_count() or 1))
