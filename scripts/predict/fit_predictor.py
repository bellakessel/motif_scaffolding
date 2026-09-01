#!/usr/bin/env python3
"""fit_predictor.py — fit the per-method success model from raw cluster data. Fully traceable.

    outcomes.csv          per-(task, method) n_designs / n_designable   (aggregate_outcomes.py)
    pool.csv              the four motif descriptors per task
    compute_costs.csv     GENERATION GPU-hours per method  (sacct-derived)
              |
              |   binomial logistic GLM, fitted SEPARATELY PER METHOD
              v
    success_model.json    mu/sd + per-method {coef, gen_gpuh_per_design}   <- the whole contract

WHAT IS AND IS NOT IN THE COST TERM. Only **generation** cost differs between methods; the AF3
refold is the identical shared job for every arm, so it is carried as ONE explicit constant
(`refold_gpuh_per_fold`) rather than folded into each method's number. That keeps the comparison
honest: `gen_gpuh_per_success` ranks generators on the part they actually control, while
`gpuh_per_success` adds the shared term back for absolute budgeting. Because the shared term
dominates (~0.045 vs 0.0005-0.0085 GPU-h/design), the total-cost ranking is very nearly the
accuracy ranking -- choosing a cheap generator buys almost nothing.

COST COVERAGE IS CHECKED, NOT ASSUMED. A method whose cost row covers fewer tasks than its score
table gets a loud warning: genie3's baseline row is measured on 60 of 314 tasks, so its GPU-h/design
is inflated ~5x and any ROI rank it wins is unverified.

    python fit_predictor.py --outcomes models/outcomes.csv --pool <pool.csv> \
        --cost lib_interface_pad5:<runs>/lib_interface_pad5/compute_costs.csv \
        --cost lib_interface_pad5_uidx:<runs>/lib_interface_pad5_uidx/compute_costs.csv \
        --out models/success_model.json --report
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from motifgo import predictor as P                                          # noqa: E402

LABELS = {"rfdiff": "RFdiffusion", "proteina": "Proteina", "genie3": "Genie3",
          "pp1c_bb": "Protpardelle-1c", "laproteina": "La-Proteina (idx)",
          "laproteina_unindexed": "La-Proteina (uidx)"}
# Derived from the one measured refold row in the baseline costs: laproteina refold
# 178.09 GPU-h over 31,400 folds = 0.00567 GPU-h/fold (~20 s), matching the design note.
DEF_REFOLD_GPUH_PER_FOLD = 178.09 / 31400.0


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def load_costs(specs):
    """{(run, method_dir): (gpu_hours, n_tasks)} for the GENERATION stage only."""
    out = {}
    for s in specs:
        run, path = s.split(":", 1)
        for r in read_rows(path):
            if r.get("stage") != "generation":
                continue
            out[(run, r["method"])] = (float(r["gpu_hours"]), int(r["n_tasks"]))
    return out


def fit_method(X, y_succ, y_tot):
    """Binomial GLM. statsmodels for CI/p; sklearn fallback keeps coefficients identical."""
    Xd = np.column_stack([np.ones(len(X)), X])
    try:
        import statsmodels.api as sm
        m = sm.GLM(np.column_stack([y_succ, y_tot - y_succ]), Xd,
                   family=sm.families.Binomial()).fit()
        ci = m.conf_int()
        return (m.params, getattr(m, "pvalues", None), ci, m.llf, m.aic, "statsmodels")
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        xs, ys, ws = [], [], []
        for i in range(len(X)):
            xs += [X[i], X[i]]; ys += [1, 0]; ws += [y_succ[i], y_tot[i] - y_succ[i]]
        lr = LogisticRegression(penalty=None, max_iter=2000).fit(np.array(xs), ys, sample_weight=ws)
        return (np.r_[lr.intercept_, lr.coef_[0]], None, None, float("nan"), float("nan"), "sklearn")


def mcfadden(y_succ, y_tot, p_hat):
    eps = 1e-12
    ll = float(np.sum(y_succ * np.log(np.clip(p_hat, eps, 1)) +
                      (y_tot - y_succ) * np.log(np.clip(1 - p_hat, eps, 1))))
    p0 = y_succ.sum() / y_tot.sum()
    ll0 = float(np.sum(y_succ * np.log(p0) + (y_tot - y_succ) * np.log(1 - p0)))
    return 1.0 - ll / ll0, ll, ll0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--cost", action="append", default=[], metavar="RUN:PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsamples", type=int, default=100, help="designs requested per task")
    ap.add_argument("--refold-gpuh-per-fold", type=float, default=DEF_REFOLD_GPUH_PER_FOLD)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)

    pool = {f"{r['pdb_name']}_m{r['motif_num']}_{r.get('variant','base')}": r
            for r in read_rows(a.pool)}
    feats = {t: P.descriptors(r) for t, r in pool.items()}
    feats = {t: d for t, d in feats.items() if d}
    mu, sd = P.standardisation(list(feats.values()))
    costs = load_costs(a.cost)

    rows = read_rows(a.outcomes)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    print(f"pool {len(pool)} tasks | descriptors ok for {len(feats)}")
    print(f"standardisation  " + "  ".join(f"{c}: mu={mu[c]:.3f} sd={sd[c]:.3f}" for c in P.CONT_FEATURES))
    print(f"refold constant  {a.refold_gpuh_per_fold:.5f} GPU-h/fold (shared by every method; "
          f"excluded from gen_* columns)\n")

    methods, warnings_ = {}, []
    hdr = (f"{'method':22s} {'tasks':>5s} {'design rate':>11s} {'pseudoR2':>8s} "
           f"{'disjoint_z':>10s} {'len_z':>7s} {'lxd':>6s} {'H':>6s} {'E':>6s} {'rsa_z':>7s} {'const':>7s}")
    print(hdr); print("-" * len(hdr))
    for meth, rs in by_method.items():
        keep = [r for r in rs if r["task"] in feats and int(r["n_designs"]) > 0]
        if not keep:
            continue
        X, ysucc, ytot = [], [], []                      # design matrix; successes and trials per motif
        for r in keep:
            d = feats[r["task"]]                         # d = this motif's four descriptors
            x = P.featurise(d["motif_length"], d["disjointed_parts"], d["dominant_sse"], d["mean_rsa"], mu, sd)
            X.append([x[f] for f in P.FINAL_FEATURES])
            ysucc.append(int(r["n_designable"])); ytot.append(int(r["n_designs"]))
        X = np.asarray(X, float); ysucc = np.asarray(ysucc, float); ytot = np.asarray(ytot, float)
        params, pvals, ci, llf, aic, backend = fit_method(X, ysucc, ytot)
        logit = params[0] + X @ params[1:]               # params[0] = intercept, rest = feature coefficients
        p_hat = 1 / (1 + np.exp(-logit))
        r2, _, _ = mcfadden(ysucc, ytot, p_hat)

        run, mdir = keep[0]["run"], keep[0]["method_dir"]
        gpuh, n_cost = costs.get((run, mdir), (float("nan"), 0))
        n_score = len(keep)
        gen_per_design = (gpuh / (n_cost * a.nsamples)) if n_cost else float("nan")
        if n_cost and n_cost < n_score:
            warnings_.append(f"{meth}: cost row covers {n_cost} tasks but scores cover {n_score} "
                             f"-> gen GPU-h/design inflated ~{n_score / n_cost:.1f}x; its ROI rank is UNVERIFIED")
        seqs_per_design = sum(int(r["n_seqs"]) for r in keep) / max(sum(int(r["n_designs"]) for r in keep), 1)

        coef = {"const": float(params[0])}
        coef.update({f: float(v) for f, v in zip(P.FINAL_FEATURES, params[1:])})
        methods[meth] = {
            "label": LABELS.get(meth, meth), "coef": coef,
            "gen_gpuh_per_design": None if gen_per_design != gen_per_design else gen_per_design,
            "seqs_per_design": seqs_per_design,
            "n_tasks": n_score, "n_designs": int(ytot.sum()), "n_designable": int(ysucc.sum()),
            "design_rate": float(ysucc.sum() / ytot.sum()),
            "pseudo_r2_mcfadden": float(r2), "aic": None if aic != aic else float(aic),
            "cost_tasks_covered": n_cost or None, "cost_tasks_total": n_score,
            "cost_gpu_hours": None if gpuh != gpuh else gpuh, "backend": backend,
            "run": run, "method_dir": mdir,
        }
        c = coef
        print(f"{meth:22s} {n_score:>5d} {ysucc.sum()/ytot.sum():>11.3f} {r2:>8.3f} "
              f"{c['disjointed_parts_z']:>10.2f} {c['motif_length_z']:>7.2f} {c['len_x_disj']:>6.2f} "
              f"{c['H']:>6.2f} {c['E']:>6.2f} {c['mean_rsa_z']:>7.2f} {c['const']:>7.2f}")

    model = P.save_model(a.out, mu=mu, sd=sd, methods=methods,
                         refold_gpuh_per_fold=a.refold_gpuh_per_fold,
                         meta={"fitted": dt.datetime.now().isoformat(timespec="seconds"),
                               "outcomes": os.path.abspath(a.outcomes),
                               "pool": os.path.abspath(a.pool),
                               "costs": [os.path.abspath(s.split(':', 1)[1]) for s in a.cost],
                               "nsamples": a.nsamples,
                               "refold_gpuh_per_fold": a.refold_gpuh_per_fold,
                               "backend": next(iter(methods.values()))["backend"] if methods else None})
    print(f"\nwrote {a.out}")
    for w in warnings_:
        print(f"  !! {w}")

    if a.report:
        print("\n=== generation cost (the only term that differs between methods) ===")
        print(f"{'method':22s} {'GPU-h':>9s} {'tasks':>6s} {'GPU-h/design':>13s} {'seqs/design':>12s}")
        for m, s in methods.items():
            g = s["gen_gpuh_per_design"]
            print(f"{m:22s} {s['cost_gpu_hours'] or float('nan'):>9.2f} {s['cost_tasks_covered'] or 0:>6d} "
                  f"{(g if g is not None else float('nan')):>13.5f} {s['seqs_per_design']:>12.2f}")
        for label, dis, ln, sse, rsa in [("typical CONTIGUOUS motif (len 8, coil, rsa 0.32)", 1, 8, "C", 0.32),
                                         ("typical DISJOINT motif  (3 fragments)", 3, 8, "C", 0.32)]:
            x = P.featurise(ln, dis, sse, rsa, mu, sd)
            sc = P.score_all(x, model)
            print(f"\n=== {label} ===")
            print(f"{'rank':>4s} {'method':22s} {'p_hat':>7s} {'gen GPU-h/succ':>15s} {'total GPU-h/succ':>17s} {'P(>=1 in 100)':>14s}")
            for i, (m, v) in enumerate(sorted(sc.items(), key=lambda kv: kv[1]["gpuh_per_success"]), 1):
                print(f"{i:>4d} {v['label']:22s} {v['p_hat']:>7.3f} {v['gen_gpuh_per_success']:>15.3f} "
                      f"{v['gpuh_per_success']:>17.2f} {P.p_at_least_one(v['p_hat'], 100):>14.3f}")


if __name__ == "__main__":
    main()
