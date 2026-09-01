#!/usr/bin/env python3
"""predict_motif.py — score a NEW interface motif against the motif-scaffolding success model.

Loads the coefficients + feature scaling exported by success_predictor.ipynb
(motif_library/analysis/success_model.json) and prints, per generator: predicted 3-gate
designability p̂, ROI (expected GPU-hours per designable design), the per-feature log-odds
contribution breakdown, and the best-ROI recommendation (incl. a fill-the-gaps what-if for
disjointed motifs). No heavy deps — stdlib + math only.

  # from explicit features:
  python predict_motif.py --length 9 --disjoint 3 --sse C --rsa 0.35
  # or look the motif up in the pool by task_name:
  python predict_motif.py --task her2_m128_base --pool <motif_pool_interface_j0.7_pad5.csv>

Model default: <lib>/analysis/success_model.json (override with --model).
"""
import argparse, csv, os, sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
from motifgo import predictor as P                                          # noqa: E402

DEF_MODEL = os.environ.get("MOTIFGO_MODEL", os.path.join(_REPO, "models", "success_model.json"))

def design_vector(length, disjoint, sse, rsa, M):
    """Delegates to motifgo.predictor so the CLI, the fit and the notebook cannot drift apart."""
    return P.featurise(length, disjoint, sse, rsa, M["mu"], M["sd"])

def score(x, M):
    """{method: (p_hat, gpuh_per_success, contributions, label)} — via motifgo.predictor."""
    sc = P.score_all(x, M)
    return {m: (v["p_hat"], v["gpuh_per_success"], v["contrib"], v["label"]) for m, v in sc.items()}

def show(title, sc, M):
    print(f"\n=== {title} ===")
    print(f"{'method':16s} {'p̂':>6s} {'GPU-h/success':>14s}")
    for meth, (p, roi, _, lab) in sorted(sc.items(), key=lambda kv: kv[1][1]):
        print(f"{lab:16s} {p:6.3f} {roi:14.2f}")
    best = min(sc.items(), key=lambda kv: kv[1][1])
    print(f"  -> best ROI: {best[1][3]}  (p̂={best[1][0]:.3f}, {best[1][1]:.2f} GPU-h per designable design)")
    return best

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--task"); ap.add_argument("--pool")
    ap.add_argument("--length", type=float); ap.add_argument("--disjoint", type=float)
    ap.add_argument("--sse", choices=["H", "E", "C"]); ap.add_argument("--rsa", type=float)
    ap.add_argument("--batch", help="features CSV (cols: motif_num,motif_length,disjointed_parts,dominant_sse,mean_rsa) -> score every row")
    ap.add_argument("--out", help="batch mode: write per-motif predictions here (default stdout)")
    ap.add_argument("--contrib-method", default=None, help="print full contribution breakdown for this method")
    a = ap.parse_args()
    M = P.load_model(a.model)

    if a.batch:                                            # ---- batch mode over a features CSV ----
        rdr = list(csv.DictReader(open(a.batch)))
        out_rows = []
        for r in rdr:
            x = design_vector(float(r["motif_length"]), float(r["disjointed_parts"]),
                              r["dominant_sse"], float(r["mean_rsa"]), M)
            sc = score(x, M)
            best = min(sc.items(), key=lambda kv: kv[1][1])
            row = {"motif_num": r.get("motif_num",""), "length": r["motif_length"],
                   "disjoint": r["disjointed_parts"], "sse": r["dominant_sse"], "rsa": r["mean_rsa"],
                   "best_method": best[1][3], "best_p_hat": round(best[1][0],4),
                   "best_gpuh_per_success": round(best[1][1],3)}
            for meth,(p,roi,_,lab) in sc.items(): row[f"p_{meth}"] = round(p,4)
            out_rows.append(row)
        cols = list(out_rows[0].keys()) if out_rows else []
        dst = open(a.out,"w",newline="") if a.out else sys.stdout
        w = csv.DictWriter(dst, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
        if a.out: dst.close(); print(f"wrote {len(out_rows)} predictions -> {a.out}")
        return

    if a.task:
        if not a.pool: ap.error("--task needs --pool <motif_pool csv>")
        row = None
        for r in csv.DictReader(open(a.pool)):
            tn = f"{r['pdb_name']}_m{int(float(r['motif_num']))}_{r['variant']}"
            if tn == a.task: row = r; break
        if row is None: ap.error(f"task {a.task} not found in {a.pool}")
        length, disjoint = float(row["motif_length"]), float(row["disjointed_parts"])
        sse, rsa = row["dominant_sse"], float(row["mean_rsa"])
        print(f"motif {a.task}: length={length:.0f} disjoint_parts={disjoint:.0f} sse={sse} mean_rsa={rsa:.3f}")
    else:
        if None in (a.length, a.disjoint, a.sse, a.rsa):
            ap.error("give --task+--pool, or all of --length --disjoint --sse --rsa")
        length, disjoint, sse, rsa = a.length, a.disjoint, a.sse, a.rsa

    x = design_vector(length, disjoint, sse, rsa, M)
    sc = score(x, M)
    best = show(f"motif as given (disjoint_parts={disjoint:.0f})", sc, M)

    if disjoint > 1:   # fill-the-gaps what-if: collapse to 1 fragment
        x2 = design_vector(length, 1, sse, rsa, M)
        sc2 = score(x2, M)
        best2 = show("what-if: gaps filled (disjoint_parts=1)", sc2, M)
        dp = best2[1][0] - best[1][0]
        print(f"\nfill-gaps effect on best-ROI p̂: {best[1][0]:.3f} -> {best2[1][0]:.3f}  (Δ {dp:+.3f})")

    cm = a.contrib_method
    if cm and cm in sc:
        print(f"\n=== per-feature log-odds contribution · {sc[cm][3]} ===")
        c = M["methods"][cm]["coef"]
        print(f"  {'intercept':16s} {c.get('const',0.0):+.3f}")
        for f, v in sorted(sc[cm][2].items(), key=lambda kv: -abs(kv[1])):
            print(f"  {f:16s} {v:+.3f}")

if __name__ == "__main__":
    main()
