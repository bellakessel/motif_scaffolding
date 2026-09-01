#!/usr/bin/env python3
"""validate_predictor.py — hold-out check: predicted p_hat vs an observed per-motif success rate.

The fit uses six antigens (her2, il1b, pdl1, tnfa, il13, il2). Any other protein is a genuine
external test. The reference case is Mcl-1 / 2mhs: RFdiffusion only, ~1000 trajectories per motif,
refolded with AF3 by this project — so the ground truth uses the IDENTICAL three gates, and the
per-motif rate is high-N rather than a noisy estimate.

Predictions come only from the motif DESCRIPTORS in `<target>_c_motif.csv` — nothing about the
designs is used — so this measures exactly what the predictor claims: can motif features alone
rank motifs before any compute is spent.

    # 1. aggregate the observed refold table to per-motif 3-gate rates (do this cluster-side:
    #    the 2mhs table is ~244 MB and streams at ~0.25 MB/s over sshfs)
    python validate_predictor.py aggregate --scores <af3_refold_2mhs_c_motif.csv> --out observed.csv

    # 2. join to predictions and report
    python validate_predictor.py report --features <mcl1_c_motif.csv> --observed observed.csv \
        --model models/success_model.json --method rfdiff --out pred_vs_obs.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from motifgo import predictor as P                                          # noqa: E402


def motif_num_of(name: str):
    """Design/prediction name -> motif number. Handles `..._motif_<N>_...` and `..._m<N>_...`."""
    for pat in (r"(?:^|_)motif[_-]?(\d+)(?:_|$)", r"(?:^|_)m(\d+)(?:_|$)"):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def cmd_aggregate(a):
    per_design = defaultdict(lambda: [0, 0])
    design_motif, vintage = {}, defaultdict(int)
    n = skipped = 0
    with open(a.scores) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") not in (None, "", "success"):
                continue
            n += 1
            v = P.outcome_gates(r)
            if v:
                vintage[v] += 1
            d = r.get("design_name") or r.get("prediction", "")
            # prefer an explicit motif_num column; fall back to parsing the design name
            mn = r.get("motif_num")                      # mn = motif number = the SASA centroid residue
            mn = int(mn) if mn not in (None, "") else motif_num_of(d)
            if mn is None:
                skipped += 1
                continue
            design_motif[d] = mn
            per_design[d][0] += 1
            per_design[d][1] += int(P.seq_passes(r.get("motifRMSD"), r.get("scRMSD"),
                                                 r.get("ranking_score")))
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for d, (ns, npass) in per_design.items():
        k = design_motif[d]
        agg[k][0] += 1
        agg[k][1] += int(npass > 0)
        agg[k][2] += ns
        agg[k][3] += npass
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["motif_num", "n_designs", "n_designable", "n_seqs", "n_seqs_pass", "obs_rate"])
        for k in sorted(agg):
            d, ok, s, sp = agg[k]
            w.writerow([k, d, ok, s, sp, f"{ok/d:.6f}" if d else ""])
    nd = sum(v[0] for v in agg.values())                 # nd = designs across all motifs
    ok = sum(v[1] for v in agg.values())                 # ok = designs with >=1 passing sequence
    vt = max(vintage, key=vintage.get) if vintage else None   # vt = gate count the source CSV used (2 or 3)
    print(f"rows {n} | motifs {len(agg)} | designs {nd} | designable {ok} ({ok/max(nd,1):.3f})")
    print(f"source csv `designable` column implemented: {vt or 'n/a'} gates (not used — gates re-derived)")
    if skipped:
        print(f"  {skipped} rows had no parseable motif number")
    print(f"wrote {a.out}")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(x), rank(y))


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n             # mx/my = means of the two series
    sx = sum((a-mx)**2 for a in x) ** 0.5                # sx/sy = uncentred spreads (sqrt of sum of squares)
    sy = sum((b-my)**2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy)


def cmd_report(a):
    model = P.load_model(a.model)
    if a.method not in model["methods"]:
        sys.exit(f"method '{a.method}' not in the model ({sorted(model['methods'])})")
    obs = {int(r["motif_num"]): r for r in csv.DictReader(open(a.observed))}
    rows = []
    for r in csv.DictReader(open(a.features)):
        if r.get("variant", "base") != "base":
            continue
        d = P.descriptors(r)
        mn = r.get("motif_num")
        if not d or mn is None:
            continue
        mn = int(mn)
        x = P.featurise(d["motif_length"], d["disjointed_parts"], d["dominant_sse"], d["mean_rsa"],
                        model["mu"], model["sd"])
        p, _ = P.predict_one(x, model["methods"][a.method])
        o = obs.get(mn)
        rows.append({"motif_num": mn, "motif_length": d["motif_length"],
                     "disjointed_parts": int(d["disjointed_parts"]), "dominant_sse": d["dominant_sse"],
                     "mean_rsa": round(d["mean_rsa"], 4), "p_hat": round(p, 6),
                     "obs_rate": float(o["obs_rate"]) if o else "",
                     "n_designs": int(o["n_designs"]) if o else 0})
    joined = [r for r in rows if r["obs_rate"] != ""]
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    ph = [r["p_hat"] for r in joined]                    # ph = predicted success probability
    ob = [r["obs_rate"] for r in joined]                 # ob = observed per-motif success rate
    print(f"\nmethod {a.method} | predicted {len(rows)} motifs | joined to ground truth {len(joined)} "
          f"| designs {sum(r['n_designs'] for r in joined):,}")
    print(f"  Spearman rho  {spearman(ph, ob):.3f}")
    print(f"  Pearson  r    {pearson(ph, ob):.3f}")
    print(f"  mean predicted {sum(ph)/len(ph):.3f}   mean observed {sum(ob)/len(ob):.3f}")
    s = sorted(joined, key=lambda r: r["p_hat"]); n = len(s); t = n // 3
    print(f"\n  {'tertile':>8s} {'n':>4s} {'mean pred':>10s} {'mean obs':>9s} {'designs':>9s}")
    for lab, part in (("low", s[:t]), ("mid", s[t:2*t]), ("high", s[2*t:])):
        print(f"  {lab:>8s} {len(part):>4d} {sum(r['p_hat'] for r in part)/len(part):>10.3f} "
              f"{sum(r['obs_rate'] for r in part)/len(part):>9.3f} {sum(r['n_designs'] for r in part):>9,}")
    by = defaultdict(list)
    for r in joined:
        by[r["disjointed_parts"]].append(r)
    print(f"\n  {'fragments':>9s} {'n':>4s} {'mean pred':>10s} {'mean obs':>9s}   <- the headline driver")
    for k in sorted(by):
        g = by[k]
        print(f"  {k:>9d} {len(g):>4d} {sum(r['p_hat'] for r in g)/len(g):>10.3f} "
              f"{sum(r['obs_rate'] for r in g)/len(g):>9.3f}")
    if a.out:
        print(f"\nwrote {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("aggregate"); g.add_argument("--scores", required=True); g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_aggregate)
    r = sub.add_parser("report")
    r.add_argument("--features", required=True); r.add_argument("--observed", required=True)
    r.add_argument("--model", required=True); r.add_argument("--method", default="rfdiff")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
