#!/usr/bin/env python3
"""aggregate_outcomes.py — the training table: raw refold scores -> per-(motif, method) successes.

    <run>/<method>/af3_refold_scores.csv      one row per REFOLDED SEQUENCE
                    |
                    |   three gates, re-derived from the raw columns (motifgo.predictor.seq_passes)
                    v
    outcomes.csv    one row per (task, method):  n_designs, n_designable, n_seqs, n_seqs_pass

Two things this does NOT do, deliberately:

1. **It never reads the `designable` column.** The baseline run's CSVs (2026-08-18) predate the
   ranking-score gate and carry a TWO-gate column; 5_score.py has emitted three gates since
   2026-08-26. Re-deriving from motifRMSD / scRMSD / ranking_score is correct for both vintages.
   The vintage each file was written with is detected and recorded in the header, so the difference
   is visible in the artefact instead of being rediscovered later.

2. **It does not parse the task out of `design_name` by token count.** Design names differ per
   method -- rfdiff `<task>_design_<N>`, proteina/genie3/laproteina `<task>_<N>`, and pp1c_bb
   `<task>_sample_<task>_<N>`, which repeats the task. The old awk aggregator took "the first three
   underscore tokens", which is right for today's names by luck and wrong for any protein tag that
   is not a single token. This prefers the `task_name` column that 5_score.py now emits, and falls
   back to longest-prefix matching against the pool for CSVs written before that column existed.

    python aggregate_outcomes.py --runs-root <runs> --pool <pool.csv> --out outcomes.csv \
        --arm rfdiff:lib_interface_pad5:rfdiff  --arm laproteina_unindexed:lib_interface_pad5_uidx:laproteina
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from motifgo import predictor as P                                          # noqa: E402

DEFAULT_ARMS = [
    "rfdiff:lib_interface_pad5:rfdiff",
    "proteina:lib_interface_pad5:proteina",
    "genie3:lib_interface_pad5:genie3",
    "pp1c_bb:lib_interface_pad5:pp1c_bb",
    "laproteina:lib_interface_pad5:laproteina",
    "laproteina_unindexed:lib_interface_pad5_uidx:laproteina",
]


def task_resolver(pool_csv: str):
    """design_name -> task. Uses `task_name` when present; else longest-prefix match on the pool."""
    tasks = set()
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            tasks.add(f"{r['pdb_name']}_m{r['motif_num']}_{r.get('variant', 'base')}")
    by_head = defaultdict(list)
    for t in sorted(tasks, key=len, reverse=True):
        by_head[t.split("_")[0]].append(t)
    cache = {}

    def resolve(design_name: str):
        if design_name in cache:
            return cache[design_name]
        hit = None
        for t in by_head.get(design_name.split("_")[0], ()):
            if design_name == t or design_name.startswith(t + "_"):
                hit = t
                break
        cache[design_name] = hit
        return hit

    return resolve, tasks


def aggregate_arm(path: str, resolve) -> tuple:
    """-> ({task: [n_designs, n_designable, n_seqs, n_seqs_pass]}, vintage, n_rows, n_unmapped)"""
    per_design = defaultdict(lambda: [0, 0])            # design -> [n_seqs refolded, n_seqs passing 3 gates]
    design_task = {}
    vintage_votes = defaultdict(int)
    n_rows = n_unmapped = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "success":
                continue
            n_rows += 1
            v = P.outcome_gates(r)
            if v:
                vintage_votes[v] += 1
            task = (r.get("task_name") or "").strip() or resolve(r.get("design_name", ""))
            if not task:
                n_unmapped += 1
                continue
            d = r.get("design_name", "")                # d = the backbone this sequence was designed onto
            design_task[d] = task
            per_design[d][0] += 1
            per_design[d][1] += int(P.seq_passes(r.get("motifRMSD"), r.get("scRMSD"),
                                                 r.get("ranking_score")))
    out = defaultdict(lambda: [0, 0, 0, 0])
    for d, (ns, npass) in per_design.items():
        t = design_task[d]                               # t = the motif task this design belongs to
        out[t][0] += 1                                   # n_designs
        out[t][1] += int(npass > 0)                      # n_designable  (ANY sequence passes)
        out[t][2] += ns                                  # n_seqs
        out[t][3] += npass                               # n_seqs_pass
    vintage = max(vintage_votes, key=vintage_votes.get) if vintage_votes else None
    return out, vintage, n_rows, n_unmapped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME:RUN:METHOD_DIR",
                    help="repeatable; defaults to the six baseline arms")
    ap.add_argument("--predictor", default="af3")
    a = ap.parse_args(argv)

    resolve, pool_tasks = task_resolver(a.pool)
    arms = a.arm or DEFAULT_ARMS
    rows, notes = [], []
    print(f"pool: {len(pool_tasks)} tasks\n")
    print(f"{'arm':22s} {'tasks':>6s} {'designs':>9s} {'designable':>11s} {'rate':>7s} {'CSV col':>9s}")
    for spec in arms:
        name, run, mdir = spec.split(":", 2)
        path = os.path.join(a.runs_root, run, mdir, f"{a.predictor}_refold_scores.csv")
        if not os.path.exists(path):
            print(f"{name:22s} {'—':>6s}  MISSING {path}")
            notes.append(f"{name}: MISSING {path}")
            continue
        agg, vintage, n_rows, n_unmapped = aggregate_arm(path, resolve)
        nd = sum(v[0] for v in agg.values())             # nd = designs generated for this arm
        ok = sum(v[1] for v in agg.values())             # ok = designs with >=1 passing sequence
        print(f"{name:22s} {len(agg):>6d} {nd:>9d} {ok:>11d} {ok / max(nd, 1):>7.3f} "
              f"{(str(vintage) + '-gate') if vintage else 'n/a':>9s}")
        if n_unmapped:
            notes.append(f"{name}: {n_unmapped} rows could not be mapped to a pool task")
        for t, v in sorted(agg.items()):
            rows.append({"task": t, "method": name, "run": run, "method_dir": mdir,
                         "n_designs": v[0], "n_designable": v[1],
                         "n_seqs": v[2], "n_seqs_pass": v[3],
                         "csv_designable_gates": vintage or ""})
    fields = ["task", "method", "run", "method_dir", "n_designs", "n_designable",
              "n_seqs", "n_seqs_pass", "csv_designable_gates"]
    with open(a.out, "w", newline="") as fh:
        fh.write(f"# 3-gate outcome re-derived from raw columns "
                 f"(motifRMSD<={P.MOTIF_RMSD_MAX}, scRMSD<={P.SC_RMSD_MAX}, rank>={P.RANK_MIN})\n")
        fh.write("# csv_designable_gates = what the SOURCE csv's own `designable` column implemented "
                 "(2 = pre-2026-08-26, 3 = current). Never used for fitting.\n")
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} (task, method) rows)")
    for n in notes:
        print(f"  NOTE: {n}")


if __name__ == "__main__":
    main()
