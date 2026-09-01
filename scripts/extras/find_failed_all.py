#!/usr/bin/env python3
"""find_failed_all.py — materialize the "failed by every method" motif subset from a benchmark run.

Reconstructs the notebook's `failed_all` set (the mount is too slow to crunch the ~250k-row score CSVs
interactively). For each method it reads <run>/<method>/af3_refold_scores.csv, marks each sequence
designable = (motifRMSD <= MRMAX) & (scRMSD <= SCMAX) & (ranking_score >= RANKMIN) — the SAME gate the
notebook uses — rolls that up to a per-task designable flag (any design of the task passes), and reports
the tasks that are designable in NONE of the methods.

design_name -> task_name is resolved by longest-prefix match against the run manifest's task_names
(method-agnostic: works whether the design name is <task>_<n>_seqNNN, <task>_design_N_seqNNN, etc.).

Writes:
  <run>/analysis/failed_all.txt          one task_name per line (the subset to re-scaffold)
  <run>/analysis/failed_all_features.csv task_name + pool features (motif_length, disjointed_parts, ...)

Runs in gen_env (stdlib + csv only; no numpy, no esp).
  python find_failed_all.py --run-dir /cluster/work/wicky/bekessel/motif_library/runs/lib_interface_pad5
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

DEFAULT_METHODS = ["rfdiff", "proteina", "genie3", "pp1c_bb", "laproteina", "laproteina_mpnn"]


def load_manifest_tasks(run_dir):
    """task_names of the run (the 314 motifs), longest-first for greedy prefix matching."""
    tasks = []
    with open(os.path.join(run_dir, "manifest.csv")) as fh:
        for r in csv.DictReader(fh):
            tasks.append(r["task_name"])
    return sorted(set(tasks), key=len, reverse=True)


def task_of(design_name, tasks_longest_first):
    """The task whose name prefixes this design (greedy longest). None if nothing matches."""
    for t in tasks_longest_first:
        if design_name.startswith(t):
            return t
    return None


def designable_tasks(csv_path, tasks, mrmax, scmax, rankmin):
    """Set of task_names with >=1 designable sequence in this method's score CSV."""
    ok = set()
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                if not (float(r["motifRMSD"]) <= mrmax and float(r["scRMSD"]) <= scmax
                        and float(r["ranking_score"]) >= rankmin):
                    continue
            except (ValueError, KeyError):
                continue
            t = task_of(r["design_name"], tasks)
            if t:
                ok.add(t)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="benchmark run dir with manifest.csv + <method>/af3_refold_scores.csv")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--pool", default=None, help="pool CSV for the features join (default <run>/../../motif_pool_interface_j0.7_pad5.csv)")
    ap.add_argument("--motif-rmsd-max", type=float, default=1.0)
    ap.add_argument("--sc-rmsd-max", type=float, default=2.0)
    ap.add_argument("--rank-min", type=float, default=0.75)
    ap.add_argument("--out", default=None, help="output txt (default <run>/analysis/failed_all.txt)")
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    tasks = load_manifest_tasks(run_dir)
    all_tasks = set(tasks)

    designable_by_method = {}
    for m in a.methods:
        p = os.path.join(run_dir, m, "af3_refold_scores.csv")
        if not os.path.isfile(p):
            print(f"[failed_all] WARNING: {m} has no score CSV at {p} — SKIPPING (not counted)", file=sys.stderr)
            continue
        ok = designable_tasks(p, tasks, a.motif_rmsd_max, a.sc_rmsd_max, a.rank_min)
        designable_by_method[m] = ok
        print(f"[failed_all] {m}: {len(ok)}/{len(all_tasks)} tasks designable", file=sys.stderr)

    if not designable_by_method:
        raise SystemExit("[failed_all] no method score CSVs found — nothing to do")

    # failed_all = designable in NONE of the (present) methods
    ever = set().union(*designable_by_method.values())
    failed_all = sorted(all_tasks - ever)

    outdir = os.path.join(run_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    out_txt = a.out or os.path.join(outdir, "failed_all.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(failed_all) + ("\n" if failed_all else ""))
    print(f"[failed_all] {len(failed_all)}/{len(all_tasks)} tasks failed ALL {len(designable_by_method)} "
          f"methods ({','.join(designable_by_method)}) -> {out_txt}")

    # feature join (optional, best-effort)
    pool = a.pool or os.path.join(os.path.dirname(os.path.dirname(run_dir)),
                                  "motif_pool_interface_j0.7_pad5.csv")
    if os.path.isfile(pool):
        fset = set(failed_all)
        rows, hdr = [], None
        with open(pool) as fh:
            rd = csv.DictReader(fh)
            hdr = rd.fieldnames
            for r in rd:
                t = f"{r['pdb_name']}_m{r['motif_num']}_{r['variant']}"
                if t in fset:
                    r = dict(r); r["task_name"] = t; rows.append(r)
        out_csv = os.path.join(outdir, "failed_all_features.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["task_name"] + hdr)
            w.writeheader(); w.writerows(rows)
        # quick composition summary to stderr
        from collections import Counter
        dj = Counter(int(r["disjointed_parts"]) for r in rows)
        print(f"[failed_all] features -> {out_csv}", file=sys.stderr)
        print(f"[failed_all] disjointedness of failures: "
              f"{dict(sorted(dj.items()))}", file=sys.stderr)
    else:
        print(f"[failed_all] pool not found at {pool} — skipped features join", file=sys.stderr)


if __name__ == "__main__":
    main()
