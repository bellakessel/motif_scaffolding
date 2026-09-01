#!/usr/bin/env python3
"""compute_costs.py — build/refresh <run>/compute_costs.csv from sacct (LOGIN NODE ONLY; needs sacct).

Schema (matches the hand-made baseline table the notebook's "Compute cost vs quality" cell reads):
    method,stage,n_tasks,gpu_hours,maxrss_g
gpu_hours = sum over array tasks of ElapsedRaw x (gres/gpu count) / 3600 ; maxrss_g = max step MaxRSS (GB).

Pass one --job per method/stage you want to (re)compute; existing rows for OTHER method/stage pairs in the
CSV are preserved (so you can add methods incrementally, or refresh one).

  # find the array jobids first (name+time disambiguates baseline vs contigscale vs uidx):
  sacct --starttime=2026-08-22 --endtime=2026-08-26 -X \
        --format=JobID,JobName%16,Submit,State,Elapsed,AllocTRES%40 | grep -E 'rfdiff|proteina|genie3|laprotei|protpard'

  python compute_costs.py --out /cluster/.../runs/lib_interface_pad5_contigscale/compute_costs.csv \
      --job generation:rfdiff:11630940 --job generation:proteina:11720001 \
      --job generation:genie3:11631115 --job generation:pp1c_bb:11631259 \
      --job generation:laproteina:11631256

A method's generation may span SEVERAL jobids (resubmits after a widen/preempt) — pass them comma-joined:
      --job generation:proteina:11631218,11720001    (summed)
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess


def parse_rss(s: str) -> float:
    """MaxRSS string (e.g. '338576K', '2.1G') -> GB."""
    s = s.strip()
    if not s:
        return 0.0
    u = s[-1]
    if u.isdigit():
        try:
            return float(s) / 1e9
        except ValueError:
            return 0.0
    try:
        x = float(s[:-1])
    except ValueError:
        return 0.0
    return {"K": x / 1e6, "M": x / 1e3, "G": x, "T": x * 1e3}.get(u.upper(), x / 1e9)


def sacct_cost(jobids: list[str]):
    """(n_completed_tasks, gpu_hours, maxrss_g) summed across one or more (array) jobids."""
    gpu_sec, ntasks, maxrss = 0.0, 0, 0.0
    for jid in jobids:
        out = subprocess.run(
            ["sacct", "-j", jid, "--format=JobID,ElapsedRaw,AllocTRES,MaxRSS,State", "-P", "-n"],
            capture_output=True, text=True).stdout
        for ln in out.splitlines():
            f = ln.split("|")
            if len(f) < 5:
                continue
            j, elapsed, tres, mrss, state = f[0], f[1], f[2], f[3], f[4]
            if "." not in j:                                  # main allocation (one per array task)
                ng = 1
                m = re.search(r"gres/gpu=(\d+)", tres or "")
                if m:
                    ng = int(m.group(1))
                try:
                    gpu_sec += int(elapsed) * ng
                except ValueError:
                    pass
                if state.startswith("COMPLETED"):
                    ntasks += 1
            else:                                             # step (.batch/.extern) carries MaxRSS
                maxrss = max(maxrss, parse_rss(mrss))
    return ntasks, round(gpu_sec / 3600, 2), round(maxrss, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="<run>/compute_costs.csv to write/refresh")
    ap.add_argument("--job", action="append", default=[], metavar="STAGE:METHOD:JOBID[,JOBID...]",
                    help="e.g. generation:rfdiff:11630940 (repeatable); multiple jobids comma-joined are summed")
    a = ap.parse_args()

    fields = ["method", "stage", "n_tasks", "gpu_hours", "maxrss_g"]
    rows = {}
    if os.path.exists(a.out):                                  # preserve existing rows for other method/stage
        with open(a.out) as fh:
            for r in csv.DictReader(fh):
                rows[(r["method"], r["stage"])] = r

    for spec in a.job:
        try:
            stage, method, jids = spec.split(":", 2)
        except ValueError:
            raise SystemExit(f"bad --job {spec!r}; want STAGE:METHOD:JOBID[,JOBID]")
        n, gh, mr = sacct_cost(jids.split(","))
        rows[(method, stage)] = {"method": method, "stage": stage, "n_tasks": n,
                                 "gpu_hours": gh, "maxrss_g": mr}
        print(f"  {method:12s} {stage:11s} n_tasks={n:4d}  gpu_hours={gh:8.2f}  maxrss_g={mr}")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows, key=lambda x: (x[1], x[0])):    # stable: by stage then method
            w.writerow({f: rows[k].get(f, "") for f in fields})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
