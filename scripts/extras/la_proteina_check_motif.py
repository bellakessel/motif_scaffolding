#!/usr/bin/env python3
"""QC a La-Proteina run: motif fidelity + chain continuity per generated design (numpy/stdlib only).

La-Proteina is an all-atom co-design model that pins the motif at native geometry and generates the rest of
the structure+sequence around it. Per design (motif recovered from each sample's motif_info.csv contig
`.../C{a}-{b}/...`, read as DESIGN chain-A positions a..b):
  motif_rmsd : native motif CA (from <run>/motifs/<task>.pdb) vs generated motif CA, Kabsch — is the motif a
               rigid native copy? (<~0.5 A = yes)
  nterm/cterm: Cα-Cα distance at the two motif<->scaffold junctions (~3.8 ideal; >~4.5 loose; >6 = real break)
  max_break  : largest consecutive Cα-Cα anywhere in the chain (>6 A = a real chain break / missing residue)

Unlike Proteina (Cα-only), La-Proteina emits real backbone geometry, so junctions should sit near 3.8 A.

Usage: python la_proteina_check_motif.py --run-dir <run> [--chain A] [--break-cutoff 4.5]
"""
import argparse
import csv
import glob
import os
import re

import numpy as np


def ca_coords(pdb, chain=None):
    xs, rid = [], []
    for ln in open(pdb):
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and (chain is None or ln[21] == chain):
            xs.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            rid.append(int(ln[22:26]))
    return np.array(xs), rid


def parse_motif_ranks(contig):
    ranks = []
    for tok in contig.split("/"):
        tok = tok.strip()
        if tok and tok[0].isalpha():
            rng = tok[1:]
            a, b = (int(x) for x in rng.split("-")) if "-" in rng else (int(rng), int(rng))
            ranks.extend(range(a, b + 1))
    return sorted(ranks)


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((Pc - Qc @ R.T) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--break-cutoff", type=float, default=4.5, help="Cα-Cα above this counts as a loose step")
    a = ap.parse_args()
    inf_root = os.path.join(a.run_dir, "laproteina", "inference")
    motifs = os.path.join(a.run_dir, "motifs")
    inf_dirs = sorted(glob.glob(os.path.join(inf_root, "inf_*")))
    if not inf_dirs:
        raise SystemExit(f"no inf_* dirs under {inf_root}")

    print(f"{'design':30s} {'L':>4s} {'motifRMSD':>9s} {'Nterm':>6s} {'Cterm':>6s} {'maxCC':>6s} {'#loose':>6s}")
    n, bad = 0, 0
    for inf_dir in inf_dirs:
        mi = glob.glob(os.path.join(inf_dir, "*_motif_info.csv"))
        if not mi:
            continue
        contig_by_sample = {int(r["sample_num"]): r["contig"].strip()
                            for r in csv.DictReader(open(mi[0]))}
        for job_dir in sorted(glob.glob(os.path.join(inf_dir, "job_*_id_*_motif_*"))):
            name = os.path.basename(job_dir)
            m = re.search(r"_id_(\d+)_motif_(.+)$", name)
            if not m:
                continue
            sample, task = int(m.group(1)), m.group(2)
            pdb = os.path.join(job_dir, name + ".pdb")
            if not os.path.exists(pdb):
                continue
            ranks = parse_motif_ranks(contig_by_sample.get(sample, ""))
            des, des_rid = ca_coords(pdb, a.chain)
            nat, _ = ca_coords(os.path.join(motifs, task + ".pdb"))
            if len(des) == 0 or len(nat) == 0 or not ranks:
                continue
            idx = {r: i for i, r in enumerate(des_rid)}
            mi_idx = [idx[r] for r in ranks if r in idx]
            dm = des[mi_idx]
            mrmsd = kabsch_rmsd(nat, dm) if len(dm) == len(nat) else float("nan")
            steps = np.linalg.norm(np.diff(des, axis=0), axis=1)
            lo, hi = min(mi_idx), max(mi_idx)
            nterm = steps[lo - 1] if lo > 0 else float("nan")
            cterm = steps[hi] if hi < len(steps) else float("nan")
            maxcc = float(steps.max()) if len(steps) else float("nan")
            nloose = int((steps > a.break_cutoff).sum())
            design_name = f"{task}_{sample}"
            print(f"{design_name:30s} {len(des):4d} {mrmsd:9.3f} {nterm:6.2f} {cterm:6.2f} "
                  f"{maxcc:6.2f} {nloose:6d}")
            n += 1
            if (mrmsd == mrmsd and mrmsd > 0.5) or (maxcc == maxcc and maxcc > 6.0):
                bad += 1
    print(f"\n{n} designs checked; {bad} flagged (motifRMSD>0.5 A or a Cα-Cα break >6 A).")


if __name__ == "__main__":
    main()
