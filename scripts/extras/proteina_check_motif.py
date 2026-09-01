#!/usr/bin/env python3
"""QC a Proteina run: motif fidelity + chain continuity per generated backbone (numpy/stdlib only).

Proteina is a Cα generator that pins the motif Cα at native geometry and generates scaffold Cα around it,
without enforcing exact Cα-Cα bond lengths. This checks, per design (motif recovered from
{task}_motif_info.csv):
  motif_rmsd : native motif CA (from <run>/motifs/<task>.pdb) vs generated motif CA, Kabsch — is the motif
               a rigid native copy? (<~0.5 A = yes)
  nterm/cterm: Cα-Cα distance at the two motif<->scaffold junctions (~3.8 ideal; >~4.5 loose; >6 = real break)
  max_break  : largest consecutive Cα-Cα anywhere in the chain (>6 A = a real chain break / missing residue)

A rigid motif + all steps < ~4.5 A = motif correctly embedded (loose junctions are a Cα-only artifact that
MPNN+refold rebuild). Systematic >6 A junctions would be a real red flag for a method.

Usage: python proteina_check_motif.py --run-dir <run> [--chain A] [--break-cutoff 4.5]
"""
import os, glob, csv, argparse
import numpy as np


def ca_coords(pdb, chain=None):
    xs = []
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and (chain is None or ln[21] == chain):
            xs.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return np.array(xs)


def parse_motif_ranks(contig):
    ranks = []
    for tok in contig.split("/"):
        tok = tok.strip()
        if tok and tok[0].isalpha():                       # motif segment <chain><start>-<end>
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--break-cutoff", type=float, default=4.5, help="Cα-Cα above this counts as a loose step")
    a = ap.parse_args()
    scaf = os.path.join(a.run_dir, "proteina", "scaffolds")
    motifs = os.path.join(a.run_dir, "motifs")
    infos = sorted(glob.glob(os.path.join(scaf, "**", "*_motif_info.csv"), recursive=True))
    if not infos:
        raise SystemExit(f"no *_motif_info.csv under {scaf}")

    print(f"{'design':26s} {'L':>4s} {'motifRMSD':>9s} {'Nterm':>6s} {'Cterm':>6s} {'maxCC':>6s} {'#loose':>6s}")
    print("-" * 72)
    rmsds, cterms, breaks = [], [], []
    for info in infos:
        d = os.path.dirname(info)
        for row in csv.DictReader(open(info)):
            task, s, contig = row["pdb_name"], row["sample_num"], row["contig"]
            gen = os.path.join(d, f"{task}_{s}.pdb")
            if not os.path.exists(gen):
                continue
            g = ca_coords(gen, a.chain)
            nat = ca_coords(os.path.join(motifs, f"{task}.pdb"))
            idx = [r - 1 for r in parse_motif_ranks(contig)]
            rmsd = kabsch_rmsd(nat, g[idx]) if len(nat) == len(idx) else float("nan")
            dd = np.linalg.norm(np.diff(g, axis=0), axis=1)
            lo, hi = idx[0], idx[-1]
            nterm = dd[lo - 1] if lo > 0 else float("nan")
            cterm = dd[hi] if hi < len(g) - 1 else float("nan")
            nloose = int((dd > a.break_cutoff).sum())
            print(f"{task+'_'+str(s):26s} {len(g):4d} {rmsd:9.3f} {nterm:6.2f} {cterm:6.2f} {dd.max():6.2f} {nloose:6d}")
            rmsds.append(rmsd); cterms.append(cterm); breaks.append(dd.max())
    if rmsds:
        print("-" * 72)
        print(f"motifRMSD  median {np.nanmedian(rmsds):.3f}  max {np.nanmax(rmsds):.3f}   "
              f"Cterm_junction median {np.nanmedian(cterms):.2f}  max {np.nanmax(cterms):.2f}   "
              f"maxCC over all {np.nanmax(breaks):.2f}")
        print(f"(motif rigid if motifRMSD small; ~3.8=bonded, >{a.break_cutoff}=loose Cα-only step, >6=real break)")


if __name__ == "__main__":
    main()
