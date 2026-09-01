#!/usr/bin/env python3
"""Normalize a La-Proteina run for the shared AF3/ESMFold2 refold + scoring downstream.

La-Proteina CO-DESIGNS structure AND sequence in one all-atom sample, so there is NO ProteinMPNN step:
the generated PDB already carries the designed sequence (real resnames + full sidechains, chain A,
renumbered 1..N). This script takes the place of BOTH proteina_mpnn.sbatch AND proteina_fixed_positions.py:
for every generated design it

  1. reads the co-designed sequence from the all-atom PDB (chain A) and writes it as a ProteinMPNN-style
     fasta <run>/laproteina/seqs/<design_name>.fa — a dropped "template" record + one designed record
     tagged `sample=1`, so esp/af3_inputs.py::build_inputs consumes it UNCHANGED (AF3 job name becomes
     <design_name>_seq001, exactly like the MPNN path);
  2. stages the all-atom design as <run>/laproteina/scaffolds/<design_name>.pdb (basename == design_name)
     so the scorer's scRMSD design-index keys it the same way as Proteina/Genie3;
  3. records the motif's DESIGN positions (1-based chain-A ranks, from each sample's motif_info.csv contig
     `.../C{a}-{b}/...`) into <run>/laproteina/positions.jsonl — the same shape as Proteina's
     fixed_positions.jsonl {design_name: {"A": [ranks]}} — which la_proteina_score.py reads to place the
     motif in the design. (The contig's chain letter is native; the numbers a..b are the design positions —
     verified: native motif CA vs design positions a..b Kabsch-RMSD ~0.01 A.)

design_name = <task>_<sample>   (e.g. 1n8z_pc_m570_base_0), matching esp/score.py's design_name_of/task split.

Runs in gen_env (stdlib only). No container, no GPU.
  python la_proteina_extract_seqs.py --run-dir /cluster/work/wicky/bekessel/motif_library/runs/<run>
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
    "SEC": "C", "PYL": "K",
}


def chain_sequence(pdb: str, chain: str = "A") -> str:
    """One-letter sequence of `chain`, residues in ascending resid order (first model only)."""
    seq = {}
    for ln in open(pdb):
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            seq[int(ln[22:26])] = THREE_TO_ONE.get(ln[17:20].strip(), "X")
    return "".join(seq[r] for r in sorted(seq))


def motif_ranks(contig: str):
    """DESIGN 1-based positions of the motif segments in a per-sample contig (e.g. '40/C41-51/43')."""
    ranks = []
    for tok in contig.split("/"):
        tok = tok.strip()
        if tok and tok[0].isalpha():                       # <chain><start>-<end> motif segment
            rng = tok[1:]
            a, b = (int(x) for x in rng.split("-")) if "-" in rng else (int(rng), int(rng))
            ranks.extend(range(a, b + 1))
    return sorted(ranks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--chain", default="A", help="La-Proteina design chain (default A)")
    a = ap.parse_args()

    inf_root = os.path.join(a.run_dir, "laproteina", "inference")
    seqs_dir = os.path.join(a.run_dir, "laproteina", "seqs")
    scaf_dir = os.path.join(a.run_dir, "laproteina", "scaffolds")
    os.makedirs(seqs_dir, exist_ok=True)
    os.makedirs(scaf_dir, exist_ok=True)

    positions: dict = {}
    n_seq = 0
    inf_dirs = sorted(glob.glob(os.path.join(inf_root, "inf_*")))
    if not inf_dirs:
        raise SystemExit(f"no inf_* dirs under {inf_root} — run generation first")

    for inf_dir in inf_dirs:
        mi = glob.glob(os.path.join(inf_dir, "*_motif_info.csv"))
        if not mi:
            print(f"[warn] no *_motif_info.csv in {inf_dir}, skipping")
            continue
        contig_by_sample = {}
        for row in csv.DictReader(open(mi[0])):
            contig_by_sample[int(row["sample_num"])] = row["contig"].strip()

        for job_dir in sorted(glob.glob(os.path.join(inf_dir, "job_*_id_*_motif_*"))):
            name = os.path.basename(job_dir)               # job_0_id_N_motif_<task>
            m = re.search(r"_id_(\d+)_motif_(.+)$", name)
            if not m:
                continue
            sample, task = int(m.group(1)), m.group(2)
            design_name = f"{task}_{sample}"
            # motif DESIGN positions come from the contig (NO pdb read) -> always recompute so the
            # end-of-run positions.jsonl is COMPLETE even for skipped/already-extracted designs.
            contig = contig_by_sample.get(sample)
            if contig is None:
                print(f"[warn] no contig for sample {sample} in {inf_dir}")
                continue
            fa_out = os.path.join(seqs_dir, design_name + ".fa")
            scaf_out = os.path.join(scaf_dir, design_name + ".pdb")

            # skip-existing (resume): if BOTH outputs already exist (non-empty), skip the pdb read + copy
            # but still record positions. Makes re-launch after a cancel do only the remaining heavy I/O.
            if os.path.exists(fa_out) and os.path.exists(scaf_out) and os.path.getsize(scaf_out) > 0:
                positions[design_name] = {a.chain: motif_ranks(contig)}
                n_seq += 1
                continue

            pdb = os.path.join(job_dir, name + ".pdb")
            if not os.path.exists(pdb):
                print(f"[warn] missing {pdb}")
                continue
            seq = chain_sequence(pdb, a.chain)
            if not seq:
                print(f"[warn] empty chain {a.chain} in {pdb}")
                continue
            # (1) MPNN-style fasta: first (template) record is dropped by build_inputs; second is the design
            with open(fa_out, "w") as fh:
                fh.write(f">template (unused), design={design_name}\n{seq}\n")
                fh.write(f">T=laproteina, sample=1, design={design_name}\n{seq}\n")
            # (2) stage all-atom design as scaffolds/<design_name>.pdb (basename == design_name)
            with open(pdb) as src, open(scaf_out, "w") as dst:
                dst.write(src.read())
            positions[design_name] = {a.chain: motif_ranks(contig)}
            n_seq += 1

    with open(os.path.join(a.run_dir, "laproteina", "positions.jsonl"), "w") as fh:
        json.dump(positions, fh)

    print(f"[extract] {n_seq} designs -> {seqs_dir}/*.fa , {scaf_dir}/*.pdb")
    print(f"[extract] motif positions -> {os.path.join(a.run_dir, 'laproteina', 'positions.jsonl')}")
    print(f"[extract] next: python la_proteina_af3_build.py {seqs_dir} "
          f"{os.path.join(a.run_dir, 'laproteina', 'refold_inputs')} 500")


if __name__ == "__main__":
    main()
