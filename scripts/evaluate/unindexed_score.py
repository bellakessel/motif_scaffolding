#!/usr/bin/env python3
"""unindexed_score.py — relocalize the motif in UNINDEXED La-Proteina co-designs, then correct positions.jsonl.

WHY THIS EXISTS
---------------
La-Proteina has two motif-scaffolding modes that differ ONLY by checkpoint (LD4 vs LD6):
  - INDEXED (LD4): the model pins the motif at the sequence indices implied by the contig. The nominal
    positions in `motif_info.csv` (written by generate_motif_indices at dataset-build time, BEFORE sampling)
    ARE the true placement -> fixed-index superposition scores it correctly (native-vs-design ~0.01 A).
  - UNINDEXED (LD6): the model embeds the motif STRUCTURE anywhere, IGNORING the contig's sequence indices
    and segment order. So `motif_info.csv` (and the positions.jsonl our extract step copies from it) is the
    NOMINAL input-order placement, NOT where the motif ended up. Scoring the refold at those nominal indices
    (5_score.py's fixed superposition) measures the WRONG residues -> inflated motifRMSD -> uidx successes
    are UNDER-COUNTED. (Verified: her2_m128 sample 0 -> Kabsch RMSD @nominal = 6.97 A, but the motif's own
    sequence `ILKGG/RG/...` appears in the co-design at entirely different positions.)

La-Proteina's OWN evaluation localizes unindexed motifs with
`proteinfoundation/utils/motif_utils.py::pad_motif_to_full_length_unindexed`: greedily match each motif
residue to the nearest generated residue of matching amino-acid type. It relies on the generated tensor being
in the motif's frame; our SAVED scaffold PDBs are in a different frame (~290 A off, arbitrary rotation), so we
first ALIGN via the motif's recovered SEQUENCE, then run the same greedy nearest+aatype assignment.

WHAT IT DOES
------------
For each co-designed scaffold `<run>/<method>/scaffolds/<design>.pdb` (chain A) and its native motif
`<run>/motifs/<task>.pdb`:
  1. split the native motif into contiguous fragments (each carries a sub-sequence + CA coords).
  2. SEED: for each fragment whose sub-sequence occurs in the co-designed sequence, Kabsch-fit the native
     fragment CA onto that scaffold window -> a rigid transform; apply it to the whole native motif.
  3. GREEDY: in that aligned frame, assign each native motif residue to the nearest UNUSED scaffold residue,
     preferring matching aa-type (fallback: ignore aa-type), exactly as pad_motif_to_full_length_unindexed.
  4. score the seed by Kabsch RMSD over the full assignment; keep the best seed across all fragments.
  -> the assigned scaffold residue numbers are the TRUE 1-based motif design positions.

Writes:
  <run>/<method>/positions_unindexed.jsonl   corrected {design: {"A": [1-based ranks]}} — feed to 5_score.py
                                             via its --positions flag (produces *_greedy scores for comparison)
  <run>/<method>/unindexed_relocalization.csv per-design diagnostics: nominal vs relocalized RMSD, seq recovery,
                                             position shift — so you can see how much of the idx-vs-uidx gap
                                             was this scoring artifact vs real geometry.

INDEXED runs are a CONTROL: relocalization should reproduce the nominal positions (shift~0, RMSD unchanged),
confirming the method is a no-op where the indices are already right.

Runs in gen_env (numpy). No container, no GPU.
  python unindexed_score.py --run-dir <run> --method laproteina
  # then re-score the refolds with the corrected positions:
  python 5_score.py --run-dir <run> --method laproteina --positions positions_unindexed.jsonl \
      --out <run>/laproteina/af3_refold_scores_greedy.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
    "SEC": "C", "PYL": "K",
}


def read_ca(pdb: str, chain: str | None = None):
    """Ordered CA records (resid, aa1, xyz) for the first model (optionally one chain)."""
    rows = []
    with open(pdb) as fh:
        for ln in fh:
            if ln.startswith("ENDMDL"):
                break
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                if chain is not None and ln[21] != chain:
                    continue
                rows.append((int(ln[22:26]), THREE_TO_ONE.get(ln[17:20].strip(), "X"),
                             np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
    return rows


def fragments(motif_ca):
    """Contiguous runs of native residues -> list of dicts {seq, xyz(N,3), idx0(list of motif-array indices)}."""
    frags, cur = [], [0]
    for i in range(1, len(motif_ca)):
        if motif_ca[i][0] == motif_ca[i - 1][0] + 1:
            cur.append(i)
        else:
            frags.append(cur); cur = [i]
    frags.append(cur)
    out = []
    for f in frags:
        out.append({"seq": "".join(motif_ca[i][1] for i in f),
                    "xyz": np.array([motif_ca[i][2] for i in f]),
                    "idx0": f})
    return out


def kabsch(P, Q):
    """Rigid transform mapping P onto Q (both (N,3)); returns (R, t) with P@R + t ~ Q."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    V, _, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    t = Qc - Pc @ R
    return R, t


def rmsd_after_fit(P, Q):
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R + t - Q) ** 2).sum(1).mean()))


def all_occurrences(hay: str, needle: str):
    starts, i = [], hay.find(needle)
    while i != -1:
        starts.append(i)
        i = hay.find(needle, i + 1)
    return starts


def greedy_assign(nat_xyz_aligned, nat_aa, scaf_xyz, scaf_aa, match_aatype=True):
    """Assign each native motif residue to the nearest UNUSED scaffold residue (aa-type match optional).
    Returns list of scaffold 0-based indices, or None if an aa-type-constrained pass finds no candidate."""
    n = len(nat_aa)
    used = set()
    out = []
    for i in range(n):
        best_j, best_d = None, float("inf")
        for j in range(len(scaf_aa)):
            if j in used:
                continue
            if match_aatype and scaf_aa[j] != nat_aa[i]:
                continue
            d = float(np.sum((nat_xyz_aligned[i] - scaf_xyz[j]) ** 2))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            return None
        out.append(best_j); used.add(best_j)
    return out


def relocalize(motif_ca, scaf_ca):
    """Best (assignment, rmsd, seq_recovery) of the native motif onto the co-designed scaffold.
    Seeds a rigid alignment from each fragment's recovered sequence, then greedy nearest+aatype assignment."""
    nat_aa = [r[1] for r in motif_ca]
    nat_xyz = np.array([r[2] for r in motif_ca])
    scaf_aa = [r[1] for r in scaf_ca]
    scaf_xyz = np.array([r[2] for r in scaf_ca])
    scaf_seq = "".join(scaf_aa)

    best = None  # (rmsd, assignment)
    # seed from every fragment whose sequence is recovered somewhere (longest first = most specific anchor)
    for frag in sorted(fragments(motif_ca), key=lambda f: -len(f["seq"])):
        for start in all_occurrences(scaf_seq, frag["seq"]):
            win = list(range(start, start + len(frag["seq"])))
            R, t = kabsch(frag["xyz"], scaf_xyz[win])         # align native fragment -> this scaffold window
            nat_aligned = nat_xyz @ R + t                      # bring the whole native motif into scaffold frame
            for ma in (True, False):                           # prefer aa-type-consistent assignment, then relax
                asg = greedy_assign(nat_aligned, nat_aa, scaf_xyz, scaf_aa, match_aatype=ma)
                if asg is not None:
                    r = rmsd_after_fit(nat_xyz, scaf_xyz[asg])
                    if best is None or r < best[0]:
                        best = (r, asg)
                    break
    if best is None:                                           # no fragment sequence recovered -> unresolved
        return None
    rmsd, asg = best
    seq_rec = float(np.mean([scaf_aa[j] == nat_aa[i] for i, j in enumerate(asg)]))
    # CRITICAL: emit positions in NATIVE-RESIDUE order (motif residues sorted by resid == the order
    # 5_score's native_motif_residues() pairs against), NOT value-sorted — because unindexed can place the
    # fragments OUT OF ORDER (e.g. a C-terminal fragment at a low index). 5_score --positions must pair
    # con_hal_idx0[k] with the k-th native residue, so the k-th emitted position must be that residue's match.
    order = sorted(range(len(motif_ca)), key=lambda i: motif_ca[i][0])
    positions = [int(scaf_ca[asg[i]][0]) for i in order]
    return {"positions": positions, "rmsd": rmsd, "seq_recovery": seq_rec}


# --- parallel relocalization: one worker per scaffold (CPU-bound Kabsch/greedy; ~31k designs too slow
# serial). Native-motif cache is per-worker (small); nominal positions shared read-only via initializer. ---
_UW: dict = {}


def _uinit(motifs_dir, chain, nominal):
    _UW.update(motifs_dir=motifs_dir, chain=chain, nominal=nominal, mcache={})


def _uwork(sp):
    motifs_dir, chain, nominal, mcache = _UW["motifs_dir"], _UW["chain"], _UW["nominal"], _UW["mcache"]
    design = os.path.basename(sp)[:-4]
    task = design.rsplit("_", 1)[0]
    if task not in mcache:
        mp_pdb = os.path.join(motifs_dir, task + ".pdb")
        mcache[task] = read_ca(mp_pdb) if os.path.isfile(mp_pdb) else None
    motif_ca = mcache[task]
    if not motif_ca:
        return (design, None, {"design": design, "task": task, "status": "no_motif_pdb"})
    scaf_ca = read_ca(sp, chain)
    if not scaf_ca:
        return (design, None, {"design": design, "task": task, "status": "empty"})

    res = relocalize(motif_ca, scaf_ca)
    nom = sorted(nominal.get(design, {}).get(chain, []))
    nom_rmsd = ""
    if nom and len(nom) == len(motif_ca):
        try:
            nom_xyz = np.array([scaf_ca[p - 1][2] for p in nom])
            nom_rmsd = round(rmsd_after_fit(np.array([r[2] for r in motif_ca]), nom_xyz), 3)
        except IndexError:
            nom_rmsd = ""
    if res is None:
        return (design, {chain: nom}, {"design": design, "task": task, "status": "unresolved_seq",
                                       "n_motif": len(motif_ca), "nominal_rmsd": nom_rmsd})
    shift = "" if not nom else int(np.mean([abs(a_ - b_) for a_, b_ in
                                            zip(sorted(nom), sorted(res["positions"]))]))
    row = {"design": design, "task": task, "status": "ok", "n_motif": len(motif_ca),
           "nominal_rmsd": nom_rmsd, "greedy_rmsd": round(res["rmsd"], 3),
           "seq_recovery": round(res["seq_recovery"], 3), "mean_pos_shift": shift,
           "nominal_positions": " ".join(map(str, nom)),
           "greedy_positions": " ".join(map(str, res["positions"]))}
    return (design, {chain: res["positions"]}, row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", default="laproteina", help="co-design method dir (default laproteina)")
    ap.add_argument("--chain", default="A", help="design chain (default A)")
    ap.add_argument("--workers", type=int, default=None, help="parallel scaffold workers (default: all CPUs)")
    ap.add_argument("--out-positions", default=None,
                    help="corrected positions.jsonl (default <run>/<method>/positions_unindexed.jsonl)")
    ap.add_argument("--out-csv", default=None,
                    help="diagnostics CSV (default <run>/<method>/unindexed_relocalization.csv)")
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    mdir = os.path.join(run_dir, a.method)
    scaf_dir = os.path.join(mdir, "scaffolds")
    motifs_dir = os.path.join(run_dir, "motifs")
    nominal = json.load(open(os.path.join(mdir, "positions.jsonl")))   # {design: {chain: [ranks]}}
    out_pos = a.out_positions or os.path.join(mdir, "positions_unindexed.jsonl")
    out_csv = a.out_csv or os.path.join(mdir, "unindexed_relocalization.csv")

    scaffolds = sorted(glob.glob(os.path.join(scaf_dir, "*.pdb")))
    if not scaffolds:
        raise SystemExit(f"no scaffolds under {scaf_dir}")

    corrected: dict = {}
    rows = []
    n_ok = n_unresolved = 0
    nproc = max(1, a.workers or os.cpu_count() or 1)
    print(f"[unindexed_score] {len(scaffolds)} scaffolds, {nproc} workers ...", flush=True)
    with mp.Pool(processes=nproc, initializer=_uinit, initargs=(motifs_dir, a.chain, nominal)) as pool:
        done = 0
        for design, pos_val, row in pool.imap_unordered(_uwork, scaffolds, chunksize=64):
            rows.append(row)
            if pos_val is not None:
                corrected[design] = pos_val
            if row.get("status") == "ok":
                n_ok += 1
            elif row.get("status") == "unresolved_seq":
                n_unresolved += 1
            done += 1
            if done % 5000 == 0 or done == len(scaffolds):
                print(f"  {done}/{len(scaffolds)} relocalized", flush=True)

    json.dump(corrected, open(out_pos, "w"))
    fields = ["design", "task", "status", "n_motif", "nominal_rmsd", "greedy_rmsd", "seq_recovery",
              "mean_pos_shift", "nominal_positions", "greedy_positions"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok"]
    import statistics as st
    def med(key):
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        return round(st.median(vals), 3) if vals else float("nan")
    print(f"[unindexed_score] {len(scaffolds)} designs: {n_ok} relocalized, {n_unresolved} seq-unresolved")
    print(f"  median RMSD  nominal={med('nominal_rmsd')} A -> greedy={med('greedy_rmsd')} A  (lower=better)")
    print(f"  median seq_recovery={med('seq_recovery')} ; median |pos shift|={med('mean_pos_shift')}")
    print(f"  corrected positions -> {out_pos}")
    print(f"  diagnostics         -> {out_csv}")
    print(f"\n  re-score refolds with the corrected positions (compare vs the nominal-position scores):")
    print(f"    python {os.path.join(os.path.dirname(os.path.abspath(__file__)), '5_score.py')} "
          f"--run-dir {run_dir} --method {a.method} \\")
    print(f"        --positions {os.path.basename(out_pos)} --out {mdir}/af3_refold_scores_greedy.csv")


if __name__ == "__main__":
    main()
