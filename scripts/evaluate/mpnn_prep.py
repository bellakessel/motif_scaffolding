#!/usr/bin/env python3
"""mpnn_prep.py — stamp the native motif identity into each passing design + emit ProteinMPNN fixed
positions. Method-agnostic: reads 1_filter's outputs (passed/<canonical>.pdb, positions.jsonl, tasks.json)
for any backbone method. Backbone generators (RFdiff/Proteina/Genie3/Protpardelle-bb) emit poly-GLY/ALA
backbones; to keep the motif at its NATIVE sequence through MPNN we (a) overwrite the motif residues'
resnames with the native identities from motifs/<task>.pdb, (b) pass those positions to ProteinMPNN as
FIXED. Writes <run>/<method>/mpnn/prepped/<canonical>.pdb + <run>/<method>/mpnn/fixed_positions.jsonl.

    python mpnn_prep.py --run-dir <run> --method <rfdiff|proteina|genie3|pp1c_bb> [--workers N]

At library scale (~31k passers) the stamping is pure per-design file I/O -> parallelize over --workers
(default = all CPUs). Skip-existing (a prepped/<canon>.pdb already present) makes a re-run resume.
"""
import argparse
import json
import os
from multiprocessing import Pool


def native_resnames(motif_pdb):
    """Native motif resnames, one per residue, ordered by (chain, res_id) — matches the ascending design
    ranks in positions.jsonl (same order 1_filter paired them for the motifRMSD)."""
    rn, order = {}, []
    with open(motif_pdb) as fh:
        for ln in fh:
            if ln.startswith("ATOM"):
                key = (ln[21], int(ln[22:26]))
                if key not in rn:
                    rn[key] = ln[17:20].strip()
                    order.append(key)
    return [rn[k] for k in sorted(order)]


def _stamp_job(job):
    """Parallel worker: (src, out, rank_to_rn) -> stamp one design (own file handles, picklable args)."""
    src, out, rank_to_rn = job
    _stamp(src, rank_to_rn, out)
    return 1


def stamp(design_pdb, ranks, resnames, out_pdb):
    """Set resname for design residues whose res_id is in ranks (ascending) to the native resnames."""
    rank_to_rn = {r: rname for r, rname in zip(sorted(ranks), resnames)}
    _stamp(design_pdb, rank_to_rn, out_pdb)


def _stamp(design_pdb, rank_to_rn, out_pdb):
    """rank_to_rn maps 1-based sequential RANK (position in the ascending (chain,resid) order) -> native
    resname. Stamp by RANK, not raw resid, so it stays correct for 0-based / gapped PDB numbering
    (e.g. Proteina numbers residues 0..N-1) — matching how 1_filter/ProteinMPNN/scorer index positions."""
    seen, order = set(), []
    with open(design_pdb) as fh:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                key = (ln[21], ln[22:26])
                if key not in seen:
                    seen.add(key); order.append(key)
    rank_of = {key: i + 1 for i, key in enumerate(sorted(order, key=lambda k: (k[0], int(k[1]))))}
    with open(design_pdb) as fh, open(out_pdb, "w") as w:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                rk = rank_of.get((ln[21], ln[22:26]))
                if rk in rank_to_rn:
                    ln = ln[:17] + f"{rank_to_rn[rk]:>3s}" + ln[20:]
            w.write(ln)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel stamping workers (default: all CPUs)")
    ap.add_argument("--positions", default=None,
                    help="override motif positions file (e.g. greedy positions_unindexed.jsonl for uidx MPNN); "
                         "absolute path, or a bare name resolving in <method>/1_filter/")
    ap.add_argument("--preserve-order", action="store_true",
                    help="positions are in NATIVE-RESIDUE order (greedy relocalization) -> do NOT sort ranks "
                         "when pairing native resnames (motif fragments may be out of order in the design)")
    a = ap.parse_args()
    mdir = os.path.join(a.run_dir, a.method)
    fdir = os.path.join(mdir, "1_filter")
    pos_path = (a.positions if (a.positions and os.path.isabs(a.positions))
                else os.path.join(fdir, a.positions) if a.positions
                else os.path.join(fdir, "positions.jsonl"))
    positions = json.load(open(pos_path))
    print(f"[mpnn_prep] positions <- {pos_path}" + ("  (preserve-order)" if a.preserve_order else ""))
    _tp = os.path.join(fdir, "tasks.json")                       # optional (co-design tracks lack it;
    tasks = json.load(open(_tp)) if os.path.exists(_tp) else {}  # task then falls out of canon.rsplit("_",1)[0])
    prepped = os.path.join(mdir, "mpnn", "prepped")
    os.makedirs(prepped, exist_ok=True)

    # 1) native resnames per TASK (only ~hundreds of tasks) — serial, cheap, shared across its designs.
    native_cache = {}
    for canon in positions:
        task = tasks.get(canon) or canon.rsplit("_", 1)[0]
        if task not in native_cache:
            native_cache[task] = native_resnames(os.path.join(a.run_dir, "motifs", f"{task}.pdb"))

    # 2) build per-design jobs (src, out, rank_to_rn), skip-existing so a re-run resumes.
    jobs, skipped = [], 0
    for canon, chains in positions.items():
        out = os.path.join(prepped, f"{canon}.pdb")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skipped += 1; continue
        src = os.path.join(fdir, "passed", f"{canon}.pdb")
        if not os.path.exists(src):
            continue
        task = tasks.get(canon) or canon.rsplit("_", 1)[0]
        ranks = [int(x) for _ch, rs in chains.items() for x in rs]     # as stored (native-residue order for greedy)
        if not a.preserve_order:
            ranks = sorted(ranks)                                       # in-order methods: sort == native order
        rank_to_rn = {r: rn for r, rn in zip(ranks, native_cache[task])}
        jobs.append((src, out, rank_to_rn))

    # 3) stamp in parallel (pure file I/O per design -> scales with CPUs).
    n = 0
    if jobs:
        with Pool(processes=max(1, a.workers)) as pool:
            for _ in pool.imap_unordered(_stamp_job, jobs, chunksize=64):
                n += 1
    # fixed_positions.jsonl == positions.jsonl ({stem: {chain: [1-based ranks]}}) — ProteinMPNN's exact shape.
    json.dump(positions, open(os.path.join(mdir, "mpnn", "fixed_positions.jsonl"), "w"))
    print(f"[mpnn_prep] {a.method}: stamped {n} designs ({skipped} already present, skipped) "
          f"-> {prepped}/ ; fixed_positions.jsonl written (workers={a.workers})")


if __name__ == "__main__":
    main()
