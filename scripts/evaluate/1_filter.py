#!/usr/bin/env python3
"""1_filter.py — the motif-geometry gate (pipeline step 1, BEFORE MPNN). ONE script, ONE code path.

"Did the scaffold hold the native motif geometry?" — answered by MEASURING, not by trusting generator
metadata. For EVERY design (rfdiff / proteina / laproteina / pp1c / genie3, all the same way) we:
  1. enumerate the contig-consistent placements of the motif in the design (flank_n / linkers / flank_c
     within their manifest ranges, summing to the design length),
  2. Kabsch-superimpose the native motif Cα at each placement, take the MIN-RMSD placement,
  3. that min-RMSD IS the motifRMSD; keep designs with motifRMSD < cutoff (default 1.0 A, MotifBench).

Cα-only throughout (proteina emits Cα-only backbones, so Cα is the one metric comparable across methods),
and a pure-numpy PDB parser (robust to every method's PDB quirks — no biotite/esp load that trips on
genie3's formatting). Reads only what we ALWAYS have: the design coords, the native motif
(motifs/<task>.pdb) and the contig grammar (manifest.csv). No .trb, no per-method resolver.

Writes everything ONE place — <run>/<method>/1_filter/ :
    motif_rmsd.csv    design, task, motifRMSD, n_motif, pass, note   (ALL designs — the single results table)
    passed/<design>.pdb   SYMLINK to each passing design PDB (not a copy; inode-safe). 2_mpnn reads this dir.
    passed.txt        one passing design PDB path per line
    positions.jsonl   PASSERS only, {design: {"A": [design_res_ids]}} — the motif ranks == ProteinMPNN's
                      fixed_positions (reused unchanged by mpnn_prep + 5_score)
    tasks.json        PASSERS only, {design: task} — maps each passer to its native motif (motifs/<task>.pdb)

    python 1_filter.py --run-dir <run> --method genie3 --workers 16
    python 1_filter.py --selftest        # synthetic geometry check, no cluster
"""
from __future__ import annotations
import argparse
import csv
import glob
import itertools
import json
import os

import numpy as np
from motifgo.contig import parse_range   # one definition, shared with the contig grammar

CUTOFF = 1.0


def parse_contig(contig):
    """From the RFdiffusion-grammar contig -> (frag_lens, (flank_n, linker, flank_c) ranges, chain).
    A token is a motif fragment iff it starts with a chain letter; numeric tokens are flank_n (first),
    flank_c (last), linkers (middle). Uses `contig` (always in the manifest), not the `fragments` col."""
    toks = contig.split("/")
    frag_lens, chain = [], None
    for t in toks:
        if t and t[0].isalpha():
            chain = t[0]
            a, b = parse_range(t[1:])
            frag_lens.append(b - a + 1)
    fn = parse_range(toks[0])
    fc = parse_range(toks[-1])
    mids = [t for t in toks[1:-1] if t and t[0].isdigit()]
    ln = parse_range(mids[0]) if mids else (1, 5)
    return frag_lens, (fn, ln, fc), chain


def ca_coords(pdb_path):
    """Ordered (res_ids, coords) of Cα atoms (first model), sorted by res_id. Pure numpy; tolerant of any
    method's PDB formatting (fixed columns, no float-cast of arbitrary fields)."""
    ids, xyz = [], []
    with open(pdb_path) as fh:
        for ln in fh:
            if ln.startswith("ENDMDL"):
                break
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                ids.append(int(ln[22:26]))
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    order = np.argsort(ids, kind="stable")
    return np.array(ids)[order], np.array(xyz, dtype=float)[order]


def kabsch_rmsd(P, Q):
    """RMSD of P optimally superimposed onto Q (both (n,3)). Returns inf for non-finite (NaN/Inf) or
    degenerate coords and on SVD non-convergence — some generators (e.g. Protpardelle) occasionally emit
    NaN/collinear backbones, and a raised LinAlgError in a worker kills the whole pool. A bad placement
    just scores inf, so localize skips it (a design where EVERY placement fails -> no_placement -> filtered)."""
    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        return np.inf
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    try:
        U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    except np.linalg.LinAlgError:
        return np.inf
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P)))


def placements(frag_lens, L, fn, ln, fc):
    """Yield lists of 0-based design positions for every contig-consistent placement of the motif."""
    F = len(frag_lens)
    motif_tot = sum(frag_lens)
    for f0 in range(fn[0], fn[1] + 1):
        for links in itertools.product(range(ln[0], ln[1] + 1), repeat=max(0, F - 1)):
            tail = L - f0 - motif_tot - sum(links)
            if not (fc[0] <= tail <= fc[1]):
                continue
            pos, p = [], f0
            for i, fl in enumerate(frag_lens):
                pos.extend(range(p, p + fl))
                p += fl
                if i < F - 1:
                    p += links[i]
            if pos[-1] < L:
                yield pos


def localize(design_ids, design_xyz, native_xyz, frag_lens, ranges, early=0.5):
    """Best (rmsd, design_res_ids) placement of the native motif in one design. Early-stop on a near-0
    (pinned-motif) placement."""
    L = len(design_ids)
    fn, ln, fc = ranges
    best_r, best_pos = np.inf, None
    for pos in placements(frag_lens, L, fn, ln, fc):
        r = kabsch_rmsd(design_xyz[pos], native_xyz)
        if r < best_r:
            best_r, best_pos = r, pos
            if r < early:
                break
    if best_pos is None:
        return None, None
    # 1-based SEQUENTIAL RANKS (position in the ordered chain), NOT raw PDB resids. ProteinMPNN's
    # fixed_positions and the refold scorer both index by rank, and some generators (Proteina) number
    # residues 0-based, so returning raw resids there is off-by-one -> the motif is fixed one residue early
    # and designed away. design_ids is sorted ascending, so best_pos entry p == the 0-based rank -> p+1.
    return best_r, [p + 1 for p in best_pos]


def load_manifest(run_dir):
    m = {}
    with open(os.path.join(run_dir, "manifest.csv")) as fh:
        for r in csv.DictReader(fh):
            frag_lens, ranges, chain = parse_contig(r["contig"])
            m[r["task_name"]] = {"frag_lens": frag_lens, "ranges": ranges, "chain": r.get("chain", chain)}
    return m


def design_iter(run_dir, method, scaffolds):
    """Yield (task, design_name, pdb_path). Task matched as a SUBSTRING (longest first) of the design's
    path RELATIVE to scaffolds — works whether the task is in the filename (proteina/pp1c/laproteina) or
    only a parent dir (genie3 output/<task>/results/<generic>.pdb; rfdiff scaffolds/<task>/design_N.pdb).
    design_name always embeds the task, so it is unique and the native is recoverable."""
    tasks = load_manifest(run_dir)
    names = sorted(tasks, key=len, reverse=True)
    for p in sorted(glob.glob(os.path.join(scaffolds, "**", "*.pdb"), recursive=True)):
        rel = os.path.relpath(p, scaffolds)
        task = next((t for t in names if t in rel), None)
        if not task:
            continue
        stem = os.path.basename(p)[:-4]
        name = stem if task in stem else f"{task}_{stem}"
        yield task, name, p


def resolve_scaffolds(run_dir, method, override):
    if override:
        return override
    base = os.path.join(run_dir, method, "scaffolds")
    if os.path.isdir(base):
        return base
    for alt in ("inference", "output"):          # la-proteina -> inference/ ; genie3 -> output/<task>/results/
        cand = os.path.join(run_dir, method, alt)
        if os.path.isdir(cand):
            return cand
    return base


_W = {}   # per-worker globals (fork shares read-only natives/tasks)


def _init_worker(tasks, natives, early):
    _W["tasks"], _W["natives"], _W["early"] = tasks, natives, early


def _gate_one(item):
    """One design -> (task, name, rmsd_or_None, resids_or_None, abspath_or_None, note). Runs in a worker."""
    task, name, pdb = item
    info = _W["tasks"][task]
    native = _W["natives"].get(task)
    if native is None:
        return (task, name, None, None, None, "no_native")
    try:
        ids, xyz = ca_coords(pdb)
    except Exception:
        return (task, name, None, None, None, "read_err")
    if len(native) != sum(info["frag_lens"]) or len(ids) < len(native):
        return (task, name, None, None, None, "size")
    try:                                            # defense-in-depth: never let one design kill the pool
        r, resids = localize(ids, xyz, native, info["frag_lens"], info["ranges"], _W["early"])
    except Exception:
        return (task, name, None, None, None, "gate_err")
    if resids is None:
        return (task, name, None, None, None, "no_placement")
    return (task, name, r, resids, os.path.abspath(pdb), "")


def run(run_dir, method, cutoff, scaffolds_override, early, workers):
    run_dir = os.path.abspath(run_dir)
    scaffolds = resolve_scaffolds(run_dir, method, scaffolds_override)
    if not os.path.isdir(scaffolds):
        raise SystemExit(f"[1_filter] {method}: no scaffolds dir at {scaffolds}")
    tasks = load_manifest(run_dir)
    items = list(design_iter(run_dir, method, scaffolds))
    natives = {t: ca_coords(os.path.join(run_dir, "motifs", f"{t}.pdb"))[1]
               for t in {x[0] for x in items} if os.path.isfile(os.path.join(run_dir, "motifs", f"{t}.pdb"))}
    print(f"[1_filter] {method}: {len(items)} designs, {workers} worker(s), cutoff {cutoff} A")

    if workers > 1 and len(items) > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker, initargs=(tasks, natives, early)) as pool:
            results = pool.map(_gate_one, items, chunksize=64)
    else:
        _init_worker(tasks, natives, early)
        results = [_gate_one(x) for x in items]

    outdir = os.path.join(run_dir, method, "1_filter")
    passed_dir = os.path.join(outdir, "passed")
    os.makedirs(passed_dir, exist_ok=True)
    for f in glob.glob(os.path.join(passed_dir, "*")):     # idempotent: clear stale symlinks
        if os.path.islink(f):
            os.unlink(f)

    rows, positions, tasks_out, passers, npass = [], {}, {}, [], 0
    for task, name, r, resids, pathval, note in results:
        if resids is None:
            rows.append({"design": name, "task": task, "motifRMSD": "", "n_motif": "", "pass": 0, "note": note})
            continue
        ok = r < cutoff
        rows.append({"design": name, "task": task, "motifRMSD": round(r, 4),
                     "n_motif": len(resids), "pass": int(ok), "note": ""})
        if ok:
            link = os.path.join(passed_dir, f"{name}.pdb")
            if not os.path.lexists(link):
                os.symlink(pathval, link)
            passers.append(link)
            positions[name] = {"A": [int(x) for x in resids]}
            tasks_out[name] = task
            npass += 1

    with open(os.path.join(outdir, "motif_rmsd.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["design", "task", "motifRMSD", "n_motif", "pass", "note"],
                           lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "positions.jsonl"), "w") as fh:
        json.dump(positions, fh)
    with open(os.path.join(outdir, "tasks.json"), "w") as fh:
        json.dump(tasks_out, fh)
    with open(os.path.join(outdir, "passed.txt"), "w") as fh:
        fh.write("\n".join(passers) + ("\n" if passers else ""))

    ntot = len(rows)
    placed = [row["motifRMSD"] for row in rows if row["motifRMSD"] != ""]
    med = float(np.median(placed)) if placed else float("nan")
    pct = 100 * npass / ntot if ntot else 0
    print(f"[1_filter] {method}: {npass}/{ntot} pass motifRMSD < {cutoff} A ({pct:.1f}%); "
          f"localized-RMSD median={med:.3f}")
    print(f"  -> {outdir}/{{motif_rmsd.csv, passed/<design>.pdb (symlinks), passed.txt, positions.jsonl, tasks.json}}")
    print(f"  next: 2_mpnn --run-dir {run_dir} --method {method}  (MPNN the passers, motif fixed to native)")
    return npass, ntot


def _selftest():
    """Synthetic: a rigidly-moved copy of a 4-res motif localizes to ~0 A (pass); a 6 A kick fails."""
    import tempfile
    def write_pdb(path, residues):     # residues: list of (resid, {atom:(x,y,z)})
        with open(path, "w") as fh:
            i = 1
            for rid, atoms in residues:
                for an, (x, y, z) in atoms.items():
                    fh.write(f"ATOM  {i:>5d}  {an:<3s} GLY A{rid:>4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {an[0]}\n")
                    i += 1
    rng = np.random.default_rng(0)
    base = {rid: {"CA": tuple(rng.normal(size=3) * 5 + [rid, 0, 0])} for rid in range(1, 5)}
    theta = 0.7
    Rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    shift = np.array([10, -5, 3])
    mv = lambda v: tuple(Rot @ np.array(v) + shift)
    d = tempfile.mkdtemp()
    run_dir = os.path.join(d, "run"); os.makedirs(os.path.join(run_dir, "motifs"))
    # native motif = 4 Cα (chain A 1-4); contig: flank 3 / motif / flank 3, design length 10.
    write_pdb(os.path.join(run_dir, "motifs", "t_m0_base.pdb"), [(rid, base[rid]) for rid in range(1, 5)])
    with open(os.path.join(run_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task_name", "contig", "chain"], lineterminator="\n")
        w.writeheader(); w.writerow({"task_name": "t_m0_base", "contig": "3-3/A1-4/3-3", "chain": "A"})
    scaf = os.path.join(run_dir, "selftest", "scaffolds", "t_m0_base"); os.makedirs(scaf)
    # design length 10, resids 1..10; motif occupies design INDICES 3..6 (contig f0=3), rigidly moved.
    ok = []
    for idx in range(10):
        rid = idx + 1
        ca = mv(base[idx - 3 + 1]["CA"]) if 3 <= idx <= 6 else (float(idx), 9.0 if idx > 6 else 0.0, 0.0)
        ok.append((rid, {"CA": ca}))
    write_pdb(os.path.join(scaf, "design_0.pdb"), ok)
    bad = list(ok)
    bx = bad[3][1]["CA"]; bad[3] = (bad[3][0], {"CA": (bx[0] + 6.0, bx[1], bx[2])})
    write_pdb(os.path.join(scaf, "design_1.pdb"), bad)
    npass, ntot = run(run_dir, "selftest", CUTOFF, None, 0.5, 1)
    rows = {r["design"]: r for r in csv.DictReader(open(os.path.join(run_dir, "selftest", "1_filter", "motif_rmsd.csv")))}
    r0 = float(rows["t_m0_base_design_0"]["motifRMSD"]); r1 = float(rows["t_m0_base_design_1"]["motifRMSD"])
    print(f"  selftest: moved-copy motifRMSD={r0:.4f} (pass), kicked motifRMSD={r1:.4f} (fail)")
    assert r0 < 1e-3 and r1 > 1.0 and npass == 1, "SELFTEST FAILED"
    print("  SELFTEST PASSED")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", help="run dir with motifs/ + manifest.csv + <method>/scaffolds/")
    ap.add_argument("--method", help="rfdiff | proteina | laproteina | pp1c_bb | pp1c_aa | genie3")
    ap.add_argument("--cutoff", type=float, default=CUTOFF, help="motifRMSD gate in A (default 1.0)")
    ap.add_argument("--scaffolds", default=None, help="override scaffolds dir")
    ap.add_argument("--early", type=float, default=0.5, help="early-stop RMSD for pinned placements")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes over designs (default $SLURM_CPUS_PER_TASK or 1)")
    ap.add_argument("--selftest", action="store_true", help="synthetic geometry check, no cluster")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest(); return
    if not a.run_dir or not a.method:
        ap.error("need --run-dir and --method (or --selftest)")
    workers = a.workers if a.workers else int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    run(a.run_dir, a.method, a.cutoff, a.scaffolds, a.early, workers)


if __name__ == "__main__":
    main()
