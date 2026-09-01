#!/usr/bin/env python3
"""5_score.py — refold designability scorer, pipeline step 5, method-agnostic (one shared scorer for all
methods). REUSES motifgo.score's metric functions (MotifBench: motifRMSD backbone N,CA,C <= 1.0 AND
scRMSD CA-only <= 2.0, + AF3 confidences / pLDDT; AF3 residue-renumbering handled). Only the INPUTS are
read from the unified layout:

  - motif position IN THE DESIGN (0-based con_hal_idx0) = the ranks in <run>/<method>/1_filter/positions.jsonl
    (== the ProteinMPNN fixed positions; identical to the geometry-gate positions).
  - task (-> native motif) = <run>/<method>/1_filter/tasks.json ; native residues = motifs/<task>.pdb.
  - design backbone (for scRMSD, Cα) = the passing design <run>/<method>/1_filter/passed/<canonical>.pdb.
The refold folds the sequence in design order, so motif position k in the design == k in the refold —
the mapping motif_rmsd_refold expects. Everything else (thresholds, confidence parse, plddt, cif load,
rollup, write_csv) is motifgo.score. Runs in gen_env (biotite); no container, no GPU.

  python 5_score.py --run-dir <run> --method <rfdiff|proteina|genie3|pp1c_bb|laproteina|pp1c_aa> [--predictor af3|esmfold2]
"""
from __future__ import annotations
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List

from motifgo import score as mscore

CIF_SUFFIX = "_model.cif"
RANK_MIN = 0.75          # AF3 confidence gate; `designable` = motifRMSD<=1 & scRMSD<=2 & ranking_score>=RANK_MIN
                         # (the notebook's 3-gate). ESMFold2 has no ranking_score -> that gate is skipped there.


def load_design_lenient(pdb_path: str):
    """Load a design PDB robustly (blank occ/B -> 1.00/0.00), first model. Works for Cα-only (Proteina)
    and full-backbone / all-atom designs alike."""
    import io
    import biotite.structure.io.pdb as pdb_io
    from biotite.structure import AtomArrayStack
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                line = line.rstrip("\n")
                if len(line) < 66:
                    line = line.ljust(66)
                occ = line[54:60] if line[54:60].strip() else "  1.00"
                bf = line[60:66] if line[60:66].strip() else "  0.00"
                line = line[:54] + occ + bf + line[66:] + "\n"
            out.append(line if line.endswith("\n") else line + "\n")
    s = pdb_io.PDBFile.read(io.StringIO("".join(out))).get_structure()
    return s[0] if isinstance(s, AtomArrayStack) else s


def native_motif_residues(motif_pdb: str) -> List[tuple]:
    """Ordered (chain, resid) of the native motif residues (ascending == the order the design ranks were
    built in), from the motif template PDB."""
    res, seen = [], []
    with open(motif_pdb) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                key = (line[21], int(line[22:26]))
                if not seen or seen[-1] != key:
                    seen.append(key)
                    res.append(key)
    return sorted(res, key=lambda t: (t[0], t[1]))


def analyze(name, cif, conf, run_dir, fixed, tasks, native_cache, design_index, design_chain,
            preserve_order=False) -> Dict:
    """Score one refold. `conf` = AF3 *_summary_confidences.json (None for ESMFold2 — pLDDT from cif either way)."""
    import numpy as np
    dname = mscore.design_name_of(name)                 # <canonical>_seqNNN -> <canonical>
    task = tasks.get(dname) or dname.rsplit("_", 1)[0]
    row = {"prediction": name, "task_name": task, "design_name": dname,
           "ptm": np.nan, "iptm": np.nan, "ranking_score": np.nan, "has_clash": np.nan,
           "fraction_disordered": np.nan, "plddt_mean": np.nan, "motifRMSD": np.nan, "scRMSD": np.nan,
           "motif_ok": np.nan, "sc_ok": np.nan, "designable": np.nan, "status": "failed"}
    try:
        if conf and os.path.exists(conf):
            row.update(mscore.parse_af3_confidence(conf))
        refold = mscore.load_cif_structure(cif) if os.path.exists(cif) else None
        row["plddt_mean"] = mscore.mean_plddt(refold)

        motif_pdb = os.path.join(run_dir, "motifs", task + ".pdb")
        if task not in native_cache:
            native_cache[task] = (mscore.load_pdb_structure(motif_pdb), native_motif_residues(motif_pdb))
        native, native_motif = native_cache[task]
        native_chain = native_motif[0][0] if native_motif else design_chain

        design_pdb = design_index.get(dname)
        design = load_design_lenient(design_pdb) if design_pdb else None

        # default positions.jsonl are native-order ascending -> sorted() is a no-op; an override file
        # (unindexed_score.py's greedy relocalization) is ALREADY in native-residue order and must NOT be
        # re-sorted, because unindexed can place fragments out of order (pos[k] pairs with native_motif[k]).
        pos = fixed.get(dname, {}).get(design_chain, [])             # 1-based design ranks
        if not preserve_order:
            pos = sorted(pos)
        trb = {"con_hal_idx0": [p - 1 for p in pos], "con_ref_pdb_idx": native_motif}
        if refold is not None and pos:
            row["motifRMSD"] = mscore.motif_rmsd_refold(refold, trb, native, native_chain)
        if refold is not None and design is not None:
            row["scRMSD"] = mscore.sc_rmsd(refold, design)
        if row["motifRMSD"] == row["motifRMSD"]:
            row["motif_ok"] = int(row["motifRMSD"] <= mscore.MOTIF_RMSD_MAX)
        if row["scRMSD"] == row["scRMSD"]:
            row["sc_ok"] = int(row["scRMSD"] <= mscore.SC_RMSD_MAX)
        if row["motif_ok"] == row["motif_ok"] and row["sc_ok"] == row["sc_ok"]:
            rs = row.get("ranking_score")
            rank_ok = (rs != rs) or (rs is None) or (float(rs) >= RANK_MIN)   # NaN/None (ESMFold2) -> not gated
            row["designable"] = int(row["motif_ok"] and row["sc_ok"] and rank_ok)  # 3-gate, matches notebook
        row["status"] = "success"
    except Exception as e:
        row["error"] = str(e)[:120]
    return row


def _predictions(tmp, predictor):
    if predictor == "af3":
        for pdir in mscore.find_prediction_dirs(tmp):
            nm = os.path.basename(str(pdir).rstrip("/"))
            yield (nm, os.path.join(str(pdir), f"{nm}{CIF_SUFFIX}"),
                   os.path.join(str(pdir), f"{nm}_summary_confidences.json"))
    else:  # esmfold2 — flat *_model.cif
        for cif in sorted(glob.glob(os.path.join(tmp, "**", "*" + CIF_SUFFIX), recursive=True)):
            yield (os.path.basename(cif)[:-len(CIF_SUFFIX)], cif, None)


# co-design methods (la-Proteina, Protpardelle-1c-aa) carry the sequence in the generated backbone and are
# NOT MPNN'd, so the refold prediction names come from *_extract_seqs's <task>_<sample> namespace (== seqs/
# == scaffolds/ == the top-level positions.jsonl), NOT the 1_filter inference namespace. Read positions +
# backbones from that layout so design_name_of(prediction) joins to both the motif positions and the Cα
# reference. Backbone methods (rfdiff/proteina/genie3/pp1c_bb) keep the 1_filter/passed layout.
CODESIGN_METHODS = {"laproteina", "pp1c_aa"}

# --- parallel scoring: one worker per archive (CPU-bound biotite RMSD; backbone methods have 8 seqs/design
# -> ~250k structures, far too slow serial). Each worker extracts its archive into NODE-LOCAL $TMPDIR (fast,
# no scratch quota) and scores it. Shared read-only state (positions, design backbones) is set
# once per worker via the Pool initializer; native motif cache is per-worker (small). ---
_W: Dict = {}


def _init_worker(run_dir, fixed, tasks, design_index, design_chain, predictor, preserve_order):
    _W.update(run_dir=run_dir, fixed=fixed, tasks=tasks, design_index=design_index,
              design_chain=design_chain, predictor=predictor, preserve_order=preserve_order, native_cache={})


def _score_archive(arch):
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or os.environ.get("SCRATCH") or "/tmp")
    rows = []
    try:
        subprocess.run(f'zstd -d -c "{arch}" | tar -xC "{tmp}"', shell=True, check=True, capture_output=True)
        for nm, cif, conf in _predictions(tmp, _W["predictor"]):
            rows.append(analyze(nm, cif, conf, _W["run_dir"], _W["fixed"], _W["tasks"],
                                _W["native_cache"], _W["design_index"], _W["design_chain"],
                                _W["preserve_order"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def score_run(run_dir, method, design_chain, predictor, workers=None,
              positions_file=None) -> List[Dict]:
    mdir = os.path.join(run_dir, method)
    if method in CODESIGN_METHODS:
        # positions_file overrides the default motif positions (e.g. unindexed_score.py's greedy-relocalized
        # positions_unindexed.jsonl) — a bare name resolves in <method>/, else an explicit/absolute path.
        pos_path = (os.path.join(mdir, positions_file) if positions_file and not os.path.isabs(positions_file)
                    else positions_file) or os.path.join(mdir, "positions.jsonl")
        fixed = json.load(open(pos_path))                                # keyed <task>_<sample>
        print(f"[5_score:{method}] motif positions <- {pos_path}", flush=True)
        tasks_path = os.path.join(mdir, "tasks.json")                    # optional; rsplit fallback covers it
        design_dir = os.path.join(mdir, "scaffolds")                     # co-designed backbone (Cα for scRMSD)
    else:
        fdir = os.path.join(mdir, "1_filter")
        pos_path = (os.path.join(mdir, positions_file) if positions_file and not os.path.isabs(positions_file)
                    else positions_file) or os.path.join(fdir, "positions.jsonl")
        fixed = json.load(open(pos_path))
        tasks_path = os.path.join(fdir, "tasks.json")
        design_dir = os.path.join(fdir, "passed")                        # canonical-named passing design
    tasks = json.load(open(tasks_path)) if os.path.exists(tasks_path) else {}
    design_index = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(design_dir, "*.pdb"))}
    subdir = "af3/archives" if predictor == "af3" else "esmfold2"
    archives = sorted(glob.glob(os.path.join(mdir, subdir, "**", "*.tar.zst"), recursive=True))
    if not archives:
        raise SystemExit(f"no {predictor} archives under {mdir}/{subdir} — run the {predictor} refold first")
    nproc = max(1, workers or os.cpu_count() or 1)
    print(f"[5_score:{method}:{predictor}] {len(archives)} archive(s), {len(design_index)} design backbones, "
          f"{len(fixed)} motif-position sets, {nproc} workers", flush=True)
    rows: List[Dict] = []
    preserve_order = positions_file is not None       # override files are pre-ordered native-residue-wise
    with mp.Pool(processes=nproc, initializer=_init_worker,
                 initargs=(run_dir, fixed, tasks, design_index, design_chain, predictor, preserve_order)) as pool:
        done = 0
        for r in pool.imap_unordered(_score_archive, archives):
            rows.extend(r); done += 1
            if done % 20 == 0 or done == len(archives):
                print(f"  [{method}] {done}/{len(archives)} archives scored ({len(rows)} preds)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="unified refold designability scorer (AF3/ESMFold2; esp/score.py metrics)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", required=True,
                    help="rfdiff | proteina | genie3 | pp1c_bb | laproteina | pp1c_aa")
    ap.add_argument("--predictor", choices=["af3", "esmfold2"], default="af3")
    ap.add_argument("--design-chain", default="A", help="design chain (positions/scRMSD key)")
    ap.add_argument("--workers", type=int, default=None, help="parallel archive workers (default: all CPUs)")
    ap.add_argument("--positions", default=None,
                    help="override motif positions file (bare name resolves in <method>/, e.g. "
                         "positions_unindexed.jsonl from unindexed_score.py; else the default positions.jsonl)")
    ap.add_argument("--out", default=None, help="output CSV (default <run>/<method>/<predictor>_refold_scores.csv)")
    args = ap.parse_args()

    rows = score_run(args.run_dir, args.method, args.design_chain, args.predictor, args.workers,
                     positions_file=args.positions)
    out = args.out or os.path.join(args.run_dir, args.method, f"{args.predictor}_refold_scores.csv")
    mscore.write_csv(out, rows)
    s = mscore.rollup(rows)
    print(f"\n{args.method} {args.predictor}-refold designability ({os.path.basename(args.run_dir)}):")
    print(f"  sequences : {s['n_seq_designable']}/{s['n_seq']} designable (motifRMSD<=1 & scRMSD<=2 & rank>=0.75)")
    print(f"  backbones : {s['n_designs_designable']}/{s['n_designs']} designable "
          f"({100 * s['design_rate']:.1f}%)   <-- designability rate")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
