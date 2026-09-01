#!/usr/bin/env python3
"""motif_rsa.py — motif burial / ΔRSA extractor for de-novo refolds (benchmark Task 2).

For every AF3 refold prediction of a method, compute FULL-STRUCTURE per-residue RSA (biotite Shrake-Rupley
SASA / Tien maxASA — the SAME method as esp.motif, so it is consistent with the native library RSA), then
read off the RSA at the motif's DESIGN positions and compare to the NATIVE reference RSA of the corresponding
crystal residues:

    ΔRSA(residue) = RSA_denovo(motif residue, in the whole refold) − RSA_native(the crystal residue)

RSA is deliberately computed on the ENTIRE refold (monomer), never on isolated motif atoms, so burial by the
surrounding de-novo scaffold is captured. Motif design positions come from the same positions.jsonl the
scorer uses (rank == residue array index); pass --positions to use a corrected set (e.g. the unindexed
greedy-relocalized positions_unindexed.jsonl).

Output: <run>/<method>/motif_rsa.csv — ONE row per design (compact; residue-level ΔRSA kept as a
space-joined string so the notebook can explode it OR use the per-motif aggregates directly):
    method, task, pdb_name, design_name, n_motif, mean_rsa_denovo, mean_rsa_native,
    median_dRSA, mean_dRSA, frac_buried (ΔRSA < -0.2), native_res_ids, dRSA_by_res, rsa_denovo_by_res

Parallel over archives (biotite SASA is CPU-bound; a method is ~250k structures), each worker extracts its
archive into node-local $TMPDIR (fast, no scratch quota), exactly like 5_score.py. Runs in gen_env (biotite),
no container, no GPU.  PYTHONPATH must include the esp pipeline (for esp.motif constants).

  export PYTHONPATH=<motif_benchmark repo>   # or: pip install -e <repo> --no-deps
  python motif_rsa.py --run-dir <run> --method <rfdiff|proteina|genie3|pp1c_bb|laproteina> \
      [--positions positions_unindexed.jsonl] [--lib /cluster/.../motif_library]
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List

CIF_SUFFIX = "_model.cif"
CODESIGN_METHODS = {"laproteina", "pp1c_aa"}


def _esp_consts():
    """MAX_ASA (Tien) + three->one, imported from the canonical esp.motif so RSA matches the library."""
    from motifgo.motif_define import MAX_ASA, THREE_TO_ONE as _THREE_TO_ONE
    return MAX_ASA, _THREE_TO_ONE


def native_motif_res_ids(motif_pdb: str) -> List[int]:
    """Ordered native crystal res_ids of the motif (ascending by (chain,res_id)) — the SAME order the design
    ranks in positions.jsonl were paired against (== 5_score's con_ref_pdb_idx order)."""
    seen = []
    with open(motif_pdb) as fh:
        for ln in fh:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                key = (ln[21], int(ln[22:26]))
                if not seen or seen[-1] != key:
                    seen.append(key)
    return [rid for _, rid in sorted(seen)]


def load_native_rsa(lib: str, pdb_name: str, cache: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    """{res_id: native rsa} from motif_library/<pdb>/<pdb>_sasa_resis.csv (cached per pdb)."""
    if pdb_name not in cache:
        p = os.path.join(lib, pdb_name, f"{pdb_name}_sasa_resis.csv")
        d = {}
        if os.path.isfile(p):
            for r in csv.DictReader(open(p)):
                try:
                    d[int(r["res_id"])] = float(r["rsa"])
                except (ValueError, KeyError):
                    pass
        cache[pdb_name] = d
    return cache[pdb_name]


def rsa_by_rank(cif_path: str, MAX_ASA, T31) -> List[float]:
    """FULL-structure per-residue RSA of the refold, in residue-array (== 1-based rank) order.
    Reuses esp.motif's biotite SASA + Tien maxASA; computed over all amino acids (monomer -> chain-agnostic)."""
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io as bsio
    arr = bsio.load_structure(cif_path)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    arr = arr[struc.filter_amino_acids(arr)]
    if arr.array_length() == 0:
        return []
    sasa_res = struc.apply_residue_wise(arr, struc.sasa(arr), np.nansum)
    starts = struc.get_residue_starts(arr)
    names = arr.res_name[starts]
    out = []
    for rname, s in zip(names, sasa_res):
        maxa = MAX_ASA.get(T31.get(rname, "X"))
        out.append(float(s) / maxa if maxa else float("nan"))
    return out


_W: Dict = {}


def _init_worker(run_dir, method, fixed, task_names, lib, pool_meta):
    _W["MAX_ASA"], _W["T31"] = _esp_consts()
    _W.update(run_dir=run_dir, method=method, fixed=fixed, lib=lib, pool_meta=pool_meta,
              task_names=sorted(set(task_names), key=len, reverse=True), native_rsa={}, native_ids={})


def _design_task(dname):
    """Longest task_name that prefixes the design (robust to <task>_N / <task>_design_N naming)."""
    for t in _W["task_names"]:
        if dname.startswith(t):
            return t
    return dname.rsplit("_", 1)[0]


def _analyze(cif, name):
    import numpy as np
    dname = name.rsplit("_seq", 1)[0] if "_seq" in name else name
    task = _design_task(dname)
    meta = _W["pool_meta"].get(task, {})
    pdb = meta.get("pdb_name") or task.split("_")[0]
    row = {"method": _W["method"], "task": task, "pdb_name": pdb, "design_name": dname,
           "n_motif": 0, "mean_rsa_denovo": "", "mean_rsa_native": "", "median_dRSA": "",
           "mean_dRSA": "", "frac_buried": "", "native_res_ids": "", "dRSA_by_res": "",
           "rsa_denovo_by_res": ""}
    pos = sorted(_W["fixed"].get(dname, {}).get("A", []))
    if not pos:
        return row
    if task not in _W["native_ids"]:
        mp_pdb = os.path.join(_W["run_dir"], "motifs", task + ".pdb")
        _W["native_ids"][task] = native_motif_res_ids(mp_pdb) if os.path.isfile(mp_pdb) else []
    nat_ids = _W["native_ids"][task]
    nat_rsa = load_native_rsa(_W["lib"], pdb, _W["native_rsa"])
    rsa = rsa_by_rank(cif, _W["MAX_ASA"], _W["T31"])
    if not rsa or len(nat_ids) != len(pos):
        return row
    d_denovo, d_native, d_delta = [], [], []
    for k, p in enumerate(pos):
        if p - 1 >= len(rsa):
            continue
        rd = rsa[p - 1]
        rn = nat_rsa.get(nat_ids[k], float("nan"))
        d_denovo.append(rd); d_native.append(rn); d_delta.append(rd - rn)
    if not d_delta:
        return row
    dd = np.array(d_delta, float)
    row.update(n_motif=len(pos),
               mean_rsa_denovo=round(float(np.nanmean(d_denovo)), 3),
               mean_rsa_native=round(float(np.nanmean(d_native)), 3),
               median_dRSA=round(float(np.nanmedian(dd)), 3),
               mean_dRSA=round(float(np.nanmean(dd)), 3),
               frac_buried=round(float(np.nanmean(dd < -0.2)), 3),
               native_res_ids=" ".join(map(str, nat_ids)),
               dRSA_by_res=" ".join(f"{x:.3f}" for x in d_delta),
               rsa_denovo_by_res=" ".join(f"{x:.3f}" for x in d_denovo))
    return row


def _score_archive(arch):
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or os.environ.get("SCRATCH") or "/tmp")
    rows = []
    try:
        subprocess.run(f'zstd -d -c "{arch}" | tar -xC "{tmp}"', shell=True, check=True, capture_output=True)
        for cif in glob.glob(os.path.join(tmp, "**", "*" + CIF_SUFFIX), recursive=True):
            rows.append(_analyze(cif, os.path.basename(cif)[:-len(CIF_SUFFIX)]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--lib", default="/cluster/work/wicky/bekessel/motif_library",
                    help="motif_library root (native <pdb>/<pdb>_sasa_resis.csv)")
    ap.add_argument("--pool", default=None, help="pool CSV for pdb_name (default <lib>/motif_pool_interface_j0.7_pad5.csv)")
    ap.add_argument("--positions", default=None,
                    help="override motif positions file (bare name resolves in <method>/, e.g. positions_unindexed.jsonl)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    mdir = os.path.join(run_dir, a.method)
    pos_default = (os.path.join(mdir, "positions.jsonl") if a.method in CODESIGN_METHODS
                   else os.path.join(mdir, "1_filter", "positions.jsonl"))
    pos_path = (os.path.join(mdir, a.positions) if a.positions and not os.path.isabs(a.positions)
                else a.positions) or pos_default
    fixed = json.load(open(pos_path))
    task_names = [r["task_name"] for r in csv.DictReader(open(os.path.join(run_dir, "manifest.csv")))]
    pool = a.pool or os.path.join(a.lib, "motif_pool_interface_j0.7_pad5.csv")
    pool_meta = {}
    if os.path.isfile(pool):
        for r in csv.DictReader(open(pool)):
            t = f"{r['pdb_name']}_m{int(r['motif_num'])}_{r['variant']}"
            pool_meta[t] = {"pdb_name": r["pdb_name"]}
    archives = sorted(glob.glob(os.path.join(mdir, "af3", "archives", "**", "*.tar.zst"), recursive=True))
    if not archives:
        raise SystemExit(f"no af3 archives under {mdir}/af3/archives")
    nproc = max(1, a.workers or os.cpu_count() or 1)
    out = a.out or os.path.join(mdir, "motif_rsa.csv")
    print(f"[motif_rsa:{a.method}] {len(archives)} archives, positions<-{pos_path}, {nproc} workers -> {out}", flush=True)

    fields = ["method", "task", "pdb_name", "design_name", "n_motif", "mean_rsa_denovo", "mean_rsa_native",
              "median_dRSA", "mean_dRSA", "frac_buried", "native_res_ids", "dRSA_by_res", "rsa_denovo_by_res"]
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        with mp.Pool(nproc, initializer=_init_worker,
                     initargs=(run_dir, a.method, fixed, task_names, a.lib, pool_meta)) as pool_:
            done = 0
            for rows in pool_.imap_unordered(_score_archive, archives):
                w.writerows(rows); n += len(rows); done += 1
                if done % 20 == 0 or done == len(archives):
                    print(f"  [{a.method}] {done}/{len(archives)} archives, {n} designs", flush=True)
    print(f"[motif_rsa:{a.method}] wrote {n} design rows -> {out}")


if __name__ == "__main__":
    main()
