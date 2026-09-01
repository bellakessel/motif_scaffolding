"""motif_build.py — Lisa-style motif-definition tables written alongside each run.

Two CSVs (mirroring her `{pdb}_SASA_resis.csv` and `merged_og_motifs.csv`):
  {pdb}_sasa_resis.csv  — every residue: res_id, res_name, aa, sasa, rsa, surface (RSA>cutoff)
  {pdb}_motifs.csv      — ONE ROW PER CANDIDATE CENTROID (each surface residue, or each explicit
                          residue): motif_num, motif_residues, motif_length, disjointed_parts,
                          dssp, in_run_motif. This is the full menu, so residues that are surface but
                          out of the run's scope still appear (with in_run_motif=0).

DSSP here is REAL hydrogen-bond DSSP (esp._dssp, a dependency-light Kabsch-Sander port of PyDSSP),
computed on the FULL chain then sliced per motif — identical to Lisa's pipeline (DsspApp/mkdssp on the
full chain, 8->3 collapse; verified 100% per-residue agreement on 2mhs). This replaced biotite
`annotate_sse` (P-SEA, Cα-only), which systematically over-called coil. Needs biotite/scipy/numpy;
callers wrap in try/except so a missing dep never breaks build-contig.
"""
from __future__ import annotations
import csv
from typing import Dict, List

from . import motif_define as motif
from . import contig


def sasa_rows(pdb: str, chain: str, rsa_cutoff: float) -> List[dict]:
    """Per-residue SASA + RSA table for the whole chain (Lisa nb1), + surface flag at rsa_cutoff.
    Derives from motif.residue_rsa_table (the single SASA/RSA source) — surface flag uses the
    UNROUNDED rsa (rounding only for display), identical to the previous inline computation."""
    rows = []
    for r in motif.residue_rsa_table(pdb, chain):
        rsa = r["rsa"]                                   # rsa = relative solvent accessibility, 0..~1
        surface = int(rsa > rsa_cutoff) if rsa == rsa else 0        # `rsa == rsa` is False for NaN -> not surface
        rows.append({"res_id": r["res_id"], "res_name": r["res_name"], "aa": r["aa"],
                     "sasa": round(r["sasa"], 2),
                     "rsa": round(rsa, 3) if rsa == rsa else float("nan"),
                     "surface": surface})
    return rows


def _sse_by_res(arr) -> Dict[int, str]:
    """Per-residue 3-state SSE {res_id: 'H'|'E'|'C'} from REAL hydrogen-bond DSSP (esp._dssp),
    matching Lisa's mkdssp method. Replaces biotite annotate_sse (P-SEA), which over-called coil."""
    from . import dssp as _dssp
    return _dssp.sse_by_res(arr)


def motif_rows(pdb: str, chain: str, centroids: List[int], cutoff: float, metric: str) -> List[dict]:
    """One motif per centroid = centroid + its Cα/Cβ contacts within `cutoff` (Lisa nb2)."""
    from scipy.spatial import distance_matrix

    arr = motif._load_chain(pdb, chain)
    res_ids, reps = motif._rep_coords(arr, metric)       # reps = one coord per residue (CA or CB)
    idx = {int(r): i for i, r in enumerate(res_ids)}     # idx = res_id -> its row in `reps`
    sse = _sse_by_res(arr)                               # sse = res_id -> H / E / C
    rows = []
    for c in centroids:
        if c not in idx:
            continue
        d = distance_matrix([reps[idx[c]]], reps)[0]     # d = distance from centroid c to every residue (A)
        mres = sorted(int(res_ids[j]) for j in range(len(res_ids)) if d[j] <= cutoff)   # mres = the motif
        runs = contig.group_runs(mres)                   # runs = its consecutive stretches -> fragment count
        rows.append({
            "motif_num": int(c),
            "motif_residues": " ".join(str(r) for r in mres),
            "motif_length": len(mres),
            "disjointed_parts": len(runs),
            "dssp": "".join(sse.get(r, "C") for r in mres),
        })
    return rows


def filled_rows(pdb: str, chain: str, base_rows: List[dict], gap_merge: int) -> List[dict]:
    """Matched `reduce_disjoint` variant of each base motif: fill sequence gaps <= gap_merge between
    the motif's runs (present residues only, so a crystallographic gap never invents coordinates) so
    the SAME surface patch becomes contiguous and longer. Recomputes motif_residues/length/
    disjointed_parts/dssp, keeps the same motif_num. Motifs that are already contiguous (nothing to
    fill, filled == base) are SKIPPED — no distinct pair to compare. Pair with add_descriptors()."""
    arr = motif._load_chain(pdb, chain)
    present = set(int(r) for r in motif._residue_ids_names(arr)[0])   # residues with actual coordinates
    sse = _sse_by_res(arr)
    out = []
    for m in base_rows:
        ids = sorted(int(x) for x in str(m["motif_residues"]).split())   # ids = the BASE motif residues
        runs = contig.merge_gaps(contig.group_runs(ids), gap_merge)      # runs = fragments after gap merging
        filled = sorted(r for a, b in runs for r in range(a, b + 1) if r in present)   # `in present` never
                                                                          # invents a crystallographic gap
        if filled == ids:                                       # already contiguous -> no pair
            continue
        out.append({
            "motif_num": int(m["motif_num"]),
            "motif_residues": " ".join(str(r) for r in filled),
            "motif_length": len(filled),
            "disjointed_parts": len(contig.group_runs(filled)),
            "dssp": "".join(sse.get(r, "C") for r in filled),
        })
    return out


def add_descriptors(pdb: str, chain: str, rows: List[dict], rsa_cutoff: float) -> List[dict]:
    """Attach per-motif STRUCTURAL descriptors in place: the shape axis (rg / linearity /
    planarity from the Cα coordinate covariance), convexity (CX, Pintar et al.), mean RSA and
    SSE fractions. Same definitions the motif-library notebook uses, computed here so the
    extraction CSV is self-sufficient for pooling and later classing."""
    import numpy as np
    from scipy.spatial import cKDTree

    CURV_RADIUS, V_ATOM = 10.0, 20.1                     # CX sphere radius (A) and mean atomic volume (A^3)
    arr = motif._load_chain(pdb, chain)
    sse = _sse_by_res(arr)
    res_ids, res_names = motif._residue_ids_names(arr)

    ca, cb = {}, {}                                      # res_id -> CA / CB coordinate
    for rid in res_ids:
        r = arr[arr.res_id == rid]
        a = r[r.atom_name == "CA"]
        b = r[r.atom_name == "CB"]
        ca[int(rid)] = a.coord[0] if len(a) else r.coord.mean(0)
        cb[int(rid)] = b.coord[0] if len(b) else ca[int(rid)]

    order = [int(r) for r in res_ids]                    # order = residues in chain order
    cb_xyz = np.array([cb[r] for r in order])            # cb_xyz = their CB coords, row-aligned to `order`
    tree = cKDTree(cb_xyz)
    n_nb = np.array([len(tree.query_ball_point(p, CURV_RADIUS)) - 1 for p in cb_xyz])   # n_nb = neighbours
                                                         #   inside the sphere (-1 drops the residue itself)
    v_int = np.maximum(n_nb, 1) * V_ATOM                 # v_int = protein volume filling the sphere
    cx = {r: float(v) for r, v in zip(order, (4 / 3 * np.pi * CURV_RADIUS ** 3 - v_int) / v_int)}
    #  cx = convexity (Pintar): empty sphere volume / filled volume. High = protruding, low = buried.

    rsa = {r["res_id"]: r["rsa"] for r in sasa_rows(pdb, chain, rsa_cutoff)}   # rsa = res_id -> exposure

    for m in rows:
        ids = [int(x) for x in str(m["motif_residues"]).split() if int(x) in ca]   # ids = this motif's residues
        if len(ids) < 2:                                 # a 1-residue motif has no shape to measure
            continue
        pts = np.array([ca[i] for i in ids])             # pts = its CA coordinates
        cen = pts - pts.mean(0)                          # cen = the same points centred on their centroid
        ev = np.sort(np.linalg.eigvalsh(np.cov(cen.T) + 1e-9 * np.eye(3)))[::-1]
        #  ev = principal spreads of the point cloud, largest first (the +1e-9 keeps a flat motif invertible).
        #  rod: ev[0] >> ev[1] ~ ev[2].  disc: ev[0] ~ ev[1] >> ev[2].  ball: all three similar.
        s = "".join(sse.get(i, "C") for i in ids)        # s = the motif's DSSP string, e.g. "CCEEEC"
        rr = [rsa.get(i, float("nan")) for i in ids]     # rr = per-residue exposure of the motif
        m.update({
            "rg": round(float(np.sqrt((cen ** 2).sum(1).mean())), 3),
            "linearity": round(float((ev[0] - ev[1]) / ev[0]), 3),
            "planarity": round(float((ev[1] - ev[2]) / ev[0]), 3),
            "mean_cx": round(float(np.mean([cx[i] for i in ids])), 3),
            "mean_rsa": round(float(np.nanmean(rr)), 3),
            "C_frac": round(s.count("C") / len(s), 3),
            "E_frac": round(s.count("E") / len(s), 3),
            "H_frac": round(s.count("H") / len(s), 3),
            "dominant_sse": max("CEH", key=lambda c: s.count(c)),
        })
    return rows


def write_motif_pdbs(pdb: str, chain: str, rows: List[dict], out_dir: str,
                     archive: bool = False, name: str = "c_motif") -> int:
    """Write one all-atom template PDB per motif, matching Lisa's motif_templates layout:
    `{out_dir}/{name}/original_motif_{motif_num}.pdb`, ORIGINAL residue numbering and chain id
    (that is what the library + RFdiffusion contigs index against).

    `archive=True` packs them into `{out_dir}/{name}.tar.zst` and removes the loose files, matching
    her `motif_templates/*.tar.zst` and keeping the cluster inode count down."""
    import os
    import shutil
    import subprocess

    import biotite.structure.io.pdb as bpdb

    arr = motif._load_chain(pdb, chain)
    dest = os.path.join(out_dir, name)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for r in rows:
        ids = [int(x) for x in str(r["motif_residues"]).split()]
        if not ids:
            continue
        sub = arr[[bool(i in set(ids)) for i in arr.res_id]]
        if len(sub) == 0:
            continue
        f = bpdb.PDBFile()
        f.set_structure(sub)
        f.write(os.path.join(dest, f"original_motif_{int(r['motif_num'])}.pdb"))
        n += 1
    if archive:
        tar = os.path.join(out_dir, f"{name}.tar")
        subprocess.run(["tar", "-C", out_dir, "-cf", tar, name], check=True)
        subprocess.run(["zstd", "-q", "-f", "--rm", tar, "-o", tar + ".zst"], check=True)
        shutil.rmtree(dest)
    return n


def write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def stage_motif_pdb(lib: str, pdb_name: str, motif_num: int, variant: str,
                    staging: str, task_name: str) -> str:
    """Extract `original_motif_<num>.pdb` from the right archive into `staging/<task_name>.pdb`.

    variant 'base' -> c_motif.tar.zst ; anything else -> c_motif_filled.tar.zst.
    Accepts both layouts: `proteins/<t>/motif_pdbs/` (what extract_motifs.py writes) and
    `<lib>/<t>/motif_templates/` (the pre-repo motif_library).

    Lives here, not copied into each generator's driver — there used to be four near-identical
    copies, so a fix to one (such as supporting a new library layout) reached only one caller.
    """
    import io
    import os
    import subprocess
    import tarfile

    arch = "c_motif" if variant == "base" else "c_motif_filled"
    cands = [os.path.join(lib, pdb_name, sub, f"{arch}.tar.zst")
             for sub in ("motif_pdbs", "motif_templates")]
    zpath = next((c for c in cands if os.path.exists(c)), None)
    if zpath is None:
        raise FileNotFoundError(
            f"no motif templates for {pdb_name}/{variant}; looked for:\n  " + "\n  ".join(cands) +
            f"\nrun: extract_motifs.py --target <proteins>/{pdb_name}/target.yaml")
    member = f"original_motif_{int(motif_num)}.pdb"
    os.makedirs(staging, exist_ok=True)
    out = os.path.join(staging, f"{task_name}.pdb")
    raw = subprocess.run(["zstd", "-d", "-c", zpath], check=True, stdout=subprocess.PIPE).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        cand = [m for m in tf.getmembers() if os.path.basename(m.name) == member]
        if not cand:
            raise KeyError(f"{member} not in {zpath}")
        with tf.extractfile(cand[0]) as fh, open(out, "wb") as w:
            w.write(fh.read())
    return out


def iter_pool(pool_csv: str, variants):
    """Yield pool rows whose `variant` is in `variants` ('base', 'filled', ...).

    The pool CSV is the task list every method reads. This lived as four identical copies —
    one in the stager and one in each generator driver — so the filter semantics could drift
    per method without anyone noticing.
    """
    import csv
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            if r.get("variant", "base") in variants:
                yield r
