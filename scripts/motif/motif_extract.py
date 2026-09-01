#!/usr/bin/env python
"""Extract library-format motifs from ANY PDB, using the SAME definition as Lisa / the library.

Consistency is the point: the surface + contact primitives come straight from `esp.motif`
(`surface_rsa` RSA>0.20 via biotite SASA + Tien maxASA, then `contacts_of` Cα ≤7 Å), which was
verified to reproduce Lisa's motifs exactly (40/40 on 3bis c_motif). What esp does NOT do is
enumerate one motif PER surface residue — `select_residues(sasa)` unions every surface residue
into ONE motif (it is built to scaffold a single interface). This script adds that loop.

  # 1) screen a candidate: chains, size, secondary-structure content (no motif extraction)
  python motif_extract.py screen --pdb 1YCR.pdb
  python motif_extract.py screen --fetch 1YCR

  # 2) extract motifs + descriptors, and check redundancy against the existing library
  python motif_extract.py extract --fetch 1TNF --chain A --tag tnfa \\
      --library /cluster/work/wicky/bekessel/motif_library --out_dir <dir>
"""
import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

# the single source of truth for "what is a motif"
from motifgo import motif_define as M

RSA_CUTOFF = 0.20
CONTACT_CUTOFF = 7.0
METRIC = "ca"


def fetch(pdb_id, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{pdb_id.upper()}.pdb")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        urllib.request.urlretrieve(url, path)
    return path


def chain_report(pdb_path):
    """Per-chain: length, SSE composition, surface residues. For screening candidates."""
    import biotite.structure as struc
    import biotite.structure.io as bsio
    arr = bsio.load_structure(pdb_path)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    arr = arr[~arr.hetero]
    arr = arr[struc.filter_amino_acids(arr)]
    rows = []
    for ch in sorted(set(str(c) for c in arr.chain_id)):
        sub = arr[arr.chain_id == ch]
        n_res = len(struc.get_residue_starts(sub))
        if n_res < 30:
            continue
        try:
            sse = struc.annotate_sse(sub)
        except Exception:
            continue
        n = len(sse) or 1
        h, e = float((sse == "a").sum()) / n, float((sse == "b").sum()) / n
        try:
            _rsa, surf = M.surface_rsa(pdb_path, ch, RSA_CUTOFF)
            n_surf = len(surf)
        except Exception:
            n_surf = -1
        rows.append(dict(chain=ch, n_res=n_res, n_surface=n_surf,
                         helix_frac=round(h, 2), sheet_frac=round(e, 2),
                         coil_frac=round(1 - h - e, 2),
                         fold="alpha" if h > 0.35 and h > 2 * e else
                              ("beta" if e > 0.25 and e > h else "mixed")))
    return pd.DataFrame(rows)


def extract_motifs(pdb_path, chain, tag):
    """One motif per surface residue — Lisa's definition, via esp primitives."""
    rsa, surface = M.surface_rsa(pdb_path, chain, RSA_CUTOFF)
    recs = []
    for c in surface:
        try:
            res = M.contacts_of(pdb_path, chain, [c], CONTACT_CUTOFF, METRIC)
        except Exception:
            continue
        if len(res) < 2:
            continue
        recs.append(dict(pdb_name=tag, motif_num=int(c), n_res=len(res),
                         residues=",".join(str(r) for r in sorted(res))))
    return pd.DataFrame(recs), rsa, surface


def descriptors(pdb_path, chain, motifs, tag):
    """Same descriptor set as motif_library_generation.ipynb (rg/linearity/planarity/SSE/...)."""
    import biotite.structure as struc
    import biotite.structure.io as bsio
    from scipy.spatial import cKDTree
    CURV_RADIUS, V_ATOM = 10.0, 20.1

    arr = bsio.load_structure(pdb_path)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    arr = arr[(arr.chain_id == chain) & ~arr.hetero]
    arr = arr[struc.filter_amino_acids(arr)]
    sse_raw = struc.annotate_sse(arr)
    sse = np.array([{"a": "H", "b": "E", "c": "C"}.get(s, "C") for s in sse_raw])
    res_ids = np.unique(arr.res_id)
    ca_m, cb_m = arr.atom_name == "CA", arr.atom_name == "CB"
    per = {}
    ca_xyz, cb_xyz, order = [], [], []
    for i, rid in enumerate(res_ids):
        rm = arr.res_id == rid
        ca = arr.coord[rm & ca_m][0] if (rm & ca_m).any() else arr.coord[rm].mean(0)
        cb = arr.coord[rm & cb_m][0] if (rm & cb_m).any() else ca
        per[int(rid)] = dict(ca=ca, sse=sse[i] if i < len(sse) else "C")
        ca_xyz.append(ca); cb_xyz.append(cb); order.append(int(rid))
    cb_xyz = np.array(cb_xyz)
    t_cb = cKDTree(cb_xyz)
    n_nb = np.array([len(t_cb.query_ball_point(p, CURV_RADIUS)) - 1 for p in cb_xyz])
    v_sphere = 4 / 3 * np.pi * CURV_RADIUS ** 3
    cxv = (v_sphere - np.maximum(n_nb, 1) * V_ATOM) / (np.maximum(n_nb, 1) * V_ATOM)
    for rid, c in zip(order, cxv):
        per[rid]["cx"] = float(c)

    rsa, _ = M.surface_rsa(pdb_path, chain, RSA_CUTOFF)
    out = []
    for r in motifs.itertuples():
        ids = [int(x) for x in r.residues.split(",") if int(x) in per]
        if len(ids) < 2:
            continue
        ca = np.array([per[i]["ca"] for i in ids])
        cen = ca - ca.mean(0)
        rg = float(np.sqrt((cen ** 2).sum(1).mean()))
        ev = np.sort(np.linalg.eigvalsh(np.cov(cen.T) + 1e-9 * np.eye(3)))[::-1]
        s = "".join(per[i]["sse"] for i in ids)
        nfr, srt = 1, sorted(ids)
        for a, b in zip(srt, srt[1:]):
            if b - a > 1:
                nfr += 1
        rr = [rsa.get(i, np.nan) for i in ids]
        out.append(dict(
            pdb_name=tag, motif_num=int(r.motif_num), n_res=len(ids), n_contacts=len(ids) - 1,
            contact_density=(len(ids) - 1) / len(ids),
            mean_rsa=float(np.nanmean(rr)), mean_burial=float(1 - np.nanmean(rr)),
            mean_cx=float(np.mean([per[i]["cx"] for i in ids])),
            std_cx=float(np.std([per[i]["cx"] for i in ids])),
            rg=rg, linearity=float((ev[0] - ev[1]) / ev[0]),
            planarity=float((ev[1] - ev[2]) / ev[0]),
            C_frac=s.count("C") / len(s), E_frac=s.count("E") / len(s),
            H_frac=s.count("H") / len(s), sse=s, n_fragments=nfr,
            residues=",".join(str(i) for i in srt)))
    df = pd.DataFrame(out)
    if len(df):
        df["dominant_sse"] = df[["C_frac", "E_frac", "H_frac"]].idxmax(axis=1).str[0]
    return df


FEATS = ["n_res", "n_contacts", "contact_density", "mean_rsa", "mean_cx",
         "rg", "linearity", "planarity"]


def redundancy_vs_library(new, lib_csv, z_thresh=0.75):
    """Cross-protein redundancy: residue-set Jaccard is meaningless between different proteins,
    so compare in standardised DESCRIPTOR space — nearest library motif per new motif.
    Small distance = structurally already represented."""
    from sklearn.preprocessing import StandardScaler
    lib = pd.read_csv(lib_csv)
    lib = lib[lib.get("is_representative", True) == True] if "is_representative" in lib else lib
    common = [f for f in FEATS if f in lib.columns and f in new.columns]
    sc = StandardScaler().fit(lib[common].fillna(0).to_numpy())
    L = sc.transform(lib[common].fillna(0).to_numpy())
    N = sc.transform(new[common].fillna(0).to_numpy())
    d = np.linalg.norm(N[:, None, :] - L[None, :, :], axis=2)
    j = d.argmin(1)
    out = new.copy()
    out["nn_dist"] = d.min(1)
    out["nn_library_motif"] = [f"{lib.iloc[k].pdb_name}_{int(lib.iloc[k].motif_num)}" for k in j]
    out["is_novel"] = out["nn_dist"] > z_thresh
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen", help="per-chain size + SSE composition (pick candidates)")
    s.add_argument("--pdb"); s.add_argument("--fetch"); s.add_argument("--cache", default="./pdb_cache")
    e = sub.add_parser("extract", help="enumerate motifs + descriptors (+ library redundancy)")
    e.add_argument("--pdb"); e.add_argument("--fetch"); e.add_argument("--cache", default="./pdb_cache")
    e.add_argument("--chain", required=True); e.add_argument("--tag", required=True)
    e.add_argument("--library", default=None, help="motif_library dir (for the redundancy check)")
    e.add_argument("--out_dir", default=None)
    a = ap.parse_args()

    path = a.pdb or fetch(a.fetch, a.cache)
    if a.cmd == "screen":
        rep = chain_report(path)
        print(f"\n{os.path.basename(path)}")
        print(rep.to_string(index=False) if len(rep) else "  no protein chain >=30 res")
        return

    motifs, rsa, surface = extract_motifs(path, a.chain, a.tag)
    print(f"{a.tag} chain {a.chain}: {len(surface)} surface residues (RSA>{RSA_CUTOFF}) "
          f"-> {len(motifs)} motifs")
    desc = descriptors(path, a.chain, motifs, a.tag)
    print(f"  dominant SSE: {desc.dominant_sse.value_counts().to_dict()}")
    print(f"  n_res median {desc.n_res.median():.0f}, fragments median {desc.n_fragments.median():.0f}")
    if a.library:
        desc = redundancy_vs_library(desc, os.path.join(a.library, "motif_features.csv"))
        print(f"  NOVEL vs library: {int(desc.is_novel.sum())}/{len(desc)} "
              f"(nn descriptor distance > 0.75)")
        print(f"  novel by SSE: {desc[desc.is_novel].dominant_sse.value_counts().to_dict()}")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, f"{a.tag}_motifs.csv")
        desc.to_csv(p, index=False)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
