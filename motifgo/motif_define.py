"""motif_define.py — select the motif residues on the antigen/interface chain.

The PRIMARY, general method is SASA-based (`motif_source: sasa`), for scaffolding protein-interface
motifs where there is often no antibody: surface residues (RSA > cutoff) are the **centroids**, and the
motif = centroids + their contacts within a tunable Cα/Cβ distance — mirroring Lisa's notebooks 1 & 2.

Other sources (kept for the antibody positive-control):
  - `explicit`  : motif = the given `motif_residues`
  - `partner`   : motif = residues on `chain` within `contact_cutoff` Å of `partner_chains`

`neighborhood_radius` (any-heavy-atom spatial shell) still applies to explicit/partner; `sasa` uses the
centroid contact cutoff instead. biotite/scipy are imported lazily so the explicit path needs no deps.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# Tien et al. 2013 theoretical maximum ASA (Å²) — same reference Lisa's RSA uses.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "E": 223.0, "Q": 225.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0,
    "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_THREE_TO_ONE = THREE_TO_ONE   # back-compat alias (was private in esp/motif.py)


def _load_chain(pdb_path: str, chain: str, strip_histag: bool = True):
    """First model, amino acids of `chain`; drop a trailing 6×HIS tag (as Lisa's nb1)."""
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io as bsio

    arr = bsio.load_structure(pdb_path)                  # arr = every atom in the file
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]                                     # NMR/multi-model file -> keep model 1 only
    arr = arr[(arr.chain_id == chain) & ~arr.hetero]     # narrow to the antigen chain, drop ligands/waters
    arr = arr[struc.filter_amino_acids(arr)]
    if len(arr) == 0:
        raise ValueError(f"no amino-acid atoms on chain {chain} of {pdb_path}")
    if strip_histag:
        res_ids = np.unique(arr.res_id)
        last6 = res_ids[-6:]                             # last6 = the C-terminal six residues (His-tag candidate)
        if len(last6) == 6 and all(arr.res_name[arr.res_id == r][0] == "HIS" for r in last6):
            arr = arr[~np.isin(arr.res_id, last6)]
    return arr


def _residue_ids_names(arr):
    import biotite.structure as struc
    starts = struc.get_residue_starts(arr)               # starts = atom index where each residue begins
    return arr.res_id[starts], arr.res_name[starts]      # one id + name per residue, in chain order


def residue_rsa_table(pdb_path: str, chain: str) -> List[dict]:
    """Canonical per-residue SASA + RSA for the whole chain (biotite SASA + Tien maxASA), UNROUNDED.
    One row per residue: res_id, res_name, aa, sasa, rsa. Non-standard residues get aa='X', rsa=NaN.
    SINGLE source of the SASA/RSA computation: both surface_rsa (scaffolding) and catalog.sasa_rows
    (library CSV) derive from it, so the surface set and the descriptor RSA can never disagree."""
    import numpy as np
    import biotite.structure as struc

    arr = _load_chain(pdb_path, chain)
    sasa_atom = struc.sasa(arr)                          # solvent-accessible area per ATOM (A^2)
    sasa_res = struc.apply_residue_wise(arr, sasa_atom, np.nansum)   # summed per RESIDUE
    res_ids, res_names = _residue_ids_names(arr)
    rows: List[dict] = []
    for rid, rname, s in zip(res_ids, res_names, sasa_res):
        aa = THREE_TO_ONE.get(rname, "X")                # aa = one-letter code; X = non-standard residue
        maxa = MAX_ASA.get(aa)                           # maxa = Tien reference max ASA for this residue type
        rsa = float(s) / maxa if maxa else float("nan")  # rsa = fraction of that maximum actually exposed
        rows.append({"res_id": int(rid), "res_name": str(rname), "aa": aa,
                     "sasa": float(s), "rsa": rsa})
    return rows


def surface_rsa(pdb_path: str, chain: str, rsa_cutoff: float = 0.20) -> Tuple[Dict[int, float], List[int]]:
    """Per-residue RSA and the surface (centroid) residues with RSA > cutoff. Standard residues only
    (non-standard excluded from the RSA dict, as before); derives from residue_rsa_table."""
    rsa: Dict[int, float] = {r["res_id"]: r["rsa"] for r in residue_rsa_table(pdb_path, chain)
                             if r["aa"] != "X"}
    surface = sorted(r for r, v in rsa.items() if v > rsa_cutoff)   # surface = the motif CENTROID menu
    return rsa, surface


def _rep_coords(arr, metric: str):
    """One representative coord per residue: Cα (metric 'ca') or Cβ with Cα for Gly (metric 'cb')."""
    import numpy as np

    res_ids, res_names = _residue_ids_names(arr)
    coords = []
    for rid, rname in zip(res_ids, res_names):
        res = arr[arr.res_id == rid]                     # res = this residue's atoms
        want = "CB" if (metric == "cb" and rname != "GLY") else "CA"   # Gly has no CB
        sel = res[res.atom_name == want]                 # sel = the single representative atom
        if len(sel) == 0:  # fall back to CA if CB missing
            sel = res[res.atom_name == "CA"]
        coords.append(sel.coord[0])
    return res_ids, np.array(coords)


def contacts_of(pdb_path: str, chain: str, centroids: List[int],
                cutoff: float = 7.0, metric: str = "ca") -> List[int]:
    """Motif = centroids + every residue within `cutoff` Å (Cα/Cβ) of any centroid (Lisa's nb2)."""
    from scipy.spatial import distance_matrix

    arr = _load_chain(pdb_path, chain)
    res_ids, reps = _rep_coords(arr, metric)             # reps = one coord per residue, row-aligned to res_ids
    idx = {int(r): i for i, r in enumerate(res_ids)}     # idx = res_id -> its row in `reps`
    cen_idx = [idx[c] for c in centroids if c in idx]    # cen_idx = rows of the requested centroids
    if not cen_idx:
        raise ValueError(f"none of the centroids {centroids} found on chain {chain}")
    dm = distance_matrix(reps[cen_idx], reps)            # dm = centroid x every-residue distances (A)
    hit = set(centroids)                                 # hit = motif residues, accumulated over centroids
    for row in dm:
        hit.update(int(res_ids[j]) for j in range(len(res_ids)) if row[j] <= cutoff)
    return sorted(hit)


def contacts_with_partner(pdb_path: str, chain: str, partner_chains: List[str], cutoff: float) -> List[int]:
    """res_ids on `chain` with any heavy atom within `cutoff` Å of any partner-chain atom."""
    import numpy as np
    from scipy.spatial import cKDTree
    import biotite.structure as struc
    import biotite.structure.io as bsio

    arr = bsio.load_structure(pdb_path)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    heavy = arr[arr.element != "H"]                      # hydrogens are absent/unreliable in crystals
    on = heavy[heavy.chain_id == chain]                  # on = atoms of the antigen chain
    part = heavy[np.isin(heavy.chain_id, list(partner_chains))]      # part = atoms of the binder chains
    if len(on) == 0 or len(part) == 0:
        raise ValueError(f"empty selection: chain {chain} n={len(on)}, partners {partner_chains} n={len(part)}")
    tree = cKDTree(part.coord)                           # spatial index over the binder atoms
    hit = tree.query_ball_point(on.coord, r=cutoff)      # hit[i] = binder atoms near antigen atom i
    return sorted({int(on.res_id[i]) for i, nbrs in enumerate(hit) if nbrs})   # nbrs non-empty -> interface


def expand_neighborhood(pdb_path: str, chain: str, seed_residues: List[int], radius: float) -> List[int]:
    """Add every res on `chain` with any heavy atom within `radius` Å of the seed residues."""
    if radius <= 0:
        return sorted(set(int(r) for r in seed_residues))
    import numpy as np
    from scipy.spatial import cKDTree
    import biotite.structure as struc
    import biotite.structure.io as bsio

    arr = bsio.load_structure(pdb_path)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    heavy = arr[(arr.chain_id == chain) & (arr.element != "H")]
    seed_atoms = heavy[np.isin(heavy.res_id, list(seed_residues))]   # seed_atoms = atoms of the motif so far
    if len(seed_atoms) == 0:
        raise ValueError(f"none of the seed residues {seed_residues} found on chain {chain}")
    tree = cKDTree(seed_atoms.coord)                     # spatial index over the seed
    hit = tree.query_ball_point(heavy.coord, r=radius)   # hit[i] = seed atoms near chain atom i
    res = {int(heavy.res_id[i]) for i, nbrs in enumerate(hit) if nbrs}   # res = seed + everything touching it
    res.update(int(r) for r in seed_residues)
    return sorted(res)


def present_residues(pdb_path: str, chain: str):
    """Set of residue ids present in the chain, or None if the PDB can't be read (dep-free path)."""
    try:
        arr = _load_chain(pdb_path, chain)
        return set(int(r) for r in _residue_ids_names(arr)[0])
    except Exception:
        return None


def _present_only(pdb_path: str, chain: str, residues: List[int]) -> List[int]:
    """Best-effort: drop residues absent from the PDB (e.g. disordered loops). If biotite/PDB can't be
    read (local dep-free run), keep the list unchanged so the explicit path still works offline."""
    import sys
    present = present_residues(pdb_path, chain)
    if present is None:
        return residues
    missing = [r for r in residues if r not in present]
    if missing:
        print(f"[motif] WARNING: {len(missing)} motif residue(s) absent from chain {chain} of the PDB "
              f"(dropped): {missing}", file=sys.stderr)
    return [r for r in residues if r in present]


def resolve_source(cfg: dict) -> str:
    """Explicit `motif_source`, else infer: explicit residues -> explicit, partner -> partner, else sasa."""
    src = cfg.get("motif_source")
    if src:
        return src
    if cfg.get("motif_residues"):
        return "explicit"
    if cfg.get("partner_chains"):
        return "partner"
    return "sasa"


def select_residues(cfg: dict) -> Tuple[str, List[int], List[int], Dict[int, float]]:
    """Resolve the motif. Returns (source, centroids, selected_residues, rsa_by_res).

    - sasa   : centroids = surface residues (RSA>cutoff) within scope (motif_residues | partner | whole
               chain); selected = centroids + Cα/Cβ contacts within `neighborhood_cutoff`.
    - explicit/partner : centroids = seed; selected = seed (+ spatial `neighborhood_radius` shell).
    """
    chain = cfg["chain"]
    source = resolve_source(cfg)

    if source == "sasa":
        rsa, surface = surface_rsa(cfg["pdb"], chain, float(cfg.get("rsa_cutoff", 0.20)))
        surf = set(surface)
        if cfg.get("motif_residues"):
            cand = set(int(r) for r in cfg["motif_residues"]) & surf   # surface subset of a region
        elif cfg.get("partner_chains"):
            partner = contacts_with_partner(cfg["pdb"], chain, cfg["partner_chains"],
                                            float(cfg.get("contact_cutoff", 5.0)))
            cand = set(partner) & surf
        else:
            cand = surf                                               # whole-chain surface
        centroids = sorted(cand)
        if not centroids:
            raise ValueError("no surface centroids (RSA>cutoff) in scope — lower rsa_cutoff or widen scope")
        selected = contacts_of(cfg["pdb"], chain, centroids,
                               float(cfg.get("neighborhood_cutoff", 7.0)),
                               str(cfg.get("neighborhood_metric", "ca")).lower())
        return source, centroids, selected, rsa

    if source == "explicit":
        seed = _present_only(cfg["pdb"], chain, sorted(set(int(r) for r in cfg["motif_residues"])))
    elif source == "partner":
        seed = contacts_with_partner(cfg["pdb"], chain, cfg["partner_chains"],
                                     float(cfg.get("contact_cutoff", 5.0)))
    else:
        raise ValueError(f"unknown motif_source {source!r} (sasa | explicit | partner)")

    radius = float(cfg.get("neighborhood_radius", 0) or 0)
    selected = expand_neighborhood(cfg["pdb"], chain, seed, radius) if radius > 0 else seed
    return source, seed, selected, {}
