#!/usr/bin/env python3
"""cif_backbone.py — pull the N/CA/C/O backbone out of an AlphaFold 3 `*_model.cif`.

Biotite's mmCIF reader costs ~22 ms per 90-residue AF3 model; at 208k structures that is over
an hour of CPU spent re-parsing a file format we control the producer of. This reads the
`_atom_site` loop directly (~2 ms) by resolving field positions from the loop header rather
than assuming a column order, so it stays correct if AF3 changes its output width.

`backbone_coords` returns coordinates in the atom order pydssp/DSSP requires -- N, CA, C, O --
which is the single thing most likely to be silently wrong in an SSE pipeline. Residues are
kept in file order. Anything that is not a complete backbone (missing atom, non-standard
residue, ligand) is dropped and counted, so callers can assert on it.
"""
from __future__ import annotations

import numpy as np

BB_ORDER = ("N", "CA", "C", "O")
_BB_IDX = {n: i for i, n in enumerate(BB_ORDER)}

# label_comp_id -> one letter. AF3 writes standard residues only for a protein-only job; anything
# unrecognised becomes 'X' so the sequence stays index-aligned with the coordinate array.
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
          "MSE": "M", "SEC": "U", "PYL": "O"}


def _atom_site_fields(fh):
    """Advance `fh` to the first _atom_site data row; return {field_name: column_index}."""
    cols = {}
    for line in fh:
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1].split()[0]] = len(cols)
        elif cols and (s.startswith("ATOM") or s.startswith("HETATM")):
            return cols, line
        elif cols and s and not s.startswith("#"):
            return cols, None          # header ended without data
    return cols, None


def backbone_coords(path: str):
    """-> (coord (L, 4, 3) float32 in N/CA/C/O order, meta dict).

    meta: n_res, n_dropped, chains, b_factor (mean over CA -- AF3 writes pLDDT there),
    and `seq`, the one-letter sequence of the KEPT residues (index-aligned with `coord`, so it
    lines up position-for-position with a DSSP string computed from the same array).
    """
    with open(path) as fh:
        cols, first = _atom_site_fields(fh)
        if not cols:
            raise ValueError(f"no _atom_site loop in {path}")
        need = ("label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
                "Cartn_x", "Cartn_y", "Cartn_z")
        missing = [k for k in need if k not in cols]
        if missing:
            raise ValueError(f"{path}: _atom_site missing {missing}")
        i_at, i_asym = cols["label_atom_id"], cols["label_asym_id"]
        i_comp, i_seq = cols["label_comp_id"], cols["label_seq_id"]
        i_x, i_y, i_z = cols["Cartn_x"], cols["Cartn_y"], cols["Cartn_z"]
        i_model = cols.get("pdbx_PDB_model_num")
        i_b = cols.get("B_iso_or_equiv")

        order, byres, comp, chains, model0 = [], {}, {}, [], None
        bsum = bn = 0.0
        lines = fh if first is None else _chain(first, fh)
        for line in lines:
            if not line.startswith("ATOM"):          # HETATM => ligand/ion, never backbone
                if line.startswith("#"):
                    break
                continue
            f = line.split()
            if i_model is not None:
                if model0 is None:
                    model0 = f[i_model]
                elif f[i_model] != model0:
                    break                            # first model only
            name = f[i_at].strip('"')
            k = _BB_IDX.get(name)
            if k is None:
                continue
            key = (f[i_asym], f[i_seq])
            slot = byres.get(key)
            if slot is None:
                slot = byres[key] = np.full((4, 3), np.nan, dtype=np.float32)
                order.append(key)
                comp[key] = AA3to1.get(f[i_comp], "X")
                if f[i_asym] not in chains:
                    chains.append(f[i_asym])
            if np.isnan(slot[k, 0]):                 # keep the first altloc only
                slot[k] = (float(f[i_x]), float(f[i_y]), float(f[i_z]))
                if k == 1 and i_b is not None:       # CA B-factor == AF3 pLDDT
                    bsum += float(f[i_b]); bn += 1

    if not order:
        raise ValueError(f"{path}: no protein backbone atoms found")
    stack = np.stack([byres[k] for k in order])      # (n, 4, 3)
    good = ~np.isnan(stack).any(axis=(1, 2))
    seq = "".join(comp[k] for k, g in zip(order, good) if g)
    meta = {"n_res": int(good.sum()), "n_dropped": int((~good).sum()), "seq": seq,
            "chains": chains, "b_factor": (bsum / bn) if bn else float("nan")}
    return stack[good], meta


def _chain(first, fh):
    yield first
    for line in fh:
        yield line
