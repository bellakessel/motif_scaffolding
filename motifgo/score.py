"""score.py — the designability metric. ONE module, no path tricks.

MotifBench (arXiv:2502.12479): a design is **designable** iff

    motifRMSD <= 1.0 A   (backbone N,CA,C, motif residues, fixed superposition at design positions)
    scRMSD    <= 2.0 A   (CA only, generated backbone vs AF3 refold)

and, for this benchmark, AF3 `ranking_score >= 0.75` as a third gate (applied by scripts/evaluate/5_score.py).

HISTORY — why this file exists in this shape. The metric used to be split across two files:
`esp/score.py` reached OUTSIDE its package, by hardcoded absolute path
(`sys.path.insert("/cluster/work/wicky/bekessel/scripts")`), into an untracked
`af3_refold_analysis.py` to get the thresholds and the structure primitives. Every published
designability number therefore depended on a file nothing tracked, at a fixed path, for one user.
Both files are absorbed here, trimmed to what is actually called: of the original 707 lines, ~480
were the v1 standalone scorer (its own argparse CLI, `analyze`/`score_run`/`_process_archive`,
`locate_motif_in_design`, `compute_motif_rmsd`, ...) superseded by scripts/evaluate/5_score.py,
which drives the parallel archive loop itself.

AF3 renumbers residues (it folds the designed sequence 1..L), so the motif is located by
DESIGN POSITION, not residue id: `con_hal_idx0[k]` is the 0-based rank of motif residue k in the
refold, `con_ref_pdb_idx[k]` is its native (chain, res_id). scripts/evaluate/5_score.py synthesises
that mapping from `positions.jsonl` — no `.trb` file is needed for any method.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx import CIFFile, get_structure

# ── the designability thresholds. Defined HERE and nowhere else in the repo. ──────────────────
# All three live HERE and nowhere else. They decide every published number, so a second copy
# that drifts silently re-defines the result. scripts/ and motifgo.predictor import them.
MOTIF_RMSD_MAX = 1.0          # designable iff motifRMSD <= 1.0 A  (backbone N, CA, C)
SC_RMSD_MAX = 2.0             # designable iff scRMSD    <= 2.0 A  (CA only)
RANK_MIN = 0.75               # designable iff AF3 ranking_score >= 0.75  (INCLUSIVE: a large mass of
                              # predictions sits exactly at 0.75, so > would change the result)
MOTIF_ATOMS = ("N", "CA", "C")
SC_ATOMS = ("CA",)


# ── structure I/O (absorbed from af3_refold_analysis.py) ─────────────────────────────────────
def load_cif_structure(cif_path: str) -> Optional[struc.AtomArray]:
    """Load an AF3 model.cif as a biotite AtomArray (model 1). B-factor column = per-atom pLDDT."""
    try:
        structure = get_structure(CIFFile.read(str(cif_path)), model=1,
                                  include_bonds=False, extra_fields=["b_factor"])
        if isinstance(structure, AtomArrayStack):
            structure = structure[0]
        return structure
    except Exception as e:
        logging.error(f"CIF load failed {cif_path}: {e}")
        return None


def load_pdb_structure(pdb_path: str) -> Optional[struc.AtomArray]:
    """Load a PDB (motif template / generated design) as a biotite AtomArray."""
    try:
        with open(pdb_path) as f:
            structure = pdb_io.PDBFile.read(f).get_structure()
        if isinstance(structure, AtomArrayStack):
            structure = structure[0]
        return structure
    except Exception as e:
        logging.error(f"PDB load failed {pdb_path}: {e}")
        return None


def mean_plddt(structure: struc.AtomArray) -> float:
    """Mean CA pLDDT from the AF3 B-factor column."""
    if structure is None:
        return float("nan")
    ca = structure[structure.atom_name == "CA"]
    return float(np.mean(ca.b_factor)) if len(ca) > 0 else float("nan")


def parse_af3_confidence(json_path: str) -> Dict[str, float]:
    """ptm / ranking_score / has_clash / fraction_disordered / iptm from *_summary_confidences.json."""
    keys = ["ptm", "ranking_score", "has_clash", "fraction_disordered", "iptm"]
    try:
        with open(json_path) as f:
            d = json.load(f)
        return {k: d.get(k, np.nan) for k in keys}
    except Exception as e:
        logging.error(f"confidence load failed {json_path}: {e}")
        return {k: np.nan for k in keys}


def find_prediction_dirs(af3_dir: str) -> List[Path]:
    """Prediction dirs are the ones holding a *_model.cif. Works whether `af3_dir` is a single
    decompressed batch dir or a parent of many."""
    af3_dir = Path(af3_dir)
    return sorted({cif.parent for cif in af3_dir.rglob("*_model.cif")})


def superimpose(mobile: struc.AtomArray, fixed: struc.AtomArray):
    """Superimpose mobile onto fixed -> (fitted, rmsd, transformation). Truncates to the shorter."""
    try:
        if len(mobile) != len(fixed):
            n = min(len(mobile), len(fixed))
            mobile, fixed = mobile[:n], fixed[:n]
        fitted, transformation = struc.superimpose(fixed, mobile)
        return fitted, float(struc.rmsd(fixed, fitted)), transformation
    except Exception as e:
        logging.error(f"superimpose error: {e}")
        return mobile, float("inf"), None


# ── naming ───────────────────────────────────────────────────────────────────────────────────
def design_name_of(pred_name: str) -> str:
    """AF3 prediction name -> design name.  design_108_seq003 -> design_108"""
    return re.sub(r"_seq\d+$", "", pred_name)


# ── the two RMSDs ────────────────────────────────────────────────────────────────────────────
def _ordered_bb(structure, res_ids, atoms=MOTIF_ATOMS, chain=None):
    """Backbone atoms of the given residues, in order, atom-aligned 1:1 (one per (res, atom))."""
    pieces = []                                          # pieces = one AtomArray per (residue, atom name)
    for rid in res_ids:
        m = structure.res_id == rid                      # m = mask selecting this residue
        if chain is not None:
            m = m & (structure.chain_id == chain)
        res = structure[m]
        for a in atoms:
            sel = res[res.atom_name == a]
            if len(sel):
                pieces.append(sel[:1])
    if not pieces:
        return structure[[]]
    out = pieces[0]
    for p in pieces[1:]:
        out += p
    return out


def motif_rmsd_refold(refold, trb: dict, native, chain: str) -> float:
    """motifRMSD via the index mapping (robust for scattered/single-residue motifs, unlike
    sequence-anchoring): the refold folds the designed sequence 1..L in design order, so motif
    residue k is the `con_hal_idx0[k]`-th residue of the refold; compare its backbone (N,CA,C)
    to the native crystal motif at `con_ref_pdb_idx[k]`."""
    con_hal = [int(i) for i in trb.get("con_hal_idx0", [])]          # con_hal = 0-based RANKS in the design
    con_ref = [(str(c), int(r)) for c, r in trb.get("con_ref_pdb_idx", [])]   # con_ref = (chain, res_id) in
    #  the native crystal. The two lists are positionally paired: con_hal[k] is where native motif residue
    #  con_ref[k] ended up. AF3 renumbers 1..L, so the motif can only be found by RANK, never by res_id.
    if not con_hal or len(con_hal) != len(con_ref):
        return float("nan")
    refold_ids = refold.res_id[struc.get_residue_starts(refold)]   # refold_ids = residues in order, 1..L
    if max(con_hal) >= len(refold_ids):
        return float("nan")
    ref_bb = _ordered_bb(refold, [int(refold_ids[k]) for k in con_hal])   # ref_bb = predicted motif backbone
    nat_bb = _ordered_bb(native, [r for _c, r in con_ref], chain=chain)   # nat_bb = native motif backbone
    n = min(len(ref_bb), len(nat_bb))                    # compare only atoms present on BOTH sides
    if n == 0:
        return float("nan")
    _, rmsd, _ = superimpose(ref_bb[:n], nat_bb[:n])
    return rmsd


def sc_rmsd(refold, design) -> float:
    """scRMSD: CA-only superposition of the generated design vs the AF3 refold (self-consistency)."""
    _, rmsd, _ = superimpose(refold[refold.atom_name == "CA"], design[design.atom_name == "CA"])
    return rmsd


# ── aggregation ──────────────────────────────────────────────────────────────────────────────
def rollup(rows: List[Dict]) -> Dict:
    """Per-backbone designability (>= 1 designable sequence) + per-run rate."""
    by_design: Dict[str, list] = {}                      # design -> its per-sequence designable flags
    for r in rows:
        d = r["designable"]
        if r["status"] == "success" and d == d:                    # not NaN
            by_design.setdefault(r["design_name"], []).append(int(d))
    n_designs = len(by_design)                           # backbones that produced at least one scored seq
    n_designable = sum(1 for v in by_design.values() if any(v))   # `any` = ONE passing sequence is enough
    return {"n_seq": len(rows),
            "n_seq_designable": sum(1 for r in rows if r.get("designable") == 1),
            "n_designs": n_designs, "n_designs_designable": n_designable,
            "design_rate": (n_designable / n_designs if n_designs else 0.0)}


CSV_FIELDS = ["prediction", "task_name", "design_name", "motifRMSD", "scRMSD",
              "plddt_mean", "ptm", "iptm", "ranking_score", "has_clash", "fraction_disordered",
              "motif_ok", "sc_ok", "designable", "status"]


def write_csv(path: str, rows: List[Dict], fields: List[str] = None) -> None:
    """One row per scored sequence. `task_name` is emitted so downstream never has to re-derive
    the design_name -> task join by per-method regex (the formats differ across generators)."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
