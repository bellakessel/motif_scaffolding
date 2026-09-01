#!/usr/bin/env python3
"""extract_motifs.py — Stage 0, ONE entry point: a protein structure + a target.yaml become the
complete motif contract for a new target.

    proteins/<t>/target.yaml
        │
        ├─ motif_define.surface_rsa   ──►  <t>_sasa_resis.csv   every residue + RSA + `surface` flag
        ├─ motif_build.motif_rows     ──►  base motifs           one per surface centroid (RSA > cutoff)
        ├─ motif_build.filled_rows    ──►  filled motifs         gaps <= gap_merge merged (optional)
        ├─ motif_build.add_descriptors──►  <t>_c_motif.csv       + SSE / shape / burial descriptors
        └─ motif_build.write_motif_pdbs ► motif_pdbs/c_motif{,_filled}.tar.zst

WHY THIS SCRIPT EXISTS. Stage 0 used to be split between two tools that each did half the job with
different output schemas: `scripts/motif_extract.py` wrote only a `{tag}_motifs.csv` descriptors
table, while the `*_sasa_resis.csv`, `*_c_motif.csv` and the `motif_templates/*.tar.zst` that
`stage_motif.py` actually reads were written by a different code path (`esp.run extract-motifs`).
Following the documented "plug in a new protein" recipe therefore produced no motif templates, and
staging a run died with `FileNotFoundError`. One script, one schema, one contract.

    python extract_motifs.py --target proteins/mcl1/target.yaml
    python extract_motifs.py --pdb 2mhs.pdb --chain A --tag mcl1 --out-dir proteins/mcl1
    python extract_motifs.py --target proteins/mcl1/target.yaml --screen     # chains + SSE, write nothing
"""
from __future__ import annotations

import argparse
import os
import sys

from motifgo import motif_build as B

SASA_FIELDS = ["res_id", "res_name", "aa", "sasa", "rsa", "surface"]
MOTIF_FIELDS = ["pdb_name", "motif_num", "variant", "gap_merge", "source_pdb", "chain",
                "motif_residues", "motif_length", "disjointed_parts", "dssp", "dominant_sse",
                "C_frac", "E_frac", "H_frac", "rg", "linearity", "planarity", "mean_cx", "mean_rsa"]


def load_target(path: str) -> dict:
    """Flat YAML target config. Falls back to a minimal parser so --screen needs no PyYAML."""
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        cfg = {}
        for line in open(path):
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                cfg[k.strip()] = v
        return cfg


def screen(pdb: str) -> None:
    """Report chains, sizes and SSE composition — pick the antigen chain before committing."""
    import warnings
    warnings.filterwarnings("ignore")
    import biotite.structure as struc
    import biotite.structure.io as bsio
    from motifgo import dssp

    arr = bsio.load_structure(pdb)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    print(f"\n{os.path.basename(pdb)}")
    print(f"  {'chain':>6} {'residues':>9} {'atoms':>8}   H/E/C composition")
    for ch in sorted(set(arr.chain_id)):
        sub = arr[(arr.chain_id == ch) & (arr.element != "H")]   # sub = this chain, heavy atoms only
        n = len(struc.get_residue_starts(sub))           # n = residues in the chain
        if n < 10:                                       # too short to be a scaffolding target
            continue
        try:
            sse = dssp.sse_by_res(sub)                   # sse = res_id -> H / E / C
            comp = "".join(sse.values())                 # comp = the chain as one DSSP string
            frac = {s: comp.count(s) / max(1, len(comp)) for s in "HEC"}   # helix/sheet/coil composition
            c = f"H {frac['H']:.2f}  E {frac['E']:.2f}  C {frac['C']:.2f}"
        except Exception as e:
            c = f"(sse failed: {e})"
        print(f"  {ch:>6} {n:>9} {len(sub):>8}   {c}")


def extract(pdb: str, chain: str, tag: str, out_dir: str, *, rsa_cutoff: float,
            nbhd_cutoff: float, nbhd_metric: str, gap_merge: int, emit_filled: bool,
            archive: bool = True) -> dict:
    """Write the full Stage-0 contract for one target. Returns a summary dict."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. surface table — every residue, with the `surface` flag that defines the centroid menu
    srows = B.sasa_rows(pdb, chain, rsa_cutoff)          # srows = one row per residue: sasa, rsa, surface
    B.write_csv(os.path.join(out_dir, f"{tag}_sasa_resis.csv"), srows, SASA_FIELDS)
    centroids = [r["res_id"] for r in srows if r["surface"]]   # centroids = every exposed residue; each one
    #  seeds exactly one motif, and its number becomes the motif number (her2_m568 = built around residue 568)

    # 2. one motif per surface centroid: centroid + its Ca/Cb contacts within nbhd_cutoff
    base = B.motif_rows(pdb, chain, centroids, nbhd_cutoff, nbhd_metric)   # base = one motif per centroid
    filled = B.filled_rows(pdb, chain, base, gap_merge) if emit_filled else []   # filled = the same patches
    #  with short sequence gaps merged; only motifs that actually change get a filled twin

    # 3. descriptors (SSE fractions, radius of gyration, linearity, planarity, convexity, burial)
    base = B.add_descriptors(pdb, chain, base, rsa_cutoff)
    if filled:
        filled = B.add_descriptors(pdb, chain, filled, rsa_cutoff)

    src = os.path.basename(pdb)                          # src = provenance stamped into every motif row
    for r in base:
        r.update(pdb_name=tag, variant="base", gap_merge="", source_pdb=src, chain=chain)
    for r in filled:
        r.update(pdb_name=tag, variant="filled", gap_merge=gap_merge, source_pdb=src, chain=chain)
    B.write_csv(os.path.join(out_dir, f"{tag}_c_motif.csv"), base + filled, MOTIF_FIELDS)

    # 4. the motif template PDBs the generators scaffold (tar.zst — inode budget)
    pdb_dir = os.path.join(out_dir, "motif_pdbs")
    n_b = B.write_motif_pdbs(pdb, chain, base, pdb_dir, archive=archive, name="c_motif")        # n_b/n_f =
    n_f = B.write_motif_pdbs(pdb, chain, filled, pdb_dir, archive=archive, name="c_motif_filled") if filled else 0
    #  template PDBs written per variant — these are the files the generators actually scaffold

    return {"n_residues": len(srows), "n_surface": len(centroids), "n_base": len(base),
            "n_filled": len(filled), "n_pdb_base": n_b, "n_pdb_filled": n_f, "out_dir": out_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="proteins/<t>/target.yaml (supplies every other option)")
    ap.add_argument("--pdb"); ap.add_argument("--chain"); ap.add_argument("--tag")
    ap.add_argument("--out-dir")
    ap.add_argument("--screen", action="store_true", help="report chains/SSE and write nothing")
    ap.add_argument("--rsa-cutoff", type=float, default=None, help="surface = RSA > this (default 0.20)")
    ap.add_argument("--neighborhood-cutoff", type=float, default=None, help="motif = centroid + contacts within this A (default 7.0)")
    ap.add_argument("--neighborhood-metric", default=None, choices=[None, "ca", "cb"])
    ap.add_argument("--gap-merge", type=int, default=None, help="filled variant: merge gaps <= this (default 4)")
    ap.add_argument("--no-filled", action="store_true", help="skip the filled variant")
    ap.add_argument("--no-archive", action="store_true", help="leave motif PDBs loose instead of tar.zst")
    a = ap.parse_args(argv)

    cfg = load_target(a.target) if a.target else {}
    pdb = a.pdb or cfg.get("pdb")
    if not pdb:
        ap.error("need --pdb or a --target yaml with a `pdb:` key")
    if not os.path.exists(pdb):
        sys.exit(f"structure not found: {pdb}")

    if a.screen:
        screen(pdb)
        return

    chain = a.chain or cfg.get("chain")
    tag = a.tag or cfg.get("target")
    if not chain or not tag:
        ap.error("need --chain/--tag (or a target yaml with `chain:` and `target:`)")
    out_dir = a.out_dir or (os.path.dirname(os.path.abspath(a.target)) if a.target else f"proteins/{tag}")

    def pick(flag, key, default, cast):
        """Precedence for every knob: command-line flag > target.yaml > built-in default."""
        if flag is not None:
            return flag
        return cast(cfg[key]) if key in cfg else default

    emit_filled = (not a.no_filled) and str(cfg.get("emit_filled", True)).lower() not in ("false", "0", "no")
    s = extract(pdb, chain, tag, out_dir,                 # s = the run summary counts printed below
                rsa_cutoff=pick(a.rsa_cutoff, "rsa_cutoff", 0.20, float),
                nbhd_cutoff=pick(a.neighborhood_cutoff, "neighborhood_cutoff", 7.0, float),
                nbhd_metric=pick(a.neighborhood_metric, "neighborhood_metric", "ca", str).lower(),
                gap_merge=pick(a.gap_merge, "gap_merge", 4, int),
                emit_filled=emit_filled, archive=not a.no_archive)

    print(f"\n{tag} chain {chain}  ({os.path.basename(pdb)})")
    print(f"  surface   : {s['n_surface']}/{s['n_residues']} residues (RSA > cutoff) -> motif centroids")
    print(f"  motifs    : {s['n_base']} base + {s['n_filled']} filled")
    print(f"  templates : {s['n_pdb_base']} base + {s['n_pdb_filled']} filled PDBs")
    print(f"  -> {s['out_dir']}/{{{tag}_sasa_resis.csv, {tag}_c_motif.csv, motif_pdbs/}}")
    print(f"  next: build_pool.py  (or point stage_motif.py --lib at the proteins/ root)")


if __name__ == "__main__":
    main()
