#!/usr/bin/env python3
"""stage_motif.py — shared motif stager for the multi-method scaffolding benchmark.

ONE front-end that turns the deduped motif pool into the shared run contract every method reads:

    <run>/motifs/<task>.pdb      all-atom motif template PDBs (staged once from motif_templates/*.tar.zst)
    <run>/manifest.csv           one row per motif: task_name, provenance, fragments, contig, length range

Both artifacts are METHOD-AGNOSTIC — Proteina / La-Proteina / Protpardelle / RFdiffusion (esp) /
Genie3 all consume the SAME <run>/motifs + manifest so every method scaffolds the IDENTICAL motif
under the IDENTICAL scaffold-length policy (fairness). The contig is built with esp.contig — the
canonical RFdiffusion-grammar builder the pipeline already uses. `config_number` == SLURM array index
== the same motif across every tool.

This is the shared half of proteina_from_library.py, factored out and with a `fragments` column added
(chain-relative start-end runs) so the Genie3 problem-set converter and the RFdiffusion adapter can read
the motif segments without re-parsing residues.

    export PYTHONPATH=<motif_benchmark repo>   # or: pip install -e <repo> --no-deps
    python stage_motif.py --pool <pool.csv> --run-dir <run> [--variants base] [--nsamples 100]
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List

try:
    from motifgo import contig as contigmod
    from motifgo import motif_build as B
except Exception as e:  # pragma: no cover
    sys.exit(f"cannot import motifgo.contig ({e}); install the repo first:\n"
            f"  pip install -e <motif_benchmark> --no-deps   # or export PYTHONPATH=<motif_benchmark>")

WICKY = "/cluster/work/wicky"
DEF_LIB = f"{WICKY}/bekessel/motif_library"

MANIFEST_FIELDS = ["config_number", "task_name", "pdb_name", "motif_num", "variant", "chain",
                   "n_fragments", "fragments", "contig", "motif_min_length", "motif_max_length",
                   "nsamples", "motif_pdb"]


def stage(rows, *, lib: str, motifs_dir: str, flank_n: str, flank_c: str, linker: str,
          nsamples: int) -> List[dict]:
    """Stage every motif PDB and build its manifest row. Deterministic: same output as each method's
    own driver, so the shared motifs/ + manifest are byte-identical regardless of who wrote them."""
    manifest = []
    for i, r in enumerate(rows):                         # i becomes config_number == the SLURM array
        #  index, so the same motif is task i in EVERY method. That identity is the fairness contract.
        pdb_name = r["pdb_name"]
        motif_num = int(r["motif_num"])
        variant = r.get("variant", "base")
        chain = r["chain"]
        residues = [int(x) for x in str(r["motif_residues"]).split()]   # residues = the motif, native numbering
        fragments = contigmod.group_runs(residues)       # fragments = its unbroken stretches
        cstr = contigmod.contig_string(chain, fragments, flank_n, flank_c, linker)   # cstr = the build order
        min_len, max_len = contigmod.length_bounds(fragments, flank_n, flank_c, linker)   # total scaffold size
        task_name = f"{pdb_name}_m{motif_num}_{variant}"  # e.g. her2_m568_base — the join key everywhere
        if r.get("pdb_src"):                        # direct PDB (smoke / positive control) — copy, don't extract
            import shutil
            os.makedirs(motifs_dir, exist_ok=True)
            motif_pdb = os.path.join(motifs_dir, f"{task_name}.pdb")
            shutil.copyfile(r["pdb_src"], motif_pdb)
        else:
            motif_pdb = B.stage_motif_pdb(lib, pdb_name, motif_num, variant, motifs_dir, task_name)
        manifest.append({
            "config_number": i, "task_name": task_name, "pdb_name": pdb_name,
            "motif_num": motif_num, "variant": variant, "chain": chain,
            "n_fragments": len(fragments),
            "fragments": ";".join(f"{a}-{b}" for a, b in fragments),
            "contig": cstr, "motif_min_length": min_len, "motif_max_length": max_len,
            "nsamples": nsamples, "motif_pdb": motif_pdb,
        })
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="deduped motif pool CSV (bulk mode)")
    src.add_argument("--single", nargs=4, metavar=("PDB_NAME", "MOTIF_NUM", "VARIANT", "CHAIN"),
                     help="one motif (smoke), e.g. 1n8z_pc 570 base C — needs --residues")
    ap.add_argument("--residues", help="space/comma residue ids for --single")
    ap.add_argument("--pdb", help="direct motif PDB for --single (bypasses archive extraction)")
    ap.add_argument("--run-dir", required=True,
                    help="run dir; writes <run>/motifs/ + <run>/manifest.csv (shared across all methods)")
    ap.add_argument("--lib", default=DEF_LIB, help="motif_library root ({pdb}/motif_templates/*.tar.zst)")
    ap.add_argument("--variants", default="base,filled", help="which variants to include in bulk mode")
    # scaffold-length policy (held IDENTICAL across all benchmark methods for fairness)
    ap.add_argument("--flank-n", default="2-10")
    ap.add_argument("--flank-c", default="35-75")
    ap.add_argument("--linker", default="1-5")
    ap.add_argument("--nsamples", type=int, default=100)
    a = ap.parse_args(argv)

    run_dir = os.path.abspath(a.run_dir)
    motifs_dir = os.path.join(run_dir, "motifs")
    manifest_csv = os.path.join(run_dir, "manifest.csv")

    if a.single:
        if not a.residues:
            ap.error("--single requires --residues")
        pdb_name, motif_num, variant, chain = a.single
        rows = [{"pdb_name": pdb_name, "motif_num": motif_num, "variant": variant, "chain": chain,
                 "motif_residues": a.residues.replace(",", " "), "pdb_src": a.pdb}]
    else:
        rows = list(B.iter_pool(a.pool, a.variants.split(",")))
    if not rows:
        ap.error("no motifs selected (check --pool / --variants)")

    os.makedirs(motifs_dir, exist_ok=True)
    manifest = stage(rows, lib=a.lib, motifs_dir=motifs_dir, flank_n=a.flank_n, flank_c=a.flank_c,
                     linker=a.linker, nsamples=a.nsamples)
    # lineterminator="\n": the rfdiff sbatch parses this manifest in shell; CRLF (csv default) leaves a
    # trailing '\r' on the last field and breaks its `[ -f motif_pdb ]` check.
    with open(manifest_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)

    n = len(manifest)
    print(f"staged {n} motifs")
    print(f"  motifs   : {motifs_dir}/  ({n} PDBs)")
    print(f"  manifest : {manifest_csv}")
    print(f"  policy   : flank_n={a.flank_n} flank_c={a.flank_c} linker={a.linker} nsamples={a.nsamples}")
    print(f"  array    : 0-{n-1}   (config_number == array index == same motif across every method)")
    print("methods now read <run>/motifs + <run>/manifest.csv:")
    print("  proteina / laproteina / pp1c_bb : their *_from_library.py (or read the shared manifest)")
    print("  rfdiff  : rfdiff_generate.sbatch  (contigmap.contigs=[<contig>], inference.input_pdb=<motif_pdb>)")
    print("  genie3  : genie3_from_library.py  (manifest -> MotifBench problems/ + REMARK-999 motifs/)")


if __name__ == "__main__":
    main()
