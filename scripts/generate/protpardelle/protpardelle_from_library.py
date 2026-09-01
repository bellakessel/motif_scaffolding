#!/usr/bin/env python3
"""protpardelle_from_library.py — turn the motif library into Protpardelle-1c motif-scaffolding jobs.

Host-side driver (gen_env, Euler login node), mirrors proteina/la-proteina drivers. For each motif in the
deduped pool (or a single named motif for the smoke test) it:
  1. stages the motif's template PDB from {lib}/{pdb}/motif_templates/c_motif[_filled].tar.zst into the SHARED
     <run>/motifs/<task>.pdb  (Protpardelle's --motif-dir; motif referenced by file stem == task_name),
  2. builds the RFdiffusion-grammar contig via esp.contig — the canonical builder — then converts it to
     Protpardelle syntax: '/' segment separator -> ';' (Protpardelle uses ';' between segments and ';/;' for
     a chain break; our single-chain motifs have no chain break, so a plain '/'->';' is exact),
  3. writes ONE Protpardelle hydra sampling config per motif, `<run>/<method>/configs/inf_<N>.yaml`, with
     single-element parallel lists (motifs / motif_contigs / total_lengths / hotspots / ssadj) — sample.py
     uses the config file STEM as run_name, so inf_<N> namespaces the output,
  4. records a manifest row (config_number -> pdb_name, motif_num, variant, contig, task_name) = full provenance.

Then `protpardelle_generate.sbatch` runs a SLURM array: task i -> config inf_<i>.yaml.

Two benchmark sub-methods share this driver via --model:
  cc58 (backbone + sidechain-conditional)  -> method dir `pp1c_bb`; poly-Gly backbone + native motif stamped
       -> shared MPNN->AF3 downstream (like Proteina).
  cc91 (all-atom crop-conditional)         -> method dir `pp1c_aa`; co-designed sequence embedded in the PDB
       -> extract-seq->AF3 downstream (like La-Proteina).
Both use sidechain conditioning, which reads the motif sidechain atoms our all-atom templates already carry.
The scaffold-length policy (--flank-n/--flank-c/--linker) and --nsamples are held IDENTICAL across all methods.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from typing import List

try:
    from motifgo import contig as contigmod
    from motifgo import motif_build as B
except Exception as e:
    raise SystemExit(f"cannot import motifgo.contig ({e}); install the repo first:\n"
            f"  pip install -e <motif_benchmark> --no-deps   # or export PYTHONPATH=<motif_benchmark>")

WICKY = "/cluster/work/wicky"
DEF_LIB = f"{WICKY}/bekessel/motif_library"
DEF_RUNS_ROOT = f"{DEF_LIB}/runs"
DEF_REPO = f"{WICKY}/bekessel/software/protpardelle-1c"

# model -> (method dir, [model_name, epoch, running-sampling-config]). The triples are the repo's own
# RFdiffusion/La-Proteina benchmark configs (examples/sampling/07_rfdiffusion.yaml & 08_rfdiffusion_allatom.yaml).
MODELS = {
    "cc58": ("pp1c_bb", ["cc58", "416", "sampling_sidechain_conditional"]),          # backbone -> MPNN
    "cc91": ("pp1c_aa", ["cc91", "383", "sampling_sidechain_conditional_allatom_s1"]),  # all-atom co-design
}


def to_pp_contig(esp_contig: str) -> str:
    """esp.contig ('40-70/C570-580/40-70', '/1-5/' linkers) -> Protpardelle (';'-delimited)."""
    return esp_contig.replace("/", ";")


def write_config(config_dir, idx, *, task, pp_contig, model_triple, min_len, max_len,
                 step_scale, schurn, cc_start) -> str:
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, f"inf_{idx}.yaml")
    mt = json.dumps(model_triple)                              # ["cc58", "416", "sampling_sidechain_conditional"]
    txt = (
        "search_space:\n"
        f"  models:\n    - {mt}\n"
        f"  step_scales: [{step_scale}]\n"
        f"  schurns: [{schurn}]\n"
        f"  crop_cond_starts: [{cc_start}]\n"
        "  translations:\n    - [0.0, 0.0, 0.0]\n"
        f"motifs:\n  - {task}\n"                                # <motif-dir>/<task>.pdb
        f"motif_contigs:\n  - \"{pp_contig}\"\n"
        f"total_lengths:\n  - [[{min_len}, {max_len}]]\n"
        "hotspots:\n  - null\n"
        "ssadj:\n  - null\n"
    )
    with open(path, "w") as fh:
        fh.write(txt)
    return path


def iter_manifest(manifest_csv):
    """Rows of a pre-built shared manifest (stage_contigscale.py): consume contig/length/motif_pdb verbatim.
    The shared contig is in RFdiffusion '/' grammar; to_pp_contig() converts it to Protpardelle's ';' syntax."""
    with open(manifest_csv) as fh:
        yield from csv.DictReader(fh)


def build(rows, a) -> List[dict]:
    manifest = []
    for i, r in enumerate(rows):
        pdb_name = r["pdb_name"]; motif_num = int(r["motif_num"]); variant = r.get("variant", "base")
        chain = r["chain"]
        if a.from_manifest:                          # consume pre-built contig/length/motif_pdb verbatim
            task = r["task_name"]
            pp_contig = to_pp_contig(r["contig"]) if "/" in r["contig"] else r["contig"]
            min_len, max_len = int(r["motif_min_length"]), int(r["motif_max_length"])
            motif_pdb = r["motif_pdb"]; n_frag = int(r["n_fragments"])
            nsamples = int(r["nsamples"]) if r.get("nsamples") else a.nsamples
        else:
            residues = [int(x) for x in str(r["motif_residues"]).split()]
            fragments = contigmod.group_runs(residues)
            esp_contig = contigmod.contig_string(chain, fragments, a.flank_n, a.flank_c, a.linker)
            pp_contig = to_pp_contig(esp_contig)
            min_len, max_len = contigmod.length_bounds(fragments, a.flank_n, a.flank_c, a.linker)
            task = f"{pdb_name}_m{motif_num}_{variant}"
            if r.get("pdb_src"):
                import shutil
                os.makedirs(a.motifs_dir, exist_ok=True)
                motif_pdb = os.path.join(a.motifs_dir, f"{task}.pdb")
                shutil.copyfile(r["pdb_src"], motif_pdb)
            else:
                motif_pdb = B.stage_motif_pdb(a.lib, pdb_name, motif_num, variant, a.motifs_dir, task)
            n_frag = len(fragments); nsamples = a.nsamples
        write_config(a.config_dir, i, task=task, pp_contig=pp_contig, model_triple=a.model_triple,
                     min_len=min_len, max_len=max_len, step_scale=a.step_scale, schurn=a.schurn,
                     cc_start=a.cc_start)
        manifest.append({"config_number": i, "task_name": task, "pdb_name": pdb_name,
                         "motif_num": motif_num, "variant": variant, "chain": chain,
                         "n_fragments": n_frag, "contig": pp_contig,
                         "motif_min_length": min_len, "motif_max_length": max_len,
                         "nsamples": nsamples, "model": a.model, "motif_pdb": motif_pdb})
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="deduped motif pool CSV (bulk mode)")
    src.add_argument("--manifest", help="pre-built shared manifest CSV (e.g. contig-scale run): consume its "
                                        "contig/length/motif_pdb verbatim, skip pool + contig_string")
    src.add_argument("--single", nargs=4, metavar=("PDB_NAME", "MOTIF_NUM", "VARIANT", "CHAIN"),
                     help="one motif (smoke), e.g. 1n8z_pc 570 base C — needs --residues")
    ap.add_argument("--residues", help="space/comma-separated residue ids for --single")
    ap.add_argument("--pdb", help="direct motif PDB for --single (bypasses archive extraction)")
    ap.add_argument("--model", choices=list(MODELS), default="cc58",
                    help="cc58 = backbone+sidechain-cond -> pp1c_bb (MPNN downstream); "
                         "cc91 = all-atom co-design -> pp1c_aa (extract-seq downstream)")
    ap.add_argument("--run-dir", default=None, help="explicit run dir (overrides --run/--runs-root)")
    ap.add_argument("--run", default=None, help="run NAME -> <runs-root>/<run>")
    ap.add_argument("--runs-root", default=DEF_RUNS_ROOT)
    ap.add_argument("--variants", default="base,filled")
    ap.add_argument("--lib", default=DEF_LIB)
    ap.add_argument("--repo", default=DEF_REPO, help="Protpardelle-1c repo (scripts + model_params live here)")
    # scaffold-length policy (held IDENTICAL across all benchmark methods for fairness)
    ap.add_argument("--flank-n", default="2-10")
    ap.add_argument("--flank-c", default="35-75")
    ap.add_argument("--linker", default="1-5")
    ap.add_argument("--nsamples", type=int, default=100)
    # Protpardelle sampling knobs (repo defaults from 07/08 RFdiffusion benchmark configs)
    ap.add_argument("--step-scale", type=float, default=1.2)
    ap.add_argument("--schurn", type=float, default=200)
    ap.add_argument("--cc-start", type=float, default=0.0)
    a = ap.parse_args(argv)

    a.method, a.model_triple = MODELS[a.model]
    a.from_manifest = bool(a.manifest)
    manifest_in = a.manifest                                   # the pre-built shared manifest we consume (input)
    if a.run_dir:
        a.run_dir = os.path.abspath(a.run_dir)
    elif a.run:
        a.run_dir = os.path.join(a.runs_root, a.run)
    else:
        ap.error("give --run-dir /full/path OR --run <name> [--runs-root <root>]")
    a.method_dir = os.path.join(a.run_dir, a.method)
    a.config_dir = os.path.join(a.method_dir, "configs")
    a.motifs_dir = os.path.join(a.run_dir, "motifs")           # SHARED motif template PDBs (staged once)
    a.manifest = os.path.join(a.method_dir, "manifest.csv")    # per-method (model differs; motifs shared) — never the shared one

    if a.single:
        if not a.residues:
            ap.error("--single requires --residues")
        pdb_name, motif_num, variant, chain = a.single
        rows = [{"pdb_name": pdb_name, "motif_num": motif_num, "variant": variant, "chain": chain,
                 "motif_residues": a.residues.replace(",", " "), "pdb_src": a.pdb}]
    elif a.from_manifest:
        rows = list(iter_manifest(manifest_in))
    else:
        rows = list(B.iter_pool(a.pool, a.variants.split(",")))

    manifest = build(rows, a)

    os.makedirs(a.method_dir, exist_ok=True)
    fields = ["config_number", "task_name", "pdb_name", "motif_num", "variant", "chain",
              "n_fragments", "contig", "motif_min_length", "motif_max_length", "nsamples", "model", "motif_pdb"]
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(manifest)
    with open(os.path.join(a.method_dir, "run.yaml"), "w") as fh:
        fh.write(f"# protpardelle_from_library.py — {a.method} ({a.model})\n")
        for k in ("run_dir", "model", "method", "pool", "variants", "nsamples",
                  "flank_n", "flank_c", "linker", "step_scale", "schurn", "cc_start"):
            v = getattr(a, k, None)
            if v is not None:
                fh.write(f"{k}: {json.dumps(v)}\n")

    n = len(manifest)
    print(f"run dir : {a.run_dir}   (model {a.model} -> method {a.method})")
    print(f"  configs : {a.config_dir}/inf_0..{n-1}.yaml")
    print(f"  motifs  : {a.motifs_dir}/  (shared, {n} PDBs)")
    print(f"  manifest: {a.manifest}")
    print(f"  -> scaffolds land under {a.method_dir}/scaffolds/inf_<N>/<suffix>/<task>/")
    print(f"\nsubmit as a SLURM array (one task per motif):")
    print(f"  sbatch --array=0-{n-1}%20 --export=ALL,RUN_DIR={a.run_dir},METHOD={a.method} "
          f"{a.repo}/scripts/protpardelle_generate.sbatch")


if __name__ == "__main__":
    main()
