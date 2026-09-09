#!/usr/bin/env python3
"""proteina_from_library.py — turn the motif library into Proteina motif-scaffolding jobs.

Host-side driver (run in gen_env on the Euler login node), mirrors genie3_from_run.py. For each motif
in the deduped pool (or a single named motif for the smoke test) it:
  1. extracts the motif's template PDB from {lib}/{pdb}/motif_templates/c_motif[_filled].tar.zst to a
     staging dir (inode-safe: one loose PDB per task, under $SCRATCH),
  2. builds the RFdiffusion-grammar contig via esp.contig — the canonical builder; Proteina consumes the
     SAME grammar ("config grammar inherited from RFDiffusion", README) and a motif-only PDB (motif_only:
     True), which our motif_templates/*.pdb already are,
  3. writes one Proteina hydra config `{config_dir}/inf_<N>.yaml` (the repo's `--config_number N` selects
     `inf_{N}`, motif_inference.py line ~290),
  4. records a manifest row (config_number -> pdb_name, motif_num, variant, contig, task_name, out_dir),
     so every generated design is traceable to source PDB / motif / variant.

Then `proteina_generate.sbatch` runs as a SLURM array: task i -> `--config_number i`.

NOTE (verify in the smoke test): this assumes Proteina resolves `--config_number N` to `inf_N.yaml` in the
hydra search dir (== configs/experiment_config, where inference_motif.yaml lives). If the repo initializes
hydra on a different dir, point --config-dir there (and/or pass --config_subdir in the sbatch).
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List

# esp is the canonical contig builder (same grammar Proteina consumes). Put the esp pipeline on PYTHONPATH.
try:
    from motifgo import contig as contigmod
    from motifgo import motif_build as B
except Exception as e:  # pragma: no cover - helpful message on the cluster
    sys.exit(f"cannot import motifgo.contig ({e}); install the repo first:\n"
            f"  pip install -e <motif_benchmark> --no-deps   # or export PYTHONPATH=<motif_benchmark>")

WICKY = "/cluster/work/wicky"
DEF_LIB = f"{WICKY}/bekessel/motif_library"
DEF_RUNS_ROOT = f"{DEF_LIB}/runs"                 # benchmark runs live WITH the library, NOT in the repo
DEF_REPO = f"{WICKY}/bekessel/software/proteina"  # source + scripts + checkpoints only
DEF_CKPT_DIR = f"{DEF_REPO}/checkpoints"
DEF_CKPT_NAME = "proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt"
METHOD = "proteina"

# Sampling / solver defaults copied verbatim from configs/experiment_config/inference_motif.yaml so each
# generated config is self-contained. Generation-only: designability/fid OFF (we run our own MPNN->AF3).
SAMPLING_BLOCK = """\
# Do not change
nres_lens: null
min_len:
max_len:
step_len:

dt: 0.0025
self_cond: False

sampling_caflow:
  sampling_mode: sc
  sc_scale_noise: 0.4
  sc_scale_score: 1.0
  gt_mode: "1/t"
  gt_p: 1.0
  gt_clamp_val: null

schedule:
  schedule_mode: log
  schedule_p: 2.0

fold_cond: False
cath_code_level: "T"
len_cath_code_path: null

guidance_weight: 1.0
autoguidance_ratio: 1.0
autoguidance_ckpt_path: null

designability_seqs_per_struct: 8
compute_designability: False
compute_fid: False
"""


def write_config(config_dir: str, idx: int, *, task_name: str, contig: str, motif_pdb: str,
                 seg_order: str, min_len: int, max_len: int, nsamples: int, per_batch: int,
                 ckpt_dir: str, ckpt_name: str, seed: int, ncpus: int, motif_only: bool = True) -> str:
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, f"inf_{idx}.yaml")
    head = (
        f'run_name_: {task_name}\n'
        f'ckpt_path: {ckpt_dir}\n'
        f'ckpt_name: {ckpt_name}\n\n'
        f'ncpus_: {ncpus}\n'
        f'seed: {seed}\n\n'
        f'nsamples: {nsamples}\n'
        f'max_nsamples_per_batch: {per_batch}\n'
        f'contig_string: "{contig}"\n'
        f'motif_pdb_path: "{motif_pdb}"\n'
        f'motif_task_name: "{task_name}"\n'
        f'motif_only: {motif_only}\n'
        f'motif_min_length: {min_len}\n'
        f'motif_max_length: {max_len}\n'
        f'segment_order: "{seg_order}"\n\n'
    )
    with open(path, "w") as fh:
        fh.write(head + SAMPLING_BLOCK)
    return path


def iter_manifest(manifest_csv: str):
    """Rows of a pre-built shared manifest (stage_contigscale.py / stage_motif.py). Each row already
    carries task_name, contig, motif_min/max_length, nsamples and the staged motif_pdb — so build() uses
    those verbatim (per-gap native contigs the pool-based contig_string can't express) and skips staging."""
    with open(manifest_csv) as fh:
        yield from csv.DictReader(fh)


def build(rows, a) -> List[dict]:
    manifest = []
    for i, r in enumerate(rows):
        pdb_name = r["pdb_name"]
        motif_num = int(r["motif_num"])
        variant = r.get("variant", "base")
        chain = r["chain"]
        if a.from_manifest:                         # consume pre-built contig/length/motif_pdb verbatim
            task_name = r["task_name"]
            contig = r["contig"]
            min_len, max_len = int(r["motif_min_length"]), int(r["motif_max_length"])
            motif_pdb = r["motif_pdb"]
            n_frag = int(r["n_fragments"])
            nsamples = int(r["nsamples"]) if r.get("nsamples") else a.nsamples
        else:
            residues = [int(x) for x in str(r["motif_residues"]).split()]
            fragments = contigmod.group_runs(residues)
            contig = contigmod.contig_string(chain, fragments, a.flank_n, a.flank_c, a.linker)
            min_len, max_len = contigmod.length_bounds(fragments, a.flank_n, a.flank_c, a.linker)
            task_name = f"{pdb_name}_m{motif_num}_{variant}"
            if r.get("pdb_src"):                    # direct PDB (smoke test / positive control) — copy, don't extract
                import shutil
                os.makedirs(a.motifs_dir, exist_ok=True)
                motif_pdb = os.path.join(a.motifs_dir, f"{task_name}.pdb")
                shutil.copyfile(r["pdb_src"], motif_pdb)
            else:
                motif_pdb = B.stage_motif_pdb(a.lib, pdb_name, motif_num, variant, a.motifs_dir, task_name)
            n_frag = len(fragments)
            nsamples = a.nsamples
        write_config(a.config_dir, i, task_name=task_name, contig=contig, motif_pdb=motif_pdb,
                     seg_order=chain, min_len=min_len, max_len=max_len, nsamples=nsamples,
                     per_batch=a.per_batch, ckpt_dir=a.ckpt_dir, ckpt_name=a.ckpt_name,
                     seed=a.seed, ncpus=a.ncpus, motif_only=a.motif_only)
        manifest.append({"config_number": i, "task_name": task_name, "pdb_name": pdb_name,
                         "motif_num": motif_num, "variant": variant, "chain": chain,
                         "n_fragments": n_frag, "contig": contig,
                         "motif_min_length": min_len, "motif_max_length": max_len,
                         "nsamples": nsamples, "motif_pdb": motif_pdb})
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="deduped motif pool CSV (bulk mode)")
    src.add_argument("--manifest", help="pre-built shared manifest CSV (e.g. contig-scale run): consume its "
                                        "contig/length/motif_pdb verbatim, skip pool + contig_string")
    src.add_argument("--single", nargs=4, metavar=("PDB_NAME", "MOTIF_NUM", "VARIANT", "CHAIN"),
                     help="one motif (smoke test), e.g. 1n8z_pc 570 base C — needs --residues")
    ap.add_argument("--residues", help="space/comma-separated residue ids for --single mode")
    ap.add_argument("--pdb", help="direct motif PDB for --single mode (bypasses archive extraction; "
                                  "use for the 1n8z_pc positive control)")
    ap.add_argument("--run-dir", default=None,
                    help="explicit full path of the run dir (e.g. /cluster/.../motif_library/runs/lib_v1); "
                         "overrides --run/--runs-root. Layout inside: {motifs,manifest.csv,run.yaml} shared + "
                         "proteina/{configs,scaffolds,mpnn,af3,logs} per tool")
    ap.add_argument("--run", default=None, help="run NAME -> <runs-root>/<run>/ (convenience alt to --run-dir)")
    ap.add_argument("--runs-root", default=DEF_RUNS_ROOT, help="parent for --run (default motif_library/runs)")
    ap.add_argument("--variants", default="base,filled", help="which variants to include in bulk mode")
    ap.add_argument("--lib", default=DEF_LIB, help="motif_library root ({pdb}/motif_templates/*.tar.zst)")
    ap.add_argument("--repo", default=DEF_REPO, help="Proteina source repo (code only; no run output here)")
    ap.add_argument("--ckpt-dir", default=DEF_CKPT_DIR)
    ap.add_argument("--ckpt-name", default=DEF_CKPT_NAME)
    # scaffold-length policy (held IDENTICAL across all benchmark methods for fairness)
    ap.add_argument("--flank-n", default="2-10")
    ap.add_argument("--flank-c", default="35-75")
    ap.add_argument("--linker", default="1-5")
    ap.add_argument("--nsamples", type=int, default=100)
    ap.add_argument("--per-batch", type=int, default=10)
    ap.add_argument("--motif-only", choices=["true", "false"], default="true",
                    help="Proteina motif_only: true = the PDB is motif-only (our template PDBs / library "
                         "default); false = the PDB is the full protein and the contig selects the motif")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ncpus", type=int, default=24)
    a = ap.parse_args(argv)

    # run-dir layout: <run_dir>/{motifs,manifest.csv} shared + <run_dir>/proteina/{configs,scaffolds,...}
    if a.run_dir:
        a.run_dir = os.path.abspath(a.run_dir)
    elif a.run:
        a.run_dir = os.path.join(a.runs_root, a.run)
    else:
        ap.error("give --run-dir /full/path  OR  --run <name> [--runs-root <root>]")
    a.from_manifest = bool(a.manifest)
    manifest_in = a.manifest                                   # the pre-built shared manifest we consume (input)
    a.method_dir = os.path.join(a.run_dir, METHOD)
    a.config_dir = os.path.join(a.method_dir, "configs")
    a.motifs_dir = os.path.join(a.run_dir, "motifs")          # SHARED motif template PDBs (staged once)
    # manifest-mode consumes the pre-built SHARED manifest -> write our provenance per-method so we don't
    # clobber it (rfdiff/genie3 still read the shared <run>/manifest.csv the stager wrote).
    a.manifest = (os.path.join(a.method_dir, "manifest.csv") if a.from_manifest
                  else os.path.join(a.run_dir, "manifest.csv"))

    # Never overwrite a shared manifest the stager already wrote. stage_motif.py emits the canonical
    # schema (it alone carries `fragments`); a driver rebuilding it from --pool drops that column and
    # silently redefines the contract that rfdiff and genie3 read. Creating it when ABSENT is still
    # allowed, so "run a driver without the stager" keeps working.
    if not a.from_manifest and os.path.exists(a.manifest):
        shared, a.manifest = a.manifest, os.path.join(a.method_dir, "manifest.csv")
        print(f"[manifest] {shared} already exists (stager-written) -> keeping it;\n"
              f"[manifest] writing this driver's provenance to {a.manifest} instead")
    a.motif_only = (a.motif_only == "true")

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

    os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
    fields = ["config_number", "task_name", "pdb_name", "motif_num", "variant", "chain",
              "n_fragments", "contig", "motif_min_length", "motif_max_length", "nsamples", "motif_pdb"]
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)

    # write the resolved run definition alongside the run (self-documenting; "what we ran")
    import json
    with open(os.path.join(a.run_dir, "run.yaml"), "w") as fh:
        fh.write("# resolved by proteina_from_library.py — what this run scaffolds\n")
        for k in ("run_dir", "run", "pool", "variants", "nsamples", "flank_n", "flank_c", "linker",
                  "ckpt_dir", "ckpt_name"):
            v = getattr(a, k, None)
            if v is not None:
                fh.write(f"{k}: {json.dumps(v)}\n")

    n = len(manifest)
    print(f"run dir : {a.run_dir}")
    print(f"  configs : {a.config_dir}/inf_0..{n-1}.yaml")
    print(f"  motifs  : {a.motifs_dir}/  (shared, {n} PDBs)")
    print(f"  manifest: {a.manifest}")
    print(f"  -> scaffolds will land in {a.method_dir}/scaffolds/")
    print(f"\nsubmit as a SLURM array (one task per motif):")
    print(f"  sbatch --array=0-{n-1}%20 --export=ALL,RUN_DIR={a.run_dir} {a.repo}/scripts/proteina_generate.sbatch")
    print(f"then MPNN the scaffolds:")
    print(f"  sbatch --export=ALL,RUN_DIR={a.run_dir} {a.repo}/scripts/proteina_mpnn.sbatch")


if __name__ == "__main__":
    main()
