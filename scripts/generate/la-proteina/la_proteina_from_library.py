#!/usr/bin/env python3
"""la_proteina_from_library.py — turn the motif library into La-Proteina motif-scaffolding jobs.

Mirrors proteina/scripts/proteina_from_library.py, adapted to La-Proteina (LD4 indexed all-atom co-design):
  - motif input is the SAME RFdiffusion contig grammar (built via esp.contig) + motif-only PDB, wrapped in
    La-Proteina's `generation.dataset.motif_dict_cfg` schema (contig_string / motif_pdb_path / motif_only /
    motif_min_length / motif_max_length / segment_order / atom_selection_mode: all_atom).
  - one hydra config per motif, `inf_<runtag>_<N>.yaml`, written into the REPO `configs/` dir with
    `defaults: [inference_motif_idx_aa, _self_]` (La-Proteina's inference config uses hydra defaults, and
    generate.py resolves configs SCRIPT-relative from <repo>/configs, so the base configs must be reachable
    there). The generate sbatch selects it with `--config_name inf_<runtag>_<N>` and `cd`s into the run dir so
    output lands there. runtag = basename(run_dir) namespaces configs per run (clean them after the run).
  - shared run-dir artifacts (like proteina): <run>/motifs/ (staged PDBs, once) + <run>/manifest.csv + run.yaml.

La-Proteina CO-DESIGNS sequence+structure, so there is NO MPNN step; the co-designed sequence is later
extracted from the output PDBs for AF3/ESMFold2 refolding (la_proteina_extract_seqs.py).
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
DEF_REPO = f"{WICKY}/bekessel/software/la-proteina"
METHOD = "laproteina"


def write_config(config_dir, cfg_name, *, task, contig, motif_pdb, seg_order, min_len, max_len, nsamples,
                 ckpt_dir, base_cfg="inference_motif_idx_aa"):
    """La-Proteina hydra config: extend inference_motif_idx_aa (LD4) with this single motif task inline.
    Pins the LD4 + AE3 checkpoint paths ABSOLUTE (the shipped config uses CWD-relative
    ./checkpoints_laproteina, which breaks once the sbatch cd's into the run dir)."""
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, cfg_name + ".yaml")
    txt = (
        "defaults:\n"
        f"  - {base_cfg}\n"
        "  - _self_\n\n"
        f"ckpt_path: \"{ckpt_dir}\"\n"
        f"autoencoder_ckpt_path: \"{ckpt_dir}/AE3_motif.ckpt\"\n\n"
        "generation:\n"
        "  dataset:\n"
        f"    motif_task_name: \"{task}\"\n"
        f"    nsamples: {nsamples}\n"
        "    motif_dict_cfg:\n"
        f"      \"{task}\":\n"
        f"        contig_string: {contig}\n"
        f"        motif_pdb_path: \"{motif_pdb}\"\n"
        "        motif_only: True\n"
        f"        motif_min_length: {min_len}\n"
        f"        motif_max_length: {max_len}\n"
        f"        segment_order: \"{seg_order}\"\n"
        "        atom_selection_mode: \"all_atom\"\n"
    )
    with open(path, "w") as fh:
        fh.write(txt)
    return path


def iter_manifest(manifest_csv):
    """Rows of a pre-built shared manifest (stage_contigscale.py): consume contig/length/motif_pdb verbatim."""
    with open(manifest_csv) as fh:
        yield from csv.DictReader(fh)


def build(rows, a, runtag) -> List[dict]:
    manifest = []
    for i, r in enumerate(rows):
        pdb_name = r["pdb_name"]; motif_num = int(r["motif_num"]); variant = r.get("variant", "base")
        chain = r["chain"]
        if a.from_manifest:                          # consume pre-built contig/length/motif_pdb verbatim
            task = r["task_name"]; contig = r["contig"]
            min_len, max_len = int(r["motif_min_length"]), int(r["motif_max_length"])
            motif_pdb = r["motif_pdb"]; n_frag = int(r["n_fragments"])
            nsamples = int(r["nsamples"]) if r.get("nsamples") else a.nsamples
        else:
            residues = [int(x) for x in str(r["motif_residues"]).split()]
            fragments = contigmod.group_runs(residues)
            contig = contigmod.contig_string(chain, fragments, a.flank_n, a.flank_c, a.linker)
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
        cfg_name = f"inf_{runtag}_{i}"
        # real config lives in the RUN dir (provenance, cleanliness); symlink it into <repo>/configs so
        # hydra's `defaults: [inference_motif_idx_aa]` still resolves against the repo's base configs.
        write_config(a.run_cfg_dir, cfg_name, task=task, contig=contig, motif_pdb=motif_pdb,
                     seg_order=chain, min_len=min_len, max_len=max_len, nsamples=nsamples,
                     ckpt_dir=os.path.join(a.repo, "checkpoints_laproteina"),
                     base_cfg=f"inference_motif_{a.variant}_aa")
        _link = os.path.join(a.config_dir, cfg_name + ".yaml")
        if os.path.lexists(_link):
            os.remove(_link)
        os.symlink(os.path.abspath(os.path.join(a.run_cfg_dir, cfg_name + ".yaml")), _link)
        manifest.append({"idx": i, "config_name": cfg_name, "task_name": task, "pdb_name": pdb_name,
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
                     help="one motif (smoke), e.g. 1n8z_pc 570 base C — needs --residues")
    ap.add_argument("--residues", help="space/comma-separated residue ids for --single")
    ap.add_argument("--pdb", help="direct motif PDB for --single (bypasses archive extraction)")
    ap.add_argument("--run-dir", default=None, help="explicit run dir (overrides --run/--runs-root)")
    ap.add_argument("--run", default=None, help="run NAME -> <runs-root>/<run>")
    ap.add_argument("--runs-root", default=DEF_RUNS_ROOT)
    ap.add_argument("--variants", default="base,filled")
    ap.add_argument("--lib", default=DEF_LIB)
    ap.add_argument("--repo", default=DEF_REPO, help="La-Proteina repo (configs written to <repo>/configs)")
    ap.add_argument("--variant", choices=["idx", "uidx"], default="idx",
                    help="LD4 indexed (idx, default) or LD6 unindexed (uidx) all-atom scaffolding")
    ap.add_argument("--flank-n", default="2-10")
    ap.add_argument("--flank-c", default="35-75")
    ap.add_argument("--linker", default="1-5")
    ap.add_argument("--nsamples", type=int, default=100)
    a = ap.parse_args(argv)

    if a.run_dir:
        a.run_dir = os.path.abspath(a.run_dir)
    elif a.run:
        a.run_dir = os.path.join(a.runs_root, a.run)
    else:
        ap.error("give --run-dir /full/path OR --run <name> [--runs-root <root>]")
    a.from_manifest = bool(a.manifest)
    manifest_in = a.manifest                                 # the pre-built shared manifest we consume (input)
    runtag = os.path.basename(a.run_dir.rstrip("/"))
    a.config_dir = os.path.join(a.repo, "configs")          # repo: base configs + our symlinks (hydra reads here)
    a.run_cfg_dir = os.path.join(a.run_dir, "laproteina", "configs")   # REAL generated configs live in the run dir
    a.motifs_dir = os.path.join(a.run_dir, "motifs")
    # manifest-mode consumes the pre-built SHARED manifest -> write our provenance per-method so we don't
    # clobber it (rfdiff/genie3 still read the shared <run>/manifest.csv the stager wrote).
    a.manifest = (os.path.join(a.run_dir, "laproteina", "manifest.csv") if a.from_manifest
                  else os.path.join(a.run_dir, "manifest.csv"))

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

    manifest = build(rows, a, runtag)

    os.makedirs(a.run_dir, exist_ok=True)
    fields = ["idx", "config_name", "task_name", "pdb_name", "motif_num", "variant", "chain",
              "n_fragments", "contig", "motif_min_length", "motif_max_length", "nsamples", "motif_pdb"]
    # lineterminator="\n": the shared manifest.csv is shell-parsed by rfdiff_generate.sbatch; CRLF (csv
    # default) leaves a trailing '\r' on the last field and breaks its `[ -f motif_pdb ]` check.
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(manifest)
    with open(os.path.join(a.run_dir, "run.yaml"), "w") as fh:
        fh.write("# la_proteina_from_library.py — what this run scaffolds\n")
        for k in ("run_dir", "pool", "variants", "nsamples", "flank_n", "flank_c", "linker"):
            v = getattr(a, k, None)
            if v is not None:
                fh.write(f"{k}: {json.dumps(v)}\n")

    n = len(manifest)
    print(f"run dir : {a.run_dir}")
    print(f"  configs : {a.config_dir}/inf_{runtag}_0..{n-1}.yaml  (in the REPO configs, namespaced by run)")
    print(f"  motifs  : {a.motifs_dir}/  (shared, {n} PDBs)")
    print(f"  manifest: {a.manifest}")
    print(f"  -> scaffolds land in {a.run_dir}/{METHOD}/inference/  (cd run dir in the sbatch)")
    print(f"\nsubmit as a SLURM array (one task per motif):")
    print(f"  sbatch --gpus=rtx_4090:1 --array=0-{n-1}%20 "
          f"--export=ALL,RUN_DIR={a.run_dir},RUNTAG={runtag},REPO={a.repo} "
          f"{a.repo}/scripts/la_proteina_generate.sbatch")


if __name__ == "__main__":
    main()
