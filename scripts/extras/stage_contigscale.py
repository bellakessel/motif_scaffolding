#!/usr/bin/env python3
"""stage_contigscale.py — turn scale_contigs.py's scaled_contigs.csv into a shared benchmark run.

The contig-scaling experiment re-scaffolds the all-failure motifs with NATIVE-geometry contigs (per-gap
linker ranges + asym/sym flanks) instead of the uniform 1-5 contig the baseline benchmark used. scale_contigs.py
already emitted those contigs, two rows per motif (native_asym / native_sym), in a schema that is a superset of
the shared run manifest. This stager materializes that CSV into the SAME <run>/motifs + <run>/manifest.csv
contract every generator reads (stage_motif.py's contract), so all five methods scaffold the identical
motifs under the identical (native) scaffold policy:

    <new-run>/motifs/<task>.pdb   copied from the baseline run's motif template (same geometry; the contig,
                                  not the motif, is what changes). task = <base>__native_asym / __native_sym.
    <new-run>/manifest.csv        canonical shared schema (config_number == array index == same motif/method):
                                  config_number, task_name, pdb_name, motif_num, variant, chain, n_fragments,
                                  fragments, contig, motif_min_length, motif_max_length, nsamples, motif_pdb.

Then, unchanged: rfdiff_generate.sbatch + genie3_from_library.py read this manifest directly; the three
config-driven drivers (proteina / la-proteina / protpardelle) read it via their new `--manifest` mode.

Runs in gen_env (stdlib only). The scaled CSV's motif_pdb points at the baseline motifs/, which we copy from.

    python stage_contigscale.py \
        --scaled /cluster/work/wicky/bekessel/motif_library/runs/lib_interface_pad5/analysis/scaled_contigs.csv \
        --run-dir /cluster/work/wicky/bekessel/motif_library/runs/lib_interface_pad5_contigscale
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import sys

# Canonical shared manifest schema (must match stage_motif.py so rfdiff/genie3 read it unchanged).
MANIFEST_FIELDS = ["config_number", "task_name", "pdb_name", "motif_num", "variant", "chain",
                   "n_fragments", "fragments", "contig", "motif_min_length", "motif_max_length",
                   "nsamples", "motif_pdb"]

DEF_SCALED = ("/cluster/work/wicky/bekessel/motif_library/runs/lib_interface_pad5/"
              "analysis/scaled_contigs.csv")
DEF_RUN = "/cluster/work/wicky/bekessel/motif_library/runs/lib_interface_pad5_contigscale"


def fragments_from_contig(contig: str) -> str:
    """Chain-relative 'a-b;c-d' fragment runs from the RFdiffusion-grammar contig (drop the chain letter).
    A token is a motif fragment iff it starts with a non-digit (the chain letter); scaffold/linker range
    tokens (2-10, 1-5, 35-75) start with a digit and are skipped."""
    frags = []
    for tok in contig.split("/"):
        if tok and not tok[0].isdigit():          # e.g. E81-82  ->  81-82
            frags.append(tok[1:])
    return ";".join(frags)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaled", default=DEF_SCALED, help="scaled_contigs.csv from scale_contigs.py")
    ap.add_argument("--run-dir", default=DEF_RUN, help="new run dir to create (motifs/ + manifest.csv)")
    ap.add_argument("--nsamples", type=int, default=None,
                    help="override designs/task (default: keep the nsamples column from the scaled CSV)")
    a = ap.parse_args(argv)

    run_dir = os.path.abspath(a.run_dir)
    motifs_dir = os.path.join(run_dir, "motifs")
    manifest_csv = os.path.join(run_dir, "manifest.csv")
    os.makedirs(motifs_dir, exist_ok=True)

    with open(a.scaled) as fh:
        scaled = list(csv.DictReader(fh))
    if not scaled:
        sys.exit(f"[stage_contigscale] no rows in {a.scaled}")

    manifest, missing = [], []
    for i, r in enumerate(scaled):
        task = r["task_name"]                      # e.g. her2_m128_base__native_asym
        src_pdb = r["motif_pdb"]                    # baseline motifs/<base>.pdb (same geometry for both modes)
        if not os.path.isfile(src_pdb):
            missing.append((task, src_pdb)); continue
        dst_pdb = os.path.join(motifs_dir, f"{task}.pdb")
        shutil.copyfile(src_pdb, dst_pdb)
        manifest.append({
            "config_number": i, "task_name": task, "pdb_name": r["pdb_name"],
            "motif_num": r["motif_num"], "variant": r["variant"], "chain": r["chain"],
            "n_fragments": r["n_fragments"], "fragments": fragments_from_contig(r["contig"]),
            "contig": r["contig"], "motif_min_length": r["motif_min_length"],
            "motif_max_length": r["motif_max_length"],
            "nsamples": (a.nsamples if a.nsamples is not None else r["nsamples"]),
            "motif_pdb": dst_pdb,
        })

    # lineterminator="\n": rfdiff_generate.sbatch shell-parses this manifest; CRLF leaves a trailing '\r'.
    with open(manifest_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(manifest)

    n = len(manifest)
    print(f"[stage_contigscale] staged {n} tasks -> {run_dir}")
    print(f"  motifs   : {motifs_dir}/  ({n} PDBs, copied from baseline)")
    print(f"  manifest : {manifest_csv}")
    print(f"  array    : 0-{n-1}   (config_number == array index == same task across every method)")
    if missing:
        print(f"[stage_contigscale] MISSING {len(missing)} baseline motif PDBs: "
              + ", ".join(f"{t}({p})" for t, p in missing[:10]) + (" ..." if len(missing) > 10 else ""),
              file=sys.stderr)
    print("\nnext (all read <new-run>/manifest.csv):")
    print(f"  rfdiff  : sbatch --array=0-{n-1}%20 --export=ALL,RUN_DIR={run_dir} \\")
    print(f"              <motif_benchmark>/scripts/generate/rfdiff/rfdiff_generate.sbatch")
    print(f"  genie3  : python .../genie3/scripts/genie3_from_library.py --run-dir {run_dir}")
    print(f"  proteina/laproteina/protpardelle : their *_from_library.py --manifest {manifest_csv} --run-dir {run_dir}")


if __name__ == "__main__":
    main()
