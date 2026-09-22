#!/usr/bin/env python3
"""genie3_from_library.py — turn the shared run contract into a Genie3 MotifBench problem set.

Genie3 does not read the pool CSV; it consumes a MotifBench-format problem set (a `motifs/` dir of
REMARK-999-annotated PDBs + a `problems/` dir of JSON problem definitions). This converter reads the
SHARED contract that stage_motif.py produced —

    <run>/motifs/<task>.pdb     all-atom motif templates (same PDBs every method scaffolds)
    <run>/manifest.csv          task_name, chain, fragments, contig, motif_min/max_length, ...

— and writes, under <run>/genie3/:

    motifs/<task>.pdb           the motif PDB with a REMARK 999 header (NAME + one INPUT line/fragment)
    problems/<task>.json        {motif_filepaths, segment_config_str, min/max total length}
    experiment.yaml             Genie3 generate config (absolute paths; all problems; n_sample = nsamples)

so Genie3 scaffolds the IDENTICAL motifs, fragments and scaffold-length policy as every other method.
Segment ids in segment_config_str are A1..Ak in fragment (residue) order; the scaffold/linker ranges
are taken verbatim from the shared contig, so the length policy is guaranteed identical.

    # (no motifgo import needed here — this converter is self-contained)
    python genie3_from_library.py --run-dir <run> [--nsamples 100]
    # then (existing launch, all problems, sample-sharded):
    sbatch --array=0-9%5 --export=ALL,CONFIG=<run>/genie3/experiment.yaml genie3_generate.sbatch
"""
from __future__ import annotations
import argparse
import csv
import json
import os


def parse_contig(contig: str):
    """Split the shared RFdiffusion-grammar contig into (segment_config_str, fragments).

    `2-10/E565-570/1-5/E574-578/35-75`  ->  ("2-10,A1,1-5,A2,35-75", [("E",565,570),("E",574,578)])
    A token is a motif fragment iff it starts with the chain letter (non-digit); it becomes A1,A2,… in
    order. Numeric range tokens (flanks/linkers) are kept verbatim, so the length policy is identical.
    Derived from the contig (present in EVERY manifest) so it works whoever wrote manifest.csv."""
    cfg, frags, idx = [], [], 0
    for tok in contig.split("/"):
        if tok and tok[0].isdigit():          # scaffold / linker range (e.g. 2-10, 1-5, 35-75)
            cfg.append(tok)
        else:                                  # motif fragment (e.g. E565-570): chain + start-end
            idx += 1
            cfg.append(f"A{idx}")
            chain, rng = tok[0], tok[1:]
            a, b = rng.split("-")
            frags.append((chain, int(a), int(b)))
    return ",".join(cfg), frags


def remark_999(task_name: str, pdb_name: str, fragments) -> str:
    """Genie3 motif header: NAME + one `INPUT <chain> <start> <end>` per contiguous fragment (order = index).
    Genie3 parses FIXED COLUMNS (pdb_utils.parse_motif_pdb): chain=line[18], start=line[19:23], end=
    line[23:27] — so chain is 1 col and start/end are 4-wide fields with NO separators. Any extra space
    shifts the columns and mis-parses (e.g. a 113-115 fragment read as '3  1'). Match the layout exactly."""
    lines = [f"REMARK 999 NAME   {task_name}", f"REMARK 999 PDB    {pdb_name}"]
    for chain, a, b in fragments:
        lines.append(f"REMARK 999 INPUT  {chain}{a:>4d}{b:>4d}")   # cols: 16-17 pad, 18 chain, 19-22 start, 23-26 end
    return "\n".join(lines) + "\n"


def convert(run_dir: str, nsamples: int | None) -> int:
    run_dir = os.path.abspath(run_dir)
    manifest = os.path.join(run_dir, "manifest.csv")
    src_motifs = os.path.join(run_dir, "motifs")
    if not os.path.exists(manifest):
        raise SystemExit(f"no manifest at {manifest} — run stage_motif.py first")
    gdir = os.path.join(run_dir, "genie3")
    mdir, pdir = os.path.join(gdir, "motifs"), os.path.join(gdir, "problems")
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)

    n = 0
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            task = r["task_name"]
            seg_cfg, fragments = parse_contig(r["contig"])   # from contig -> robust to any manifest author
            src_pdb = r.get("motif_pdb") or os.path.join(src_motifs, f"{task}.pdb")
            if not os.path.exists(src_pdb):
                src_pdb = os.path.join(src_motifs, f"{task}.pdb")
            with open(src_pdb) as f:
                body = f.read()
            # sanity: every fragment's chain must exist in the PDB, else Genie3 finds no motif coords
            pdb_chains = {ln[21:22] for ln in body.splitlines() if ln.startswith(("ATOM", "HETATM"))}
            miss = {c for c, _, _ in fragments} - pdb_chains
            if miss:
                raise SystemExit(f"{task}: contig chain(s) {miss} not in {src_pdb} — chain mismatch")
            header = remark_999(task, r["pdb_name"], fragments)
            out_pdb = os.path.join(mdir, f"{task}.pdb")
            with open(out_pdb, "w") as w:
                w.write(header + body)
            ns = int(r.get("nsamples") or 0)
            prob = {
                "motif_filepaths": [os.path.abspath(out_pdb)],
                "segment_config_str": seg_cfg,
                "maximum_total_length": int(r["motif_max_length"]),
                "minimum_total_length": int(r["motif_min_length"]),
            }
            with open(os.path.join(pdir, f"{task}.json"), "w") as w:
                json.dump(prob, w, indent=2)
            n += 1
            nsamples = nsamples or (ns if ns else None)

    ns = nsamples or 100
    # absolute paths: even if Genie3 os.path.join(repo, path)s them, an absolute 2nd arg wins.
    exp = os.path.join(gdir, "experiment.yaml")
    with open(exp, "w") as w:
        w.write(
            f"experiment:\n  name: {os.path.basename(run_dir)}_genie3\n\n"
            f"paths:\n  rootdir: {os.path.join(gdir, 'output')}\n  dataset: {gdir}\n\n"
            f"generation:\n  dataset:\n    source: motif\n"
            f"    # selections omitted -> all {n} problems in dataset (README). For 1-task-1-motif\n"
            f"    # parity, set selections: <task_name> per array task instead.\n"
            f"    n_sample: {ns}\n"
            f"  sampler:\n    sampler:\n      direction_scale: 0.1\n"
        )
    print(f"genie3 problem set: {n} problems  ({ns} samples each)")
    print(f"  motifs   : {mdir}/<task>.pdb   (REMARK-999 annotated)")
    print(f"  problems : {pdir}/<task>.json")
    print(f"  config   : {exp}   (absolute rootdir/dataset)")
    print(f"  outputs  : {os.path.join(gdir, 'output')}/<task>/results/")
    print("launch (existing sbatch, all problems, sample-sharded):")
    print(f"  sbatch --array=0-9%5 --export=ALL,CONFIG={exp} "
          f"<motif_benchmark>/scripts/generate/genie3/genie3_generate.sbatch")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="run dir with <run>/manifest.csv + <run>/motifs/")
    ap.add_argument("--nsamples", type=int, default=None,
                    help="designs per problem (default: nsamples from the manifest)")
    a = ap.parse_args(argv)
    convert(a.run_dir, a.nsamples)


if __name__ == "__main__":
    main()
