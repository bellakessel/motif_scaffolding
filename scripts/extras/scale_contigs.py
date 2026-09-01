#!/usr/bin/env python3
"""scale_contigs.py — build native-geometry-scaled RFdiffusion contigs for a motif subset.

Motivation: the benchmark handed every method the SAME fixed contig (`2-10/<frag>/1-5/.../35-75`) — a
uniform `1-5` linker regardless of how far apart the fragments sit in the native structure. MotifBench
instead scales the scaffold to the native geometry: each linker is a RANGE centered on the native residue
gap (`native_gap +/- W`, never capped), and flanks are wide ranges (asymmetric or symmetric from 0). This
script emits, per motif, two such contigs so we can re-scaffold the hard (all-failure) motifs with a
geometry-aware scaffold and ask whether it recovers designs while keeping the motif surface-exposed.

Per motif (from the pool's native `motif_residues` + the extracted motif PDB's coordinates):
  fragments = contiguous runs of motif_residues (native numbering) -> `{chain}{a}-{b}` (always range form).
  per gap i between fragment end e and next start s:
     g     = native sequence gap = s - e - 1
     D     = CA-CA distance between the two fragment endpoints (from the motif PDB)
     n_min = ceil(D / 3.8)                          # physical reach floor (extended-chain)
     linker_i = [ max(n_min, g - W), g + W ]        # MotifBench-style range, NO cap  (W=5 default)
  native_asym : flanks 2-10 / 35-75  (the OLD padding) + per-gap native linkers
  native_sym  : symmetric flanks 0-60 / 0-60        + per-gap native linkers
  total_min/max = flank_lo/hi + motif_len + sum(linker_lo/hi) + flank_lo/hi  (motif_min/max_length)

Emits <out> (default motif_library/runs/<run>/analysis/scaled_contigs.csv) in the run-manifest schema
(drop-in for the from_library drivers' `--manifest` consume-mode) + native-gap diagnostics. Two rows per
motif (one per mode). Runs in gen_env (stdlib only).

  python scale_contigs.py --run-dir /cluster/work/wicky/bekessel/motif_library/runs/lib_interface_pad5 \
      --failed-all <run>/analysis/failed_all.txt
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
from motifgo.contig import parse_range   # one definition, shared with the contig grammar


def fragments_of(residues):
    """Contiguous runs of sorted native residue numbers -> [(a,b), ...]."""
    ids = sorted(int(x) for x in residues.split())
    runs = [[ids[0], ids[0]]]
    for r in ids[1:]:
        if r == runs[-1][1] + 1:
            runs[-1][1] = r
        else:
            runs.append([r, r])
    return [(a, b) for a, b in runs], len(ids)


def ca_by_resid(pdb_path):
    """{res_id: (x,y,z)} of CA atoms (first model)."""
    ca = {}
    with open(pdb_path) as fh:
        for ln in fh:
            if ln.startswith("ENDMDL"):
                break
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                ca[int(ln[22:26])] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return ca


def build_contig(chain, frags, linker_tokens, flank_n, flank_c):
    parts = [flank_n]
    for i, (a, b) in enumerate(frags):
        parts.append(f"{chain}{a}-{b}")
        if i < len(frags) - 1:
            parts.append(linker_tokens[i])
    parts.append(flank_c)
    return "/".join(parts)


def totals(flank_n, flank_c, motif_len, linkers):
    (nlo, nhi), (clo, chi) = parse_range(flank_n), parse_range(flank_c)
    lo = nlo + motif_len + sum(l[0] for l in linkers) + clo
    hi = nhi + motif_len + sum(l[1] for l in linkers) + chi
    return lo, hi


def contig_for_mode(chain, frags, linkers, flank_n, flank_c):
    toks = [f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in linkers]
    return build_contig(chain, frags, toks, flank_n, flank_c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="benchmark run dir (has motifs/ + manifest.csv)")
    ap.add_argument("--failed-all", default=None, help="txt of task_names (default <run>/analysis/failed_all.txt)")
    ap.add_argument("--pool", default=None, help="pool CSV (default motif_pool_interface_j0.7_pad5.csv next to runs/)")
    ap.add_argument("--motifs-dir", default=None, help="dir of extracted motif PDBs (default <run>/motifs)")
    ap.add_argument("--out", default=None, help="output CSV (default <run>/analysis/scaled_contigs.csv)")
    ap.add_argument("--window", type=int, default=5, help="linker half-window W: linker=[max(n_min,g-W), g+W]")
    ap.add_argument("--flank-n", default="2-10", help="native_asym N-flank (old padding)")
    ap.add_argument("--flank-c", default="35-75", help="native_asym C-flank (old padding)")
    ap.add_argument("--sym-flank", default="0-60", help="native_sym symmetric flank (both ends)")
    ap.add_argument("--nsamples", type=int, default=50)
    ap.add_argument("--burial-total", type=int, default=200, help="diagnostic flag when total_max exceeds this")
    ap.add_argument("--drop-gap-ge", type=int, default=None,
                    help="DROP a motif entirely if any native sequence gap >= this (domain-spanning "
                         "epitopes whose 'linker' would rebuild a whole native domain). Default: keep all.")
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    motifs_dir = a.motifs_dir or os.path.join(run_dir, "motifs")
    failed_txt = a.failed_all or os.path.join(run_dir, "analysis", "failed_all.txt")
    pool_csv = a.pool or os.path.join(os.path.dirname(os.path.dirname(run_dir)),
                                      "motif_pool_interface_j0.7_pad5.csv")
    out_csv = a.out or os.path.join(run_dir, "analysis", "scaled_contigs.csv")

    with open(failed_txt) as fh:
        failed = [ln.strip() for ln in fh if ln.strip()]
    fset = set(failed)

    pool = {}
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            t = f"{r['pdb_name']}_m{r['motif_num']}_{r['variant']}"
            if t in fset:
                pool[t] = r

    W = a.window
    fields = ["idx", "config_name", "task_name", "base_task", "pdb_name", "motif_num", "variant", "mode",
              "chain", "n_fragments", "contig", "motif_min_length", "motif_max_length", "nsamples",
              "motif_pdb", "native_gaps", "endpoint_dists", "min_linkers", "linker_ranges",
              "total_min", "total_max", "burial_flag", "disjointed_parts", "motif_length", "mean_rsa",
              "source_pdb"]
    rows = []
    idx = 0
    missing = []
    for base in failed:
        r = pool.get(base)
        if r is None:
            missing.append((base, "not in pool")); continue
        motif_pdb = os.path.join(motifs_dir, f"{base}.pdb")
        if not os.path.isfile(motif_pdb):
            missing.append((base, "no motif pdb")); continue
        chain = r["chain"]
        frags, motif_len = fragments_of(r["motif_residues"])
        ca = ca_by_resid(motif_pdb)

        gaps, dists, nmins, linkers = [], [], [], []
        for (a0, b0), (a1, b1) in zip(frags, frags[1:]):
            g = a1 - b0 - 1
            if b0 in ca and a1 in ca:
                D = math.dist(ca[b0], ca[a1]); n_min = max(1, math.ceil(D / 3.8))
            else:
                D = float("nan"); n_min = 1
            lo = max(n_min, g - W); hi = g + W
            if lo > hi:
                lo = hi = n_min
            gaps.append(g); dists.append(round(D, 1)); nmins.append(n_min); linkers.append((lo, hi))

        if a.drop_gap_ge is not None and gaps and max(gaps) >= a.drop_gap_ge:
            missing.append((base, f"domain-spanning gap={max(gaps)}>={a.drop_gap_ge}")); continue

        for mode, fn, fc in (("native_asym", a.flank_n, a.flank_c),
                             ("native_sym", a.sym_flank, a.sym_flank)):
            contig = contig_for_mode(chain, frags, linkers, fn, fc)
            tlo, thi = totals(fn, fc, motif_len, linkers)
            task = f"{base}__{mode}"
            rows.append({
                "idx": idx, "config_name": f"inf_contigscale_{idx}", "task_name": task, "base_task": base,
                "pdb_name": r["pdb_name"], "motif_num": r["motif_num"], "variant": r["variant"], "mode": mode,
                "chain": chain, "n_fragments": len(frags), "contig": contig,
                "motif_min_length": tlo, "motif_max_length": thi, "nsamples": a.nsamples,
                "motif_pdb": motif_pdb,
                "native_gaps": " ".join(map(str, gaps)), "endpoint_dists": " ".join(map(str, dists)),
                "min_linkers": " ".join(map(str, nmins)),
                "linker_ranges": " ".join(f"{lo}-{hi}" for lo, hi in linkers),
                "total_min": tlo, "total_max": thi, "burial_flag": int(thi > a.burial_total),
                "disjointed_parts": r["disjointed_parts"], "motif_length": r["motif_length"],
                "mean_rsa": r["mean_rsa"], "source_pdb": r["source_pdb"],
            })
            idx += 1

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    n_motifs = len({row["base_task"] for row in rows})
    n_buried = sum(row["burial_flag"] for row in rows) // 1
    print(f"[scale_contigs] {n_motifs} motifs x 2 modes = {len(rows)} contigs -> {out_csv}")
    print(f"[scale_contigs] W={W}, asym flanks {a.flank_n}/{a.flank_c}, sym flank {a.sym_flank}, "
          f"nsamples={a.nsamples}; {n_buried} rows total_max>{a.burial_total} (burial_flag)", file=sys.stderr)
    if missing:
        print(f"[scale_contigs] SKIPPED {len(missing)}: " + ", ".join(f"{t}({why})" for t, why in missing[:20])
              + (" ..." if len(missing) > 20 else ""), file=sys.stderr)
    # show a couple of examples
    for row in rows[:4]:
        print(f"  {row['task_name']:38s} nf={row['n_fragments']} gaps=[{row['native_gaps']}] "
              f"links=[{row['linker_ranges']}] total={row['total_min']}-{row['total_max']}", file=sys.stderr)


if __name__ == "__main__":
    main()
