#!/usr/bin/env python3
"""3_af3_build.py — build AF3 monomer-refold JSONs from a run's sequences (shared by AF3 + ESMFold2).

Pipeline step 3, method-agnostic. Reuses the shared builder (motifgo.af3_inputs.build_inputs) by
path so every method gets the IDENTICAL Lisa-format monomer JSONs (single chain, empty MSA, seed 42,
template dropped) — so the refold + scorer read them the same way regardless of generator.

Sequence source:
  backbone methods (rfdiff/proteina/genie3/pp1c_bb) : <run>/<method>/mpnn/seqs   (from 2_mpnn)
  co-design methods (laproteina/pp1c_aa)            : <run>/<method>/seqs         (co-designed, no MPNN)

PARALLEL: build_inputs is single-threaded and a backbone method is ~250k JSONs (8 seqs/design) -> serial
is hours. We shard the seq files across BUILD_WORKERS processes (each builds its shard with the SAME
build_inputs into its own out dir), then renumber-merge every shard's bNNNN batch dir into one global
b0001.. sequence in <out_base>. Sharding is by DESIGN FILE, so a design's 8 seqs stay in one batch. The
last batch of each shard may be <batch_size — harmless, 4_af3 folds whatever JSONs a batch contains.

Usage:  python 3_af3_build.py <seqs_dir> <out_base> [batch_size=500]
Env:    BUILD_WORKERS (default: all CPUs; 1 = serial fallback, no sharding).
Writes  <out_base>/b0001/ (b0002/ …) and prints (n_json, n_batch).
"""
import glob
import os
import shutil
import sys
from multiprocessing import Pool

from motifgo import af3_inputs


def _build_shard(job):
    """Worker: build one shard's seqs into out_shard with the shared esp builder."""
    seqs_shard, out_shard, batch = job
    os.makedirs(out_shard, exist_ok=True)
    n_json, _ = af3_inputs.build_inputs(seqs_shard, out_shard, batch_size=batch,
                                     seed=42, chain="A", n_per_design=None)
    return out_shard, n_json


def main() -> None:
    seqs_dir, out_base = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    workers = int(os.environ.get("BUILD_WORKERS") or os.cpu_count() or 1)
    os.makedirs(out_base, exist_ok=True)

    print(f"[3_af3_build] listing seq files in {seqs_dir} ...", flush=True)
    files = sorted(glob.glob(os.path.join(seqs_dir, "*.fa")) or glob.glob(os.path.join(seqs_dir, "*")))
    print(f"[3_af3_build] {len(files)} seq files; workers={workers}", flush=True)
    if workers <= 1 or len(files) <= batch:                      # serial fallback (small / single-worker)
        n_json, n_batch = af3_inputs.build_inputs(seqs_dir, out_base, batch_size=batch,
                                               seed=42, chain="A", n_per_design=None)
        print(f"[3_af3_build] {n_json} monomer JSONs in {n_batch} batch(es) -> {out_base} (serial)")
        return

    # 1) shard the seq files round-robin into <out_base>/_shards/sW/seqs (symlinks -> inode-light)
    tmp = os.path.join(out_base, "_shards")
    shutil.rmtree(tmp, ignore_errors=True)
    shard_seqs = []
    for w in range(workers):
        sd = os.path.join(tmp, f"s{w}", "seqs")
        os.makedirs(sd, exist_ok=True)
        shard_seqs.append(sd)
    for i, f in enumerate(files):
        os.symlink(os.path.abspath(f), os.path.join(shard_seqs[i % workers], os.path.basename(f)))
    print(f"[3_af3_build] sharded {len(files)} files -> {workers} shards; launching workers ...", flush=True)

    # 2) build each shard in parallel
    jobs = [(shard_seqs[w], os.path.join(tmp, f"s{w}", "out"), batch) for w in range(workers)]
    results = []
    with Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_build_shard, jobs):
            results.append(r)
            print(f"[3_af3_build] shard {len(results)}/{workers} done ({r[1]} JSONs)", flush=True)

    # 3) renumber-merge: concatenate every shard's bNNNN dir into one global b0001.. in out_base
    gi, total = 0, 0
    for out_shard, n_json in sorted(results):
        total += n_json
        for b in sorted(glob.glob(os.path.join(out_shard, "b*"))):
            if not os.path.isdir(b):
                continue
            gi += 1
            shutil.move(b, os.path.join(out_base, f"b{gi:04d}"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[3_af3_build] {total} monomer JSONs in {gi} batch(es) -> {out_base} ({workers} workers)")


if __name__ == "__main__":
    main()
