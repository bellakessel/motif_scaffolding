"""af3_inputs.py — build AF3 monomer-refold JSONs from ProteinMPNN fastas.

Each design's .fa has 1 template record (poly-Gly scaffold + fixed motif) followed by the designed
sequences (`>T=0.2, sample=M, …`). We SKIP the template and emit one AF3 monomer JSON per designed
sequence — matching Lisa's af3_job_generator (single protein chain, empty MSA, seed 42):
    {"name": …, "modelSeeds": [42], "sequences": [{"protein": {"id":"A","sequence":…,
      "unpairedMsa":"","pairedMsa":"","templates":[]}}], "dialect":"alphafold3","version":1}

INODE-SAFE: JSONs are written under $SCRATCH into fixed-size BATCH sub-dirs (b0001/, b0002/ …) so the
AF3 array can point one `--input_dir` per task and pack one tar.zst per batch (few archives on
/cluster/work instead of tens of thousands).
"""
from __future__ import annotations
import glob
import json
import os
from typing import List, Tuple


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    recs, h, s = [], None, []                            # recs = (header, sequence); h/s = the one being read
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(s)))
            h, s = line[1:], []
        else:
            s.append(line)
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


def _sample(h: str) -> str:
    for tok in h.split(","):
        if "sample=" in tok:
            return tok.split("=")[1].strip()
    return "1"


def _global_score(h: str) -> float:
    for tok in h.split(","):
        if "global_score=" in tok:
            try:
                return float(tok.split("=")[1])
            except ValueError:
                return 1e9
    return 1e9


def build_inputs(seqs_dir: str, out_base: str, batch_size: int = 100,
                 seed: int = 42, chain: str = "A", n_per_design: int | None = None) -> Tuple[int, int]:
    """Write monomer JSONs into $out_base/b0001/, b0002/, … (batch_size each).

    n_per_design=None -> all designed sequences; else the best N by MPNN global_score (lower=better).
    Returns (n_json, n_batch).
    """
    os.makedirs(out_base, exist_ok=True)
    jobs = []                                            # jobs = (prediction name, sequence) to fold
    for fa in sorted(glob.glob(os.path.join(seqs_dir, "*.fa"))):
        base = os.path.splitext(os.path.basename(fa))[0]           # base = the design, e.g. design_17
        recs = parse_fasta(fa)[1:]     # [0] is MPNN's echo of the INPUT backbone, not a designed sequence
        if n_per_design:
            recs = sorted(recs, key=lambda r: _global_score(r[0]))[:n_per_design]
        for h, seq in recs:
            jobs.append((f"{base}_seq{int(_sample(h)):03d}", seq))  # design_N_seq001 … seq008

    n_batch = 0
    for i in range(0, len(jobs), batch_size):
        n_batch += 1
        bdir = os.path.join(out_base, f"b{n_batch:04d}")  # bdir = one AF3 batch = one SLURM array task
        os.makedirs(bdir, exist_ok=True)
        for name, seq in jobs[i:i + batch_size]:
            js = {"name": name, "modelSeeds": [seed],     # js = the AF3 job: ONE chain, no MSA, no template
                  #      (empty MSA is what makes this a de-novo refold rather than a lookup)
                  "sequences": [{"protein": {"id": chain, "sequence": seq,
                                             "unpairedMsa": "", "pairedMsa": "", "templates": []}}],
                  "dialect": "alphafold3", "version": 1}
            json.dump(js, open(os.path.join(bdir, name + ".json"), "w"))
    return len(jobs), n_batch
