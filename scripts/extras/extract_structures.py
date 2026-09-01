#!/usr/bin/env python3
"""extract_structures.py — pull the SCAFFOLD pdb + AF3 REFOLD cif for chosen designs into one folder.

For qualitative inspection (e.g. the motifs La-Proteina unindexed rescues that every other method failed):
picks the generated scaffold and its AF3 refold structure(s) and drops them together, so you can open them
in PyMOL/ChimeraX. Runs on the cluster (reads the af3/archives .tar.zst). gen_env / stdlib only.

  scaffold: backbone methods -> <run>/<method>/1_filter/passed/<design>.pdb
            co-design (laproteina/pp1c_aa) -> <run>/<method>/scaffolds/<design>.pdb
  refold:   scanned out of <run>/<method>/af3/archives/*.tar.zst  (member <pred>/<pred>_model.cif,
            pred = <design>_seqNNN); up to --seqs-per-design cif(s) per design.

Select designs directly (--designs) OR by motif/task (--tasks, N representative designs each). Output ->
<run>/<method>/<out>/  (default 'picked_structures'):  <design>_scaffold.pdb , <pred>_model.cif .

  python extract_structures.py --run-dir <run> --method laproteina \
      --designs her2_m591_base_12,her2_m591_base_40 --seqs-per-design 1
  python extract_structures.py --run-dir <run> --method rfdiff \
      --tasks her2_m591_base,il2_m65_base --per-task 2 --out rescued_examples
"""
from __future__ import annotations
import argparse
import glob
import os
import shutil
import subprocess
import tempfile

CODESIGN = {"laproteina", "pp1c_aa"}


def scaffold_src(mdir: str, method: str, design: str) -> str | None:
    cand = (os.path.join(mdir, "scaffolds", design + ".pdb") if method in CODESIGN
            else os.path.join(mdir, "1_filter", "passed", design + ".pdb"))
    if os.path.exists(cand):
        return cand
    # fallback: rfdiff scaffolds live under scaffolds/<task>/design_N.pdb (design = <task>_design_N)
    if "_design_" in design:
        task, num = design.split("_design_")
        alt = os.path.join(mdir, "scaffolds", task, f"design_{num}.pdb")
        if os.path.exists(alt):
            return alt
    return None


def designs_for_tasks(mdir: str, method: str, tasks: list[str], per_task: int) -> list[str]:
    """Up to per_task design_names for each task, from the scaffold layout (passers for backbone)."""
    out = []
    base = (os.path.join(mdir, "scaffolds") if method in CODESIGN
            else os.path.join(mdir, "1_filter", "passed"))
    for t in tasks:
        got = 0
        for p in sorted(glob.glob(os.path.join(base, "**", "*.pdb"), recursive=True)):
            stem = os.path.basename(p)[:-4]
            if method not in CODESIGN and "design_" in stem and os.path.basename(os.path.dirname(p)) == t:
                dn = f"{t}_{stem}"                                  # rfdiff scaffolds/<task>/design_N.pdb
            elif stem.startswith(t):
                dn = stem                                          # passed/<task>_..pdb or scaffolds/<task>_N.pdb
            else:
                continue
            out.append(dn); got += 1
            if got >= per_task:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--designs", help="comma list OR @file of design_names")
    g.add_argument("--tasks", help="comma list OR @file of task_names (motifs); use --per-task")
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--seqs-per-design", type=int, default=1, help="refold cif(s) per design to pull")
    ap.add_argument("--out", default="picked_structures")
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    mdir = os.path.join(run_dir, a.method)
    outdir = os.path.join(mdir, a.out)
    os.makedirs(outdir, exist_ok=True)

    def _list(s):
        return [x.strip() for x in (open(s[1:]).read().split() if s.startswith("@") else s.split(",")) if x.strip()]

    designs = _list(a.designs) if a.designs else designs_for_tasks(mdir, a.method, _list(a.tasks), a.per_task)
    if not designs:
        raise SystemExit("no designs resolved")
    print(f"[extract] {a.method}: {len(designs)} designs -> {outdir}")

    n_sc = 0
    for d in designs:
        src = scaffold_src(mdir, a.method, d)
        if src:
            shutil.copyfile(src, os.path.join(outdir, d + "_scaffold.pdb")); n_sc += 1
        else:
            print(f"  [warn] no scaffold for {d}")
    print(f"  scaffolds copied: {n_sc}/{len(designs)}")

    want = {d: 0 for d in designs}
    n_rf = 0
    for arch in sorted(glob.glob(os.path.join(mdir, "af3", "archives", "**", "*.tar.zst"), recursive=True)):
        if all(v >= a.seqs_per_design for v in want.values()):
            break
        listing = subprocess.run(f'zstd -d -c "{arch}" | tar -t', shell=True,
                                 capture_output=True, text=True).stdout
        take = []
        for m in (l for l in listing.splitlines() if l.endswith("_model.cif")):
            pred = os.path.basename(m)[:-len("_model.cif")]        # <design>_seqNNN
            dn = pred.rsplit("_seq", 1)[0] if "_seq" in pred else pred
            if dn in want and want[dn] < a.seqs_per_design:
                take.append(m); want[dn] += 1
        if take:
            tmp = tempfile.mkdtemp()
            subprocess.run(f'zstd -d -c "{arch}" | tar -xC "{tmp}" ' + " ".join(f'"{m}"' for m in take),
                           shell=True, check=True)
            for m in take:
                shutil.copyfile(os.path.join(tmp, m), os.path.join(outdir, os.path.basename(m))); n_rf += 1
            shutil.rmtree(tmp, ignore_errors=True)
    miss = [d for d, v in want.items() if v == 0]
    print(f"  refolds extracted: {n_rf}  (designs with no refold found: {len(miss)})")
    if miss:
        print("   missing refold:", ", ".join(miss[:8]) + (" ..." if len(miss) > 8 else ""))
    print(f"[extract] done -> {outdir}/  (open <design>_scaffold.pdb + <pred>_model.cif together)")


if __name__ == "__main__":
    main()
