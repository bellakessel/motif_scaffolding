#!/usr/bin/env python3
"""designable_sse_extract.py — DSSP secondary structure for every designable AF3 refold, plus the
CIFs of the designable designs, in ONE pass over the refold archives.

WHY. The backbone Foldseek phase had to reduce all five generators to CA and rewrite them as
poly-Gly (Proteina and Genie3 emit CA-only files; residue naming was perfectly method-correlated),
which threw away the inverse-folding step entirely and gave a weak signal. The AF3 refolds are
uniform across methods by construction -- all-atom, real sequence, one chain -- so no
normalisation is needed, and they are the structures that actually passed designability.

THE GATE. `designable` in af3_refold_scores.csv is only TWO gates (motifRMSD <= 1, scRMSD <= 2);
those CSVs predate 5_score.py growing RANK_MIN. The project definition is THREE gates,
    motif_ok AND sc_ok AND ranking_score >= 0.75          (note >=, not >)
and `ranking_score` is stored rounded to 2 dp, so a large mass sits exactly at 0.75 -- in
lib_interface_pad5 that is 12,102 predictions, i.e. `>` silently drops ~14% of the set.

WHAT COMES OUT. SSE is annotated for the WIDER two-gate set and flagged with `conf_ok`, while
only the three-gate best-per-design CIFs are written for clustering. Annotating the wider set is
free -- the archive is already decompressed -- and it is exactly the comparison the reference
preprint makes: what does selecting on confidence do to the SSE distribution? Without it that
figure would need a second full pass.

Archive handling mirrors software/downstream/5_score.py: one worker per archive, decompressed
into NODE-LOCAL $TMPDIR (fast, no scratch inode quota), shared read-only state set once per
worker by the Pool initializer.

  python designable_sse_extract.py --run <run> --method rfdiff \
      --out-tsv <out>/pred_sse.tsv --cif-out-dir $SCRATCH/.../cifs --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cif_backbone import backbone_coords          # noqa: E402
from dssp_numpy import assign_c3                  # noqa: E402

CIF_SUFFIX = "_model.cif"
RANK_MIN = 0.75          # 5_score.py:RANK_MIN. INCLUSIVE -- see module docstring.

FIELDS = ["method", "prediction", "design_name", "task_name", "pdb_name", "n_res",
          "H_frac", "E_frac", "C_frac", "dssp3", "seq", "motifRMSD", "scRMSD", "plddt_mean",
          "ranking_score", "ptm", "has_clash", "fraction_disordered", "conf_ok", "is_best"]


def resolve_tasks(run: str, method: str, design_names):
    """design_name -> (task_name, pdb_name), WITHOUT parsing the name for a separator.

    Every generator names its designs differently and only rfdiff uses `_design_`:
        rfdiff           her2_m239_base_design_13
        genie3           il13_m24_base_71
        proteina         her2_m602_filled_0
        pp1c_bb          her2_m239_base_sample_her2_m239_base_31
        laproteina_mpnn  her2_m262_base_job_0_id_93_motif_her2_m262_base
    A `_design_` split silently produced a bogus task for four of the five (caught by the smoke
    run, job 11833718). What they DO all share is that the design name begins with its task
    name, so resolve by longest-prefix match against the manifest, and cross-check against the
    method's own tasks.json where it exists. Both sources agree 100% on all five methods.
    """
    man = pd.read_csv(os.path.join(run, "manifest.csv"))
    pdb_of = dict(zip(man["task_name"], man["pdb_name"]))
    tasks = sorted(man["task_name"].unique(), key=len, reverse=True)   # longest first

    def by_prefix(dn):
        for t in tasks:
            if dn.startswith(t):
                return t
        return None

    uniq = pd.Index(pd.unique(pd.Series(list(design_names))))
    pref = uniq.map(by_prefix)

    src = "longest-prefix (manifest)"
    for cand in (os.path.join(run, method, "1_filter", "tasks.json"),
                 os.path.join(run, method, "tasks.json")):
        if os.path.exists(cand):
            j = json.load(open(cand))
            auth = uniq.map(lambda d: j.get(d))
            cov = float(pd.notna(auth).mean())
            if cov > 0.999:
                agree = float((pd.Series(auth) == pd.Series(pref)).mean())
                print(f"[sse:{method}] task map <- {cand}  (covers {cov:.2%}, "
                      f"agrees with longest-prefix {agree:.2%})")
                assert agree > 0.999, "tasks.json and longest-prefix disagree -- inspect before trusting"
                pref, src = auth, cand
            break
    else:
        print(f"[sse:{method}] task map <- {src} (no tasks.json)")

    bad = int(pd.isna(pref).sum())
    assert bad == 0, f"{bad} design names could not be resolved to a task ({src})"
    t = dict(zip(uniq, pref))
    return t, {d: pdb_of[tk] for d, tk in t.items()}


def select(scores_csv: str, conf_min: float, run: str, method: str,
           profile: str = "manifest", annotate: str = "all2g"):
    """-> (wanted: prediction -> dict of score fields, n2, n3, d2, d3).

    `wanted` covers the TWO-gate set; `is_best` marks the one prediction per design that carries
    that design into the clustering -- lowest scRMSD among its THREE-gate passers.
    """
    d = pd.read_csv(scores_csv)
    for c in ("designable", "ranking_score", "scRMSD"):
        if c not in d.columns:
            raise SystemExit(f"{scores_csv}: missing column '{c}'")
    g2 = d[d["designable"] == 1].copy()
    g2["conf_ok"] = (g2["ranking_score"] >= conf_min).astype(int)     # >= is deliberate
    g3 = g2[g2["conf_ok"] == 1]
    best = set(g3.loc[g3.groupby("design_name")["scRMSD"].idxmin(), "prediction"])
    g2["is_best"] = g2["prediction"].isin(best).astype(int)

    # task / pdb come from the manifest, never from splitting the design name.
    # Lisa's 1n8z/2mhs/3bis runs predate the manifest and carry pdb_name + motif_num as COLUMNS,
    # so there is nothing to resolve there -- profile="csv" reads them directly.
    if profile == "csv":
        need = [c for c in ("pdb_name", "motif_num") if c not in g2.columns]
        if need:
            raise SystemExit(f"{scores_csv}: profile=csv needs columns {need}")
        g2["task_name"] = g2["pdb_name"] + "_motif_" + g2["motif_num"].astype(str)
    else:
        t_of, p_of = resolve_tasks(run, method, g2["design_name"])
        g2["task_name"] = g2["design_name"].map(t_of)
        g2["pdb_name"] = g2["design_name"].map(p_of)

    # Annotation scope. `all2g` keeps the pre-gate rows that make the confidence-gate figure
    # possible; for a set as large as Lisa's (1.98M two-gate predictions) that is a 500 MB TSV
    # of rows no figure uses, so `best` (one per design) is the right default there.
    # capture the full-scope counts BEFORE narrowing, so the log reports what the gates did to
    # the whole set rather than to whatever subset we chose to annotate
    n2_all, d2_all = len(g2), g2["design_name"].nunique()
    n3_all, d3_all = len(g3), g3["design_name"].nunique()
    if annotate == "best":
        g2 = g2[g2["is_best"] == 1]
    elif annotate == "3g":
        g2 = g2[g2["conf_ok"] == 1]
    elif annotate != "all2g":
        raise SystemExit(f"--annotate must be all2g|3g|best, got {annotate!r}")

    keep = ["prediction", "design_name", "task_name", "pdb_name", "motif_num", "motifRMSD", "scRMSD",
            "plddt_mean", "ranking_score", "ptm", "has_clash", "fraction_disordered",
            "conf_ok", "is_best"]
    keep = [c for c in keep if c in g2.columns]
    wanted = {r["prediction"]: r for r in g2[keep].to_dict("records")}
    return wanted, n2_all, n3_all, d2_all, d3_all


def expect_rows(annotate: str, n2: int, n3: int, d3: int) -> int:
    """How many rows this --annotate scope should produce."""
    return {"all2g": n2, "3g": n3, "best": d3}[annotate]


_W: Dict = {}


def _init_worker(method, wanted, cif_dir, with_seq):
    _W.update(method=method, wanted=wanted, cif_dir=cif_dir, with_seq=with_seq)


def _do_archive(arch):
    method, wanted, cif_dir = _W["method"], _W["wanted"], _W["cif_dir"]
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or os.environ.get("SCRATCH") or "/tmp")
    rows, errs = [], []
    try:
        subprocess.run(f'zstd -d -c "{arch}" | tar -xC "{tmp}"', shell=True, check=True,
                       capture_output=True)
        # Archive layouts differ: motif_library extracts prediction dirs at the top level,
        # Lisa's runs wrap them in a `batch_NNNN/` dir. Descend one level when a top-level dir
        # is not itself a prediction, so one code path handles both without configuration.
        def pred_dirs(root):
            with os.scandir(root) as it:
                for e in it:
                    if not e.is_dir():
                        continue
                    if e.name in wanted:
                        yield e.name, e.path
                    else:
                        with os.scandir(e.path) as it2:
                            for e2 in it2:
                                if e2.is_dir() and e2.name in wanted:
                                    yield e2.name, e2.path

        for nm, path in pred_dirs(tmp):
                s = wanted.get(nm)
                if s is None:
                    continue
                cif = os.path.join(path, nm + CIF_SUFFIX)
                try:
                    coord, meta = backbone_coords(cif)
                    sse = assign_c3(coord)
                    n = len(sse) or 1
                    dn = s["design_name"]
                    rows.append({
                        "method": method, "prediction": nm, "design_name": dn,
                        "task_name": s["task_name"], "pdb_name": s["pdb_name"],
                        "n_res": meta["n_res"], "n_dropped": meta["n_dropped"],
                        "H_frac": sse.count("H") / n, "E_frac": sse.count("E") / n,
                        "C_frac": sse.count("-") / n,
                        "dssp3": sse.replace("-", "C"),          # C for coil, to match the motif pool
                        "seq": meta["seq"] if _W["with_seq"] else "",
                        "motifRMSD": s.get("motifRMSD"), "scRMSD": s.get("scRMSD"),
                        "plddt_mean": s.get("plddt_mean"), "ranking_score": s.get("ranking_score"),
                        "ptm": s.get("ptm"), "has_clash": s.get("has_clash"),
                        "fraction_disordered": s.get("fraction_disordered"),
                        "conf_ok": s["conf_ok"], "is_best": s["is_best"],
                    })
                    if s["is_best"] and cif_dir:
                        shutil.copyfile(cif, os.path.join(cif_dir, f"{method}__{dn}.cif"))
                except Exception as ex:              # one bad structure must not kill the archive
                    errs.append(f"{nm}: {type(ex).__name__}: {str(ex)[:100]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows, errs


def main() -> None:
    ap = argparse.ArgumentParser(description="DSSP SSE for designable AF3 refolds + designable CIFs")
    ap.add_argument("--run", required=True, help="run dir, e.g. .../runs/lib_interface_pad5")
    ap.add_argument("--method", required=True)
    ap.add_argument("--scores", default=None, help="default <run>/<method>/af3_refold_scores.csv")
    ap.add_argument("--out-tsv", required=True, help="appended to; header written if absent")
    ap.add_argument("--cif-out-dir", default=None, help="best three-gate CIF per design goes here")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: N archives EVENLY SPACED across the list. Not the first N -- "
                         "archives are written in task order, so the head of the list covers only "
                         "2-3 motifs and can contain zero designable predictions (batch_0001 of "
                         "rfdiff does exactly that). Same trap as the backbone smoke run.")
    ap.add_argument("--conf-min", type=float, default=RANK_MIN, help="ranking_score gate, INCLUSIVE")
    ap.add_argument("--archives", default=None,
                    help="archive dir (default <run>/<method>/af3/archives)")
    ap.add_argument("--profile", choices=["manifest", "csv"], default="manifest",
                    help="manifest = resolve task via tasks.json/manifest (motif_library); "
                         "csv = pdb_name/motif_num are columns of the scores CSV (Lisa's runs)")
    ap.add_argument("--annotate", choices=["all2g", "3g", "best"], default="all2g",
                    help="which designable rows to DSSP. best = one per design.")
    ap.add_argument("--with-seq", action="store_true", help="also emit the one-letter sequence")
    ap.add_argument("--label", default=None, help="value for the `method` column (default --method)")
    a = ap.parse_args()

    a.label = a.label or a.method
    mdir = os.path.join(a.run, a.method)
    scores = a.scores or os.path.join(mdir, "af3_refold_scores.csv")
    adir = a.archives or os.path.join(mdir, "af3", "archives")
    archives = sorted(os.path.join(adir, f) for f in os.listdir(adir) if f.endswith(".tar.zst"))
    if not archives:
        raise SystemExit(f"no archives under {adir}")
    if a.limit and a.limit < len(archives):
        step = len(archives) / a.limit                      # evenly spaced, never the head
        archives = [archives[int(i * step)] for i in range(a.limit)]

    wanted, n2, n3, d2, d3 = select(scores, a.conf_min, a.run, a.method, a.profile, a.annotate)
    nproc = max(1, a.workers or os.cpu_count() or 1)
    print(f"[sse:{a.label}] {len(archives)} archive(s) in {adir}, {nproc} workers, "
          f"annotate={a.annotate}\n"
          f"[sse:{a.label}] 2-gate {n2:,} preds / {d2:,} designs  ->  "
          f"3-gate (ranking_score >= {a.conf_min}) {n3:,} preds / {d3:,} designs\n"
          f"[sse:{a.label}] annotating {expect_rows(a.annotate, n2, n3, d3):,} of them "
          f"({a.annotate})", flush=True)
    if a.cif_out_dir:
        os.makedirs(a.cif_out_dir, exist_ok=True)

    t0 = time.time()
    rows, errs, done = [], [], 0
    with mp.Pool(nproc, initializer=_init_worker,
                 initargs=(a.label, wanted, a.cif_out_dir, a.with_seq)) as pool:
        for r, e in pool.imap_unordered(_do_archive, archives):
            rows.extend(r); errs.extend(e); done += 1
            if done % 20 == 0 or done == len(archives):
                el = time.time() - t0
                print(f"  [{a.method}] {done}/{len(archives)} archives  {len(rows):,} preds  "
                      f"{el:.0f}s ({el / done:.2f} s/archive)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"[sse:{a.label}] NO rows produced -- archives or scores CSV mismatched")

    # --- assertions: a silent archive read failure is the realistic bug here
    expect = expect_rows(a.annotate, n2, n3, d3)
    miss = expect - len(df)
    dropped = int(df["n_dropped"].sum())
    bad = df[(df[["H_frac", "E_frac", "C_frac"]].sum(axis=1) - 1.0).abs() > 1e-6]
    print(f"\n[sse:{a.label}] rows {len(df):,} / expected {expect:,}"
          f"{'' if miss == 0 else f'   *** {miss:,} MISSING ***'}")
    exp_conf = {"all2g": n3, "3g": n3, "best": d3}[a.annotate]
    print(f"[sse:{a.label}] conf_ok {int(df.conf_ok.sum()):,} / expected {exp_conf:,}   "
          f"is_best {int(df.is_best.sum()):,} / expected {d3:,}")
    print(f"[sse:{a.label}] incomplete-backbone residues dropped: {dropped}   "
          f"frac-sum violations: {len(bad)}   parse errors: {len(errs)}")
    for e in errs[:10]:
        print(f"    ERR {e}")
    if not a.limit:
        assert miss == 0, f"{miss} designable predictions were not found in the archives"
        assert int(df.is_best.sum()) == d3
        if a.annotate == "all2g":
            assert int(df.conf_ok.sum()) == n3
    assert len(bad) == 0, "H+E+C != 1"

    g = df[df.conf_ok == 1]
    print(f"[sse:{a.label}] three-gate mean SSE:  H {g.H_frac.mean():.3f}  "
          f"E {g.E_frac.mean():.3f}  C {g.C_frac.mean():.3f}   (two-gate: "
          f"H {df.H_frac.mean():.3f}  E {df.E_frac.mean():.3f}  C {df.C_frac.mean():.3f})")

    df = df.drop(columns=["n_dropped"])[FIELDS]
    hdr = not os.path.exists(a.out_tsv) or os.path.getsize(a.out_tsv) == 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out_tsv)), exist_ok=True)
    df.to_csv(a.out_tsv, sep="\t", index=False, mode="a", header=hdr, float_format="%.6g")
    print(f"[sse:{a.label}] {'wrote' if hdr else 'appended to'} {a.out_tsv}  "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
