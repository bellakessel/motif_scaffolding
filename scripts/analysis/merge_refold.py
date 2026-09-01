#!/usr/bin/env python3
"""
merge_refold.py  --  Part A of the Foldseek clustering phase.

Two jobs:
  1. REPRESENTATIVES (cheap, always): for each design_name pick the prediction with the
     max ranking_score AMONG its designable==1 rows -> exactly one structure per design
     that has >=1 success. Designs with zero designable predictions contribute nothing.
     -> reports/foldseek/representatives.csv   (pdb_name,motif_num,design_name,prediction,ranking_score)
     -> reports/foldseek/keep_{pdb}.txt        (one prediction/line; the extractor's keep-set)
  2. MASTER TABLE (heavy, skippable): vertical-concat all per-protein refold CSVs and stream
     them to ONE Parquet (~10x smaller, columnar) for the Part-D join + colouring.
     -> reports/combined_af3_refold.parquet
     Set SKIP_PARQUET=1 to compute only the representatives/keep-lists (fast, no big write).

Reps are computed directly from the lazy CSV scan (no Parquet dependency), so SKIP_PARQUET=1
still produces everything Part B (extract_structures.sh) needs.

Run ON EULER (CSVs are local + fast there), in the gen_env venv:
    module load stack/2024-06 python/3.11.6
    source /cluster/work/wicky/bekessel/gen_env/bin/activate
    python -c "import polars" 2>/dev/null || pip install polars   # if missing
    python merge_refold.py                      # reps + keep-lists + parquet, all 3 proteins
    SKIP_PARQUET=1 python merge_refold.py        # reps + keep-lists only (fast)
    PROTEINS=3bis python merge_refold.py         # one protein
"""
import os
import polars as pl

MINE     = os.environ.get("MINE", "/cluster/work/wicky/bekessel")
MOTIF_T  = os.environ.get("MOTIF_TYPE", "c_motif")
PROTEINS = os.environ.get("PROTEINS", "1n8z,2mhs,3bis").split(",")
SKIP_PARQUET = os.environ.get("SKIP_PARQUET", "0") == "1"
REPORTS  = os.path.join(MINE, "reports")
OUT_PARQ = os.path.join(REPORTS, "combined_af3_refold.parquet")
REPS_DIR = os.path.join(REPORTS, "foldseek")
REPS_CSV = os.path.join(REPS_DIR, "representatives.csv")

os.makedirs(REPS_DIR, exist_ok=True)


def csv_path(pdb: str) -> str:
    return os.path.join(REPORTS, f"af3_refold_{pdb}_{MOTIF_T}.csv")


paths = [csv_path(p) for p in PROTEINS]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    raise SystemExit("ERROR: missing refold CSV(s):\n  " + "\n  ".join(missing))

# vertical_relaxed tolerates minor dtype differences across files (e.g. all-empty iptm)
lf = pl.concat([pl.scan_csv(p) for p in paths], how="vertical_relaxed")

# ── representatives: best designable prediction per design (direct from the lazy scan) ──
# carry the per-prediction quality metrics into representatives.csv too: Part D then reads ONE
# small CSV (no giant-parquet-over-sshfs read, which is slow + flaky on the mount).
REP_METRICS = ["plddt_mean", "ptm", "iptm", "scRMSD", "fraction_disordered"]
reps = (
    lf.filter(pl.col("designable") == 1)
    .sort("ranking_score", descending=True, nulls_last=True)
    .group_by(["pdb_name", "design_name"], maintain_order=True)
    .first()                                    # top ranking_score row per design
    .select(["pdb_name", "motif_num", "design_name", "prediction", "ranking_score", *REP_METRICS])
    .collect()
)
reps.write_csv(REPS_CSV)
print(f"wrote {REPS_CSV}  ({reps.height:,} representatives)")

# per-protein keep lists (prediction only) — the extractor's keep-set
for p in PROTEINS:
    kp = os.path.join(REPS_DIR, f"keep_{p}.txt")
    sub = reps.filter(pl.col("pdb_name") == p).select("prediction")
    sub.write_csv(kp, include_header=False)
    print(f"wrote {kp}  ({sub.height:,} structures to extract for {p})")

# ── master table: stream all rows to Parquet (heavy; skip with SKIP_PARQUET=1) ──
if SKIP_PARQUET:
    print("SKIP_PARQUET=1 -> not writing combined_af3_refold.parquet")
else:
    print(f"streaming master table -> {OUT_PARQ}")
    lf.sink_parquet(OUT_PARQ)                   # out-of-core
    total = pl.scan_parquet(OUT_PARQ).select(pl.len()).collect().item()
    print(f"  parquet rows total: {total:,}")

# ── verification ────────────────────────────────────────────────────────────
print("\nVERIFY (representatives = structures to extract):")
rep_by_pdb = reps.group_by("pdb_name").len().sort("pdb_name")
for r in rep_by_pdb.iter_rows(named=True):
    print(f"    {r['pdb_name']:6} {r['len']:,}")
print("\nnext: extract these representatives' .cif from the AF3 archives (extract_structures.sh)")
