#!/bin/bash
#SBATCH --job-name=rfdiff
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/generate/%x_%A_%a.out
#SBATCH --gpus=1                    # NO gpu-TYPE pin -> eligible for EVERY GPU family (RFdiffusion runs on
#SBATCH --gres=gpumem:24g          #   Turing too); only require >=24G VRAM. Widest node eligibility (Leo lesson 3).
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00            # per MOTIF (100 designs ~ mins-1h). Keep <=4h -> eligible for the 4h+24h+120h
#SBATCH --requeue                  #   pools at once (Leo lesson 1). Bulk = higher CONCURRENCY, reached by relocate.
#SBATCH --open-mode=append         # bulk/preempt-safe: requeue on preemption, skip-existing (below) resumes clean.
set -euo pipefail

# ============================================================================
# RFdiffusion motif-scaffolding GENERATION on Euler — pool-driven, mirrors the other 4 methods.
# One SLURM ARRAY task == one motif == one manifest row (config_number == array index == same motif per tool).
# Reads the SHARED contract that stage_motif.py wrote (<run>/motifs + <run>/manifest.csv); RFdiffusion's
# contigmap.contigs is esp.contig's grammar VERBATIM and input_pdb is the motif-only template PDB, so this
# reuses esp's proven run_inference.py invocation (templates/rfdiff.sbatch.tmpl) at pool scale. Generation
# ONLY — backbones feed the SAME shared MPNN->AF3->score downstream as every method (fairness).
#
#   # (host, gen_env) stage the pool once -> <run>/motifs + <run>/manifest.csv:
#   pip install -e <motif_benchmark> --no-deps    # or: export PYTHONPATH=<motif_benchmark>
#   python <repo>/scripts/motif/stage_motif.py --pool <pool.csv> --run-dir <run> --variants base --nsamples 100
#   # then submit (N = number of manifest rows; 0-indexed):
#   sbatch --array=0-<N-1>%20 --export=ALL,RUN_DIR=<run> rfdiff_generate.sbatch
# ============================================================================

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run> (the dir holding manifest.csv + motifs/)}"
SIF="${SIF:-/cluster/work/wicky/containers/RFdiffusion.sif}"
CKPT="${CKPT:-/cluster/work/wicky/params/RFParams/Base_ckpt.pt}"
CFGNUM="${CFGNUM:-${SLURM_ARRAY_TASK_ID:-0}}"
MANIFEST="$RUN_DIR/manifest.csv"

mkdir -p /cluster/scratch/$USER/motifgo_logs/generate
[ -f "$SIF" ]      || { echo "[rfdiff] container not found: $SIF"; exit 1; }
[ -f "$CKPT" ]     || { echo "[rfdiff] checkpoint not found: $CKPT"; exit 1; }
[ -f "$MANIFEST" ] || { echo "[rfdiff] no manifest at $MANIFEST — run stage_motif.py first"; exit 1; }

# Parse this array task's row by HEADER NAME — robust to WHICH driver wrote manifest.csv. stage_motif
# uses `config_number, task_name, ...` (task_name col 2); the la-proteina driver uses `idx, config_name,
# task_name, ...` so task_name shifts to col 3 — a fixed `cut -f2` there silently grabs config_name and the
# scaffolds land under inf_<runtag>_N instead of the motif name. Header lookup fixes that; the array-index
# column is `config_number` (stage) or `idx` (la-proteina), both at col 1. gsub(/\r/) strips CRLF too.
# (manifest has NO commas inside fields — contig uses '/', fragments ';' — so -F, is safe.)
{ read -r TASK; read -r CONTIG; read -r LMIN; read -r LMAX; read -r NDES; read -r MOTIF_PDB; } < <(
  awk -F, -v c="$CFGNUM" '
    NR==1 { for(i=1;i<=NF;i++){v=$i; gsub(/\r/,"",v); h[v]=i}
            ic = (("config_number" in h)?h["config_number"]:(("idx" in h)?h["idx"]:1)); next }
    { key=$ic; gsub(/\r/,"",key);
      if (key==c) { for(i=1;i<=NF;i++) gsub(/\r/,"",$i);
        print $(h["task_name"]); print $(h["contig"]); print $(h["motif_min_length"]);
        print $(h["motif_max_length"]); print $(h["nsamples"]); print $(h["motif_pdb"]); exit } }
  ' "$MANIFEST")
[ -n "$TASK" ] && [ -n "$MOTIF_PDB" ] || { echo "[rfdiff] no row for index=$CFGNUM (or missing columns) in $MANIFEST"; exit 1; }
LENGTH="${LMIN}-${LMAX}"
OUT="$RUN_DIR/rfdiff/scaffolds/$TASK"
[ -f "$MOTIF_PDB" ] || { echo "[rfdiff] motif pdb missing: $MOTIF_PDB"; exit 1; }

# skip-existing (idempotent -> safe for requeue / bulk): already have all NDES backbones? done.
# NB guard with [ -d ] + use find — a bare `ls "$OUT"/design_*.pdb` on a missing dir / no match exits
# non-zero and pipefail+set -e would abort the job BEFORE generating.
if [ -d "$OUT" ]; then
  have=$(find "$OUT" -maxdepth 1 -name 'design_*.pdb' | wc -l)
  [ "${have:-0}" -ge "$NDES" ] && { echo "[rfdiff] $TASK: $have/$NDES designs present -> skip"; exit 0; }
fi

mkdir -p "$OUT" "$OUT/schedules/0"
# in-container python must use the container's own stdlib (host module-load leaks PYTHONHOME/PYTHONPATH).
unset PYTHONHOME PYTHONPATH

echo "[rfdiff] $TASK  config=$CFGNUM  ndesigns=$NDES  contig=[$CONTIG]  length=$LENGTH -> $OUT"

# Verified esp invocation (NVIDIA base image -> exec + python3; run_inference.py under /app). write_trajectory
# =False drops per-design traj PDBs (inode saver). design_startnum=0: one task does ALL NDES backbones/motif.
singularity exec --nv -B /cluster/work/wicky:/cluster/work/wicky "$SIF" \
  python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.input_pdb="$MOTIF_PDB" \
  inference.output_prefix="$OUT/design" \
  inference.num_designs="$NDES" \
  inference.design_startnum=0 \
  inference.write_trajectory=False \
  inference.ckpt_override_path="$CKPT" \
  inference.schedule_directory_path="$OUT/schedules/0" \
  "contigmap.contigs=[$CONTIG]" \
  "contigmap.length=$LENGTH"

echo "[rfdiff] done. backbones -> $OUT/design_*.pdb  (feed the shared MPNN->AF3->score downstream)."
