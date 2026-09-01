#!/bin/bash
#SBATCH --job-name=proteina
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/generate/%x_%A_%a.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G           # ~32G total
#SBATCH --time=03:00:00            # per MOTIF (one array task). Keep SHORT (GPU-QOS). 100 samples ~ mins-1h.
#SBATCH --requeue                  # bulk/preemptible-safe: requeue on preemption; skip-existing (below) resumes
#SBATCH --open-mode=append
set -euo pipefail

# ============================================================================
# Proteina motif-scaffolding GENERATION on Euler (mirror of genie3_generate.sbatch).
#
# One SLURM ARRAY task == one motif == one generated config `inf_<N>.yaml`, selected by
# `--config_number N` (motif_inference.py resolves it to inf_{N}). proteina_from_library.py writes
# those configs + a manifest first; then submit the array over them:
#
#   # (host, gen_env) build configs from the library:
#   python proteina_from_library.py --pool /cluster/work/wicky/bekessel/motif_library/motif_pool_dedup_j0.7.csv
#   # then submit (N = number of configs it reported):
#   sbatch --array=0-<N-1>%20 proteina_generate.sbatch
#
# Smoke (single motif / 1n8z_pc): the driver's --single mode writes inf_0.yaml, then:
#   sbatch --array=0-0 proteina_generate.sbatch
#
# Generation ONLY (compute_designability/fid False in the configs) — MPNN->AF3 downstream is separate,
# matching the lean Genie3 setup.
# ============================================================================

REPO="${REPO:-/cluster/work/wicky/bekessel/software/proteina}"
SIF="${SIF:-/cluster/work/wicky/containers/proteina.sif}"     # existing lab sif; override with bekessel build
METHOD="proteina"
RUN_DIR="${RUN_DIR:-}"                                         # <runs>/<run>/ ; outputs -> $RUN_DIR/proteina/
CONFIG_NAME="${CONFIG_NAME:-}"                                 # OR select a shipped/base config by NAME (Smoke A)
CFGNUM="${CFGNUM:-${SLURM_ARRAY_TASK_ID:-0}}"

mkdir -p /cluster/scratch/$USER/motifgo_logs/generate
HFCACHE="${SCRATCH}/hf"; mkdir -p "$HFCACHE"                  # transformers/HF cache -> writable scratch

[ -f "$SIF" ] || { echo "[proteina] container not found: $SIF (see RUN_ON_EULER.md)"; exit 1; }

# Two modes: RUN_DIR (library run, all artifacts in the run dir) or CONFIG_NAME (shipped example in the repo).
# CFG_ENV/OUT_ENV drive PROTEINA_CONFIG_DIR / PROTEINA_OUT_DIR (bekessel patches) so configs + scaffolds live
# in the run dir, NOT in the source repo. Empty -> the repo defaults (relative ./configs & ./inference).
if [ -n "$CONFIG_NAME" ]; then
    WORKDIR="$REPO"; SEL="--config_name ${CONFIG_NAME}"; OUTNAME="${CONFIG_NAME}"
    CHECK="$REPO/configs/experiment_config/${CONFIG_NAME}.yaml"; CFG_ENV=""; OUT_ENV=""
elif [ -n "$RUN_DIR" ]; then
    WORKDIR="$RUN_DIR/$METHOD"; SEL="--config_number ${CFGNUM}"; OUTNAME="inf_${CFGNUM}"
    CHECK="$WORKDIR/configs/inf_${CFGNUM}.yaml"; CFG_ENV="$WORKDIR/configs"; OUT_ENV="$WORKDIR/scaffolds"
    mkdir -p "$OUT_ENV"
else
    echo "[proteina] set RUN_DIR=<runs>/<run>  (library run)  OR  CONFIG_NAME=<name>  (shipped example)"; exit 1
fi
[ -f "$CHECK" ] || { echo "[proteina] config missing: $CHECK — run proteina_from_library.py first"; exit 1; }

# skip-existing (bulk/requeue-safe): this motif's samples already generated? nothing to do -> exit 0.
# NB guard the find with [ -d ] — on a not-yet-created output dir `find` exits non-zero and pipefail+set -e
# would abort the whole job BEFORE generating (that emptied the first launch).
if [ -n "${OUT_ENV:-}" ] && [ -d "$OUT_ENV/$OUTNAME" ]; then
  NS=$(grep -oE 'nsamples: *[0-9]+' "$CHECK" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true); NS=${NS:-100}
  have=$(find "$OUT_ENV/$OUTNAME" -maxdepth 1 -name '*.pdb' | wc -l)
  [ "${have:-0}" -ge "$NS" ] && { echo "[proteina] $OUTNAME: $have/$NS samples present -> skip"; exit 0; }
fi

# The container ships its own python; host PYTHONHOME/PYTHONPATH (module load / gen_env) must NOT leak in via
# singularity or the in-container python dies with "No module named 'encodings'" (verified rfdiff/genie3).
# PYTHONPATH=$REPO inside the exec resolves `from proteinfoundation...` (source bind-mounted, never installed).
unset PYTHONHOME PYTHONPATH

echo "[proteina] sif=$SIF workdir=$WORKDIR select='$SEL' out=${OUT_ENV:-$REPO/inference}"

# --nv: GPU. Bind /cluster/work/wicky (repo, run dir, checkpoints) + $SCRATCH (HF cache/logs). We `cd $WORKDIR`
# so CWD-relative temp writes (save_motif_csv's {task}_motif_info.csv) land in the run dir, not the repo.
# HF_HOME/TRANSFORMERS_CACHE -> scratch (designability module imports transformers even when off).
singularity exec --nv \
  --bind /cluster/work/wicky:/cluster/work/wicky \
  --bind "${SCRATCH}:${SCRATCH}" \
  --env DATA_PATH="${DATA_PATH:-/cluster/work/wicky/bekessel/software/proteina/data}" \
  --env HF_HOME="$HFCACHE" --env TRANSFORMERS_CACHE="$HFCACHE" --env HF_HUB_CACHE="$HFCACHE" \
  --env PROTEINA_CONFIG_DIR="$CFG_ENV" --env PROTEINA_OUT_DIR="$OUT_ENV" \
  "$SIF" \
  bash -c "cd '$WORKDIR' && PYTHONPATH='$REPO' python '$REPO'/proteinfoundation/motif_inference.py ${SEL}"

# stray per-task motif_info.csv (written CWD-relative; already copied into scaffolds/$OUTNAME/)
find "$WORKDIR" -maxdepth 1 -name '*_motif_info.csv' -delete 2>/dev/null || true

echo "[proteina] done. samples under ${OUT_ENV:-$REPO/inference}/${OUTNAME}/ (see manifest for provenance)."
