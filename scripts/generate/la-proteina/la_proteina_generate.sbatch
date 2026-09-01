#!/bin/bash
#SBATCH --job-name=laproteina
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/generate/%x_%A_%a.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --requeue                  # bulk/preemptible-safe: requeue on preemption; skip-existing (below) resumes
#SBATCH --open-mode=append
set -euo pipefail

# ============================================================================
# La-Proteina motif-scaffolding GENERATION on Euler (LD4 indexed all-atom co-design).
# One SLURM ARRAY task == one motif == one config `inf_<RUNTAG>_<TASK>.yaml` (in the repo configs/, written by
# la_proteina_from_library.py). We `cd` into <run>/laproteina so generate.py's CWD-relative ./inference/ output
# lands in the run dir; hydra reads configs SCRIPT-relative from <repo>/configs (defaults resolve there); the
# LD4/AE3 checkpoint paths are pinned ABSOLUTE in the generated config so cd doesn't break them.
# La-Proteina CO-DESIGNS sequence+structure -> output PDBs carry the sequence; NO MPNN step.
#
#   python scripts/la_proteina_from_library.py --single 1n8z_pc 570 base C --residues "570 ... 580" \
#       --pdb .../1n8z_pc_motif_C570_580.pdb --run-dir <run> --flank-n 40-70 --flank-c 40-70 --nsamples 10
#   sbatch --array=0-0 --export=ALL,RUN_DIR=<run>,RUNTAG=$(basename <run>) scripts/la_proteina_generate.sbatch
# ============================================================================

REPO="${REPO:-/cluster/work/wicky/bekessel/software/la-proteina}"
SIF="${SIF:-/cluster/work/wicky/containers/la-proteina.sif}"
RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
RUNTAG="${RUNTAG:-$(basename "$RUN_DIR")}"
TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
CONFIG_NAME="inf_${RUNTAG}_${TASK}"

mkdir -p /cluster/scratch/$USER/motifgo_logs/generate
HFCACHE="${SCRATCH}/hf"; mkdir -p "$HFCACHE"
WORKDIR="$RUN_DIR/laproteina"; mkdir -p "$WORKDIR"
# the config interpolates ${oc.env:DATA_PATH} for its metric_factory (eval only); must be SET so hydra can
# compose. Generation does not use the metric, so an empty writable dir suffices.
DATA_PATH="${DATA_PATH:-$REPO/data}"; mkdir -p "$DATA_PATH"

[ -f "$SIF" ] || { echo "[laproteina] container not found: $SIF (build la-proteina.def — see RUN_ON_EULER.md)"; exit 1; }
[ -f "$REPO/configs/${CONFIG_NAME}.yaml" ] || {
    echo "[laproteina] config missing: $REPO/configs/${CONFIG_NAME}.yaml — run la_proteina_from_library.py first"; exit 1; }

# skip-existing (bulk/requeue-safe): guard the find with [ -d ] — on a not-yet-created dir `find` exits
# non-zero and pipefail+set -e would abort BEFORE generating.
if [ -d "$WORKDIR/inference" ]; then
  NS=$(grep -oE 'nsamples: *[0-9]+' "$REPO/configs/${CONFIG_NAME}.yaml" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true); NS=${NS:-100}
  have=$(find "$WORKDIR/inference" -path "*${CONFIG_NAME}_*" -name '*.pdb' | wc -l)
  [ "${have:-0}" -ge "$NS" ] && { echo "[laproteina] ${CONFIG_NAME}: $have/$NS designs present -> skip"; exit 0; }
fi

# host python env must NOT leak into the container (verified rfdiff/genie3/proteina pattern)
unset PYTHONHOME PYTHONPATH

echo "[laproteina] sif=$SIF workdir=$WORKDIR config=$CONFIG_NAME"

singularity exec --nv \
  --bind /cluster/work/wicky:/cluster/work/wicky \
  --bind "${SCRATCH}:${SCRATCH}" \
  --env HF_HOME="$HFCACHE" --env TRANSFORMERS_CACHE="$HFCACHE" --env HF_HUB_CACHE="$HFCACHE" \
  --env DATA_PATH="$DATA_PATH" \
  "$SIF" \
  bash -c "cd '$WORKDIR' && PYTHONPATH='$REPO' python '$REPO'/proteinfoundation/generate.py --config_name ${CONFIG_NAME}"

echo "[laproteina] done. samples under $WORKDIR/inference/${CONFIG_NAME}_<motif_task>/ (co-designed all-atom + seq)."
