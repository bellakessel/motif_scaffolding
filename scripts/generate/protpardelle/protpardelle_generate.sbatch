#!/bin/bash
#SBATCH --job-name=protpardelle
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/generate/%x_%A_%a.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --requeue                  # bulk/preemptible-safe: requeue on preemption; skip-existing (below) resumes
#SBATCH --open-mode=append
set -euo pipefail

# ============================================================================
# Protpardelle-1c motif-scaffolding GENERATION on Euler (mirror of proteina/la-proteina generate sbatch).
# One SLURM ARRAY task == one motif == one config `inf_<N>.yaml` (written by protpardelle_from_library.py).
# The config's search_space.models triple selects the model, so this sbatch is model-agnostic:
#   METHOD=pp1c_bb -> cc58 backbone (poly-Gly + native motif stamped)  -> shared MPNN->AF3 downstream
#   METHOD=pp1c_aa -> cc91 all-atom (co-designed sequence embedded)     -> extract-seq->AF3 downstream
# Generation ONLY: --num-mpnn-seqs 0 skips Protpardelle's own ProteinMPNN + ESMFold self-consistency, so we
# score every method through ONE shared downstream (fairness), exactly like the lean Genie3/Proteina setup.
#
#   # (host, gen_env) build configs from the library / a single smoke motif:
#   python scripts/protpardelle_from_library.py --single 1n8z_pc 570 base C --residues "570 ... 580" \
#       --pdb .../1n8z_pc_motif_C570_580.pdb --model cc58 --run-dir <run> --flank-n 40-70 --flank-c 40-70 --nsamples 10
#   # then submit (N = number of configs it reported):
#   sbatch --array=0-<N-1>%20 --export=ALL,RUN_DIR=<run>,METHOD=pp1c_bb protpardelle_generate.sbatch
# ============================================================================

REPO="${REPO:-/cluster/work/wicky/bekessel/software/protpardelle-1c}"
SIF="${SIF:-/cluster/work/wicky/containers/protpardelle.sif}"
RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=pp1c_bb (cc58) or pp1c_aa (cc91)}"
CFGNUM="${CFGNUM:-${SLURM_ARRAY_TASK_ID:-0}}"
SEED="${SEED:-5}"
BATCH="${BATCH:-8}"

METHOD_DIR="$RUN_DIR/$METHOD"
CONFIG="$METHOD_DIR/configs/inf_${CFGNUM}.yaml"
MOTIF_DIR="$RUN_DIR/motifs"
OUT_DIR="$METHOD_DIR/scaffolds"
MODEL_PARAMS="${PROTPARDELLE_MODEL_PARAMS:-$REPO/model_params}"

mkdir -p /cluster/scratch/$USER/motifgo_logs/generate "$OUT_DIR"
HFCACHE="${SCRATCH}/hf"; mkdir -p "$HFCACHE"

[ -f "$SIF" ]    || { echo "[protpardelle] container not found: $SIF (build protpardelle.def — see RUN_ON_EULER.md)"; exit 1; }
[ -f "$CONFIG" ] || { echo "[protpardelle] config missing: $CONFIG — run protpardelle_from_library.py first"; exit 1; }
[ -d "$MODEL_PARAMS/weights" ] || { echo "[protpardelle] model weights missing under $MODEL_PARAMS (bash download_model_params.sh)"; exit 1; }

# samples per motif from the manifest (kept consistent with the driver's --nsamples; contig has no commas so
# awk -F, is safe). Column 11 == nsamples.
NSAMPLES="$(awk -F, -v c="$CFGNUM" 'NR>1 && $1==c {print $11}' "$METHOD_DIR/manifest.csv")"
NSAMPLES="${NSAMPLES:-8}"

# skip-existing (bulk/requeue-safe): guard the find with [ -d ] — on a not-yet-created dir `find` exits
# non-zero and pipefail+set -e would abort BEFORE generating.
if [ -d "$OUT_DIR/inf_${CFGNUM}" ]; then
  have=$(find "$OUT_DIR/inf_${CFGNUM}" -name 'sample_*.pdb' | wc -l)
  [ "${have:-0}" -ge "$NSAMPLES" ] && { echo "[protpardelle] inf_${CFGNUM}: $have/$NSAMPLES samples present -> skip"; exit 0; }
fi

# host python env must NOT leak into the container (verified rfdiff/genie3/proteina/la-proteina pattern).
unset PYTHONHOME PYTHONPATH

echo "[protpardelle] sif=$SIF method=$METHOD config=inf_${CFGNUM} nsamples=$NSAMPLES -> $OUT_DIR"

# --nv GPU; bind /cluster/work/wicky (repo, run dir, weights) + $SCRATCH. PYTHONPATH=<repo>/src resolves
# `import protpardelle` (source bind-mounted, never installed). PROJECT_ROOT_DIR + PROTPARDELLE_MODEL_PARAMS
# point env.py at the repo's model_params; PROTPARDELLE_OUTPUT_DIR routes samples into the run dir.
singularity exec --nv \
  --bind /cluster/work/wicky:/cluster/work/wicky \
  --bind "${SCRATCH}:${SCRATCH}" \
  --env PROJECT_ROOT_DIR="$REPO" \
  --env PROTPARDELLE_MODEL_PARAMS="$MODEL_PARAMS" \
  --env PROTPARDELLE_OUTPUT_DIR="$OUT_DIR" \
  --env HF_HOME="$HFCACHE" --env TRANSFORMERS_CACHE="$HFCACHE" --env HF_HUB_CACHE="$HFCACHE" \
  "$SIF" \
  bash -c "cd '$REPO' && PYTHONPATH='$REPO/src' python -m protpardelle.sample '$CONFIG' \
             --motif-dir '$MOTIF_DIR' --num-samples ${NSAMPLES} --num-mpnn-seqs 0 \
             --seed ${SEED} --batch-size ${BATCH}"

echo "[protpardelle] done. samples under $OUT_DIR/inf_${CFGNUM}/<model-suffix>/<task>/ (scaffold_info.csv + *.pdb)."
