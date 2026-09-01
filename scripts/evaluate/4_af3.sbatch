#!/bin/bash
#SBATCH --job-name=refold_af3
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/af3/%x_%A_%a.out
#SBATCH --gpus=1                    # NO model pin -> 142 gpuhe nodes (4090+3090+quadro_rtx_6000+titan_rtx) vs 80
#SBATCH --gres=gpumem:24g          # >=24G filters to AF3-capable cards without pinning a type (Leo review #1)
#SBATCH --cpus-per-task=4          # AF3 inference is GPU-bound (no data pipeline); 4 CPUs is plenty
#SBATCH --mem-per-cpu=4G           # 16G/task: at 64G the gpuhe mem cap (1770G) binds at 27 jobs < 96-GPU cap;
                                   # at 16G -> 110 > 96 so the GPU cap binds instead. ~3.5x more jobs (review #4)
#SBATCH --time=04:00:00
#SBATCH --requeue                  # restart-safe: requeue on preemption / node-fail (Leo GPU-maxxing L2)
#SBATCH --open-mode=append         # keep the log across requeues
#SBATCH --signal=B:USR1@180        # 180s before walltime: USR1 -> checkpoint + self-requeue (survive TIMEOUT)
set -euo pipefail
unset PYTHONHOME PYTHONPATH        # don't leak host python env into the container

# GPU POOL + BULK (Leo review): --gpus=1 --gres=gpumem:24g reaches all 142 AF3-capable gpuhe nodes (the
# rtx_4090 pin used only 80, and excluded the currently-idle quadro_rtx_6000 nodes). The gpuhe.4h QOS cap is
# 96 GPUs — enough for ~92 batches. Euler's submit plugin routes by WALLTIME and OVERRIDES --partition, so
# --partition=gpuhe.bulk does NOT work; to reach the 240-GPU bulk cap (only if scaling past ~96 concurrent),
# submit then relocate:  scontrol update jobid=<id> partition=gpuhe.bulk qos=es_wicky/gpuhe/bulk
# NOTE: create the log dir on the login node BEFORE submitting (Slurm opens --output before the script runs):
#   mkdir -p /cluster/scratch/$USER/motifgo_logs/af3
#   sbatch --array=1-<NBATCH>%92 --export=ALL,RUN_DIR=<run>,METHOD=<m> 4_af3.sbatch

# ============================================================================
# 4_af3.sbatch — AF3 monomer-REFOLD, pipeline step 4, method-agnostic (one shared script for all methods).
# Fold each designed sequence ALONE (no MSA/template) -> designability scoring (5_score: motifRMSD<=1 &
# scRMSD<=2). Reads the SHARED refold JSONs at <run>/<METHOD>/refold_inputs/bNNNN (built by 3_af3_build.py;
# ESMFold2 reads the SAME JSONs). One array task folds one batch -> <method>/af3/archives/batch_NNNN.tar.zst.
#
#   python 3_af3_build.py <run>/<METHOD>/{mpnn/seqs|seqs} <run>/<METHOD>/refold_inputs 500
#   sbatch --array=1-<NBATCH> --export=ALL,RUN_DIR=<run>,METHOD=<method> 4_af3.sbatch
# ============================================================================

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|genie3|pp1c_bb|laproteina|pp1c_aa}"
AF3_SIF="${AF3_SIF:-/cluster/work/wicky/containers/alphafold3.sif}"
AF3_PARAMS="${AF3_PARAMS:-/cluster/work/wicky/params/af3_params}"
TASK="${SLURM_ARRAY_TASK_ID:-1}"
BATCH="b$(printf '%04d' "$TASK")"

IN="$RUN_DIR/$METHOD/refold_inputs/$BATCH.tar"    # inode-safe: one tar per batch (3_af3_build packs these)
# NODE-LOCAL working dir: inputs+outputs live on $TMPDIR, NOT scratch. Scratch has a ~1M-FILE quota that
# the AF3 outputs (500 designs x ~6 files x concurrent batches) blow; $TMPDIR is per-job local SSD, wiped
# at job end. Only the final batch tar.zst lands on /cluster/work. (Was $SCRATCH -> Errno 122 quota kills.)
WORK="${TMPDIR:?TMPDIR unset}/af3work/${METHOD}_$BATCH"
ARCH_DIR="$RUN_DIR/$METHOD/af3/archives"
mkdir -p "$WORK" "$ARCH_DIR" /cluster/scratch/$USER/motifgo_logs/af3

[ -f "$IN" ] || { echo "[4_af3] no input batch tar $IN — build first: sbatch --export=ALL,RUN_DIR=$RUN_DIR,METHOD=$METHOD 3_af3_build.sbatch"; exit 1; }

# skip-existing (idempotent for requeue/array-resume): batch already packed? done.
if [ -f "$ARCH_DIR/batch_$(printf '%04d' "$TASK").tar.zst" ]; then
  echo "[4_af3] $BATCH already archived -> skip"; exit 0
fi

# extract the batch's JSONs into node-local scratch
INDIR="$WORK/in"; mkdir -p "$INDIR"
tar -xf "$IN" -C "$INDIR"
# AF3 globs *.json; macOS sshfs writes drop AppleDouble ._*.json that AF3 chokes on (UnicodeDecodeError).
find "$INDIR" -maxdepth 1 -name '._*' -delete 2>/dev/null || true

BINDS="-B /cluster/work/wicky:/cluster/work/wicky -B ${TMPDIR}:${TMPDIR}"
BOUT="$WORK/out"; mkdir -p "$BOUT"          # node-local; batch runs to completion in one task (8h walltime)
TODO="$WORK/todo"; rm -rf "$TODO"; mkdir -p "$TODO"

# checkpoint-before-wall (Leo L2): on USR1 (@180s before TIMEOUT) requeue ourselves. $BOUT is now NODE-LOCAL
# ($TMPDIR, wiped on requeue), so a requeue redoes the whole batch — fine because 8h walltime >> ~2.8h/batch
# means we don't hit the wall; the trap is just a backstop for the rare preemption. (Batch-level skip-existing
# on the packed tar.zst still prevents redoing a COMPLETED batch.)
_ckpt() { echo "[4_af3] USR1: $(find "$BOUT" -maxdepth 2 -iname '*_model.cif' 2>/dev/null | wc -l) preds done (node-local, will redo batch) -> requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 0; }
trap _ckpt USR1

# RESUME: fold only the sequences whose prediction isn't already in $BOUT (skip-existing per-prediction, so
# a dropped+requeued job continues from where it stopped instead of redoing the whole batch). Erring toward
# re-folding is safe; a false "done" is not, so we match the actual AF3 output cif (case-insensitive).
ntot=0
for j in "$INDIR"/*.json; do
  [ -e "$j" ] || continue
  ntot=$((ntot+1)); nm=$(basename "$j" .json)
  if find "$BOUT" -maxdepth 2 -iname "${nm}_model.cif" 2>/dev/null | grep -q .; then :; else cp "$j" "$TODO/"; fi
done
ntodo=$(find "$TODO" -maxdepth 1 -name '*.json' | wc -l)   # find (not ls glob): empty TODO must not trip pipefail (review #2)
echo "[4_af3] $METHOD $BATCH: $ntot seqs, $((ntot-ntodo)) already folded, $ntodo to fold (resume-safe)"

if [ "$ntodo" -gt 0 ]; then
  # Turing cards (quadro_rtx_6000 / titan_rtx = sm_75) can't run AF3's default Triton FlashAttention (needs
  # sm_80+; it dies with "implementation='triton' ... unsupported"). Pick the XLA kernel on <sm_80 so widening
  # the GPU pool to those cards is safe; Ampere/Ada keep triton. (Leo review #1 catch.)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)
  FA=triton
  if [ "${CC:-80}" -lt 80 ] 2>/dev/null; then
    FA=xla
    # sm 7.x (Turing: quadro_rtx_6000/titan_rtx) also REQUIRES this XLA flag or run_alphafold aborts with
    # "the ENV XLA_FLAGS must include --xla_disable_hlo_passes=custom-kernel-fusion-rewriter". Inject into the
    # container via the SINGULARITYENV_/APPTAINERENV_ prefix (portable across singularity & apptainer builds).
    export SINGULARITYENV_XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
    export APPTAINERENV_XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
  fi
  echo "[4_af3] GPU compute_cap=${CC:-?} -> flash_attention=$FA XLA_FLAGS=${SINGULARITYENV_XLA_FLAGS:-<none>}"

  # RESILIENT FOLD: AF3 folds the whole $TODO in ONE process, so a single design whose diffusion yields
  # NaN/inf coords (ValueError "Column x must not contain NaN/inf" — a known sm_7.x/Turing numerical issue,
  # but also possible on any card for a degenerate sequence) crashes the process and would sink the WHOLE
  # 500-batch. Loop: fold -> on crash, quarantine the design it died on (the first still-missing its cif in
  # AF3's sorted processing order) and retry with the rest. Converges (each crash removes >=1). Quarantined
  # designs are logged to <method>/af3/quarantine_bNNNN.txt and simply skipped in the packed batch.
  QUAR="$WORK/quarantine"; mkdir -p "$QUAR"
  QLOG="$ARCH_DIR/../quarantine_${BATCH}.txt"
  for attempt in $(seq 1 30); do
    rm -rf "$TODO"; mkdir -p "$TODO"; left=0        # TODO = designs neither folded nor quarantined
    for j in "$INDIR"/*.json; do
      [ -e "$j" ] || continue; nm=$(basename "$j" .json)
      [ -e "$QUAR/$nm.json" ] && continue
      if find "$BOUT" -maxdepth 2 -iname "${nm}_model.cif" 2>/dev/null | grep -q .; then :; else cp "$j" "$TODO/"; left=$((left+1)); fi
    done
    [ "$left" -eq 0 ] && break
    set +e
    singularity exec --nv $BINDS "$AF3_SIF" \
      python3 /app/alphafold/run_alphafold.py \
      --model_dir="$AF3_PARAMS" --run_data_pipeline=False --run_inference=True \
      --flash_attention_implementation="$FA" \
      --num_diffusion_samples=1 --input_dir="$TODO" --output_dir="$BOUT" &
    wait $!; rc=$?            # background + wait so the USR1 checkpoint trap can still fire mid-fold
    set -e
    [ "$rc" -eq 0 ] && break
    culprit=""               # first TODO design with no output cif == the one AF3 died on
    for j in "$TODO"/*.json; do
      [ -e "$j" ] || continue; nm=$(basename "$j" .json)
      find "$BOUT" -maxdepth 2 -iname "${nm}_model.cif" 2>/dev/null | grep -q . || { culprit="$nm"; break; }
    done
    [ -n "$culprit" ] || { echo "[4_af3] crash rc=$rc but no culprit found — abort"; exit 1; }
    echo "[4_af3] AF3 crashed (rc=$rc) on $culprit — quarantine + retry ($attempt/30)"
    cp "$INDIR/$culprit.json" "$QUAR/"; echo "$culprit" >> "$QLOG"
  done
fi

# pack when every NON-quarantined prediction is present; else exit non-zero so a requeue finishes the rest.
nquar=$(find "$WORK/quarantine" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)   # 0 if none / dir absent
ndone=$(find "$BOUT" -maxdepth 2 -iname "*_model.cif" 2>/dev/null | wc -l)
need=$((ntot - nquar))
[ "$ndone" -ge "$need" ] || { echo "[4_af3] incomplete: $ndone/$need folded (quarantined $nquar) — exit (requeue will finish)"; exit 1; }
[ "$nquar" -gt 0 ] && echo "[4_af3] $BATCH: packed $ndone folds, $nquar design(s) quarantined (NaN) -> $ARCH_DIR/../quarantine_${BATCH}.txt"

# pack outputs into one tar.zst per batch (inode-safe; 5_score reads tar.zst)
tar -C "$BOUT" -cf - . | zstd -q -T0 -o "$ARCH_DIR/batch_$(printf '%04d' "$TASK").tar.zst.tmp"
mv "$ARCH_DIR/batch_$(printf '%04d' "$TASK").tar.zst.tmp" "$ARCH_DIR/batch_$(printf '%04d' "$TASK").tar.zst"
rm -rf "$BOUT" "$TODO"      # clean $SCRATCH only after the batch is fully packed
echo "[4_af3] done -> $ARCH_DIR/batch_$(printf '%04d' "$TASK").tar.zst ($ntot folds)"
