#!/bin/bash
#SBATCH --job-name=genie3
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/generate/%x_%A_%a.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G           # ~32G total: headroom for the model + sampling
#SBATCH --time=03:00:00            # per SHARD. Keep SHORT — long GPU walltimes queue for hours / can hit a
                                   # GPU-QOS cap. 100 designs/shard ~2.5h worst case; size shards to fit this.
set -euo pipefail

# ============================================================================
# Genie3 motif-scaffolding GENERATION on Euler (lean gen-only container).
#
# Runs `genie3 generate` inside genie3.sif. Env-var driven with defaults (override with
# --export=...), esmfold2_array.sh-style, so it is directly `sbatch`-able for the smoke
# test and reusable for real configs. Generation ONLY — no evaluation (that stack is not
# in this image); the `evaluation:` block in a config is simply ignored by `generate`.
#
# Smoke test (shipped example: problem 22_1BCF, n_sample 5):
#   sbatch genie3_generate.sbatch
#
# Own config (must be repo-relative, like the examples — rootdir/dataset resolve from repo root):
#   sbatch --export=ALL,CONFIG=examples/motif_scaffolding/experiment.yaml genie3_generate.sbatch
#
# Multi-GPU on one node: --export=ALL,NUM_DEVICES=4 + #SBATCH --gpus=rtx_4090:4
#
# SHARDING for large runs (e.g. 1000 designs): launch as a SLURM ARRAY. Each task generates one
# shard of the config's n_sample (Genie3 slices it into non-overlapping, uniquely-named chunks via
# --shard-id/--num-shards), so 1000 designs finish in parallel instead of one long job. Set the
# config's n_sample to the TOTAL (1000) and pick the shard count with --array:
#   sbatch --array=0-4  --export=ALL,CONFIG=<abs experiment.yaml> genie3_generate.sbatch   # 5x200
#   sbatch --array=0-9%5 --export=ALL,CONFIG=<...> genie3_generate.sbatch                  # 10x100, <=5 at once
# num_shards is taken from the array size automatically; a non-array run is a single shard.
# ============================================================================

REPO="${REPO:-/cluster/work/wicky/containers/genie3}"
SIF="${SIF:-/cluster/work/wicky/containers/genie3.sif}"
CONFIG="${CONFIG:-examples/motif_scaffolding/experiment.yaml}"   # repo-relative path
NUM_DEVICES="${NUM_DEVICES:-1}"

# Sharding: from the SLURM array (override with NUM_SHARDS). Non-array run -> 1 shard (no slicing).
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_ARGS=""
[ "${NUM_SHARDS}" -gt 1 ] && SHARD_ARGS="--shard-id ${SHARD_ID} --num-shards ${NUM_SHARDS}"

mkdir -p /cluster/scratch/$USER/motifgo_logs/generate

[ -f "$SIF" ] || { echo "[genie3] container not found: $SIF  (build it first — see RUN_ON_EULER.md)"; exit 1; }
[ -d "$REPO/pretrained/v1" ] || echo "[genie3] WARNING: $REPO/pretrained/v1 missing — run scripts/setup/download.sh --weights"

# The container ships its own python; the host's PYTHONHOME/PYTHONPATH (module load / gen_env)
# must NOT leak in via singularity or the in-container python dies with
# "ModuleNotFoundError: No module named 'encodings'". Strip them (verified rfdiff pattern).
unset PYTHONHOME PYTHONPATH

echo "[genie3] repo=$REPO  sif=$SIF  config=$CONFIG  num_devices=$NUM_DEVICES  shard ${SHARD_ID}/${NUM_SHARDS}"

# rootdir/dataset in the config are repo-relative -> run from the repo root.
# PYTHONPATH=src makes `genie3` importable without installing it into the image.
# singularity exec --nv --bind ... matches the rfdiff/mpnn/af3 pipeline templates
# (on Euler `singularity` and `apptainer` are the same binary; .sif is interchangeable).
singularity exec --nv \
  --bind /cluster/work/wicky:/cluster/work/wicky \
  "$SIF" \
  bash -c "cd '$REPO' && PYTHONPATH=src python -m genie3.cli generate -c '$CONFIG' --num-devices $NUM_DEVICES ${SHARD_ARGS} --verbose"

echo "[genie3] done. outputs under the config's paths.rootdir (repo-relative)."