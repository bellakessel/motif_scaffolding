#!/bin/bash
#SBATCH --job-name=unindexed_score
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/score/%x_%A.out
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G           # light: per-design Kabsch/greedy on small CA sets, parallel over designs
#SBATCH --time=02:00:00            # ~31k designs / 32 workers -> minutes; CPU job -> normal.* partition
set -euo pipefail

# ============================================================================
# unindexed_score.sbatch — relocalize UNINDEXED La-Proteina motif positions (CPU-only; do NOT run the full
# 31k on the login node). Sequence-seeded Kabsch + greedy nearest/aatype match -> corrected positions.jsonl.
# Runs in the gen_env venv (numpy). PREREQ: <run>/<METHOD>/{scaffolds/, positions.jsonl} + <run>/motifs/.
#
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=laproteina unindexed_score.sbatch
#   # then re-score the refolds at the corrected positions:
#   python 5_score.py --run-dir <run> --method laproteina \
#       --positions positions_unindexed.jsonl --out <run>/laproteina/af3_refold_scores_greedy.csv
# ============================================================================

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:-laproteina}"
DS="${DS:-/cluster/work/wicky/bekessel/motif_benchmark/scripts/evaluate}"
GEN_ENV="${GEN_ENV:-/cluster/work/wicky/bekessel/gen_env}"

source "$GEN_ENV/bin/activate"                   # gen_env venv (numpy) — NOT conda
mkdir -p /cluster/scratch/$USER/motifgo_logs/score

python "$DS/unindexed_score.py" --run-dir "$RUN_DIR" --method "$METHOD" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "[unindexed_score] $METHOD done -> $RUN_DIR/$METHOD/positions_unindexed.jsonl (+ unindexed_relocalization.csv)"
