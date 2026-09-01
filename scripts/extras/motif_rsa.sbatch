#!/bin/bash
#SBATCH --job-name=motif_rsa
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/score/%x_%A.out
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G           # biotite Shrake-Rupley SASA is CPU-bound; parallel over af3 archives
#SBATCH --time=08:00:00            # ~250k structures / 48 workers; CPU job -> normal.* partition
set -euo pipefail

# ============================================================================
# motif_rsa.sbatch — Task 2 de-novo motif burial / ΔRSA extractor for ONE method (CPU-only).
# Full-structure biotite SASA/RSA on every AF3 refold, ΔRSA vs native library RSA, parallel over archives
# into node-local $TMPDIR (no scratch quota). Runs in the gen_env venv (biotite) with the esp pipeline on
# PYTHONPATH. Output -> <run>/<METHOD>/motif_rsa.csv (one row/design + per-residue ΔRSA string).
#
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=rfdiff motif_rsa.sbatch
#   # unindexed La-Proteina — use the greedy-relocalized positions:
#   sbatch --export=ALL,RUN_DIR=<uidx>,METHOD=laproteina,POSITIONS=positions_unindexed.jsonl motif_rsa.sbatch
# ============================================================================

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|genie3|pp1c_bb|laproteina}"
POSITIONS="${POSITIONS:-}"                        # override motif positions (e.g. positions_unindexed.jsonl)
LIB="${LIB:-/cluster/work/wicky/bekessel/motif_library}"
REPO="${REPO:-/cluster/work/wicky/bekessel/motif_benchmark}"
GEN_ENV="${GEN_ENV:-/cluster/work/wicky/bekessel/gen_env}"

source "$GEN_ENV/bin/activate"                   # gen_env venv (biotite) — NOT conda
export PYTHONPATH="$REPO:${PYTHONPATH:-}"        # motifgo (if not pip-installed into the venv)
mkdir -p /cluster/scratch/$USER/motifgo_logs/score

python "$REPO/scripts/extras/motif_rsa.py" --run-dir "$RUN_DIR" --method "$METHOD" --lib "$LIB" \
    ${POSITIONS:+--positions "$POSITIONS"} --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "[motif_rsa] $METHOD done -> $RUN_DIR/$METHOD/motif_rsa.csv"
