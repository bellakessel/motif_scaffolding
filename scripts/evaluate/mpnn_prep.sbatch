#!/bin/bash
#SBATCH --job-name=mpnn_prep
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/mpnn_prep/%x_%A.out
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00
set -euo pipefail

# mpnn_prep.sbatch — parallel motif-identity stamping for one method's passers (pipeline step, host-side).
# Pure per-design file I/O, so it scales with CPUs; skip-existing makes it resumable. Runs in gen_env
# directly (no container, no esp import). One CPU job, many workers (NOT a SLURM array).
#
#   mkdir -p /cluster/scratch/$USER/motifgo_logs/mpnn_prep      # Slurm opens --output before the script runs
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=genie3 mpnn_prep.sbatch
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=rfdiff mpnn_prep.sbatch
# Outputs -> <run>/<METHOD>/mpnn/{prepped/, fixed_positions.jsonl}. PREREQ: 1_filter must have run.

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|genie3|pp1c_bb}"
POSITIONS="${POSITIONS:-}"        # override positions file (e.g. greedy positions_unindexed.jsonl for uidx MPNN)
PRESERVE="${PRESERVE:-}"          # 1 -> --preserve-order (greedy positions are in native-residue order)
PY="${GEN_PY:-/cluster/work/wicky/bekessel/gen_env/bin/python}"
DS="${DS:-/cluster/work/wicky/bekessel/motif_benchmark/scripts/evaluate}"

mkdir -p /cluster/scratch/$USER/motifgo_logs/mpnn_prep
"$PY" "$DS/mpnn_prep.py" --run-dir "$RUN_DIR" --method "$METHOD" \
    ${POSITIONS:+--positions "$POSITIONS"} ${PRESERVE:+--preserve-order} \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
