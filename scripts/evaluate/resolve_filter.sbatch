#!/bin/bash
#SBATCH --job-name=resolve_filter
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/filter/%x_%A.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G           # light: 1_filter is pure numpy/Kabsch (no biotite/esp), parallel over designs
#SBATCH --time=02:00:00            # (peak RSS ~2-3G; 16G total is plenty). CPU job -> normal.* partition.
set -euo pipefail

# ============================================================================
# Motif-geometry filter for ONE method (CPU-only; do NOT run on the login node for the full library).
#   1_filter.py -> for EVERY design (all methods, incl. rfdiff): localize the native motif, Cα-superimpose,
#                  take min-RMSD = motifRMSD, keep motifRMSD < 1.0 A. Pure numpy, no esp, no .trb.
# Outputs: <run>/<method>/1_filter/{motif_rmsd.csv, passed/<design>.pdb (symlinks), passed.txt,
#                                    positions.jsonl (passers), tasks.json}  (passed/ = preserved motif).
#
# Runs in the gen_env venv (numpy). Defaults to the standard path; override with GEN_ENV=<dir>.
#   sbatch --cpus-per-task=16 --export=ALL,RUN_DIR=<run>,METHOD=<m> resolve_filter.sbatch
# ============================================================================

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|laproteina|pp1c_bb|genie3}"
DS="${DS:-/cluster/work/wicky/bekessel/motif_benchmark/scripts/evaluate}"
GEN_ENV="${GEN_ENV:-/cluster/work/wicky/bekessel/gen_env}"
CUTOFF="${CUTOFF:-1.0}"                           # motifRMSD gate (A). Set CUTOFF=99 to pass ALL designs
                                                  # (e.g. co-design MPNN-all for a symmetric comparison).

source "$GEN_ENV/bin/activate"                   # gen_env venv (numpy) — NOT conda
mkdir -p /cluster/scratch/$USER/motifgo_logs/filter

# ONE script, ONE code path (localize -> motifRMSD -> gate) for EVERY method incl. rfdiff. Pure numpy,
# no esp import, no .trb. Parallel over designs with all allocated CPUs.
python "$DS/1_filter.py" --run-dir "$RUN_DIR" --method "$METHOD" --cutoff "$CUTOFF" --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "[resolve_filter] $METHOD done -> $RUN_DIR/$METHOD/1_filter/ (passed/ = designs that preserved the motif <1A)"
