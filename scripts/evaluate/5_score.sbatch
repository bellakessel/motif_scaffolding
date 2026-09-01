#!/bin/bash
#SBATCH --job-name=refold_score
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/score/%x_%A.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
set -euo pipefail

# 5_score.sbatch — refold designability scoring for one method (pipeline step 5, host-side, CPU-only).
# Reuses esp/score.py metrics (MotifBench: motifRMSD<=1 & scRMSD<=2). Reads <run>/<METHOD>/af3/archives/
# *.tar.zst; PARALLEL over archives (--workers = SLURM_CPUS_PER_TASK), each worker extracts one archive into
# NODE-LOCAL $TMPDIR and scores it. Backbone methods have 8 seqs/design (~250k structures) -> serial is ~7h;
# 16 workers brings it to ~40min. Runs gen_env directly (biotite; no container, no GPU, no esp PYTHONPATH —
# 5_score.py loads esp/score.py by file PATH). Writes <run>/<METHOD>/af3_refold_scores.csv + rollup.
#
#   mkdir -p /cluster/scratch/$USER/motifgo_logs/score
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=laproteina 5_score.sbatch
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=proteina,PREDICTOR=af3 5_score.sbatch

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|genie3|pp1c_bb|laproteina|pp1c_aa}"
PREDICTOR="${PREDICTOR:-af3}"
DESIGN_CHAIN="${DESIGN_CHAIN:-A}"
POSITIONS="${POSITIONS:-}"                        # override motif positions (e.g. positions_unindexed.jsonl)
OUT="${OUT:-}"                                    # output CSV (default <run>/<method>/<predictor>_refold_scores.csv)
PY="${GEN_PY:-/cluster/work/wicky/bekessel/gen_env/bin/python}"
DS="${DS:-/cluster/work/wicky/bekessel/motif_benchmark/scripts/evaluate}"

mkdir -p /cluster/scratch/$USER/motifgo_logs/score
"$PY" "$DS/5_score.py" --run-dir "$RUN_DIR" --method "$METHOD" \
    --predictor "$PREDICTOR" --design-chain "$DESIGN_CHAIN" \
    ${POSITIONS:+--positions "$POSITIONS"} ${OUT:+--out "$OUT"} \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
