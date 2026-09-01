#!/bin/bash
#SBATCH --job-name=af3_build
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/af3_build/%x_%A.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
set -euo pipefail
# PARALLEL build: 3_af3_build.py shards the seqs across BUILD_WORKERS (=cpus) -> ~250k JSONs in ~minutes,
# not hours (the single-threaded builder used to race the walltime on backbone methods).

# 3_af3_build.sbatch — build AF3/ESMFold2 monomer-refold JSONs for one method (pipeline step 3, host-side).
# Backbone methods emit ~31k designs x NSEQ seqs => ~250k tiny JSONs: too slow/inode-heavy for the login
# node. Build onto $SCRATCH (inode-safe), then symlink <run>/<METHOD>/refold_inputs -> the scratch dir so
# 4_af3.sbatch (which reads <run>/<METHOD>/refold_inputs/bNNNN) finds them without touching /cluster/work.
# Runs in gen_env directly (3_af3_build.py loads esp/af3_inputs.py by file PATH -> no container, no PYTHONPATH).
#
#   mkdir -p /cluster/scratch/$USER/motifgo_logs/af3_build
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=proteina 3_af3_build.sbatch
#   sbatch --export=ALL,RUN_DIR=<run>,METHOD=laproteina 3_af3_build.sbatch   # co-design: seqs at top level
# Prints n_batch; then:  sbatch --array=1-<n_batch>%92 --export=ALL,RUN_DIR=<run>,METHOD=<m> 4_af3.sbatch

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|genie3|pp1c_bb|laproteina|pp1c_aa}"
BATCH="${BATCH:-500}"
PY="${GEN_PY:-/cluster/work/wicky/bekessel/gen_env/bin/python}"
DS="${DS:-/cluster/work/wicky/bekessel/motif_benchmark/scripts/evaluate}"

# co-design methods carry the seq in the generated PDB (seqs/); backbone methods get it from MPNN (mpnn/seqs).
case "$METHOD" in
  laproteina|pp1c_aa) SEQS="$RUN_DIR/$METHOD/seqs" ;;
  *)                  SEQS="$RUN_DIR/$METHOD/mpnn/seqs" ;;
esac
[ -d "$SEQS" ] || { echo "[3_af3_build] no seqs dir $SEQS"; exit 1; }

REFROOT="${REFROOT:-$SCRATCH/refold_inputs/$(basename "$RUN_DIR")}"
OUT="$REFROOT/$METHOD"                     # holds one bNNNN.tar per batch (INODE-SAFE: ~2k files, not ~250k)
mkdir -p "$OUT" /cluster/scratch/$USER/motifgo_logs/af3_build

# INODE-SAFE build: emit the ~250k loose JSONs into NODE-LOCAL $TMPDIR (not scratch — scratch has a ~1M
# file quota that ~1M input JSONs across methods blows), then collapse each batch into a single tar on
# scratch. 4_af3.sbatch extracts the batch tar into its own node-local $TMPDIR at fold time.
BUILD="${TMPDIR:-$SCRATCH/tmp}/af3build_${METHOD}_$$"
mkdir -p "$BUILD"
export BUILD_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
echo "[3_af3_build] $METHOD: $(find "$SEQS" -maxdepth 1 -type f | wc -l) seq files -> $OUT (batch=$BATCH, ${BUILD_WORKERS} workers, build in $BUILD)"
"$PY" "$DS/3_af3_build.py" "$SEQS" "$BUILD" "$BATCH"

# tar each batch dir down to one file on scratch, then drop the loose JSONs
for d in "$BUILD"/b*/; do
  [ -d "$d" ] || continue
  b=$(basename "$d")
  tar -cf "$OUT/$b.tar" -C "$d" .
done
rm -rf "$BUILD"

# point the run dir at the scratch input tars so 4_af3.sbatch resolves them
ln -sfn "$OUT" "$RUN_DIR/$METHOD/refold_inputs"
NB=$(ls "$OUT"/b*.tar 2>/dev/null | wc -l)
echo "[3_af3_build] $METHOD done: $NB batches (as bNNNN.tar). Next:"
echo "  sbatch --time=08:00:00 --array=1-$NB%92 --export=ALL,RUN_DIR=$RUN_DIR,METHOD=$METHOD $DS/4_af3.sbatch"
