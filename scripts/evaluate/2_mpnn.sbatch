#!/bin/bash
#SBATCH --job-name=refold_mpnn
#SBATCH --output=/cluster/scratch/%u/motifgo_logs/mpnn/%x_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
unset PYTHONHOME PYTHONPATH        # don't leak host python env into the container

# ============================================================================
# 2_mpnn.sbatch — ProteinMPNN sequence design, pipeline step 2 (BACKBONE methods only; co-design methods
# skip this — their seq comes from the model). SHARDED SLURM ARRAY: one task = one slice of the passers.
#
# WHY sharded: at library scale a method yields ~31k passing backbones. Parsing + designing ALL of them in
# ONE CPU job stalls in parse_multiple_chains and never fits the walltime (the failure this replaces). Each
# array task parses + MPNNs ONLY its slice; sequences land in the SHARED <method>/mpnn/seqs/ (unique per-
# design filenames, so shards never collide). Skip-existing (seqs/<name>.fa) makes it requeue/bulk-safe.
#
# The generator kind is a knob (AUTO-DETECTED below, so a wrong value can't silently design from a Cα-only
# trace with the full model — the genie3 0%-designable bug):
#   CA_ONLY=1 (Proteina AND Genie3 — Cα-only backbone) -> --ca_only + ca_model_weights
#   CA_ONLY=0 (RFdiff / Protpardelle-bb — full N,Cα,C,O) -> vanilla ProteinMPNN (default)
# TEMP/NSEQ/SEED/OMIT match esp/genie3 so designs stay comparable across methods.
#
# PREREQ — run mpnn_prep ONCE (already done if <method>/mpnn/{prepped/,fixed_positions.jsonl} exist):
#   singularity exec -B /cluster/work/wicky:/cluster/work/wicky <openfold.sif> \
#       python $DS/mpnn_prep.py --run-dir <run> --method <m>
#
# THEN submit the array (pick NSHARDS so each shard is ~a few hundred designs; array 0..NSHARDS-1):
#   sbatch --array=0-99%50 --export=ALL,RUN_DIR=<run>,METHOD=proteina,CA_ONLY=1,NSHARDS=100 2_mpnn.sbatch
#   sbatch --array=0-99%50 --export=ALL,RUN_DIR=<run>,METHOD=pp1c_bb,NSHARDS=100          2_mpnn.sbatch
# GPU (~10x faster; ProteinMPNN has matmuls): add `--gpus=1 --gres=gpumem:8g` to sbatch and set USE_GPU=1.
# Outputs -> <run>/<METHOD>/mpnn/seqs/<design>.fa ; next: 3_af3_build on that seqs/ dir.
# ============================================================================

RUN_DIR="${RUN_DIR:?set RUN_DIR=<runs_root>/<run>}"
METHOD="${METHOD:?set METHOD=rfdiff|proteina|genie3|pp1c_bb}"
CA_ONLY="${CA_ONLY:-0}"            # 1 for Proteina (Cα-only generator)
NSHARDS="${NSHARDS:-50}"          # number of array shards; each shard ~ (#passers / NSHARDS) designs
SID="${SLURM_ARRAY_TASK_ID:-0}"
NSEQ="${NSEQ:-8}"                  # sequences per backbone (MotifBench: 8)
TEMP="${TEMP:-0.2}"              # Lisa/esp/genie3 proven sampling temp (comparability)
OMIT="${OMIT:-C}"                # no free Cys in the DESIGNED scaffold (motif Cys are fixed -> kept)
SEED="${SEED:-42}"
USE_GPU="${USE_GPU:-0}"          # 1 -> add --nv (also add #SBATCH --gpus=1 --gres=gpumem:8g when submitting)

DS="${DS:-/cluster/work/wicky/bekessel/motif_benchmark/scripts/evaluate}"
MP="${MP:-/cluster/work/wicky/bekessel/software/ProteinMPNN}"
SIF="${SIF:-/cluster/work/wicky/containers/openfold.sif}"

MDIR="$RUN_DIR/$METHOD"
OUT="$MDIR/mpnn"
mkdir -p "$OUT/seqs" "$OUT/shards" /cluster/scratch/$USER/motifgo_logs/mpnn
[ -f "$OUT/fixed_positions.jsonl" ] || { echo "[2_mpnn] no $OUT/fixed_positions.jsonl — run mpnn_prep.py once first"; exit 1; }
[ -d "$OUT/prepped" ]              || { echo "[2_mpnn] no $OUT/prepped — run mpnn_prep.py once first"; exit 1; }

NV=""; [ "$USE_GPU" = "1" ] && NV="--nv"
L="singularity exec $NV -B /cluster/work/wicky:/cluster/work/wicky $SIF"

# design list built ONCE from fixed_positions.jsonl (in-memory json read — NO 31k-file dir listing).
# atomic create (own tmp + mv) so concurrent shards on the first run don't corrupt it.
LIST="$OUT/design_list.txt"
if [ ! -f "$LIST" ]; then
  $L python -c "import json;print('\n'.join(sorted(json.load(open('$OUT/fixed_positions.jsonl')))))" > "$LIST.$SID.$$" && mv "$LIST.$SID.$$" "$LIST"
fi

# this shard's designs = every NSHARDS-th line (round-robin), MINUS any whose seqs already exist (resume).
SHARDDIR="$OUT/shards/$SID"; rm -rf "$SHARDDIR"; mkdir -p "$SHARDDIR"
awk -v s="$SID" -v n="$NSHARDS" '((NR-1)%n)==s' "$LIST" > "$SHARDDIR/names.txt"
todo=0
while IFS= read -r name; do
  [ -z "$name" ] && continue
  [ -f "$OUT/seqs/${name}.fa" ] && continue                     # skip-existing -> requeue-safe
  src="$OUT/prepped/${name}.pdb"
  [ -f "$src" ] && { ln -sf "$src" "$SHARDDIR/${name}.pdb"; todo=$((todo+1)); }
done < "$SHARDDIR/names.txt"
echo "[2_mpnn] $METHOD shard $SID/$NSHARDS: $todo designs to do (skip-existing applied)"
[ "$todo" -gt 0 ] || { echo "[2_mpnn] shard $SID nothing to do -> done"; rm -rf "$SHARDDIR"; exit 0; }

# AUTO-DETECT Cα-only backbones: vanilla ProteinMPNN needs N-Cα-C, but Genie3/Proteina emit a Cα-only trace
# (no backbone N/C). Designing those with the full model yields garbage sequences that refold nowhere near
# the scaffold (scRMSD ~12 A -> 0% designable). If the prepped PDBs carry no backbone N/C atom, force the
# Cα-only model regardless of the passed CA_ONLY, so the metric is never fed junk sequences.
if [ "$CA_ONLY" != "1" ]; then
  _probe=$(find "$SHARDDIR" -maxdepth 1 -name '*.pdb' -print -quit 2>/dev/null)
  if [ -n "$_probe" ] && [ "$(awk 'substr($0,1,4)=="ATOM"{a=substr($0,13,4);gsub(/ /,"",a);if(a=="N"||a=="C")c++}END{print c+0}' "$_probe")" = "0" ]; then
    echo "[2_mpnn] $METHOD: prepped backbones are Cα-only (no N/C) -> forcing CA_ONLY=1"
    CA_ONLY=1
  fi
fi

# 1) parse this shard's backbones into a shard-local jsonl (--ca_only only for Cα-only generators).
#    parse_chains.py = drop-in for ProteinMPNN's parse_multiple_chains.py, but reads each PDB ONCE
#    (stock re-reads every file 352x — once per chain letter — which stalls the parse at library scale).
CA_FLAG=""; [ "$CA_ONLY" = "1" ] && CA_FLAG="--ca_only"
$L python "$DS/parse_chains.py" \
    --input_path "$SHARDDIR" --output_path "$SHARDDIR/pdbs.jsonl" $CA_FLAG

# 2) ProteinMPNN with the motif FIXED to native seq. out_folder = the SHARED $OUT so every shard writes
#    seqs/<design>.fa into one place. The GLOBAL fixed_positions.jsonl is fine — MPNN only uses the keys
#    present in this shard's jsonl.
MODEL_ARGS="--model_name v_48_020"
[ "$CA_ONLY" = "1" ] && MODEL_ARGS="--ca_only --path_to_model_weights $MP/ca_model_weights --model_name v_48_020"
$L python "$MP/protein_mpnn_run.py" \
    --jsonl_path "$SHARDDIR/pdbs.jsonl" \
    --fixed_positions_jsonl "$OUT/fixed_positions.jsonl" \
    $MODEL_ARGS \
    --out_folder "$OUT" \
    --num_seq_per_target "$NSEQ" \
    --sampling_temp "$TEMP" \
    --omit_AAs "$OMIT" \
    --seed "$SEED" \
    --batch_size 1

rm -rf "$SHARDDIR"     # drop the shard's symlinks + jsonl (inode-light); seqs are in $OUT/seqs
echo "[2_mpnn] $METHOD shard $SID done -> $OUT/seqs/ ($todo designs). Next (after all shards): 3_af3_build on $OUT/seqs"
