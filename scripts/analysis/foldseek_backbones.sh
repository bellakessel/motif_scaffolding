#!/bin/bash
#SBATCH --job-name=fs_backbones
#SBATCH --output=/cluster/work/wicky/bekessel/scripts/logs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#
# Cross-method structural clustering of the motif_library de novo BACKBONES.
#
# QUESTION: do the 5 generators (rfdiff, proteina, pp1c_bb=Protpardelle, laproteina,
# genie3) occupy distinct fold space? Backbones are clustered directly -- NOT the
# ProteinMPNN sequences or their AF3 refolds. Clustering refolds would mostly measure
# AF3's ability to recapitulate an intended backbone (8 seqs x 5 methods x 31.4k = 1.26M
# structures of mostly noise); Foldseek's 3Di alphabet reads the backbone geometry itself.
#
# INPUT: each method's motif-RMSD-passing designs, already flat and path-listed:
#     $RUN/{method}/1_filter/passed.txt   ->  .../1_filter/passed/{design}.pdb
#   counts: proteina 31400  laproteina 31400  pp1c_bb 31263  genie3 31154  rfdiff 24401
#           = 149,618 backbones  (rfdiff loses ~22% at the motif-RMSD gate)
#
# THE CATCH THAT MAKES normalize_backbones.py NECESSARY: the 5 generators write
# mutually incompatible PDBs. proteina and genie3 are CA-ONLY, so their 3Di cannot be
# computed at all and createdb drops/garbage-encodes them WITHOUT erroring. And residue
# naming (poly-Gly / all-ALA / all-UNK / real sequence) is perfectly correlated with the
# generator, so the AA term in Foldseek's score would separate the methods on naming
# rather than fold. `--min-seq-id 0.0` does NOT fix that -- it drops the seq-id filter,
# not the AA term. Every structure is therefore reduced to its CA trace, rebuilt with the
# SAME fitted geometry, and renamed to poly-GLY. See that script's header.
#
# OUTPUT (small, persisted to /work; heavy DBs stay on $SCRATCH):
#     reports/foldseek_bb/clusters_methods.tsv  representative <TAB> member
#                                               member = {method}__{design}
#     reports/foldseek_bb/reps_hits.tsv         query <TAB> target <TAB> alntmscore  (UMAP graph)
#     reports/foldseek_bb/bb_lengths.tsv        method <TAB> design <TAB> n_res
#     reports/foldseek_bb/bb_coeffs.json        the fitted rebuild geometry (provenance)
#
# The member ID carries the generator, and {design} joins to
# {method}/1_filter/motif_rmsd.csv (design -> task -> motifRMSD) and thence to the run's
# manifest.csv (task_name -> motif_num / variant / n_fragments / contig). That is what the
# notebook uses to colour clusters by generation mode and by motif.
#
# SUBMIT (in order; each stage is resumable and re-runnable):
#     sbatch --export=ALL,STAGE=smoke     foldseek_backbones.sh   # 5 min, PROVE IT FIRST
#     sbatch --export=ALL,STAGE=all       foldseek_backbones.sh   # fit + normalize + cluster
#     sbatch --export=ALL,STAGE=graph     foldseek_backbones.sh   # rep-rep TM graph for UMAP
#   individual stages: STAGE=fit | normalize | cluster | graph
#
# SIZING: measured on the previous Foldseek phase (job 3608643) at the same scale --
# 181k structures gave createdb 32s, cluster ~10s, peak RSS ~1.4-5G. Normalisation is
# ~2-3 ms/structure, so 149.6k over 16 workers is a few minutes, I/O-bound on reading
# /work. 16 cpu / 32G / 4h is generous on purpose; `graph` is the only slow stage.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
MINE="/cluster/work/wicky/bekessel"
RUN="${RUN:-$MINE/motif_library/runs/lib_interface_pad5}"
METHODS="${METHODS:-rfdiff proteina pp1c_bb laproteina genie3}"
FULLBB="${FULLBB:-rfdiff pp1c_bb laproteina}"   # the 3 that ship real N/CA/C/O -> fit on these
STAGE="${STAGE:-all}"
PY="${PY:-$MINE/gen_env/bin/python}"            # gen_env is a VENV, not conda
NORMALIZER="${NORMALIZER:-$MINE/scripts/normalize_backbones.py}"
SIF="${SIF:-/cluster/work/wicky/containers/foldseek.sif}"

# clustering params. --min-seq-id is moot once everything is poly-Gly, but is passed so
# the run is explicit about ignoring sequence. --cov-mode 1 = coverage of target.
COV="${COV:-0.8}"
COVMODE="${COVMODE:-1}"
MINSEQID="${MINSEQID:-0.0}"
SENS="${SENS:-7.5}"

# rep-rep graph params. DELIBERATELY permissive: the reps were chosen because they are
# mutually dissimilar, so a strict -e makes the 3Di prefilter reject nearly every rep-rep
# pair and the graph collapses to self-hits (learned the hard way in foldseek_search_reps.sh).
# We threshold on TM-score in the notebook, not on e-value here.
GSENS="${GSENS:-9.5}"
GEVAL="${GEVAL:-10}"
GMAXSEQS="${GMAXSEQS:-1000}"
ALNTYPE="${ALNTYPE:-1}"                          # 1 = TMalign -> gives alntmscore

FIT_N="${FIT_N:-400}"                            # structures per list for the geometry fit
CHECK_N="${CHECK_N:-200}"                         # held-out structures for selfcheck
SMOKE_N="${SMOKE_N:-200}"                         # designs per method in the smoke run

SCRATCH_ROOT="${SCRATCH:-/cluster/scratch/bekessel}"
WORK="${SCRATCH_ROOT}/foldseek_bb"
NORM="${WORK}/bb_norm"                            # ONE flat dir, method-prefixed names
FSDB="${WORK}/fsdb"
TMP="${WORK}/tmp"
COEFFS="${WORK}/bb_coeffs.json"
LENGTHS="${WORK}/bb_lengths.tsv"
OUT="${MINE}/reports/foldseek_bb"
CPUS="${SLURM_CPUS_PER_TASK:-16}"

if [ "$STAGE" = "smoke" ]; then                   # keep the smoke run fully separate
    WORK="${SCRATCH_ROOT}/foldseek_bb_smoke"
    NORM="${WORK}/bb_norm"; FSDB="${WORK}/fsdb"; TMP="${WORK}/tmp"
    COEFFS="${WORK}/bb_coeffs.json"; LENGTHS="${WORK}/bb_lengths.tsv"
    OUT="${WORK}/reports"
fi

mkdir -p "$WORK" "$FSDB" "$TMP" "$OUT" "$MINE/scripts/logs"

echo "=========================================================================="
echo "STAGE=$STAGE  node=${SLURMD_NODENAME:-$(hostname)}  cpus=$CPUS"
echo "run=$RUN"
echo "methods=$METHODS"
echo "scratch work=$WORK"
echo "persisted out=$OUT"
echo "=========================================================================="

[ -x "$PY" ]        || { echo "ERROR: python not executable: $PY"; exit 1; }
[ -r "$NORMALIZER" ] || { echo "ERROR: normalizer not readable: $NORMALIZER"; exit 1; }

list_of () { echo "${RUN}/$1/1_filter/passed.txt"; }

# every method's path list must exist before we do anything expensive
for m in $METHODS; do
    l="$(list_of "$m")"
    [ -r "$l" ] || { echo "ERROR: missing path list for '$m': $l"; exit 1; }
done

# ── foldseek runner ───────────────────────────────────────────────────────────
# This foldseek.sif has NO plain 'foldseek' on PATH -- only /usr/local/bin/foldseek_{avx2,
# sse41} plus an 'entrypoint' that auto-selects by CPU flags. So invoke via `singularity
# run` (which goes through the entrypoint), NOT `exec foldseek`.
setup_foldseek () {
    module load eth_proxy 2>/dev/null || true
    module load stack/2024-06 2>/dev/null || true
    RUNNER="$(command -v singularity || command -v apptainer || true)"
    [ -n "$RUNNER" ] || { echo "ERROR: no singularity/apptainer on PATH (module load?)"; exit 1; }
    [ -r "$SIF" ]    || { echo "ERROR: foldseek .sif not readable: $SIF"; exit 1; }
    FOLDSEEK=("$RUNNER" run -B /cluster/work/wicky -B /cluster/scratch "$SIF")
    echo "runner: $RUNNER   sif: $SIF"
    "${FOLDSEEK[@]}" version || { echo "ERROR: foldseek not runnable via 'run'"; exit 1; }
}

# ── stage: fit the rebuild geometry ───────────────────────────────────────────
do_fit () {
    local n="${1:-$FIT_N}"
    local lists=()
    for m in $FULLBB; do lists+=("$(list_of "$m")"); done
    echo ""
    echo "### fit rebuild geometry on the full-backbone methods ($FULLBB), ${n}/list"
    "$PY" "$NORMALIZER" fit --lists "${lists[@]}" --out "$COEFFS" --nsample "$n"
    echo ""
    echo "### selfcheck on HELD-OUT structures (skip=${n})"
    "$PY" "$NORMALIZER" selfcheck --lists "${lists[@]}" --coeffs "$COEFFS" \
        --nsample "$CHECK_N" --skip "$n"
}

# ── stage: normalise every method into ONE flat dir ───────────────────────────
do_normalize () {
    local limit_arg=()
    [ -n "${1:-}" ] && limit_arg=(--limit "$1")
    [ -r "$COEFFS" ] || { echo "ERROR: $COEFFS missing -- run STAGE=fit first"; exit 1; }
    # full rebuild every time: a partial NORM dir would silently under-populate the DB
    rm -rf "$NORM"; mkdir -p "$NORM"
    : > "$LENGTHS"
    echo ""
    echo "### normalise -> $NORM   (CA-only -> rebuilt N/C/O -> poly-GLY)"
    for m in $METHODS; do
        "$PY" "$NORMALIZER" apply \
            --list "$(list_of "$m")" --method "$m" --coeffs "$COEFFS" \
            --out-dir "$NORM" --stats "$LENGTHS" --workers "$CPUS" "${limit_arg[@]}"
    done
    local n
    n=$(find "$NORM" -maxdepth 1 -name '*.pdb' | wc -l)
    echo "normalised total: $n .pdb in $NORM"
    echo ""
    echo "### n_res distribution per method (length is a CONFOUND for -c ${COV}:"
    echo "### if methods differ systematically in length, clusters split on length, not fold)"
    awk -F'\t' '{n[$1]++; s[$1]+=$3; if(!($1 in mn)||$3<mn[$1])mn[$1]=$3;
                 if(!($1 in mx)||$3>mx[$1])mx[$1]=$3; v[$1]=v[$1]" "$3}
        END{printf "%-12s %8s %7s %7s %7s\n","method","n","min","mean","max";
            for(k in n) printf "%-12s %8d %7d %7.1f %7d\n",k,n[k],mn[k],s[k]/n[k],mx[k]}' \
        "$LENGTHS" | sort
    cp -f "$LENGTHS" "$OUT/bb_lengths.tsv"
    cp -f "$COEFFS"  "$OUT/bb_coeffs.json"
}

# ── stage: cluster ────────────────────────────────────────────────────────────
do_cluster () {
    setup_foldseek
    local db="${FSDB}/bb_DB" clu="${FSDB}/bb_clu" tsv="${OUT}/clusters_methods.tsv"
    local n
    n=$(find "$NORM" -maxdepth 1 -name '*.pdb' | wc -l)
    [ "$n" -gt 0 ] || { echo "ERROR: no .pdb in $NORM -- run STAGE=normalize first"; exit 1; }
    echo ""
    echo "### createdb on $n normalised backbones"
    # foldseek refuses to overwrite existing DBs -> clear first (makes the stage re-runnable)
    rm -rf "${db}"* "${clu}"* "${TMP}/clu" "$tsv"
    # NORM is already a single flat dir with globally-unique method-prefixed names, so no
    # symlink farm is needed (and a *.pdb glob would overflow ARG_MAX at 149k files anyway --
    # createdb takes the DIRECTORY, not a glob).
    "${FOLDSEEK[@]}" createdb "$NORM" "$db"
    echo ""
    echo "### cluster (-c $COV --cov-mode $COVMODE --min-seq-id $MINSEQID -s $SENS, $CPUS threads)"
    "${FOLDSEEK[@]}" cluster "$db" "$clu" "${TMP}/clu" \
        --threads "$CPUS" -c "$COV" --cov-mode "$COVMODE" \
        --min-seq-id "$MINSEQID" -s "$SENS"
    "${FOLDSEEK[@]}" createtsv "$db" "$db" "$clu" "$tsv"

    local nmem nrep
    nmem=$(wc -l < "$tsv"); nrep=$(cut -f1 "$tsv" | sort -u | wc -l)
    echo ""
    echo "members=$nmem  clusters=$nrep  -> $tsv"
    if [ "$nmem" -ne "$n" ]; then
        echo "WARN: $nmem members but $n input structures -- $((n - nmem)) were DROPPED by foldseek."
        echo "      (this is exactly the silent-drop failure the normalisation exists to prevent)"
    fi
    echo ""
    echo "### members per method (sanity: all 5 must be present in roughly their input counts)"
    cut -f2 "$tsv" | awk -F'__' '{c[$1]++} END{for(k in c) printf "  %-12s %8d\n",k,c[k]}' | sort
}

# ── stage: rep-rep TM-score graph (for the notebook's UMAP) ───────────────────
do_graph () {
    setup_foldseek
    local tsv="${OUT}/clusters_methods.tsv"
    local farm="${WORK}/reps" db="${FSDB}/reps_DB" aln="${FSDB}/reps_aln"
    local hits="${OUT}/reps_hits.tsv"
    [ -r "$tsv" ] || { echo "ERROR: $tsv missing -- run STAGE=cluster first"; exit 1; }

    rm -rf "$farm" "${db}"* "${aln}"* "${TMP}/reps" "$hits"
    mkdir -p "$farm" "${TMP}/reps"
    echo ""
    echo "### gather cluster representatives -> $farm"
    local missing=0
    while read -r name; do
        if [ -f "${NORM}/${name}.pdb" ]; then ln -s "${NORM}/${name}.pdb" "${farm}/${name}.pdb"
        else missing=$((missing + 1)); fi
    done < <(cut -f1 "$tsv" | sort -u)
    local nrep
    nrep=$(find "$farm" -maxdepth 1 -name '*.pdb' | wc -l)
    echo "linked $nrep representatives (missing $missing)"

    "${FOLDSEEK[@]}" createdb "$farm" "$db"
    echo ""
    echo "### all-vs-all TMalign among reps (-s $GSENS -e $GEVAL --max-seqs $GMAXSEQS)"
    "${FOLDSEEK[@]}" search "$db" "$db" "$aln" "${TMP}/reps" \
        --threads "$CPUS" -s "$GSENS" -e "$GEVAL" --max-seqs "$GMAXSEQS" \
        --alignment-type "$ALNTYPE" -a 1
    "${FOLDSEEK[@]}" convertalis "$db" "$db" "$aln" "$hits" \
        --format-output "query,target,alntmscore"
    local nh
    nh=$(wc -l < "$hits")
    echo ""
    echo "$nh rep-rep hits -> $hits  (avg $(awk -v n="$nrep" -v h="$nh" \
        'BEGIN{if(n)printf "%.1f", h/n; else print "0"}') neighbours/rep)"
    if [ "$nh" -le "$((nrep * 2))" ]; then
        echo "WARN: graph looks collapsed (~1 neighbour/rep = self-hits only)."
        echo "      Raise sensitivity/permissiveness: GEVAL=100 GMAXSEQS=2000, or EXHAUSTIVE."
    fi
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "$STAGE" in
    smoke)
        echo "SMOKE RUN: ${SMOKE_N} designs/method, isolated under $WORK"
        do_fit 100
        do_normalize "$SMOKE_N"
        do_cluster
        echo ""
        echo "SMOKE OK. Check above that: (a) selfcheck virtual_centre error is small,"
        echo "(b) all 5 methods appear in the members-per-method table with ~${SMOKE_N} each,"
        echo "(c) no 'DROPPED' warning. Then: sbatch --export=ALL,STAGE=all $0"
        ;;
    fit)        do_fit ;;
    normalize)  do_normalize ;;
    cluster)    do_cluster ;;
    graph)      do_graph ;;
    all)        do_fit; do_normalize; do_cluster ;;
    *)          echo "ERROR: unknown STAGE='$STAGE' (smoke|fit|normalize|cluster|graph|all)"; exit 1 ;;
esac

echo ""
echo "DONE (STAGE=$STAGE). Persisted: $OUT"
