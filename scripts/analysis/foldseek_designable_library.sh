#!/bin/bash
#SBATCH --job-name=fs_designable
#SBATCH --output=/cluster/work/wicky/bekessel/scripts/logs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#
# SSE + fold-space characterisation of the DESIGNABLE library (motif_library/runs/lib_interface_pad5).
#
# QUESTION: what secondary-structure space do the generators actually cover once the design has
# been through ProteinMPNN and refolded -- and how much of the apparent helix dominance is a
# selection artefact rather than a property of the generators? Replicates Figure 2 of
# doi:10.64898/2026.02.16.706194 (ternary coil/helix/sheet + Foldseek cluster sampling).
#
# WHY THE REFOLDS AND NOT THE BACKBONES. foldseek_backbones.sh had to reduce all five generators
# to CA and rewrite them poly-Gly (proteina/genie3 are CA-ONLY; residue naming was perfectly
# method-correlated), which discarded the inverse-folding step and gave a weak signal
# (AMI(cluster, generator) = 0.041). The AF3 refolds are uniform across methods by construction
# -- all-atom, real sequence, one chain -- so NO normalisation is needed here at all.
#
# THE GATE (project-wide, see 5_score.py:RANK_MIN):
#     designable = motif_ok AND sc_ok AND ranking_score >= 0.75          <- INCLUSIVE
# The `designable` column in af3_refold_scores.csv is only the first TWO gates; those CSVs predate
# 5_score.py growing RANK_MIN. `ranking_score` is rounded to 2 dp so a large mass sits exactly at
# 0.75 -- in this run, 12,102 predictions, i.e. `>` would silently drop ~14% of the set.
#
# INPUT   $RUN/{method}/af3/archives/*.tar.zst   +   $RUN/{method}/af3_refold_scores.csv
#   2-gate / 3-gate preds and designs (verified 2026-08-26, all five methods present):
#     genie3          54,763 / 10,997  ->  26,119 / 6,805      rfdiff   50,270 / 10,261 -> 17,199 / 5,141
#     laproteina_mpnn 55,620 / 10,881  ->  26,149 / 6,273      pp1c_bb  47,937 / 10,946 -> 15,107 / 5,060
#     proteina        10,323 /  3,428  ->  10,323 / 3,428      <- ALREADY 3-GATE, see below
#     TOTAL          218,913 / 46,513  ->  94,897 / 26,707
#
#   PROTEINA IS PRE-GATED. It was refolded last and scored with the CURRENT 5_score.py, whose
#   `designable` column already applies RANK_MIN -- its CSV has no sub-0.75 rows at all (min
#   ranking_score exactly 0.750). The other four predate that change and carry the full two-gate
#   set. Consequences: every THREE-gate number is comparable across all five, but proteina
#   contributes nothing to the two-gate set, so the notebook's gate-effect figure detects
#   pre-gated methods and drops them from that comparison only. Nothing to do here -- the
#   extractor is agnostic, it just annotates whatever `designable == 1` rows exist.
#
#   To add or re-do ONE method later without redoing the rest:
#     sbatch --export=ALL,STAGE=sse,METHODS=<m>,APPEND=1 foldseek_designable_library.sh
#     sbatch --export=ALL,STAGE=cluster                  foldseek_designable_library.sh
#   APPEND=1 keeps the existing TSV and CIF dir and refuses to re-add a method already in the TSV;
#   without it, STAGE=sse rebuilds both from scratch (which is what you want for a full re-run).
#
# OUTPUT (persisted; heavy DBs stay on $SCRATCH)
#   $OUT/pred_sse.tsv          one row per TWO-gate designable prediction, with conf_ok / is_best
#   $OUT/clusters_tm050.tsv    representative <TAB> member, THREE-gate best-per-design only
#                              (26,707 designs across the five generators)
#   $OUT/designable_cifs.tar.zst   the clustered structures (inode discipline: 23k files -> 1 archive)
#
# SSE is annotated on the WIDER two-gate set on purpose: it is free (the archive is already
# decompressed) and it is what lets the notebook show what the confidence gate does to the SSE
# distribution -- the reference preprint's central claim -- without a second full pass.
#
# SUBMIT
#     sbatch --export=ALL,STAGE=smoke   foldseek_designable_library.sh   # ~10 min, PROVE IT FIRST
#     sbatch --export=ALL,STAGE=all     foldseek_designable_library.sh   # sse + cluster + graph
#   individual stages: STAGE=sse | cluster | graph
#
#   Lisa's 1n8z/2mhs/3bis campaign (5x bigger -- ask for more wall time at submit):
#     sbatch --time=12:00:00 --export=ALL,DATASET=lisa,STAGE=smoke foldseek_designable_library.sh
#     sbatch --time=12:00:00 --export=ALL,DATASET=lisa,STAGE=all   foldseek_designable_library.sh
#
# SIZING: extraction is I/O-bound on reading 27.3 GB of archives off /cluster/work, ~4 s/archive
# measured; 2,407 archives over 16 workers is ~15-25 min. DSSP is 2.5 ms/structure (vendored
# NumPy kernel, no torch). Clustering 27k structures with TMalign is 10-40 min. 4 h is generous.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
MINE="/cluster/work/wicky/bekessel"

# DATASET selects which body of work to characterise. Both use the SAME extractor and the SAME
# TM-0.5 clustering, so their numbers are directly comparable -- which is the whole point of
# re-running Lisa's proteins rather than reusing the 2026-06 clusters (those used foldseek's
# default 3Di+AA alignment with -c 0.8 --cov-mode 0 -s 7.5 and NO TM threshold).
#   lib  = motif_library/runs/lib_interface_pad5, 5 generators x 314 interface motifs
#   lisa = the original 1n8z / 2mhs / 3bis campaign (RFdiffusion only, c_motif)
DATASET="${DATASET:-lib}"
case "$DATASET" in
  lib)
    RUN="${RUN:-$MINE/motif_library/runs/lib_interface_pad5}"
    METHODS="${METHODS:-rfdiff genie3 proteina pp1c_bb laproteina_mpnn}"
    PROFILE="${PROFILE:-manifest}"      # task via tasks.json + manifest
    ANNOTATE="${ANNOTATE:-all2g}"       # keep pre-gate rows -> confidence-gate figure
    OUT_DEFAULT="$MINE/motif_library/foldseek"
    WORK_NAME="foldseek_designable"
    ;;
  lisa)
    # 2,079 archives / 152 GB / 6.3M predictions -> 126,531 three-gate designs.
    # ANNOTATE=best: the two-gate set here is 1.98M rows (~500 MB of TSV) that no figure uses.
    RUN="${RUN:-$MINE/data}"
    METHODS="${METHODS:-1n8z 2mhs 3bis}"
    PROFILE="${PROFILE:-csv}"           # pdb_name + motif_num are COLUMNS of the scores CSV
    ANNOTATE="${ANNOTATE:-best}"
    OUT_DEFAULT="$MINE/analysis/reports/foldseek_tm050"
    WORK_NAME="foldseek_lisa"
    ;;
  *) echo "ERROR: DATASET must be lib|lisa (got '$DATASET')"; exit 1 ;;
esac
STAGE="${STAGE:-all}"
PY="${PY:-$MINE/gen_env/bin/python}"              # gen_env is a VENV, not conda
EXTRACT="${EXTRACT:-$MINE/scripts/designable_sse_extract.py}"
SIF="${SIF:-/cluster/work/wicky/containers/foldseek.sif}"
CONF_MIN="${CONF_MIN:-0.75}"                      # INCLUSIVE ranking_score gate
APPEND="${APPEND:-0}"                             # 1 = add methods to an existing TSV/CIF dir
                                                  #     (proteina, once its refold finishes)

# clustering params: the MotifBench / Genie 2 / Proteina convention, verbatim.
# --alignment-type 1 = TMalign, --tmscore-threshold 0.5 = the "same fold" line everyone uses.
# --min-seq-id 0 is moot under TMalign (the alignment is purely structural) but is passed so the
# run is explicit that sequence identity plays no part.
ALNTYPE="${ALNTYPE:-1}"
TMTHRESH="${TMTHRESH:-0.5}"
COVMODE="${COVMODE:-0}"
MINSEQID="${MINSEQID:-0}"
SENS="${SENS:-7.5}"

# graph stage (rep-rep TM edges for the notebook UMAP). DELIBERATELY permissive: the reps were
# chosen BECAUSE they are mutually dissimilar, so a strict -e makes the 3Di prefilter reject
# nearly every rep-rep pair and the graph collapses to self-hits (learned in foldseek_search_reps.sh).
GSENS="${GSENS:-9.5}"
GEVAL="${GEVAL:-10}"
GMAXSEQS="${GMAXSEQS:-1000}"

SCRATCH_ROOT="${SCRATCH:-/cluster/scratch/bekessel}"
WORK="${SCRATCH_ROOT}/${WORK_NAME}"
CIFS="${WORK}/cifs"
FSDB="${WORK}/fsdb"
TMP="${WORK}/tmp"
OUT="${OUT:-$OUT_DEFAULT}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
SMOKE_ARCHIVES="${SMOKE_ARCHIVES:-8}"             # per method, EVENLY SPACED (never the head)

if [ "$STAGE" = "smoke" ]; then                   # keep the smoke run fully separate
    WORK="${SCRATCH_ROOT}/${WORK_NAME}_smoke"
    CIFS="${WORK}/cifs"; FSDB="${WORK}/fsdb"; TMP="${WORK}/tmp"; OUT="${WORK}/reports"
fi

TSV="${OUT}/pred_sse.tsv"
CLU="${OUT}/clusters_tm050.tsv"
HITS="${OUT}/reps_hits.tsv"
REPCIF="${WORK}/reps_cif"

mkdir -p "$WORK" "$CIFS" "$FSDB" "$TMP" "$OUT" "$MINE/scripts/logs"

echo "=========================================================================="
echo "STAGE=$STAGE  node=${SLURMD_NODENAME:-$(hostname)}  cpus=$CPUS"
echo "run=$RUN"
echo "dataset=$DATASET  methods=$METHODS  profile=$PROFILE  annotate=$ANNOTATE"
echo "conf gate: ranking_score >= $CONF_MIN"
echo "scratch work=$WORK"
echo "persisted out=$OUT"
echo "=========================================================================="

[ -x "$PY" ]       || { echo "ERROR: python not executable: $PY"; exit 1; }
[ -r "$EXTRACT" ]  || { echo "ERROR: extractor not readable: $EXTRACT"; exit 1; }

archives_of () {                                  # $1 = method / protein
    if [ "$DATASET" = "lisa" ]; then echo "$RUN/$1/AF3/af3_output"
    else echo "$RUN/$1/af3/archives"; fi
}
scores_of () {
    if [ "$DATASET" = "lisa" ]; then echo "$MINE/analysis/reports/af3_refold_$1_c_motif.csv"
    else echo "$RUN/$1/af3_refold_scores.csv"; fi
}
for m in $METHODS; do
    [ -d "$(archives_of "$m")" ] || { echo "ERROR: no archives for '$m': $(archives_of "$m")"; exit 1; }
    [ -r "$(scores_of "$m")" ]   || { echo "ERROR: no scores CSV for '$m': $(scores_of "$m")"; exit 1; }
done

# ── foldseek runner ───────────────────────────────────────────────────────────
# This foldseek.sif has NO plain 'foldseek' on PATH -- only /usr/local/bin/foldseek_{avx2,sse41}
# plus an 'entrypoint' that auto-selects by CPU flags. Invoke via `singularity run` (which goes
# through the entrypoint), NOT `exec foldseek`.
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

# ── stage: DSSP + designable CIFs ─────────────────────────────────────────────
do_sse () {
    local limit_arg=()
    [ -n "${1:-}" ] && limit_arg=(--limit "$1")
    if [ "$APPEND" = "1" ]; then
        echo "### APPEND=1 -- keeping the existing $TSV and $CIFS"
        mkdir -p "$CIFS"
        for m in $METHODS; do
            grep -q -m1 "^$m$(printf '\t')" "$TSV" 2>/dev/null && \
                { echo "ERROR: '$m' already has rows in $TSV -- refusing to duplicate."; exit 1; }
        done
    else
        # a partial CIF dir or a half-written TSV would silently under-populate the DB
        rm -rf "$CIFS"; mkdir -p "$CIFS"
        rm -f "$TSV"
    fi
    echo ""
    echo "### DSSP over the designable refolds -> $TSV ; CIFs -> $CIFS"
    for m in $METHODS; do
        echo ""
        "$PY" "$EXTRACT" --run "$RUN" --method "$m" \
            --archives "$(archives_of "$m")" --scores "$(scores_of "$m")" \
            --profile "$PROFILE" --annotate "$ANNOTATE" --with-seq \
            --out-tsv "$TSV" --cif-out-dir "$CIFS" \
            --workers "$CPUS" --conf-min "$CONF_MIN" "${limit_arg[@]}"
    done
    local n
    n=$(find "$CIFS" -maxdepth 1 -name '*.cif' | wc -l)
    echo ""
    echo "### designable CIFs written: $n   (rows in $TSV: $(( $(wc -l < "$TSV") - 1 )))"
    echo "### per-method CIF counts:"
    find "$CIFS" -maxdepth 1 -name '*.cif' -printf '%f\n' | awk -F'__' '{c[$1]++}
        END{for(m in c) printf "    %-18s %6d\n", m, c[m]}' | sort
}

# ── stage: field-standard TM-0.5 clustering ───────────────────────────────────
do_cluster () {
    setup_foldseek
    local n
    n=$(find "$CIFS" -maxdepth 1 -name '*.cif' | wc -l)
    [ "$n" -gt 0 ] || { echo "ERROR: no CIFs in $CIFS -- run STAGE=sse first"; exit 1; }
    echo ""
    echo "### createdb over $n designable refolds"
    rm -rf "$FSDB" "$TMP"; mkdir -p "$FSDB" "$TMP"
    "${FOLDSEEK[@]}" createdb "$CIFS" "$FSDB/db" --threads "$CPUS"

    echo ""
    echo "### cluster  --alignment-type $ALNTYPE --tmscore-threshold $TMTHRESH"
    echo "###          --cov-mode $COVMODE --min-seq-id $MINSEQID -s $SENS"
    echo "### (MotifBench 'unique solutions' / Genie 2 + Proteina 'diversity' convention)"
    "${FOLDSEEK[@]}" cluster "$FSDB/db" "$FSDB/db_clu" "$TMP" \
        --alignment-type "$ALNTYPE" --tmscore-threshold "$TMTHRESH" \
        --cov-mode "$COVMODE" --min-seq-id "$MINSEQID" -s "$SENS" --threads "$CPUS"
    "${FOLDSEEK[@]}" createtsv "$FSDB/db" "$FSDB/db" "$FSDB/db_clu" "$CLU"

    echo ""
    echo "### sanity: member IDs as foldseek actually wrote them (the notebook parses these)"
    head -3 "$CLU"
    local members clusters
    members=$(wc -l < "$CLU"); clusters=$(cut -f1 "$CLU" | sort -u | wc -l)
    echo ""
    echo "### members $members  (inputs $n)   clusters $clusters"
    [ "$members" -eq "$n" ] || echo "*** WARNING: members != inputs -- foldseek dropped structures ***"

    echo ""
    echo "### diversity = clusters / designable designs, per generator"
    echo "### (higher = more distinct folds per design; directly comparable to published values)"
    printf "    %-18s %8s %9s %10s\n" "method" "designs" "clusters" "diversity"
    awk -F'\t' '{split($2,a,"__"); m=a[1]; tot[m]++; k=m SUBSEP $1;
                 if(!(k in seen)){seen[k]=1; clu[m]++}}
        END{for(m in tot) printf "    %-18s %8d %9d %10.3f\n", m, tot[m], clu[m], clu[m]/tot[m]}' \
        "$CLU" | sort
}

# ── stage: rep-rep TM graph (feeds the notebook UMAP) ─────────────────────────
do_graph () {
    setup_foldseek
    [ -r "$CLU" ] || { echo "ERROR: $CLU missing -- run STAGE=cluster first"; exit 1; }
    # A DB of the representatives only. Build a symlink farm rather than copying: the reps are a
    # small subset of $CIFS and foldseek createdb takes exactly one directory.
    rm -rf "$REPCIF"; mkdir -p "$REPCIF"
    cut -f1 "$CLU" | sort -u > "$WORK/reps.txt"
    local nrep; nrep=$(wc -l < "$WORK/reps.txt")
    while read -r r; do ln -sf "$CIFS/${r}.cif" "$REPCIF/${r}.cif"; done < "$WORK/reps.txt"
    local nlink; nlink=$(find "$REPCIF" -maxdepth 1 -name '*.cif' -type l | wc -l)
    echo ""
    echo "### rep-rep TM graph over $nrep representatives ($nlink symlinks resolved)"
    [ "$nlink" -eq "$nrep" ] || echo "*** WARNING: $((nrep - nlink)) rep CIFs missing from $CIFS ***"

    rm -rf "$FSDB/reps"* "$TMP/graph"; mkdir -p "$TMP/graph"
    "${FOLDSEEK[@]}" createdb "$REPCIF" "$FSDB/reps_db" --threads "$CPUS"
    "${FOLDSEEK[@]}" search "$FSDB/reps_db" "$FSDB/reps_db" "$FSDB/reps_aln" "$TMP/graph" \
        -s "$GSENS" -e "$GEVAL" --max-seqs "$GMAXSEQS" \
        --alignment-type "$ALNTYPE" -a 1 --threads "$CPUS"
    "${FOLDSEEK[@]}" convertalis "$FSDB/reps_db" "$FSDB/reps_db" "$FSDB/reps_aln" "$HITS" \
        --format-output "query,target,alntmscore" --threads "$CPUS"
    local nh
    nh=$(wc -l < "$HITS")
    echo ""
    echo "### $nh rep-rep hits -> $HITS   ($(awk -v n="$nh" -v r="$nrep" 'BEGIN{printf "%.1f", n/r}') neighbours/rep)"
    # ~1.0 neighbours/rep means the prefilter rejected everything -> raise GEVAL / GMAXSEQS
    awk -v n="$nh" -v r="$nrep" 'BEGIN{if (n/r < 2) print "*** WARNING: graph collapsed to self-hits -- raise GEVAL/GMAXSEQS ***"}'
}

archive_cifs () {
    local n; n=$(find "$CIFS" -maxdepth 1 -name '*.cif' | wc -l)
    [ "$n" -gt 0 ] || return 0
    echo ""
    echo "### archiving $n CIFs -> $OUT/designable_cifs.tar.zst (inode discipline)"
    tar -C "$CIFS" -cf - . | zstd -T"$CPUS" -3 -o "$OUT/designable_cifs.tar.zst" -f
    ls -la "$OUT/designable_cifs.tar.zst"
}

case "$STAGE" in
    smoke)   do_sse "$SMOKE_ARCHIVES"; do_cluster; do_graph ;;
    sse)     do_sse ;;
    cluster) do_cluster ;;
    graph)   do_graph ;;
    all)     do_sse; do_cluster; do_graph; archive_cifs ;;
    *)       echo "ERROR: unknown STAGE='$STAGE' (smoke|sse|cluster|graph|all)"; exit 1 ;;
esac

echo ""
echo "=========================================================================="
echo "STAGE=$STAGE done."
echo "  $TSV"
[ -r "$CLU" ] && echo "  $CLU"
[ -r "$HITS" ] && echo "  $HITS"
echo "next: sbatch --export=ALL,STAGE=all /cluster/work/wicky/bekessel/scripts/foldseek_designable_library.sh"
echo "=========================================================================="
