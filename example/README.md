# The worked example — seven tasks, five methods, end to end

Seven motifs, one per benchmark protein plus one base/filled pair, run through the whole pipeline at
full depth. This directory holds the outputs; **it is `.gitignore`d** — the data lives on disk next to
the repo, not in git. Rebuild it with `scripts/extras/make_example_bundle.py`, or reproduce it from
scratch with the recipe below.

| task | L | parts | SSE | variant | rfdiff | proteina | genie3 | pp1c_bb | laproteina |
|---|---|---|---|---|---|---|---|---|---|
| `il13_m103_base` | 9 | 1 | H | base | 724/800 | 294/800 | 728/800 | 552/800 | 85/100 |
| `il2_m101_base` | 6 | 1 | H | base | 621/800 | 185/800 | 573/800 | 561/800 | 80/100 |
| `il1b_m21_base` | 7 | **2** | E | base | 418/792 | 54/800 | 483/800 | 306/800 | 65/100 |
| `il1b_m21_filled` | 9 | **1** | E | **filled** | 513/800 | 190/800 | 624/800 | 584/800 | 84/100 |
| `pdl1_m113_base` | 11 | 3 | E | base | 276/696 | 28/800 | 405/800 | 165/800 | 39/100 |
| `her2_m568_base` | 15 | 4 | E | base | 2/792 | 1/800 | 4/800 | 2/800 | 0/100 |
| `tnfa_m27_base` | 9 | 4 | C | base | 0/176 | 0/800 | 28/800 | 3/800 | 0/100 |

(designable = motifRMSD ≤ 1 Å ∧ scRMSD ≤ 2 Å ∧ ranking_score ≥ 0.75)

Difficulty spans 724 → 0. `il1b_m21` base-vs-filled is the controlled pair: merging one gap takes it
from 2 fragments to 1 and designability rises for **all five** methods.

---

## What `stage_motif.py` is actually for

It does **not** write a contig per method. It writes **one method-agnostic contract**, and the methods
consume it three different ways:

```
                      motifs/pool.csv
                            │
                  stage_motif.py          ← runs ONCE per run
                            │
              ┌─────────────┴──────────────┐
      <run>/motifs/<task>.pdb        <run>/manifest.csv
      motif template, staged once    config_number, task_name, chain, fragments,
                                     contig, motif_min/max_length, nsamples, motif_pdb
                            │
        ┌───────────────────┼────────────────────────┐
        ▼                   ▼                        ▼
   READS it directly   TRANSLATES it            RE-DERIVES it
   rfdiff              genie3                   proteina · protpardelle · la-proteina
   `contig` column IS  manifest → MotifBench    their drivers call the same
   contigmap.contigs   problems/*.json +        motifgo.contig functions on the same
   verbatim; no        REMARK-999 PDBs          pool rows and get the same string
   driver at all       (different syntax,       (or pass --manifest to consume the
                        same fragments)          staged one verbatim)
```

So its real job is to fix four things **once**, so no method can silently differ:

1. **the task list and its order** — `config_number` == SLURM array index == the same motif in every
   method. This is what makes a per-task cross-method comparison meaningful.
2. **the motif template** — each `<task>.pdb` is extracted from the archive exactly once and shared,
   so every method scaffolds identical coordinates.
3. **the length policy** — flanks `2-10`/`35-75`, linker `1-5`, and the derived
   `motif_min_length`/`motif_max_length`. Held identical across methods; this is the fairness constraint.
4. **the contig** — the fragment decomposition in RFdiffusion grammar.

**Verified, not assumed.** Running Proteina's driver in `--pool` mode on the same pool and diffing its
generated configs against the staged manifest gives 3/3 identical on contig, length bounds *and* motif
template path.

> **The one hazard.** Because three drivers *re-derive* rather than read, an override passed to the
> stager but not to a driver (or vice versa) silently changes what that method builds. All four default
> to `2-10 / 35-75 / 1-5 / 100`. If you change any of them, either pass the same flags everywhere, or
> use `--manifest <run>/manifest.csv` so the driver consumes the staged contig verbatim.

---

## Running the seven example tasks end to end

Placeholders: `<repo>` the checkout, `<runs>` where runs live, `<venv>` your environment.

### 0. Once per shell

```bash
REPO=<repo>                                    # e.g. /cluster/work/wicky/bekessel/motif_benchmark
RUN=<runs>/example7                            # e.g. /cluster/work/wicky/bekessel/motif_library/runs/example7
POOL=$REPO/example/pool_example7.csv        # cut from motifs/pool.csv — see step 2
source <venv>/bin/activate                     # needs motifgo: pip install -e $REPO --no-deps
mkdir -p /cluster/scratch/$USER/motifgo_logs/{generate,filter,mpnn_prep,mpnn,af3_build,af3,score}
```

The `mkdir` is not optional: Slurm opens `--output` **before** the job script runs.

### 1. Motifs from each protein  *(skip if `proteins/<t>/` is already populated)*

Six proteins back the seven tasks. One command each, driven by that protein's `target.yaml`
(`pdb`, `chain`, `rsa_cutoff: 0.20`, `neighborhood_cutoff: 7.0`, `gap_merge: 4`, `emit_filled: true`):

```bash
for T in her2 pdl1 il13 il1b il2 tnfa; do
  python $REPO/scripts/motif/extract_motifs.py --target $REPO/proteins/$T/target.yaml
done
# -> proteins/<T>/{<T>_sasa_resis.csv, <T>_c_motif.csv, motif_pdbs/c_motif{,_filled}.tar.zst}
```

Knobs (all default from the yaml): `--rsa-cutoff 0.20` · `--neighborhood-cutoff 7.0` ·
`--neighborhood-metric ca|cb` · `--gap-merge 4` · `--no-filled` · `--screen` (report chains/SSE, write
nothing) · `--no-archive` (leave motif PDBs loose).

### 2. The task list

The seven tasks are a subset of the benchmark pool. To cut the example pool from the full one:

```bash
python - <<'PY'
import csv
KEEP = {("il13","103","base"), ("il2","101","base"), ("il1b","21","base"),
        ("il1b","21","filled"), ("pdl1","113","base"), ("her2","568","base"),
        ("tnfa","27","base")}
rows = [r for r in csv.DictReader(open("<repo>/motifs/pool.csv"))
        if (r["pdb_name"], r["motif_num"], r["variant"]) in KEEP]
w = csv.DictWriter(open("<repo>/example/pool_example7.csv","w",newline=""),
                   fieldnames=rows[0].keys(), lineterminator="\n")
w.writeheader(); w.writerows(rows)
print(len(rows), "tasks")
PY
```

To build a pool from scratch instead — **not yet written (phase P5)**; today the dedup + interface +
padding step still lives in `notebooks/motif_library_generation.ipynb`:

```bash
python $REPO/scripts/motif/build_pool.py --jaccard 0.7 --pad 5 \
       --interface-from-proteins $REPO/proteins --out $REPO/motifs/pool.csv     # P5
```

### 3. Stage the run — the shared contract

```bash
python $REPO/scripts/motif/stage_motif.py \
    --pool $POOL --run-dir $RUN --lib $REPO/proteins \
    --variants base,filled --nsamples 100 \
    --flank-n 2-10 --flank-c 35-75 --linker 1-5
# -> $RUN/motifs/<task>.pdb  +  $RUN/manifest.csv     array bound: 0..6
N=$(( $(wc -l < $RUN/manifest.csv) - 2 ))
```

### 4. Generate — the only per-method stage

```bash
G=$REPO/scripts/generate

# RFdiffusion — no driver; reads manifest.csv directly
sbatch --job-name=ex7.rfdiff.gen --array=0-$N%8 --export=ALL,RUN_DIR=$RUN \
       $G/rfdiff/rfdiff_generate.sbatch
#   knobs: SIF= CKPT= CFGNUM=

# Proteina
python $G/proteina/proteina_from_library.py --pool $POOL --run-dir $RUN --lib $REPO/proteins \
       --variants base,filled --nsamples 100
sbatch --job-name=ex7.proteina.gen --array=0-$N%8 --export=ALL,RUN_DIR=$RUN \
       $G/proteina/proteina_generate.sbatch

# Genie3 — converter turns the manifest into a MotifBench problem set
python $G/genie3/genie3_from_library.py --run-dir $RUN
sbatch --job-name=ex7.genie3.gen --array=0-9%5 --export=ALL,CONFIG=$RUN/genie3/experiment.yaml \
       $G/genie3/genie3_generate.sbatch
#   NOTE: genie3 shards the SAMPLE dim, not the task dim — the array is shards, not motifs

# Protpardelle-1c  (cc58 = backbone + sidechain-conditional -> method dir pp1c_bb)
python $G/protpardelle/protpardelle_from_library.py --pool $POOL --run-dir $RUN --lib $REPO/proteins \
       --model cc58 --variants base,filled --nsamples 100
sbatch --job-name=ex7.pp1c.gen --array=0-$N%8 --export=ALL,RUN_DIR=$RUN,METHOD=pp1c_bb \
       $G/protpardelle/protpardelle_generate.sbatch

# La-Proteina — co-design (indexed). For unindexed: --uidx and a _uidx run dir.
python $G/la-proteina/la_proteina_from_library.py --pool $POOL --run-dir $RUN --lib $REPO/proteins \
       --variants base,filled --nsamples 100
sbatch --job-name=ex7.lapro.gen --array=0-$N%8 --export=ALL,RUN_DIR=$RUN,RUNTAG=$(basename $RUN) \
       $G/la-proteina/la_proteina_generate.sbatch
```

All five drivers share `--pool --run-dir --lib --variants --nsamples --flank-n --flank-c --linker`,
and accept `--manifest $RUN/manifest.csv` instead of `--pool` to consume the staged contig verbatim.

### 5. Evaluate — one script per stage, method as a variable

Backbone methods (`rfdiff`, `proteina`, `genie3`, `pp1c_bb`):

```bash
EV=$REPO/scripts/evaluate
for M in rfdiff proteina genie3 pp1c_bb; do
  sbatch --job-name=ex7.$M.filter --export=ALL,RUN_DIR=$RUN,METHOD=$M           $EV/resolve_filter.sbatch
  sbatch --job-name=ex7.$M.prep   --export=ALL,RUN_DIR=$RUN,METHOD=$M           $EV/mpnn_prep.sbatch
  CA=""; [ "$M" = proteina ] && CA=",CA_ONLY=1"; [ "$M" = genie3 ] && CA=",CA_ONLY=1"
  sbatch --job-name=ex7.$M.mpnn --array=0-9%5 --export=ALL,RUN_DIR=$RUN,METHOD=$M,NSHARDS=10$CA \
                                                                                $EV/2_mpnn.sbatch
  sbatch --job-name=ex7.$M.build  --export=ALL,RUN_DIR=$RUN,METHOD=$M           $EV/3_af3_build.sbatch
  # 3_af3_build prints NBATCH; then:
  # sbatch --job-name=ex7.$M.af3 --array=1-<NBATCH>%92 --export=ALL,RUN_DIR=$RUN,METHOD=$M $EV/4_af3.sbatch
  # sbatch --job-name=ex7.$M.score --export=ALL,RUN_DIR=$RUN,METHOD=$M          $EV/5_score.sbatch
done
```

`CA_ONLY=1` is required for **Proteina and Genie3** — both emit Cα-only backbones, so MPNN must use
`ca_model_weights`. Other knobs: `CUTOFF=1.0` (filter gate; `99` passes all) · `NSEQ=8` · `TEMP=0.2` ·
`BATCH=500` (sequences per AF3 batch) · `PREDICTOR=af3|esmfold2` · `POSITIONS=` + `PRESERVE=1`.

Co-design (`laproteina`) skips filter/prep/MPNN and rejoins at step 3:

```bash
python $G/la-proteina/la_proteina_extract_seqs.py --run-dir $RUN
sbatch --job-name=ex7.lapro.build --export=ALL,RUN_DIR=$RUN,METHOD=laproteina   $EV/3_af3_build.sbatch
sbatch --job-name=ex7.lapro.af3 --array=1-<NBATCH>%92 --export=ALL,RUN_DIR=$RUN,METHOD=laproteina $EV/4_af3.sbatch
sbatch --job-name=ex7.lapro.score --export=ALL,RUN_DIR=$RUN,METHOD=laproteina   $EV/5_score.sbatch

# unindexed only: the model places the motif itself, so recover where it put it first
sbatch --export=ALL,RUN_DIR=${RUN}_uidx,METHOD=laproteina                       $EV/unindexed_score.sbatch
sbatch --export=ALL,RUN_DIR=${RUN}_uidx,METHOD=laproteina,\
POSITIONS=positions_unindexed.jsonl,PRESERVE=1                                  $EV/5_score.sbatch
```

### Running one method on its own settings

Nothing forces the five methods to share settings — the shared contract is the *motif and the length
policy*, not the sampler. Each generate sbatch is standalone: give it a run dir and a config and it
runs, with no reference to any other method.

**Every generator is already config-driven.** The driver's only job is to write those configs:

| method | what the sbatch reads per array task | so you can |
|---|---|---|
| rfdiff | the `manifest.csv` **row** (parsed in-shell by `config_number`) | edit the row, or point `RUN_DIR` at a hand-made manifest |
| proteina | `$RUN/proteina/configs/inf_<N>.yaml` | hand-write / edit `inf_<N>.yaml` |
| protpardelle | `$RUN/<METHOD>/configs/inf_<N>.yaml` | same, plus `SEED= BATCH= NSAMPLES= MODEL_PARAMS=` |
| la-proteina | `$REPO/configs/inf_<RUNTAG>_<N>.yaml` (a symlink to the real file in the run dir) | hand-write the yaml, or change its `defaults:` |
| genie3 | one `experiment.yaml` for the whole problem set | edit that file; the array shards samples, not motifs |

So yes — a single hand-made yaml plus one `sbatch` is a legitimate way to run, and it is exactly what
the drivers produce. To sample one motif harder than the rest:

```bash
# 40 designs for task 3 only, everything else untouched
sed -i 's/^nsamples:.*/nsamples: 40/' $RUN/proteina/configs/inf_3.yaml
sbatch --job-name=ex7.proteina.gen3 --array=3-3 --export=ALL,RUN_DIR=$RUN \
       $REPO/scripts/generate/proteina/proteina_generate.sbatch
```

Per-method knobs at submit time, no config edit needed:

```bash
sbatch --export=ALL,RUN_DIR=$RUN,SIF=<other.sif>,CKPT=<other.pt>  .../rfdiff_generate.sbatch
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=pp1c_bb,SEED=7,BATCH=16,NSAMPLES=40 \
                                                                  .../protpardelle_generate.sbatch
sbatch --export=ALL,CONFIG=<abs experiment.yaml>,NUM_DEVICES=2    .../genie3_generate.sbatch
```

**La-Proteina unindexed is a flag, not a second pipeline.** `--variant uidx` swaps the hydra base
config from `inference_motif_idx_aa` to `inference_motif_uidx_aa`, and those two base configs name
different diffusion checkpoints (`LD4_motif_idx_aa.ckpt` vs `LD6_motif_uidx_aa.ckpt`) over the same
autoencoder (`AE3_motif.ckpt`) and the same image. One flag reaches all three differences:

```bash
python $G/la-proteina/la_proteina_from_library.py --pool $POOL --run-dir ${RUN}_uidx \
       --lib $REPO/proteins --variant uidx --variants base,filled --nsamples 100
sbatch --job-name=ex7.lapro_uidx.gen --array=0-$N%8 \
       --export=ALL,RUN_DIR=${RUN}_uidx,RUNTAG=$(basename ${RUN}_uidx) \
       $G/la-proteina/la_proteina_generate.sbatch
```

Use a **separate run dir** (`${RUN}_uidx`): idx and uidx produce different scaffolds for the same
motif, and the run dir is keyed by task name, not by model. Scoring uidx additionally needs
`unindexed_score.sbatch` first — the unindexed model chooses where to place the motif, so its nominal
positions are wrong until they are recovered.

**The one thing that must stay identical.** `--flank-n / --flank-c / --linker` and the motif templates
define *what is being asked for*; the sampler settings define *how hard you look*. Vary the second
freely per method. Vary the first and the methods are no longer solving the same problem — and because
three drivers re-derive the contig rather than read it, that divergence is silent. If you must change
the length policy, pass `--manifest $RUN/manifest.csv` so every driver consumes the staged contig
verbatim instead of re-deriving its own.

### 6. Results

```
$RUN/<method>/af3_refold_scores.csv     one row per sequence:
                                        prediction, task_name, design_name, motifRMSD, scRMSD,
                                        plddt_mean, ptm, ranking_score, motif_ok, sc_ok, designable
$RUN/<method>/1_filter/motif_rmsd.csv   ALL generated designs + pass flag
                                        (the post-generation denominator — use this, not the passers,
                                         when computing a designability rate)
```

```bash
python $REPO/scripts/extras/motif_rsa.py --run-dir $RUN --method <M> --lib $REPO/proteins   # ΔRSA burial
python $REPO/scripts/extras/compute_costs.py --out $RUN/compute_costs.csv --job generation:<M>:<jobid>
# then point scripts/analysis/motif_benchmark.ipynb at $RUN
```

---

## A smaller smoke test

The full seven-task run is ~3,500 designs per method. To exercise every stage in ~30 minutes on one
method, use a 3-task pool at `--nsamples 4` — that is what validated this pipeline end to end
(mcl1/2mhs: extract → stage → rfdiff → filter → mpnn_prep → MPNN → AF3 build → AF3 → score, with
Stage 0 reproducing the published catalog byte-for-byte).
