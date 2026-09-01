# Pilot Runbook — 3-motif compute pilot across all 5 scaffolding methods

Goal: run the **same 3 library motifs** through every method at the real benchmark sample count
(`nsamples=100`), then `sacct` each job to get per-motif generation time + peak VRAM/RSS, so the
full 734-motif pass can be right-sized (GPU class, `--mem`, walltime tier) per Leo's GPU-maxxing docs.

**This runbook changes no existing driver/sbatch** — it just collects the launch commands in one
consistent shape. Each block shows: **DIR** (where to run) · **DRIVER** (build configs from the pilot
pool) · **LAUNCH** (sbatch) · **METHOD** / **MODEL** · **OUTPUT** (where results land) · **GPU**.

## Pilot motifs (span size × disjointedness × protein)
`pilot_pool_j07_3motif.csv` (base variant, array index = row order):

| idx | motif | len | parts | chain | character |
|-----|-------|-----|-------|-------|-----------|
| 0 | il1b m23  | 4  | 1 | A | small, contiguous (easy) |
| 1 | il13 m20  | 8  | 2 | C | median, 2 fragments |
| 2 | her2 m568 | 15 | 4 | E | large, 4 fragments (worst-case) |

## Shared env (paste once)
```bash
POOL=/cluster/work/wicky/bekessel/motif_library/pilot_pool_j07_3motif.csv
RUN=/cluster/work/wicky/bekessel/motif_library/runs/pilot_j07_3motif
PIPE=/cluster/work/wicky/bekessel/epitope_scaffolding/pipeline    # esp lives here
# 3 base motifs -> SLURM array 0-2 (all pool drivers are 0-indexed).
# nsamples=100, flanks 2-10 / 35-75, linker 1-5 — held IDENTICAL across methods (fair benchmark).
```

## Step 0 — shared stager (build the contract every method reads)
Stages each motif PDB once + writes the shared manifest (contig + fragments + length range). Genie3 and
RFdiff read this; the 3 pool drivers re-derive it identically, so it's safe to run either way.
```bash
export PYTHONPATH=$PIPE:$PYTHONPATH
python $PIPE/scripts/stage_motif.py --pool $POOL --run-dir $RUN --variants base --nsamples 100
# -> $RUN/motifs/<task>.pdb   +   $RUN/manifest.csv   (config_number == array index == same motif everywhere)
```

---

## 1. Proteina  ·  METHOD=proteina  ·  MODEL=proteina_v1.7_60M_motif  ·  backbone
```bash
cd /cluster/work/wicky/bekessel/software/proteina                                   # DIR
python scripts/proteina_from_library.py --pool $POOL --run-dir $RUN --variants base --nsamples 100   # DRIVER
sbatch --array=0-2 --export=ALL,RUN_DIR=$RUN scripts/proteina_generate.sbatch       # LAUNCH  (GPU rtx_4090:1)
# OUTPUT -> $RUN/proteina/scaffolds/inf_<0..2>/...
```

## 2. La-Proteina  ·  METHOD=laproteina  ·  MODEL=laproteina_allatom  ·  co-design (no MPNN)
```bash
cd /cluster/work/wicky/bekessel/software/la-proteina                                # DIR
python scripts/la_proteina_from_library.py --pool $POOL --run-dir $RUN --variants base --nsamples 100 # DRIVER
sbatch --array=0-2 --export=ALL,RUN_DIR=$RUN,RUNTAG=$(basename $RUN) scripts/la_proteina_generate.sbatch  # LAUNCH (GPU rtx_4090:1)
# OUTPUT -> $RUN/laproteina/inference/inf_<RUNTAG>_<0..2>_<task>/   (co-designed all-atom + sequence)
```

## 3. Protpardelle-bb  ·  METHOD=pp1c_bb  ·  MODEL=cc58 (backbone+sidechain-cond)  ·  backbone
```bash
cd /cluster/work/wicky/bekessel/software/protpardelle-1c                             # DIR
python scripts/protpardelle_from_library.py --pool $POOL --model cc58 --run-dir $RUN --variants base --nsamples 100  # DRIVER
sbatch --array=0-2 --export=ALL,RUN_DIR=$RUN,METHOD=pp1c_bb scripts/protpardelle_generate.sbatch  # LAUNCH (GPU rtx_4090:1)
# OUTPUT -> $RUN/pp1c_bb/scaffolds/inf_<0..2>/<model-suffix>/<task>/
```

## 4. RFdiffusion  ·  METHOD=rfdiff  ·  MODEL=RFdiffusion (Lisa baseline)  ·  backbone
**Now pool-driven** off the Step-0 shared contract (no per-protein YAML). `rfdiff_generate.sbatch` reads
each manifest row and runs esp's proven `run_inference.py` invocation (`contigmap.contigs=[<contig>]`,
`inference.input_pdb=<motif_pdb>`, `contigmap.length=<min>-<max>`) — one array task = one motif.
```bash
cd /cluster/work/wicky/bekessel/software/rfdiff                                      # DIR
# (Step 0 already staged $RUN/motifs + $RUN/manifest.csv — no driver needed here)
sbatch --array=0-2 --export=ALL,RUN_DIR=$RUN rfdiff_generate.sbatch                  # LAUNCH (GPU rtx_4090:1)
# METHOD=rfdiff | MODEL=RFdiffusion Base_ckpt | container containers/RFdiffusion.sif
# OUTPUT -> $RUN/rfdiff/scaffolds/<task>/design_*.pdb
```

## 5. Genie3  ·  METHOD=genie3  ·  MODEL=Genie3 (diffusion)  ·  backbone
**Now pool-driven** via the converter (`genie3_from_library.py`): reads the Step-0 shared contract →
writes a MotifBench problem set (REMARK-999 PDBs + problem JSONs) + `experiment.yaml` with absolute paths.
```bash
cd /cluster/work/wicky/bekessel/software/genie3                                      # DIR
python scripts/genie3_from_library.py --run-dir $RUN                                 # CONVERTER (reads $RUN/manifest.csv)
sbatch --array=0-9%5 --export=ALL,CONFIG=$RUN/genie3/experiment.yaml genie3_generate.sbatch  # LAUNCH (GPU rtx_4090:1)
# OUTPUT -> $RUN/genie3/output/<task>/results/
# NOTE: genie3 shards the SAMPLE dim (all problems per task). For 1-task-1-motif parity, set
#       `selections: <task>` per array task (needs a small sbatch tweak — deferred).
```

---

## After the jobs finish — measure, then scale (Leo L4)
```bash
sacct -j <jobid> -o JobID,Elapsed,MaxRSS,ReqMem,State,TRESUsageInMax%80
#  Elapsed / 100  -> sec/design  (× 100 × 734 for the full-pass GPU-h per method)
#  TRESUsageInMax gres/gpumem  -> smallest GPU class that fits (L3: rtx_2080_ti 11G? rtx_3090 24G?)
#  MaxRSS × 1.2   -> --mem for the full array (L2)
```
Then the full pass: `--pool motif_pool_dedup_j0.7.csv`, `--array=0-733%<cap>`, spread across GPU
types + walltime tiers, bulk QOS + skip-existing (to be added) + tar.zst packing (inode budget).

**All 5 pool-driven off the shared Step-0 contract:** Proteina, La-Proteina, Protpardelle-bb, Genie3
(via converter), RFdiffusion (via `rfdiff_generate.sbatch`, reusing esp's `run_inference.py` call).
