# motif_benchmark

Motif-scaffolding benchmark: take an interface epitope out of a protein, ask five generative models to
build a scaffold around it, and measure how often the refolded design puts the motif back where it was.

```
proteins/          →   motifs/        →   runs/<run>/<method>/        →   scores
target PDB+yaml        pool + PDBs        scaffolds → seqs → refolds      designable?
  scripts/motif/       scripts/motif/     scripts/generate/ + evaluate/   scripts/analysis/
```

Five generators — **RFdiffusion, Proteina, Genie3, Protpardelle-1c, La-Proteina** (indexed and
unindexed) — share one motif definition, one contig grammar and one scorer. Generation is the *only*
method-specific stage; everything downstream is one script per stage, with the method as a runtime
variable. That is what makes the comparison fair.

## Install

```bash
source <your-venv>/bin/activate          # needs numpy, scipy, biotite
pip install -e /path/to/motif_benchmark --no-deps
python -c "import motifgo; print(motifgo.__file__)"
```

`--no-deps` is deliberate: the three dependencies are *declared* in `pyproject.toml` so the repo is
honest about what it needs, but letting pip resolve them can silently upgrade `biotite` or `numpy`
under a working pipeline. No install at all also works: `export PYTHONPATH=/path/to/motif_benchmark`.

## Layout

| path | what |
|---|---|
| `motifgo/` | the shared library — six modules, the only importable code |
| `proteins/<target>/` | source structure, `target.yaml`, SASA table, motif catalog, motif PDBs |
| `motifs/` | the scaffolding tasks: `pool.csv` + one PDB per task |
| `scripts/` | every executable, by stage — see [`scripts/README.md`](scripts/README.md) |
| `containers/` | `.def` build recipes; the `.sif` images live in `/cluster/work/wicky/containers` |
| `example/` | a full-depth worked example, on disk, not in git |
| `docs/` | runbook, pipeline walkthrough, repo plan |

## Models and weights

The repo ships **no weights and no images** — only the `.def` build recipes in `containers/`. Two
locations, deliberately separate:

| what | where | why |
|---|---|---|
| **container images** (`.sif`) | `/cluster/work/wicky/containers/` | shared lab store; one copy per image for everyone |
| **model repos + weights** | `/cluster/work/wicky/containers/<method>/` | one directory per generator (`genie3`, `proteina`, `la-proteina`, `la-proteina-uidx`, `protpardelle-1c`, `ProteinMPNN`): code, configs and weights, plus the upstream `.git`, so `git status` inside a model directory shows exactly what was patched for this benchmark. The generate sbatches take it as `REPO` and hydra resolves configs relative to it, so a model directory moves as a whole, never piecemeal |

`site.yaml` records both (`containers:` and `model_repos:`). The `*.sif` links inside some model
directories are back-compat symlinks to the flat images; nothing reads them by default.

### Where each model's weights came from

**RFdiffusion** — `/cluster/work/wicky/params/RFParams/Base_ckpt.pt` (462 MB, base model).
Pre-existing shared lab copy from the upstream release; not downloaded by this project.
Image: `/cluster/work/wicky/containers/RFdiffusion.sif` (the shared store — this repo's
`containers/` holds only `.def` build recipes).

**Proteina** — `software/proteina/checkpoints/proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt`
(812 MB). The 60M no-triangle DFS model fine-tuned for motif scaffolding, from the NVIDIA **NGC public
resource API** — no NGC CLI or auth needed for public resources:

```bash
wget --content-disposition "https://api.ngc.nvidia.com/v2/resources/nvidia/clara/proteina_v1.7_dfs_60m_notri_motif_scaffolding/versions/1.0/files/proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt"
```

**Genie3** — `software/genie3/pretrained/v1/checkpoints/step=600000.ckpt` (331 MB; a `legacy/`
step=400000 checkpoint is also present but unused). From HuggingFace `yeqinglin/genie3`. Download
**through the container** — the host has no `hf` CLI, and the Genie3 repo's own `scripts/setup/download.sh`
checks for the retired `huggingface-cli` name:

```bash
HF_HOME=$SCRATCH/hf singularity exec /cluster/work/wicky/containers/genie3.sif   hf download yeqinglin/genie3 --include "pretrained/**" --local-dir <repo>
```

`HF_HOME=$SCRATCH/hf` is required: HuggingFace stages downloads under `$HOME/.cache/huggingface`,
and `/cluster/home` is read-only inside the container.

**Protpardelle-1c** — `software/protpardelle-1c/model_params/weights/` (4.4 GB, many epochs).
The benchmark uses **`cc58` epoch 416** with the `sampling_sidechain_conditional` config
(backbone + sidechain-conditional → the `pp1c_bb` method dir). From **Zenodo record 16817230**:

```bash
wget -O protpardelle-1c.tar.gz "https://zenodo.org/records/16817230/files/protpardelle-1c.tar.gz?download=1"
tar -xzvf protpardelle-1c.tar.gz --strip-components=1     # -> model_params/{weights,configs}/
```

Only the diffusion weights are needed. The repo's `download_model_params.sh` also pulls ESMFold /
LigandMPNN / Foldseek — those are Protpardelle's own *evaluation* dependencies and we do not use them.

**La-Proteina** — `software/la-proteina/checkpoints_laproteina/` (9.3 GB), three checkpoints, all from
the NVIDIA NGC `laproteina_weights_data` collection (downloaded manually per the upstream README):

| file | size | role |
|---|---|---|
| `AE3_motif.ckpt` | 3.8 GB | the autoencoder — always loaded |
| `LD4_motif_idx_aa.ckpt` | 2.7 GB | latent diffusion, **indexed** all-atom (the `laproteina` idx runs) |
| `LD6_motif_uidx_aa.ckpt` | 2.7 GB | latent diffusion, **unindexed** (the `_uidx` runs — the model chooses where to place the motif) |

idx and uidx share **one image** (`/cluster/work/wicky/containers/la-proteina.sif`); they differ only by checkpoint and
base config, which is why `la-proteina-uidx/` is mostly symlinks into `la-proteina/`.

**ProteinMPNN** — `software/ProteinMPNN/vanilla_model_weights/v_48_020.pt` (6.7 MB), vendored with the
git checkout. Cα-only backbones (Proteina, Genie3) use `ca_model_weights/v_48_020.pt` via `CA_ONLY=1`.
Runs inside `/cluster/work/wicky/containers/openfold.sif`, which supplies python/torch only.

**AlphaFold3** — `/cluster/work/wicky/params/af3_params` (1.1 GB), shared lab copy. Access is gated by
Google DeepMind's terms; obtain your own grant rather than redistributing these.

## How a motif is named

`her2_m568_base` = the motif built around **HER2 surface residue 568**, base variant. `motif_num` is
always the SASA centroid: a residue with **RSA > 0.20**, plus every residue whose Cα falls within
**7 Å** of it. (Verified across all 265 base motifs: the centroid is in its own residue list, is
flagged `surface=1`, and has RSA ≥ 0.201.) `_filled` is the same motif with short sequence gaps
merged — see below.

## Reducing disjointedness: `gap_merge`

A motif grown around one centroid is often **disjoint** — several sequence stretches. `gap_merge` is
the knob that joins them:

```python
from motifgo.contig import group_runs, merge_gaps
merge_gaps(group_runs([81,82, 91, 126,127,128,129,130, 155, 157]), gap_merge=4)
# [(81,82), (91,91), (126,130), (155,157)]      5 fragments -> 4
```

Two fragments merge when fewer than `gap_merge` residues separate them **in sequence numbering**, and
those residues join the motif. Each gap is measured from the *growing* fragment, so a run of short
gaps collapses in one pass (`[(10,12),(16,18),(22,24)]` at `gap_merge=4` → `[(10,24)]`).

No distance check is needed at this scale: measured across the whole library (5,532 inter-fragment gap
residues), a gap of ≤ 5 residues is **always** spatially adjacent to the motif — 0 % farther than 7 Å,
with a 1.33 Å floor that is simply the peptide C–N bond. Above ~6 that stops holding (5 % at gap 6,
43 % at gap 10), and `contig.merge_gaps_spatial(..., coords_of=, max_dist=)` adds the guard.

The cost is fidelity: filled residues become part of the motif you scaffold, so it is no longer
exactly the epitope you measured. Of 163 disjoint motifs, `gap_merge=3` improves 44 and makes 18 fully
contiguous; `gap_merge=4` improves 49 and makes 22 contiguous.

*In the published benchmark:* `gap_merge=4` was applied at catalog time to emit a second `filled` row
beside each `base` motif — 49 of the 314 tasks — so base and filled could be scored head to head.

*Adjacent but different:* the contig **linker** (`1-5` by default) lets the generator invent residues
*between* fragments without adding them to the motif — disjointedness is bridged, not reduced. That is
what every benchmark run actually scaffolded.

## Running it

Set once:

```bash
REPO=/cluster/work/wicky/bekessel/motif_benchmark
RUN=/cluster/work/wicky/bekessel/motif_library/runs/my_run
POOL=$REPO/motifs/pool_mcl1_smoke.csv     # the shipped 3-task demo pool; build_pool.py (P5) will
                                          # generate the full one — see "Known gaps" below
mkdir -p /cluster/scratch/$USER/motifgo_logs/{generate,filter,mpnn_prep,mpnn,af3_build,af3,score}
```

Logs default to `/cluster/scratch/<you>/motifgo_logs/<stage>/`. **Slurm opens `--output` before the
script runs and expands only `%` patterns there — never environment variables** — so the directory
must exist at submit time, and relocating means `-o/-e` on the command line. Name the job and every
log is attributable:

```bash
sbatch --job-name=<run>.<method>.<stage> ...     # %x picks it up; so do squeue and sacct
```

### Stage 0 — motifs

```bash
python $REPO/scripts/motif/motif_extract.py screen  --fetch 1XYZ
python $REPO/scripts/motif/motif_extract.py extract --fetch 1XYZ --chain A --tag mytarget \
       --out_dir $REPO/proteins/mytarget
python $REPO/scripts/motif/stage_motif.py --pool $POOL --run-dir $RUN \
       --variants base --nsamples 100 \
       [--flank-n 2-10 --flank-c 35-75 --linker 1-5]        # length policy, identical for all methods
```

### Stage 1 — generate (the only per-method stage)

```bash
M=rfdiff        # rfdiff | proteina | genie3 | pp1c_bb | laproteina
N=$(( $(wc -l < $RUN/manifest.csv) - 2 ))                    # 0-indexed array bound

# rfdiff reads the manifest directly — no driver
sbatch --job-name=run.rfdiff.gen --array=0-$N%20 --export=ALL,RUN_DIR=$RUN \
       $REPO/scripts/generate/rfdiff/rfdiff_generate.sbatch

# the other four: build per-motif configs first, then submit
python $REPO/scripts/generate/proteina/proteina_from_library.py --pool $POOL --run-dir $RUN --variants base --nsamples 100
sbatch --job-name=run.proteina.gen --array=0-$N%20 --export=ALL,RUN_DIR=$RUN \
       $REPO/scripts/generate/proteina/proteina_generate.sbatch

python $REPO/scripts/generate/genie3/genie3_from_library.py --run-dir $RUN
sbatch --job-name=run.genie3.gen --array=0-9%5 --export=ALL,CONFIG=$RUN/genie3/experiment.yaml \
       $REPO/scripts/generate/genie3/genie3_generate.sbatch

python $REPO/scripts/generate/protpardelle/protpardelle_from_library.py --pool $POOL --model cc58 --run-dir $RUN
sbatch --job-name=run.pp1c.gen --array=0-$N%20 --export=ALL,RUN_DIR=$RUN,METHOD=pp1c_bb \
       $REPO/scripts/generate/protpardelle/protpardelle_generate.sbatch

python $REPO/scripts/generate/la-proteina/la_proteina_from_library.py --pool $POOL --run-dir $RUN
sbatch --job-name=run.lapro.gen --array=0-$N%20 --export=ALL,RUN_DIR=$RUN,RUNTAG=$(basename $RUN) \
       $REPO/scripts/generate/la-proteina/la_proteina_generate.sbatch
```

Knobs common to all five: `SIF=` (image), `REPO=` (model repo), `CKPT=`, `NSAMPLES=`.
Images resolve from `/cluster/work/wicky/containers` by default.

### Stages 2–5 — evaluate (one script per stage, method as a variable)

```bash
EV=$REPO/scripts/evaluate

sbatch --job-name=run.$M.filter    --export=ALL,RUN_DIR=$RUN,METHOD=$M          $EV/resolve_filter.sbatch
#   CUTOFF=1.0   motifRMSD gate in A;  CUTOFF=99 passes everything

sbatch --job-name=run.$M.prep      --export=ALL,RUN_DIR=$RUN,METHOD=$M          $EV/mpnn_prep.sbatch
#   POSITIONS=<file>  override motif positions;  PRESERVE=1  they are already in native order

sbatch --job-name=run.$M.mpnn      --array=0-49%25 --export=ALL,RUN_DIR=$RUN,METHOD=$M  $EV/2_mpnn.sbatch
#   NSHARDS=50  NSEQ=8  TEMP=0.2  CA_ONLY=1 (Proteina)   MP=<ProteinMPNN checkout>
#   delete mpnn/design_list.txt before re-running after mpnn_prep grows — it caches the design set

sbatch --job-name=run.$M.build     --export=ALL,RUN_DIR=$RUN,METHOD=$M          $EV/3_af3_build.sbatch
#   BATCH=500  sequences per AF3 batch;  prints NBATCH for the next line

sbatch --job-name=run.$M.af3 --array=1-<NBATCH>%92 --export=ALL,RUN_DIR=$RUN,METHOD=$M  $EV/4_af3.sbatch
#   AF3_SIF=  AF3_PARAMS=   ; for the bulk pool:
#   scontrol update jobid=<id> partition=gpuhe.bulk qos=es_wicky/gpuhe/bulk

sbatch --job-name=run.$M.score     --export=ALL,RUN_DIR=$RUN,METHOD=$M          $EV/5_score.sbatch
#   PREDICTOR=af3|esmfold2   DESIGN_CHAIN=A   POSITIONS=<file>   OUT=<csv>
```

**Co-design branch (La-Proteina)** skips stages 2 entirely and rejoins at `3_af3_build`:

```bash
python $REPO/scripts/generate/la-proteina/la_proteina_extract_seqs.py --run-dir $RUN
sbatch --export=ALL,RUN_DIR=$RUN_uidx,METHOD=laproteina $EV/unindexed_score.sbatch     # uidx only
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=laproteina      $EV/3_af3_build.sbatch
sbatch --array=1-<NBATCH>%92 --export=ALL,RUN_DIR=$RUN,METHOD=laproteina $EV/4_af3.sbatch
sbatch --export=ALL,RUN_DIR=$RUN_uidx,METHOD=laproteina,POSITIONS=positions_unindexed.jsonl,PRESERVE=1 \
       $EV/5_score.sbatch
```

## Designability

A design counts as designable iff **motifRMSD ≤ 1.0 Å** (backbone N,CA,C at the design positions)
**and scRMSD ≤ 2.0 Å** (CA only) **and AF3 `ranking_score` ≥ 0.75**. The first two thresholds are
MotifBench (arXiv:2502.12479) and are defined in exactly one place, `motifgo/score.py`.

## Predicting success before you run

`scripts/predict/` fits a per-method logistic model on the finished benchmark and scores a motif that
has never been run — from its four descriptors alone, before any compute is spent. It is an
**addition, not a gate**: no pipeline stage depends on it.

```bash
python scripts/predict/predict_motif.py --length 8 --disjoint 3 --sse C --rsa 0.32
```

**Disjointedness dominates everything.** Per SD, it costs −1.4 to −1.8 log-odds for every
index-conditioned method; length, the interaction, SSE and exposure add little. **La-Proteina
unindexed is the structural exception** — coefficient −0.36, because it places fragments freely
instead of being told where they go.

| motif | best method | p̂ | why |
|---|---|---|---|
| disjoint (>1 fragment) | **La-Proteina unindexed** | 0.11 vs 0.003–0.013 | the only arm that does not collapse on fragments |
| contiguous | uidx, then genie3 / pp1c_bb / rfdiff | 0.23 / 0.31 / 0.24 / 0.25 | close enough to choose on wall-clock or diversity |

**Merging gaps is the largest single lever the benchmark found**: 3 fragments → 1 moves a typical motif
from p̂ 0.11 to 0.23 and changes which method you should use. See
[Reducing disjointedness](#reducing-disjointedness-gap_merge).

Fitted on the baseline run (314 motifs × 6 arms) and checked on a held-out protein — Mcl-1/2mhs, which
is **not** in the training pool: Spearman ρ 0.60 over 78 motifs and 65,606 designs, with the
fragment-count effect reproducing on unseen data (1 fragment predicted 0.50 / observed 0.63; 2 frag
0.13 / 0.17; 3 frag 0.016 / 0.018). It **under-predicts by roughly a constant**, so use it to rank and
triage, and read the absolute p̂ as a lower bound. Details and caveats:
[`scripts/predict/README.md`](scripts/predict/README.md).

## Known gaps (what is documented but not yet built)

| missing | referenced by | consequence |
|---|---|---|
| `scripts/motif/build_pool.py` | README quickstart, `example/README.md` | the dedup + interface + padding step still lives in `notebooks/motif_library_generation.ipynb`; `motifs/pool.csv` therefore does not exist and only the 3-task `pool_mcl1_smoke.csv` ships |
| `scripts/extras/make_example_bundle.py` | `example/README.md`, `.gitignore` | `example/` cannot be regenerated by script yet |
| six `complex:` paths + her2 `fab_chains` | `proteins/*/target.yaml` | every `complex_cif:` points at a file that does not exist, and her2 is `[A,B]` where the published pool used `[A,B,C,D]`; interface filtering from `target.yaml` will be wrong until fixed |

## Known caveat: the published pool is not deduplicated

`motif_pool_interface_j0.7_pad5.csv` — the 314-task pool behind the published numbers — carries `j0.7`
in its name but was built from the **raw** pool with only the interface+padding filter applied. 84 motif
pairs at Jaccard ≥ 0.7 remain, 5 of them identical residue sets. Cross-method comparison is unaffected
(every method scaffolded the same 314 tasks), but aggregate and per-protein rates are weighted by that
redundancy. `scripts/motif/build_pool.py --jaccard 0.7` will produce a genuinely deduplicated pool of
unique residue spans — **not yet written, see Known gaps**.
