# Motif-Scaffolding Benchmark — End-to-End Pipeline Walkthrough

**Purpose of this document.** A complete, script-by-script map of the motif-scaffolding benchmark as it
actually ran (HER2/PD-L1/IL-13/IL-1β/IL-2/TNF-α interface epitopes × 5 generators × AF3 refold scoring),
written so it can be handed to Claude to build a single, reproducible GitHub repo. It covers: where every
script lives today, what it does, what it consumes, what it emits and where that lands, which stages are
**shared/method-agnostic** vs **per-method**, and which scripts are **legacy/superseded** and should *not*
be carried into the unified repo. Nothing here rewrites or deletes anything — it is analysis only.

All absolute paths are on ETH Euler under `/cluster/work/wicky` (locally mounted at `eulermount/`).
Shorthand used below: `WICKY = /cluster/work/wicky`, `BEK = $WICKY/bekessel`.

---

## 0. The big picture — three code layers + one data contract

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │  esp/  — CORE LIBRARY (shared primitives, single source of    │
                       │  truth for "what is a motif", contig grammar, AF3 inputs,     │
                       │  scoring metric).  Every stage imports from here.             │
                       └─────────────────────────────────────────────────────────────┘
   STAGE 0 (library)          STAGE 1 (generate, PER-METHOD)      STAGES 2–5 (SHARED downstream)
   ┌───────────────┐          ┌──────────────────────────┐        ┌────────────────────────────┐
   │ extract-motifs│  pool +  │ <method>_from_library.py │ scaff- │ 1_filter → mpnn_prep →      │
   │ build-pool    │ ───────▶ │ <method>_generate.sbatch │ ─olds▶ │ 2_mpnn → 3_af3_build →      │
   │ motif_extract │ templates│ (5 generators)           │        │ 4_af3 → 5_score             │
   └───────────────┘          └──────────────────────────┘        └────────────────────────────┘
                                                                             │
                                                                             ▼
                                                                   ANALYSIS (notebook + helpers)
```

**The one data contract that ties everything together** (produced once per run, read by every method and
every downstream stage):

| artifact | produced by | consumed by |
|---|---|---|
| `<run>/motifs/<task>.pdb` | Stage 0 stager | every generator (motif template) + `mpnn_prep` + `1_filter` + `5_score` (native motif) |
| `<run>/manifest.csv` | Stage 0 stager / `<method>_from_library.py` | `1_filter` (contig grammar), every generator, contig-scaling |
| `<run>/<method>/scaffolds/…` | Stage 1 generator | `1_filter` (backbone) / `unindexed_score` (uidx) |

`task_name = <pdb>_m<motif_num>_<variant>` (e.g. `her2_m128_base`) is the join key everywhere.

**Two critical design facts (verified against the run data):**
1. **Generation is the ONLY method-specific stage.** Everything after a scaffold exists (filter → MPNN →
   AF3 build → AF3 → score) is ONE shared, method-agnostic script per stage in `software/downstream/`.
2. **Co-design methods (La-Proteina idx & uidx) skip MPNN.** They branch: instead of `1_filter → mpnn_prep
   → 2_mpnn`, they run `la_proteina_extract_seqs.py` (+ `unindexed_score.py` for uidx), then rejoin the
   shared path at `3_af3_build`.

---

## 1. `esp/` — the core library

**Location:** `BEK/epitope_scaffolding/pipeline/esp/` (a Python package). Put its parent on `PYTHONPATH`:
`export PYTHONPATH=$WICKY/bekessel/epitope_scaffolding/pipeline:$PYTHONPATH`. Every host-side script imports
from here; this is the single source of truth and must be the shared library in the new repo.

| module | what it provides | key functions |
|---|---|---|
| `motif.py` | **Motif definition.** SASA-based: surface residues (RSA > 0.20 via biotite Shrake-Rupley + Tien maxASA) are centroids; motif = centroid + Cα/Cβ contacts ≤ 7 Å. Verified to reproduce Lisa's motifs exactly. | `surface_rsa`, `contacts_of`, `expand_neighborhood`, `select_residues` |
| `contig.py` | **Contig grammar** (RFdiffusion-style, consumed by ALL generators). Three modes over one residue list: `as_is`, `reduce_disjoint` (fill gaps ≤ threshold), `full_stretch`. | `group_runs`, `merge_gaps`, `contig_string`, `disjointedness`, `make_manifest` |
| `catalog.py` | **Library catalog** (Lisa-style tables). Writes `{pdb}_sasa_resis.csv` + `{pdb}_motifs.csv` (one row per centroid: motif_num, residues, length, disjointed_parts, dssp) + per-motif template PDBs. | `sasa_rows`, `motif_rows`, `filled_rows`, `add_descriptors`, `write_motif_pdbs` |
| `af3_inputs.py` | **AF3 monomer-refold JSON builder.** One AF3 job per designed sequence (single chain, empty MSA, seed 42, no templates). Consumes ProteinMPNN/`extract_seqs` fastas. | `parse_fasta`, `build_inputs` |
| `score.py` | **Designability metric** (MotifBench). `motif_rmsd_refold` (fixed superposition of refold motif backbone N,CA,C at design positions vs native ≤ 1.0 Å) + `sc_rmsd` (CA superposition ≤ 2.0 Å) + AF3 confidences/pLDDT; handles AF3 residue-renumbering. **This is the metric `downstream/5_score.py` reuses.** | `motif_rmsd_refold`, `sc_rmsd`, `analyze`, `score_run`, `rollup`, `write_csv` |
| `rmsd.py` | Kabsch superposition + `filter_run` (used by the esp.run RFdiffusion campaign path). | `filter_run` |
| `stages.py`, `run.py`, `config.py`, `compute.py`, `_dssp.py` | The **`esp.run` CLI** (see §6): single-target RFdiffusion campaign orchestration + sbatch templating + flat-YAML config loader + vendored DSSP. | `esp.run` subcommands |

> **Repo note.** `esp/` should be a top-level installable package (`pip install -e .`). Stages 0/1/2–5 all
> import it; keeping it central is what guarantees "the motif is defined the same way everywhere".

---

## 2. Stage 0 — Library generation (motifs → pool → templates)

Goal: from a set of target PDBs, produce (a) per-protein motif catalogs, (b) one **pool CSV** (the master
list of scaffolding tasks), and (c) **motif template PDB archives** the generators scaffold.

### 2.1 Per-PDB motif extraction

Two equivalent entry points (both use `esp.catalog`/`esp.motif`, so definitions are identical):

- **`esp.run extract-motifs --config configs/<target>.yaml`** — for a configured target. Writes under
  `<lib>/<target>/`: `{target}_sasa_resis.csv`, `{target}_c_motif.csv`, `motif_templates/c_motif[.tar.zst]`
  (one all-atom PDB per motif). Config is a flat YAML: `target, pdb, chain, rsa_cutoff (0.20),
  neighborhood_cutoff (7.0), neighborhood_metric (ca|cb)`. Example configs:
  `BEK/epitope_scaffolding/pipeline/configs/{1n8z_her2,3bis_pdl1}.yaml`.
- **`scripts/motif_extract.py`** — the **generalized extractor for ANY new PDB** (fetch from RCSB or local).
  `screen` (report chains/SSE/surface) then `extract` (motifs + descriptors + **cross-protein redundancy
  check** vs the existing library, `redundancy_vs_library`, z-threshold 0.75). This is the script to run
  when plugging in a brand-new protein. Location: `BEK/scripts/motif_extract.py`.

### 2.2 Pool assembly + dedup + templates

- **`esp.run build-pool`** (`esp/run.py::build_pool`) concatenates every protein's `c_motif.csv` into one
  provenance-tagged pool (`pdb_name` column). This is the raw pool.
- **Dedup + interface filtering + padding** (the `motif_pool_interface_j0.7_pad5.csv` used by the
  benchmark) is done in the notebook **`BEK/notebooks/motif_library_generation.ipynb`** (Jaccard-0.7
  redundancy removal, interface-residue restriction via `motif_library/interface_residues.json`, pad5
  neighborhood). The pool variants live at `BEK/motif_library/`:
  `motif_pool.csv` → `motif_pool_deduped.csv` → **`motif_pool_interface_j0.7_pad5.csv`** (the 314-task
  benchmark pool; the one the notebook loads).

> **Repo note / the one gap to formalize.** The dedup+pad step currently lives in a notebook. For
> reproducibility it should become a script `build_pool.py --jaccard 0.7 --pad 5 --interface
> interface_residues.json` so a new protein flows pool→dedup without a notebook. Everything else is already
> scripted.

**Pool columns (the contract for Stage 1):** `pdb_name, motif_num, variant (base|filled), chain,
motif_residues, motif_length, disjointed_parts, dominant_sse, dssp` (+ provenance).

**Outputs of Stage 0 land at:**
`BEK/motif_library/{<pdb>/…, motif_pool_interface_j0.7_pad5.csv, <pdb>/motif_templates/*.tar.zst}`.

---

## 3. Stage 0.5 — Stage the pool into a run (the shared stager)

Before any generator runs, the pool is materialized into a **run directory** — the shared contract.

- **`BEK/epitope_scaffolding/pipeline/scripts/stage_motif.py`**
  `python stage_motif.py --pool $POOL --run-dir $RUN --variants base --nsamples 100`
  → writes `$RUN/motifs/<task>.pdb` (each motif template staged once) + `$RUN/manifest.csv`
  (`config_number, task_name, pdb_name, motif_num, variant, chain, n_fragments, fragments, contig,
  motif_min_length, motif_max_length, nsamples, motif_pdb`). `config_number == SLURM array index == the
  same motif in every method`, which is what makes the benchmark fair.

RFdiffusion reads this manifest directly (no per-method driver). The other four generators' `from_library`
drivers **re-derive the identical manifest** (via `esp.contig`) as they build their configs, so running the
stager is optional for them but recommended for a single canonical `manifest.csv`.

`<run>` convention: `BEK/motif_library/runs/<run_name>/` (e.g. `lib_interface_pad5`,
`lib_interface_pad5_uidx`, `lib_interface_pad5_contigscale`).

---

## 4. Stage 1 — Generation (PER-METHOD; the only method-specific stage)

Pattern for all backbone + co-design generators:
`<method>_from_library.py` (host, gen_env) builds per-motif configs + manifest → `<method>_generate.sbatch`
(SLURM array, one task = one motif = one `--config_number N`) runs the model container → **scaffolds land in
`$RUN/<method>/scaffolds/`** (co-design: `$RUN/<method>/inference/…`, normalized later).

All drivers share flags: `--pool $POOL --run-dir $RUN --variants base[,filled] --nsamples N`. All generate
configs use the **same contig** built by `esp.contig` (flanks `2-10`/`35-75`, linker `1-5`) — held identical
across methods for fairness. Container `.sif` files live in each method dir or `$WICKY/containers/`.

| # | method | driver (`software/<dir>/scripts/…`) | generate sbatch | output dir | MPNN? | notes |
|---|---|---|---|---|---|---|
| 1 | **RFdiffusion** (`rfdiff`) | *(none — reads `$RUN/manifest.csv`)* | `software/rfdiff/rfdiff_generate.sbatch` | `$RUN/rfdiff/scaffolds/<task>/design_*.pdb` | yes | reuses esp's `run_inference.py` call; `containers/RFdiffusion.sif` |
| 2 | **Proteina** (`proteina`) | `proteina_from_library.py` | `proteina_generate.sbatch` | `$RUN/proteina/scaffolds/inf_<N>/…` | yes | consumes RFdiff contig grammar; `containers/proteina.sif`; **0-based residue numbering** (handled downstream) |
| 3 | **Genie3** (`genie3`) | `scripts/genie3_from_library.py` (converter → MotifBench problem set + `experiment.yaml`) | `genie3_generate.sbatch` (`--export CONFIG=$RUN/genie3/experiment.yaml`) | `$RUN/genie3/output/<task>/results/` | yes | CA-only backbones; `software/genie3/genie3.sif` |
| 4 | **Protpardelle-1c-bb** (`pp1c_bb`) | `scripts/protpardelle_from_library.py --model cc58` | `protpardelle_generate.sbatch` (`--export METHOD=pp1c_bb`) | `$RUN/pp1c_bb/scaffolds/inf_<N>/<model>/<task>/` | yes | `software/protpardelle-1c/protpardelle.sif` |
| 5 | **La-Proteina idx** (`laproteina`) | `scripts/la_proteina_from_library.py` | `la_proteina_generate.sbatch` (`--export RUNTAG=$(basename $RUN)`) | `$RUN/laproteina/inference/inf_<RUNTAG>_<N>_<task>/` | **no (co-design)** | all-atom seq+structure; `software/la-proteina/la-proteina.sif` |
| 5b | **La-Proteina uidx** (`laproteina` in a `_uidx` run) | `software/la-proteina-uidx/scripts/la_proteina_from_library.py` | same, unindexed base config | `$RUN_uidx/laproteina/inference/…` | **no (co-design)** | separate repo/checkpoint; model chooses motif placement |

**Reference recipe with exact commands:** `BEK/motif_library/PILOT_RUNBOOK.md` (the 3-motif pilot) shows
DIR · DRIVER · LAUNCH · OUTPUT · GPU for every method. This is the canonical launch reference — mirror it in
the repo's README.

**Environment:** drivers run in **`gen_env`** — a **venv, not conda**: `source $WICKY/bekessel/gen_env/bin/activate`;
they need `PYTHONPATH=$WICKY/bekessel/epitope_scaffolding/pipeline`. Generators run inside their `.sif`
containers on GPU (`rtx_4090:1` at pilot; right-sized per method after `sacct`).

---

## 5. Stages 2–5 — Shared downstream (`software/downstream/`, method-agnostic)

**Location:** `BEK/software/downstream/`. This is the canonical, current, ONE-script-per-stage pipeline
(timestamps 2026-08-14 → 08-27). Every backbone method runs the full chain; co-design methods branch at 1–2.
Host-side steps run in `gen_env` (no container); AF3 runs in `alphafold3.sif`.

### Step 1 — `1_filter.py` (+ `resolve_filter.sbatch`) — motif-geometry gate [BACKBONE only]
- **Reads:** `$RUN/<method>/scaffolds/**.pdb`, `$RUN/manifest.csv` (contig grammar), `$RUN/motifs/<task>.pdb`.
- **Does:** for every design, enumerate contig-consistent placements of the motif, Cα-superimpose, take
  min-RMSD = scaffold motifRMSD; keep `< cutoff` (default 1.0 Å; `CUTOFF=99` passes all). Pure NumPy, no
  `.trb`, method-agnostic `design_iter` handles each generator's output naming. Fixes Proteina 0-based
  numbering by indexing positions **by rank**.
- **Writes** `$RUN/<method>/1_filter/`: `motif_rmsd.csv` (ALL scaffolds + `pass` flag — the post-generation
  denominator), `passed/<design>.pdb` (symlinks; `2_mpnn` reads this), `passed.txt`, `positions.jsonl`
  (passers, `{design:{"A":[ranks]}}`), `tasks.json` (passer→task).
- **Launch:** `sbatch --export=ALL,RUN_DIR=$RUN,METHOD=<m> resolve_filter.sbatch` (CPU, parallel workers).

### Step 1.5 — `mpnn_prep.py` (+ `.sbatch`) — stamp native motif identity [BACKBONE only]
- **Reads:** `1_filter/{passed/, positions.jsonl, tasks.json}` + `$RUN/motifs/<task>.pdb`.
- **Does:** backbones are poly-Gly/Ala; overwrites the motif residues' resnames with the **native identities**
  and emits ProteinMPNN **fixed positions** (by rank → 0-based-PDB safe). Flags: `--positions <file>`
  (override, e.g. uidx greedy positions), `--preserve-order` (positions already in native-residue order).
- **Writes:** `$RUN/<method>/mpnn/prepped/<design>.pdb` + `$RUN/<method>/mpnn/fixed_positions.jsonl`.
- **Launch:** `sbatch --export=ALL,RUN_DIR=$RUN,METHOD=<m> mpnn_prep.sbatch` (CPU, many workers, resumable).

### Step 2 — `2_mpnn.sbatch` — ProteinMPNN sequence design [BACKBONE only]
- **Reads:** `mpnn/prepped/` + `mpnn/fixed_positions.jsonl` (+ `parse_chains.py`, an O(files) drop-in for
  ProteinMPNN's `parse_multiple_chains.py`).
- **Does:** SHARDED SLURM array (one task = a slice of passers; sharding is required at ~31k backbones or the
  single-job parse stalls). 8 sequences/design, motif fixed.
- **Writes:** `$RUN/<method>/mpnn/seqs/<design>.fa`.
- **Note:** delete `mpnn/design_list.txt` before a re-run after `mpnn_prep` grows (it caches the design set).

### Step 2′ — `la_proteina_extract_seqs.py` — co-design normalizer [CO-DESIGN branch; replaces Steps 1,1.5,2]
- **Location:** `software/la-proteina/scripts/la_proteina_extract_seqs.py`.
- **Reads:** `$RUN/laproteina/inference/…` co-designed all-atom PDBs (+ each sample's `motif_info.csv`).
- **Writes (mirroring the backbone layout so downstream is identical):** `$RUN/laproteina/seqs/<design>.fa`
  (co-designed sequence as an MPNN-style fasta), `$RUN/laproteina/scaffolds/<design>.pdb`,
  `$RUN/laproteina/positions.jsonl` (`{design:{"A":[ranks]}}`, from the contig — nominal placement).
- **uidx add-on — `downstream/unindexed_score.py` (+ `.sbatch`):** the unindexed model doesn't tell you where
  it placed the motif, so its nominal positions are wrong. This does **greedy relocalization** (sequence-seeded
  Kabsch anchor + greedy nearest/aatype match, mirroring La-Proteina's `pad_motif_to_full_length_unindexed`)
  → `positions_unindexed.jsonl` (native-residue order) + `unindexed_relocalization.csv`. **These greedy
  positions are what `5_score`/`mpnn_prep` must use for uidx** (via `--positions … --preserve-order`).
  Validated: idx control is a no-op (recovers true positions); uidx median motifRMSD 4.06 Å (nominal) →
  0.035 Å (greedy).

### Step 3 — `3_af3_build.py` (+ `.sbatch`) — AF3 monomer-refold JSONs [SHARED, all methods]
- **Reads:** `mpnn/seqs` (backbone) **or** `seqs` (co-design). Same script, one flag decides.
- **Does:** builds one AF3 monomer JSON per designed sequence via `esp.af3_inputs.build_inputs` (empty MSA,
  seed 42). Parallel: shards seq files across CPU workers (backbone = ~250k JSONs), renumber-merges into
  global batches; a design's 8 seqs stay in one batch. Built on `$SCRATCH` (inode-safe), then tar'd per batch.
- **Writes:** `$RUN/<method>/refold_inputs/b0001.tar …` (one tar per batch of 500).
- **Launch:** `sbatch --export=ALL,RUN_DIR=$RUN,METHOD=<m> 3_af3_build.sbatch`.

### Step 4 — `4_af3.sbatch` — AF3 inference [SHARED, all methods]
- **Reads:** `refold_inputs/b<NNNN>.tar`. **Does:** SLURM array (one task = one batch), folds each sequence
  alone in `alphafold3.sif` (`--run_data_pipeline=False --run_inference=True`), NODE-LOCAL `$TMPDIR` (scratch
  ~1M-file quota), GPU-pool-portable (Turing→XLA kernel), resilient (quarantines NaN designs), requeue-safe.
- **Writes:** `$RUN/<method>/af3/archives/batch_<NNNN>.tar.zst` (one packed tar.zst per batch; inode-safe).
- **Launch:** `sbatch --array=1-<NBATCH>%92 --export=ALL,RUN_DIR=$RUN,METHOD=<m> 4_af3.sbatch`
  (to reach the bulk pool: `scontrol update jobid=<id> partition=gpuhe.bulk qos=es_wicky/gpuhe/bulk`).

### Step 5 — `5_score.py` (+ `.sbatch`) — designability scorer [SHARED, all methods]
- **Reads:** `af3/archives/*.tar.zst` + `positions.jsonl` (design positions) + `$RUN/motifs/<task>.pdb`
  (native). Reuses `esp.score` metric functions unchanged.
- **Does:** per sequence, `motifRMSD` (backbone N,CA,C, fixed superposition at design positions) + `scRMSD`
  (CA) + `ranking_score`/pLDDT. **3-gate designable = motifRMSD ≤ 1 & scRMSD ≤ 2 & ranking_score ≥ 0.75**
  (`RANK_MIN`). Parallel over archives, node-local `$TMPDIR`. Flags: `--positions <file>` (uidx greedy),
  `--preserve-order`.
- **Writes:** `$RUN/<method>/af3_refold_scores.csv` (one row per sequence:
  `design_name, motifRMSD, scRMSD, ranking_score, …, designable`). **This is the file the analysis notebook
  loads for every method.**
- **Launch:** `sbatch --export=ALL,RUN_DIR=$RUN,METHOD=<m>[,POSITIONS=<greedy.jsonl>] 5_score.sbatch`.

**Full downstream call sequence (copy-paste shape):**
```bash
RUN=$WICKY/bekessel/motif_library/runs/lib_interface_pad5 ; DS=$WICKY/bekessel/software/downstream
# BACKBONE (rfdiff|proteina|genie3|pp1c_bb):
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=$M            $DS/resolve_filter.sbatch     # 1
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=$M            $DS/mpnn_prep.sbatch          # 1.5
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=$M            $DS/2_mpnn.sbatch             # 2
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=$M            $DS/3_af3_build.sbatch        # 3
sbatch --array=1-<NB>%92 --export=ALL,RUN_DIR=$RUN,METHOD=$M $DS/4_af3.sbatch       # 4
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=$M            $DS/5_score.sbatch            # 5
# CO-DESIGN (laproteina): run extract_seqs (+ unindexed_score for uidx), then 3→4→5 with POSITIONS for uidx.
python $WICKY/bekessel/software/la-proteina/scripts/la_proteina_extract_seqs.py --run-dir $RUN
sbatch --export=ALL,RUN_DIR=$RUN_uidx,METHOD=laproteina $DS/unindexed_score.sbatch  # uidx only
sbatch --export=ALL,RUN_DIR=$RUN,METHOD=laproteina    $DS/3_af3_build.sbatch
sbatch --array=1-<NB>%92 --export=ALL,RUN_DIR=$RUN,METHOD=laproteina $DS/4_af3.sbatch
sbatch --export=ALL,RUN_DIR=$RUN_uidx,METHOD=laproteina,POSITIONS=$RUN_uidx/laproteina/positions_unindexed.jsonl $DS/5_score.sbatch
```

---

## 6. `esp.run` CLI — the ORIGINAL single-target RFdiffusion campaign (distinct from the benchmark)

`esp/run.py` is a CLI (`python -m esp.run <subcommand> --config configs/<target>.yaml`) for a **single target,
RFdiffusion-only** epitope-scaffolding campaign (the `1n8z_her2`, `3bis_pdl1` work). Subcommands: `build-pool`,
`extract-motifs`, `build-contig`, `launch-rfdiff`, `filter-designs`, `launch-mpnn`, `build-af3`, `launch-af3`,
`score-af3`, `status`, `archive`. It renders sbatch from `esp/../templates/*.tmpl` via `esp/stages.py`.

**Relationship to the benchmark:** its `build-pool`/`extract-motifs` ARE reused for Stage 0. Its
`filter-designs`/`launch-mpnn`/`build-af3`/`launch-af3`/`score-af3` are the **RFdiffusion-only** downstream and
were **superseded for the multi-method benchmark by `software/downstream/`** (which is method-agnostic and what
produced all the benchmark CSVs). Keep `esp.run` for single-target campaigns; use `software/downstream/` for
the benchmark. Its output layout differs slightly (`rfdiff_passed/`, `mpnn/seqs`, `af3_refold_scores.csv`).

---

## 7. Are the shared stages really the same for all methods? — YES

The user's key requirement. Verified:

- **MPNN, AF3-build, AF3, scoring are ONE script each in `software/downstream/`**, explicitly method-agnostic
  (their own docstrings say "ONE script, ONE code path" / "method-agnostic (one shared scorer for all
  methods)"). Method is a runtime `METHOD=` env var, not a different script. The only branch is co-design
  skipping MPNN (via `la_proteina_extract_seqs.py`), then rejoining at `3_af3_build`.
- **The scoring metric is identical for every method** — all call `esp.score` (motifRMSD ≤ 1 & scRMSD ≤ 2 &
  rank ≥ 0.75). Only the *design-position mapping* differs: backbone/Proteina/Genie3/pp1c use `1_filter`'s
  Kabsch localization (verified == RFdiffusion `.trb` for rfdiff, 400/400); La-Proteina idx uses the contig
  nominal positions; La-Proteina uidx uses greedy relocalization. Same gates, same code.

### Legacy / superseded scripts — DO NOT carry these into the unified repo
Each method dir still ships its **v1 per-method downstream** (older timestamps 2026-07-24 → 08-11), fully
replaced by `software/downstream/`:

| legacy (per-method) | replaced by (unified) |
|---|---|
| `proteina_mpnn.sbatch`, `genie3_mpnn.sbatch`, `genie3_mpnn_array.sbatch` | `downstream/2_mpnn.sbatch` |
| `proteina_fixed_positions.py`, `genie3_fixed_positions.py` | `downstream/mpnn_prep.py` |
| `proteina_af3_build.py`, `genie3_af3_build.py`, `la_proteina_af3_build.py` | `downstream/3_af3_build.py` |
| `proteina_af3.sbatch`, `genie3_af3.sbatch`, `la_proteina_af3.sbatch` | `downstream/4_af3.sbatch` |
| `proteina_score.py`, `genie3_score.py`, `la_proteina_score.py` | `downstream/5_score.py` |
| `genie3_from_run.py` | `genie3_from_library.py` (pool-driven) |

Keep only each method's **`<method>_from_library.py` + `<method>_generate.sbatch`** (+ La-Proteina's
`extract_seqs`). Everything else per-method is legacy.

---

## 8. Analysis layer

**Primary:** `BEK/notebooks/motif_benchmark.ipynb` — loads every `<run>/<method>/af3_refold_scores.csv`,
merges pool metadata, computes 3-gate designability normalized by the post-generation denominator
(`1_filter/motif_rmsd.csv` for backbone, `positions.jsonl` for co-design), and renders §1–§21 (overall,
disjointedness, SSE, length, protein, metric distributions, cost, base-vs-filled, coverage, idx-vs-uidx,
contig-scaling, ΔRSA burial, per-task Fig-7). See `motif_scaffolding_evaluation_pipeline.md` (docs) for the
figure-level write-up.

**Supporting scripts** (`BEK/scripts/` unless noted):
| script | role | output |
|---|---|---|
| `motif_rsa.py` (+ `.sbatch`) | Task-2 motif **burial ΔRSA** (biotite SASA / Tien maxASA; refold vs native `<pdb>_sasa_resis.csv`) | `<run>/<method>/motif_rsa.csv` |
| `extract_structures.py` (+ `.sh`) | pull scaffold PDB + AF3 refold CIF for chosen designs | `<run>/<method>/<out>/` |
| `motif_library/scripts/compute_costs.py` | sacct-based GPU-hours/method (LOGIN NODE) | `<run>/compute_costs.csv` |
| `af3_refold_analysis.py` | the original metric functions `esp.score` reuses | — |
| `designable_sse_extract.py`, `dssp_numpy.py`, `cif_backbone.py` | DSSP/backbone helpers for the designable set | — |
| `foldseek_*.sh`, `merge_refold.py`, `normalize_backbones.py` | cross-method Foldseek fold-space clustering (uniqueness of scaffolds) | `reports/foldseek/…` |

---

## 9. Contig-scaling sub-pipeline (rescue the all-failure motifs)

`BEK/motif_library/scripts/`:
| script | role | output |
|---|---|---|
| `find_failed_all.py` | materialize the motifs designable by NO method (0 across all) | `<run>/analysis/failed_all.txt` (+ features) |
| `scale_contigs.py` | build **native-geometry** contigs: per-gap linker = native-gap range `[max(n_min, g−5), g+5]`; `native_asym` (old flanks) & `native_sym` (0-60 flanks) | `scaled_contigs.csv` (run-manifest schema) |
| `stage_contigscale.py` | turn `scaled_contigs.csv` into a shared benchmark run (`manifest.csv` = the contract) | `runs/lib_interface_pad5_contigscale/{motifs/, manifest.csv}` |

Then the standard Stage 1 generators consume it via `--manifest` mode (Proteina/La-Proteina/Protpardelle got a
small `--manifest` consume-mode; RFdiff/Genie3 read the manifest as-is), followed by the same `downstream/`.

---

## 10. "Plug in a new protein" — the reproducible recipe

```bash
export PYTHONPATH=$WICKY/bekessel/epitope_scaffolding/pipeline:$PYTHONPATH
source $WICKY/bekessel/gen_env/bin/activate

# 0. Extract motifs from the new PDB (screen first, then extract + redundancy check)
python $WICKY/bekessel/scripts/motif_extract.py screen  --fetch 1XYZ
python $WICKY/bekessel/scripts/motif_extract.py extract --fetch 1XYZ --chain A --tag mytarget \
       --library $WICKY/bekessel/motif_library --out_dir $WICKY/bekessel/motif_library
# -> motif_library/mytarget/{mytarget_sasa_resis.csv, mytarget_c_motif.csv, motif_templates/*.tar.zst}

# 1. Rebuild the pool (currently notebook motif_library_generation.ipynb; formalize as build_pool.py)
#    -> motif_pool_interface_j0.7_pad5.csv  (adds mytarget's tasks)

# 2. Stage a run (shared contract)
POOL=$WICKY/bekessel/motif_library/motif_pool_interface_j0.7_pad5.csv
RUN=$WICKY/bekessel/motif_library/runs/mytarget_run
python $PIPE/scripts/stage_motif.py --pool $POOL --run-dir $RUN --variants base --nsamples 100

# 3. Generate on each method (see PILOT_RUNBOOK.md for the exact per-method block)
#    proteina/laproteina/pp1c: <method>_from_library.py --pool $POOL --run-dir $RUN ; then <method>_generate.sbatch
#    rfdiff: rfdiff_generate.sbatch (reads manifest) ; genie3: genie3_from_library.py --run-dir $RUN then generate

# 4. Downstream per method (see §5 call sequence): filter→mpnn_prep→2_mpnn→3_af3_build→4_af3→5_score
#    (co-design: extract_seqs [+unindexed_score] → 3→4→5)

# 5. Analyze: point motif_benchmark.ipynb RUN at $RUN and run top-to-bottom
```

---

## 11. Environments & containers (for repo packaging)

- **`gen_env`** — a **venv** (NOT conda): `source $WICKY/bekessel/gen_env/bin/activate`. Host-side scripts
  (drivers, downstream host steps, analysis) run here; they need `PYTHONPATH=…/epitope_scaffolding/pipeline`.
  Deps: numpy, pandas, biotite, scipy, scikit-learn, matplotlib, seaborn, PyYAML.
- **Generator containers (`.sif`):** RFdiffusion (`$WICKY/containers/RFdiffusion.sif`), Proteina
  (`containers/proteina.sif`), Genie3 (`software/genie3/genie3.sif`), Protpardelle
  (`software/protpardelle-1c/protpardelle.sif`), La-Proteina (`software/la-proteina/la-proteina.sif`,
  uidx: `software/la-proteina-uidx/la-proteina.sif`).
- **AF3:** `$WICKY/containers/alphafold3.sif` + params `$WICKY/params/af3_params`.
- **ProteinMPNN:** vendored at `software/ProteinMPNN` (+ `downstream/parse_chains.py` fast parser).
- **SLURM specifics:** GPU jobs on `gpuhe.4h`/`gpuhe.bulk`; AF3 uses node-local `$TMPDIR` (scratch has a ~1M
  inode quota); all heavy outputs are tar.zst-packed (inode budget); never run the full library filter/score
  on the login node.

---

## 12. Suggested unified-repo layout

```
motif-scaffolding-benchmark/
├── esp/                         # core library (pip install -e .) — §1
├── library/
│   ├── motif_extract.py         # per-PDB extractor — §2.1
│   ├── build_pool.py            # (NEW: script-ify the notebook dedup+pad — §2.2)
│   └── configs/<target>.yaml
├── stage/stage_motif.py     # §3
├── generators/<method>/{<method>_from_library.py, <method>_generate.sbatch}   # §4 (5 methods)
│   └── la-proteina/la_proteina_extract_seqs.py
├── downstream/                  # §5 — the SHARED chain, verbatim from software/downstream/
│   ├── 1_filter.py resolve_filter.sbatch mpnn_prep.py mpnn_prep.sbatch parse_chains.py
│   ├── 2_mpnn.sbatch 3_af3_build.py 3_af3_build.sbatch 4_af3.sbatch 5_score.py 5_score.sbatch
│   └── unindexed_score.py unindexed_score.sbatch
├── contigscale/{find_failed_all.py, scale_contigs.py, stage_contigscale.py}   # §9
├── analysis/{motif_benchmark.ipynb, motif_rsa.py, extract_structures.py, compute_costs.py, foldseek_*}  # §8
├── containers/                  # .def files (build recipes) for each .sif
└── docs/PILOT_RUNBOOK.md        # exact per-method launch commands
```
**Do NOT include** the per-method legacy downstream from §7 (the `*_af3_build.py`, `*_score.py`,
`*_mpnn.sbatch`, `*_fixed_positions.py`, `*_af3.sbatch`, `genie3_from_run.py`).
```
```
