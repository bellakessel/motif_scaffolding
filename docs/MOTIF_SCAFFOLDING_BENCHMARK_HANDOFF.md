# Motif-Scaffolding Benchmark — Full Handoff / Report Reference

**Project:** Motif Scaffolding for exploration of Protein–Protein Interaction Interfaces
(ETH Zürich, Wicky Lab, D-BSSE). MSc semester project, bekessel.
**Snapshot date:** 2026-08-31. This document is the single-source handoff for a new session and
the technical backbone for the written report. It records *what we did, how every model was set
up (containers, checkpoints, sources, parameters), the analysis, the results, and every issue /
failed job / problem→solution* encountered.

> Everything below reflects the state of `/cluster/work/wicky/bekessel/` (mounted locally at
> `eulermount/`). The author never SSHes to Euler — files are staged over the sshfs mount and all
> `sbatch`/generation commands are run by the user on the Euler login node.

---

## 0. One-paragraph summary

We built a **non-redundant, class-balanced library of PPI-interface epitope motifs** (734 motifs
from 7 antigen–antibody complexes), then scaffolded each motif with **five de-novo backbone/all-atom
generators** (RFdiffusion, Proteina, Genie3, Protpardelle-1c, La-Proteina) under an **identical
contig specification**, ran **one shared downstream** (ProteinMPNN → AlphaFold3 monomer refold),
and scored **designability** with the MotifBench three-gate criterion (motifRMSD ≤ 1 Å & scRMSD ≤ 2 Å
& AF3 ranking_score ≥ 0.75). We then characterised *which motifs* are designable and *which method
suits which motif* (disjointedness dominates), added an **all-atom motif-recapitulation cross-check**,
a **motif-burial/exposure (ΔRSA) analysis**, **Foldseek fold-space clustering + coil/helix/sheet
ternary** of the designable set, a **La-Proteina unindexed** placement study, and a **contig-scaling
rescue experiment** on the motifs that failed every method.

---

## 1. Research questions

1. **Pipeline success rate** — what fraction of interface motifs can be scaffolded into de-novo
   backbones and survive an AF3 refold (3-gate designable)?
2. **Predictors of scaffoldability** — which motif properties (size, disjointedness / number of
   fragments, dominant secondary structure, burial) predict designability?
3. **Method × motif** — does a given generator suit a given motif class; can any method scaffold what
   the others cannot?
4. **Motif classes that systematically fail**, and whether **native-geometry contigs** rescue them.

---

## 2. The motif library (the benchmark input)

### 2.1 Definition
A **motif** = one surface residue + its contacts within **7 Å** (Cα–Cα), i.e. one local interface
patch, extracted with the `esp` primitives (`esp/motif.py`: `surface_rsa()` = biotite SASA + Tien
maxASA, surface = **RSA > 0.20**; `contacts_of()` = Cα ≤ 7 Å). This reproduces Lisa Brandenburg's
`c_motif` definition **exactly** (verified: 40/40 3bis motifs identical; surface set differs only on
4 borderline residues at the RSA cutoff = SASA numerical noise).

### 2.2 Source complexes (7 antigens, all antibody-validated)
Extracted per protein into one flat provenance-tagged pool (`pdb_name` column). Configs:
`epitope_scaffolding/pipeline/library/configs/<tag>.yaml`.

| tag  | PDB / antigen chain | Fab chains | antibody               |
|------|---------------------|-----------|------------------------|
| her2 | **8PWH** / E        | A,B & C,D | trastuzumab + pertuzumab |
| tnfa | **3WD5** / A        | H,L       | adalimumab             |
| mcl1 | **2mhs** / A        | —         | none                   |
| il1b | **4G6J** / A        | H,L       | canakinumab            |
| il2  | **5LQB** / A        | H,L       | NARA1 (research)       |
| pdl1 | **5XXY** / A        | H,L       | atezolizumab           |
| il13 | **5L6Y** / C        | H,L       | tralokinumab           |

**ID traps that were verified against PDB metadata (do not recall from memory):** 3BIS = PD-L1
(*not* MCL-1; MCL-1 = 2mhs); the PD-L1+atezolizumab entry is **5XXY** (5XXX = microtubule/KIF5C);
3MK8 = MCL-1 (redundant with 2mhs); ustekinumab 3HMX binds p40 not the helical p35; 1BJ1 antigen =
chains V/W not the Fab H/J/K/L. Every Fab was confirmed to actually contact the extracted chain
(16–23 epitope residues each). 8PWH is a clean upgrade over 1n8z for HER2 (624 vs 581 res, two
antibodies, same Cys-rich ectodomain epitope; numbering aligned to 1n8z).

### 2.3 Secondary-structure labelling (DSSP)
**Bug found + fixed:** `catalog._sse_by_res` used biotite `annotate_sse` (**P-SEA**, Cα-geometry
only) while columns were named `dssp`/`dominant_sse` → systematically **over-called coil**. Replaced
with a **vendored dependency-light Kabsch–Sander DSSP** (`esp/_dssp.py`, a numpy rewrite of pydssp —
no torch/einops). Verified **100 % per-residue identical to Lisa's real mkdssp on shared 2mhs
(151/151)** and to reference pydssp on all 7 chains. Impact on the deduped pool:
H 90→117, E 65→113, C 473→398 (coil 75 %→63 %; sheet nearly doubled). Coil is *part real* (surface is
loop-rich) and *part artifact* (now fixed).

### 2.4 Dedup and balancing
- **Dedup = residue-set Jaccard ≥ 0.7, WITHIN (protein, variant)** (same surface patch = true
  redundancy). Jaccard is only meaningful within a protein; cross-protein redundancy would use
  standardised descriptor-space distance. SSE is *not* a second Jaccard axis — it is a label/balancing
  axis applied *after* dedup.
- Dedup must be **per-(protein,variant)**, never global: a filled motif is its own base + gap-fillers
  (self-Jaccard ≈ 0.86) so protein-only dedup would wrongly drop 78 % of filled motifs as
  self-duplicates.

### 2.5 Matched-pair variants (`base` vs `filled`) — the disjointedness control
The two extraction knobs move disjointedness in **opposite** directions: a larger neighborhood cutoff
grows length *and* disjointedness; `reduce_disjoint` gap-fill grows length but makes motifs
*contiguous*. To get a clean causal test of disjointedness controlling for the patch, each centroid
emits a matched pair:
- `variant=base`  — `c_motif` @ 7 Å.
- `variant=filled` — its `reduce_disjoint` version, gaps ≤ `gap_merge=4` bridged (present residues
  only; already-contiguous motifs skipped so there is no duplicate pair).

### 2.6 Final library
**734 motifs = 628 base + 106 filled**, saved to
`motif_library/motif_pool_dedup_j0.7.csv` (raw undeduped = 926). Per-protein (deduped):
her2 336 / tnfa 84 / il1b 76 / mcl1 71 / il2 65 / il13 52 / pdl1 50. SSE H = 133 / E = 131 / C = 470
(helix ≈ sheet, coil 64 %). The `filled` variant successfully populated the previously-empty
large(≥11)+contiguous cell (1→16 motifs) and reaches 16–21 res (base capped ~15).

> **The pool actually used for the benchmark runs** is `motif_pool_interface_j0.7_pad5.csv`
> (the interface-restricted, pad-5 variant referenced by the notebook as `POOL`), which drives the
> `lib_interface_pad5*` runs (task set = 314 tasks after the run's own filtering; base/filled are
> separate `task_name`s).

**Data bug that cost a wrong conclusion (watch when adding motif types):** `2mhs_c_motif.csv` ships
an extra unnamed pandas index column; reading the centre positionally (`iloc[0]`) silently returns
the *row index* not the residue id, corrupting every 2mhs descriptor without raising. Fix = select
**by column name** and raise if absent. 1n8z/3bis have no such column.

---

## 3. Benchmark design & the shared contract

### 3.1 One contig for every method (fairness)
Every motif is handed the **same RFdiffusion contig grammar**, emitted once by `esp.contig`:

```
2-10 / <frag> / 1-5 / … / <frag> / 35-75
```

i.e. N-flank sampled 2–10, uniform 1–5 linkers between fragments, C-flank 35–75. Scaffold-length
policy (`--flank-n 2-10 --flank-c 35-75 --linker 1-5`) and **nsamples/motif are held identical across
all methods**. Single-residue fragments are always written as ranges (`C579-579`) — RFdiffusion
requires it.

### 3.2 The shared run contract
All methods read two shared artifacts, staged once:
- `<run>/motifs/<task>.pdb` — all-atom motif, original chain + numbering (staged from the tar.zst
  template archives).
- `<run>/manifest.csv` — **the shared contract**: `config_number == SLURM array index == the same
  motif for every tool`; columns `task_name, pdb_name, motif_num, variant, chain, n_fragments,
  fragments, contig, motif_min_length, motif_max_length, nsamples, motif_pdb`.

Per-tool outputs live under `<run>/<method>/{configs,scaffolds,mpnn,1_filter,refold_inputs,af3,
af3_refold_scores.csv,motif_rsa.csv,…}`. Repos hold **source + scripts + checkpoints only**.

Shared stager: `epitope_scaffolding/pipeline/scripts/stage_from_pool.py`
(`--pool <csv> --run-dir <run> [--variants base] [--flank-*] [--nsamples 100]`).

### 3.3 The runs
| run dir | what | methods present |
|---|---|---|
| `lib_interface_pad5` | **baseline** — full library, fixed contig | rfdiff, proteina, genie3, pp1c_bb, laproteina (idx), laproteina_mpnn (idx) |
| `lib_interface_pad5_uidx` | La-Proteina **unindexed** (LD6) | laproteina, laproteina_mpnn |
| `lib_interface_pad5_contigscale` | 56 all-failure motifs × {native_asym, native_sym} | rfdiff, genie3, pp1c_bb, proteina, laproteina |
| `hardest_0pass` | earlier 23 RFdiff-0pass motifs | proteina, laproteina, pp1c_bb (probe run) |
| `smokes` | per-method single-motif smoke tests | all |

---

## 4. The five generators — setup, containers, checkpoints, sources, parameters, patches

All models run as **singularity/apptainer containers** (lab standard — no conda). Each model is
self-contained under `bekessel/software/<model>/`; source is **bind-mounted, not installed**
(`PYTHONPATH=<repo>/src`). Every run uses `singularity exec --nv -B /cluster/work/wicky …`,
`unset PYTHONHOME PYTHONPATH`, HF caches redirected to `$SCRATCH`, SLURM array = one motif per task.

### 4.1 RFdiffusion (baseline reference)
- **Role:** baseline; also Lisa's original pipeline. Full-backbone (N,Cα,C,O) generator.
- **Container / checkpoint:** `containers/RFdiffusion.sif` (3.9 GB, NVIDIA base);
  `params/RFParams/Base_ckpt.pt` (484 MB).
- **Driver:** none needed — `software/rfdiff/rfdiff_generate.sbatch` reads the shared manifest row
  for `$SLURM_ARRAY_TASK_ID` (awk over cols) and calls the **exact esp** entrypoint:
  `singularity exec --nv … RFdiffusion.sif python3 /app/RFdiffusion/scripts/run_inference.py
  inference.input_pdb=<motif_pdb> inference.output_prefix=<OUT>/design
  inference.num_designs=<N> inference.ckpt_override_path=<CKPT> inference.write_trajectory=False
  "contigmap.contigs=[<contig>]" "contigmap.length=<min>-<max>"`.
- **`write_trajectory=False`** = inode saver. Motif positions read later from the design **`.trb`**
  (`con_hal_idx0`) — ground-truth, no search.
- **Output:** `<run>/rfdiff/scaffolds/<task>/design_*.pdb`. GPU rtx_4090.

### 4.2 Proteina (NVIDIA proteinfoundation)
- **Model:** Cα-only backbone diffusion (400-step SDE). Output is **Cα-only, all-ALA** (sequence
  comes from MPNN downstream). Motif pinned at native geometry (0.12–0.21 Å Cα RMSD verified).
- **Container:** existing `containers/proteina.sif` (7.9 GB; compiled PyG 2.6.1 +
  pytorch-sparse/scatter/cluster @ torch 2.4.1 — reused rather than rebuilt; a lean `proteina.def`
  is a deferred reproducibility fallback).
- **Checkpoint / source:** `proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt` (851 MB) from
  **NVIDIA NGC** (public files API, no auth):
  `wget --content-disposition "https://api.ngc.nvidia.com/v2/resources/nvidia/clara/proteina_v1.7_dfs_60m_notri_motif_scaffolding/versions/1.0/files/…ckpt"` → `checkpoints/`.
  Also needs `data/pdb_raw/cath_label_mapping.pt` from `proteina_additional_files.zip` (NGC, 199 MB).
- **Entrypoint:** `proteinfoundation/motif_inference.py` (hydra `--config_subdir lib --config_number N`
  → `inf_N.yaml`); `compute_designability/fid: False` (we use our own downstream).
- **Driver:** `software/proteina/scripts/proteina_from_library.py` (consumes shared manifest;
  `--manifest` consume-mode for contig-scaling).
- **4 required source patches to `motif_inference.py` (must persist if re-cloned):**
  (1) guard `df.to_csv` under `if cfg.compute_designability` (was NameError on gen-only runs);
  (2) replace `cfg.motif_task_name.split('_')[0]` → `cfg.motif_task_name` at lines ~360/364 (our
  task_names are `{pdb}_m{num}_{variant}`, the split assumed a bare PDB id → FileNotFoundError);
  (3) config dir from `PROTEINA_CONFIG_DIR`; (4) root_path from `PROTEINA_OUT_DIR`.
- **Consumes the RFdiffusion contig grammar directly** (its big integration win).

### 4.3 Genie3 (aqlaboratory, commit d77ae5a)
- **Model:** all-atom SE(3) diffusion. Output PDB is **Cα-only** (scaffold Cα trace = UNK; motif =
  Cα + sidechain, no N/C/O) → must be inverse-folded with ProteinMPNN in **`--ca_only`** mode.
- **Container:** lean generation-only `genie3.def` (base `pytorch/pytorch:2.7.1-cuda12.6-cudnn9`,
  pip: lightning, ml-collections, biopython<1.86, numpy≥2, scipy, pandas, tqdm, zstandard,
  huggingface_hub). `generate` lazily imports the eval stack, so gen-only is clean.
- **Checkpoint / source:** HF `yeqinglin/genie3`, `pretrained/v1/checkpoints/step=600000.ckpt` +
  `pretrained/v1/config.yaml`, pulled **through the container** (HF caches → `$SCRATCH/hf`; host has
  no `hf` CLI and `/cluster/home` is read-only inside the container).
- **Input:** motif PDB with `REMARK 999 INPUT <chain><start><end>` (**fixed columns**, 4-wide, NO
  separators — see §8 bug) + a 4-key problem JSON (`motif_filepaths`, `segment_config_str`,
  `min/max_total_length`). `segment_config_str` = the RFdiff contig with `/`→`,` and each motif token
  auto-indexed `A1,A2,…` (e.g. `2-10,A1,1-5,A2,35-75`).
- **Driver:** `software/genie3/scripts/genie3_from_library.py` (reads shared manifest+motifs; derives
  segments from the **contig**, robust to whichever driver wrote the manifest). Launch via
  `genie3_generate.sbatch` with `CONFIG=<run>/genie3/experiment.yaml`. `sampler.direction_scale: 0.1`
  (motif default). GPU rtx_4090/a100 (sm_80+).

### 4.4 Protpardelle-1c (Lu & Shuai bioRxiv 2025, PMC12393353)
- **Two sub-methods** (both used): **cc58** → backbone method **`pp1c_bb`** (writes N/Cα/C/O poly-Gly
  with native motif residues stamped → shared MPNN→AF3); **cc91** → all-atom co-design method
  **`pp1c_aa`** (writes full atoms + co-designed sequence → extract-seq→AF3, like La-Proteina).
  Model triples: `["cc58","416","sampling_sidechain_conditional"]`,
  `["cc91","383","sampling_sidechain_conditional_allatom_s1"]`. Both condition on motif backbone +
  sidechain atoms (crop-conditional guidance, classifier-free at γ_t=1).
- **Container:** `protpardelle.def` (base `pytorch 2.6.0-cuda12.4`, pure-pip pyproject deps, **no
  compiled PyG → easiest container**; source bind-mounted `PYTHONPATH=<repo>/src`).
- **Checkpoint / source:** Zenodo record **16817230** tarball (cc58 + cc91 in
  `model_params/{weights,configs}`). ESMFold/MPNN/LigandMPNN/Foldseek in the repo = eval-only, skipped.
- **Entrypoint:** `python -m protpardelle.sample <cfg.yaml> --motif-dir <dir> --num-samples N
  --num-mpnn-seqs 0`. Motif contigs are the RFdiff grammar but **`;`-delimited**
  (`2-10;C570-580;35-75`) + `total_lengths: [[min,max]]`.
- **Driver:** `software/protpardelle-1c/scripts/protpardelle_from_library.py` (`--model cc58|cc91`;
  `/`→`;`). Per-sample motif positions from the output `scaffold_info.csv` (`motif_placements`).

### 4.5 La-Proteina (NVIDIA, all-atom co-design)
- **Model:** **all-atom co-design** (structure + sequence jointly) → **NO MPNN** (sequence read from
  the output PDB). Two checkpoints define two tracks that differ **only by checkpoint** (contig format
  identical):
  - **LD4 = indexed** (`laproteina`, run `lib_interface_pad5`): pins the motif at the contig indices
    (~0.01 Å).
  - **LD6 = unindexed** (`laproteina` in run `lib_interface_pad5_uidx`): embeds the motif *structure*
    anywhere, **ignoring sequence indices, linker, padding, and fragment order** — only the motif
    fragments + total-length range are consumed.
- **Container:** `la-proteina.def` (base `pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime`; pip pins per
  `environment.yaml`: torch 2.7.0+cu118, graphein 1.7.7 `--no-deps`, PyG wheels for 2.7.0+cu118,
  numpy 1.23.5, +loralib/pydantic/sklearn/seaborn/xarray; **gcc/g++ added** for the cpdb-protein
  C-extension; `proteina.sif` does **not** work — no graphein). `DATA_PATH` env must be set (config
  interpolates `${oc.env:DATA_PATH}` for its eval-only metric factory).
- **Checkpoints / source:** `LD4_motif_idx_aa.ckpt` + `AE3_motif.ckpt` (autoencoder), and the LD6
  unindexed checkpoint, from **NGC** → `checkpoints_laproteina/` (paths **pinned absolute** in the
  generated config — shipped config is CWD-relative and breaks on `cd`).
- **Entrypoint:** `proteinfoundation/generate.py` (same hydra interface as Proteina;
  `defaults:[inference_motif_idx_aa]` → base configs must live in repo `configs/`). **No code edit
  needed** (the all-atom `bulk_all_atom_xmotif` branch is already active).
- **Driver + co-design downstream (replaces MPNN):**
  `software/la-proteina/scripts/la_proteina_from_library.py`, then
  `la_proteina_extract_seqs.py` (co-designed chain-A seq → MPNN-style fasta in `<method>/seqs/`,
  stages all-atom design as `scaffolds/<design_name>.pdb`, writes motif design positions into
  `positions.jsonl`). Output layout:
  `inference/inf_<runtag>_<idx>_<task>/job_0_id_<N>_motif_<task>/…pdb`, all-atom chain A renumbered
  1..N with real co-designed sidechains. Junctions near-ideal (3.74–3.92 Å; better than Proteina's
  Cα-only ~4.3 Å).

### 4.6 Cα-only is the common denominator
Because Proteina and Genie3 emit Cα-only backbones, **Cα is the only cross-method-comparable metric**;
`1_filter` and the universal localizer operate on Cα throughout.

---

## 5. Shared downstream (identical for all methods)

Location `bekessel/software/downstream/`, numbered by pipeline order. Runs in **`gen_env`** (a venv,
not conda: `source /cluster/work/wicky/bekessel/gen_env/bin/activate`; esp imports need
`PYTHONPATH=/cluster/work/wicky/bekessel/epitope_scaffolding/pipeline`).

```
generate ─▶ [resolve positions] ─▶ 1_filter ─▶ [2_mpnn backbone | extract_seqs co-design]
         ─▶ 3_af3_build ─▶ 4_af3 ─▶ 5_score
```

- **Motif-position resolver** — how each method's design motif positions are recovered:
  - rfdiff: design `.trb` `con_hal_idx0` (ground truth).
  - laproteina (idx) / pp1c_aa: co-design `positions.jsonl` from `extract_seqs`.
  - proteina / pp1c_bb / genie3: `resolve_positions.py` — a **metadata-free universal localizer**:
    enumerate contig-consistent placements, Kabsch-superimpose the native motif Cα at each, take the
    min-RMSD placement. Validated to reproduce ground truth.
  - laproteina **unindexed**: **greedy relocalization** (`mpnn/fixed_positions.jsonl` for the MPNN
    track; `positions_unindexed.jsonl` for the co-design track) — see §7.

- **`1_filter.py`** — post-generation motif-geometry gate: superimpose the design motif backbone onto
  the original template, keep `motifRMSD < CUTOFF`. Baseline pass rates (methods pin the motif):
  **proteina 100 %, genie3 99.2 %, pp1c_bb 99.6 %, rfdiff 77.7 %** at CUTOFF=1.0 → baseline effectively
  MPNN'd ~all designs. Contig-scale (harder motifs) drops to 46–60 %, so backbone methods there run
  **CUTOFF=99** (MPNN everything, let AF3 decide). Writes `1_filter/{motif_rmsd.csv, positions.jsonl,
  passed/<canonical>.pdb, tasks.json}`. Canonical design name = `<task>_<local>` (RFdiff `design_N`
  collides across motifs).

- **`mpnn_prep.py` + `2_mpnn.sbatch`** — stamp native motif identity into the poly-Gly/ALA passers at
  the design ranks + emit ProteinMPNN `fixed_positions.jsonl`; then **8 sequences/design**
  ProteinMPNN, TEMP 0.2, SEED 42, omit-Cys on scaffold (motif Cys kept). `CA_ONLY` knob: proteina &
  genie3 = `--ca_only` v_48_020 (openfold.sif); rfdiff & pp1c_bb = vanilla full-backbone MPNN.
  Co-design methods (laproteina, pp1c_aa) **skip 2_mpnn**.

- **`3_af3_build.py` + `4_af3.sbatch`** — build **AF3 monomer JSONs** (empty MSA, seed 42, chain A;
  via `esp/af3_inputs.py`), then AF3 inference-only (`--run_data_pipeline=False`) → per batch one
  `af3/archives/batch_NNNN.tar.zst`. **Node-local `$TMPDIR`** build+fold (Euler personal scratch has a
  ~1,000,000-**file** quota; ~1M tiny JSONs blew it — everything now tars per batch, ~4k files total).
  AF3 ≈ 20 s/fold, ~2.8 h per 500-design batch; `--time=08:00:00`, `--array=1-<NB>%92`; per-batch
  skip-existing keeps arrays resumable. Turing sm_75 needs `FA=xla` + XLA custom-kernel flags.

- **`5_score.py`** — computes the metrics from the AF3 archives + positions + native motif:
  - **motifRMSD** = fixed backbone superposition of the refold motif (`con_hal_idx0`) onto the native
    motif (`con_ref_pdb_idx`), atoms **N,Cα,C** (Kabsch). Identical for all methods; only the
    position mapping differs.
  - **scRMSD** = Cα-only whole-chain refold-vs-design (independent of positions).
  - `--predictor {af3,esmfold2}`, `--positions <file>` override, `CODESIGN_METHODS={laproteina,
    pp1c_aa}` branch (reads top-level `positions.jsonl` + `scaffolds/`; backbone methods read
    `1_filter/positions.jsonl` + `1_filter/passed/`). Output `<method>/af3_refold_scores.csv`.
  - The CSV `designable` column is **only 2 gates** (motifRMSD ≤ 1 & scRMSD ≤ 2). The 3rd gate
    (ranking_score ≥ 0.75) is applied in analysis (5_score was later patched to include it, but treat
    the analysis 3-gate as canonical).

### 5.1 Designability = THREE gates (project-wide)
```
designable  ⇔  motifRMSD ≤ 1.0 Å  AND  scRMSD ≤ 2.0 Å  AND  ranking_score ≥ 0.75   (inclusive ≥)
```
The `≥` matters: ranking_score is stored rounded to 2 dp, so a large mass sits **exactly** at 0.75
(12,102 predictions in `lib_interface_pad5`); using `>` silently drops ~14 %. Under only 2 gates the
generators are within ~1.5 % of each other — **the confidence gate is what separates them.** Note
`iptm` is all-null (monomers) and `ranking_score ≠ ptm` (agree only ~53 %) — never substitute.

MotifBench provenance (arXiv:2502.12479): success = ≥1 of 8 MPNN seqs passes motifRMSD ≤ 1 Å (N,Cα,C)
AND scRMSD ≤ 2 Å (Cα). We substitute **AF3** for their ESMFold/AF2-no-MSA refolder (lab verified
ESMFold over-passes vs AF3, so it is the looser, untrusted gate).

---

## 6. Analysis

All analysis is in `bekessel/notebooks/motif_benchmark.ipynb` (main, ~84 cells) and
`notebooks/foldseek_cluster_analysis.ipynb` (Parts A–F). Cell 1 defines RUN/RUN_UIDX, the 8-method
`LOAD` map, LABELS/PALETTE, the three thresholds, and `prettify()`. `per_design.designable` is the
**3-gate** flag (`seq_pass` = all three; design designable ⇔ any of its ≤8 seqs passes).

**Methods loaded (8):** rfdiff, proteina, genie3, pp1c_bb, laproteina (idx native), laproteina_mpnn
(idx +MPNN), laproteina_unindexed (uidx native), laproteina_unindexed_mpnn (uidx +MPNN).

### 6.1 Core designability (§1, §11)
Per-method designability rate = motifs with ≥1 designable design / **post-generation** scaffolds
(the true denominator: a generated-but-never-refolded scaffold counts as a non-success). Post-gen
denominator = `1_filter/motif_rmsd.csv` (backbone methods) or `positions.jsonl` keys (co-design).

### 6.2 motifRMSD / scRMSD distributions (§6) + all-atom cross-check (§6B)
`scripts/motif_atom_crosscheck.py` recomputes, per **passing** prediction, three atom sets from ONE
Kabsch over the motif atoms (paired by atom name): **bb3** (N,Cα,C = our metric, ≤1 by construction),
**bb4** (+O = Genie3 backbone), **heavy** (all non-H = Genie3/MotifBench all-atom, their gate < 2 Å).
Fast because it opens only each passing design's single AF3 `.cif` + does one small SVD (no MSA, no
refold, no placement search).
- **What it validates:** it reproduces the scorer (sanity) and, via **heavy** atoms (native sidechain
  identity, MPNN-fixed), gives a mild *independent* corroboration that the motif residues really sit
  where `positions.jsonl` claims (backbone-only bb3 could match a wrong-but-similar loop; sidechains
  can't). It does **not** independently re-derive the alignment (reuses the same positions.jsonl).
- **Kabsch / short-motif note:** Kabsch needs ≥3 non-collinear atoms; bb3 = 3 atoms/residue and passer
  motifs span **4–21 res (median 7, min 4 = 12 atoms)** → never under-determined.

### 6.3 Motif burial / exposure — ΔRSA (§17)
`scripts/motif_rsa.py` + `motif_rsa.sbatch` (48 CPU, biotite Shrake–Rupley over every AF3 refold,
node-local `$TMPDIR`). Output `<method>/motif_rsa.csv` (cols incl. `mean_rsa_denovo`,
`mean_rsa_native`, `mean_dRSA`, `frac_buried`, per-residue ΔRSA strings). Exposure metric =
`mean_rsa_denovo` (0 buried → 1 exposed). De-novo designs **tend to expose** motifs (median ΔRSA
≈ +0.06). Supports `POSITIONS` override for unindexed (`positions_unindexed.jsonl`).

### 6.4 Foldseek fold-space clustering + ternary (designable library)
Two passes exist. The **field-standard, current** one clusters **AF3 refolds** (uniform by
construction — all-atom, real seq, chain A) at **TM-0.5** (`--alignment-type 1 --tmscore-threshold
0.5 --cov-mode 0 --min-seq-id 0`), scoring exactly one design/cluster (lowest scRMSD). Scripts:
`scripts/{dssp_numpy.py, cif_backbone.py, designable_sse_extract.py, foldseek_designable_library.sh}`;
notebook Part F. DSSP for the coil/helix/sheet ternary = the vendored numpy pydssp kernel (verified
bit-identical to upstream on 36,480 residues).
- **26,707 designable designs → 2,343 folds.** Diversity (clusters/design): **Proteina 0.307** ≫
  rfdiff 0.202 > genie3 0.184 > laproteina_mpnn 0.167 > **pp1c_bb 0.117** — Proteina is *least*
  designable but *most* fold-diverse (the headline tension; holds under rarefaction to equal n).
- Mean SSE: pp1c_bb is by far the most beta (E .35); genie3 the most helical (H .62).
- Motif SSE → design SSE is a strong clean signal (helical motif → design 76 % H).
- On refolds the **motif** predicts fold better than the generator (AMI: motif identity 0.099 >
  target protein 0.082 > generator 0.080), the reverse of the backbone-clustering pass.
- An earlier **backbone** clustering pass (`foldseek_cluster_analysis.ipynb` Part E) had to CA-reduce
  + poly-Gly everything (Proteina/Genie3 are CA-only, names method-correlated) and gave a weak signal
  (AMI 0.041) — kept only as a cross-check; refolds are the reportable version.
- **Lisa's 3-protein AF3 set** (re-run at TM-0.5): 126,531 designable → 3,204 folds; antigen does not
  dictate fold (56 % of folds span >1 antigen, holding 98 % of designs).

### 6.5 La-Proteina unindexed — placement study (§22)
`scripts/motif_placement.py` (stdlib-only): scaffold length L = chain-A Cα count, group motif
positions into contiguous fragments, one row/fragment with `motif_center_rel` (0=N→1=C),
`frag_dist_to_end` (0=terminus→0.5=centre), padding, side. Fig A = scaffold-length violin + whole-motif
centre histogram (uidx designable vs not, contig-scale reference); Fig B = for disjoint motifs, where
each fragment lands by size (lonely / short / long). Compared against the contig-scale run (closest
padding regime).

### 6.6 Unindexed native vs +MPNN rescue (§14 / §14B) — *this session*
For the unindexed track, native co-design gives **1** sequence/design, the MPNN track gives **8**.
Per-sequence pass rate is the *same* (~23.6 %), but 8 shots nearly double the design-level rate and
pull borderline motifs over the line. See §7 Results.

### 6.7 Contig-scaling rescue (run `lib_interface_pad5_contigscale`)
`motif_library/scripts/{find_failed_all.py, scale_contigs.py, stage_contigscale.py}`. Tests whether
**native-geometry contigs** rescue the motifs that failed every baseline method.
- `find_failed_all.py` → **68 tasks** failed all 6 baseline methods (her2 40, tnfa 8, il1b 7, pdl1 7,
  il13 3, il2 3).
- Two regimes: **local-disjoint** (~56, native gaps ≤ 30, fragments close in 3D) vs **domain-spanning**
  (10 motifs, native gap 90–120 → whole-domain "linkers", buries the motif). **Dropped the 10**
  (`--drop-gap-ge 90`) → **56 tasks × 2 modes = 112 contigs**.
- Linker policy = MotifBench-style native-gap **range**, no cap: `[max(n_min, g−W), g+W]`, W=5, floored
  at 3D reach `n_min=ceil(D/3.8)`. Two modes: `native_asym` (old flanks 2-10/35-75 + native linkers),
  `native_sym` (symmetric 0-60/0-60 + native linkers; burial reported not capped).
- `manifest.csv` is the shared contract: rfdiff + genie3 consume it with **no code edit**;
  proteina/laproteina/protpardelle got a small `--manifest` consume-mode (use the row's contig
  verbatim, skip `contig_string`/`length_bounds`, write provenance to their own `<method>/manifest.csv`).
  nsamples = 50.

---

## 7. Results

> Numbers are 3-gate unless noted. "designs" = per-backbone (≥1 of 8 seqs passes); "preds" =
> per-sequence.

### 7.1 Baseline `lib_interface_pad5` — designs (2-gate → 3-gate)
| method | 2-gate designs | **3-gate designs** | kept |
|---|---|---|---|
| genie3 | 10,997 | **6,805** | 61.9 % |
| laproteina_mpnn (idx) | 10,881 | **6,273** | 57.7 % |
| rfdiff | 10,261 | **5,141** | 50.1 % |
| pp1c_bb | 10,946 | **5,060** | 46.2 % |
| proteina | 3,428 | **3,428** | 100 %* |
| **total** | 46,513 | **26,707 designs / 94,897 preds** | |

\*Proteina was refolded last with the 3-gate `5_score.py`, so its `designable` column already includes
the confidence gate (min rs = 0.750). The other four predate that and carry the 2-gate set.
Under 2 gates the four backbone methods are within ~1.5 % — the confidence gate is the separator.

### 7.2 All-atom cross-check (baseline passers)
bb3 ≤ 1 = 100 % (by construction). bb4 ≤ 1 = 84.6–89.8 %. **heavy ≤ 2** (Genie3's stricter all-atom
bar): rfdiff 85.0 % (med 1.52) · proteina 77.0 % (1.65, weakest — poly-Ala scaffold) · genie3 87.4 %
(1.44) · pp1c_bb 84.4 % (1.47) · **laproteina_mpnn 89.1 % (1.38, strongest)**. By motif length,
heavy ≤ 2 = 84 % (4–5 res) → 98 % (13+ res) — short motifs are **not** inflating the gate.

### 7.3 La-Proteina indexed vs unindexed
The early "unindexed is worse" observation was **entirely a scoring artifact** (nominal contig indices
vs LD6's structure-placed motif). After **greedy relocalization**, per-design 3-gate designability:
idx control **10.9 %** (no-op, 10.95→10.9) vs **uidx 23.64 %** (was 2.9 % nominal; median motifRMSD
3.86→1.04). Only the motif ≤ 1 gate changed (sc ≤ 2 and rank ≥ 0.75 byte-identical), confirming the
fix is isolated to the motif-position metric. **Conclusion flips sign: unindexed is more designable
than indexed.**

### 7.4 Unindexed native vs +MPNN rescue (this session, run `lib_interface_pad5_uidx`)
| level | native uidx (1 seq) | uidx + MPNN (8 seqs) |
|---|---|---|
| per-sequence pass | 23.6 % | 23.6 % |
| per-**design** (≥1 seq) | 23.6 % | **40.6 %** |
| per-**motif** (≥1 design) | 88.5 % (278/314) | **90.5 % (284/314)** |

**MPNN rescues net +6 motifs** (9 rescued, 3 marginal losses). The lever is 8× sequence sampling, not
better individual sequences (per-seq rate unchanged).
- **Rescued (designable only after MPNN):** her2_m243_base, her2_m568_base, her2_m580_base,
  her2_m582_base, her2_m595_filled, il13_m21_base, il13_m25_base, il2_m68_base, tnfa_m110_filled.
- **"Lost" (all marginal, not systematic):** her2_m310_base (batch-199 artifact — 7 designs dropped,
  native rank 0.760); her2_m268_base (native rank 0.790); il13_m20_filled (native rank 0.770).

**Sequence-count audit (this session):** `laproteina_mpnn/af3_refold_scores.csv` = 249,804 rows =
31,225 designs × 8 seqs + 1 truncated design × 4. Exactly 8/design everywhere else. **62 designs
missing** vs native (batch-199 quarantine cap), spread over 10 contiguous her2_m30x/m31x tasks, ≤ 7 %
of any task's designs → **no motif dropped**, comparison unbiased. Healthy 23.6 % seq-pass confirms
scoring used the correct greedy `mpnn/fixed_positions.jsonl` (wrong 1_filter positions would collapse
to ~5 %).

### 7.5 Predictors of designability
- **Disjointedness dominates** (held-out RFdiff pass-rate on the library axes): contiguous 0.99 vs
  disjointed 0.46; SSE-stable 0.66 vs loose 0.56; short 0.63 vs long 0.51. Contiguous motifs are
  near-ceiling for RFdiffusion — **discrimination between methods lives in the disjointed classes.**
- Fold space is hub-dominated; motif identity is the strongest fold predictor on refolds; selection
  (confidence gate + one-per-cluster sampling) **inflates helix** at ~72-res scaffold length.

### 7.6 Contig-scaling
68 failed-all → 56 local-disjoint tasks re-scaffolded on all 5 generators × {asym, sym}. (Δ-designability
vs the failing baseline + the motif-exposure metric are the open comparison — see §9.)

---

## 8. Issues, failed jobs, problems & solutions

| # | Problem | Root cause | Solution / status |
|---|---|---|---|
| 1 | Proteina/La-Proteina designability **under-counted** (baseline + contigscale) | Proteina PDBs are **0-based**; `1_filter.localize` returned raw resids as fixed positions but MPNN/stamp/score treat them as 1-based ranks → MPNN fixed the residue *before* the motif → motif designed away | `1_filter.localize` returns `p+1`; `mpnn_prep._stamp` stamps by rank not resid. No-op for 1-based methods. **Re-run proteina both runs** filter→prep→(rm design_list)→mpnn→af3→score |
| 2 | Unindexed La-Proteina motifRMSD **inflated** (uidx successes under-counted ~8×) | Scorer used nominal contig indices; LD6 places the motif *structure* anywhere, ignoring indices/order | Greedy relocalization (`unindexed_score.py` → `positions_unindexed.jsonl`; MPNN track → `mpnn/fixed_positions.jsonl`) + `5_score --positions` override. idx control = perfect no-op (proves matcher recovers truth). uidx 2.9→23.6 % |
| 3 | Whole `1_filter` job **killed** by one bad design | `kabsch_rmsd` raised `SVD did not converge` on a pp1c_bb NaN/collinear backbone; exception escaped the mp.Pool worker; output written only at end | `kabsch_rmsd` returns inf on non-finite; `_gate_one` try/except; broken design → no_placement → filtered out. Shared file protects every method |
| 4 | AF3 array died `OSError 122 Disk quota exceeded`, nothing packed | Euler personal scratch has a **~1,000,000-file** quota; ~1M tiny JSONs blew it | Build+fold entirely in node-local `$TMPDIR`; only per-batch `batch_NNNN.tar.zst` on `/cluster/work` (~4k files total); `--time=08:00:00` |
| 5 | Genie3 **crashed** on generation | `genie3_from_library.py` wrote `REMARK 999 INPUT` with spaces; Genie3 `parse_motif_pdb` reads **fixed 4-wide columns** (`line[19:23]`, `[23:27]`) → `113-115` fragment mis-parsed | Write `f"REMARK 999 INPUT  {chain}{a:>4d}{b:>4d}"` (round-trips incl. 113-115 & single-res). Re-run `genie3_from_library.py` + resubmit |
| 6 | Genie3 scRMSD silently **all-NaN** (first run 0/160) | Genie3 PDBs leave occupancy/B-factor blank; biotite `float('')`→ValueError→design=None | `load_design_lenient()` patches blank occ/bf to 1.00/0.00 in-memory before parse |
| 7 | `2_mpnn` finishes clean with **no new seqs** after growing the design set | `2_mpnn.sbatch` caches `mpnn/design_list.txt`, rebuilds only if absent | `rm -f <m>/mpnn/design_list.txt` before re-running whenever prepped grew. For an all-scaffolds redo also `rm -rf <m>/{refold_inputs,af3/archives,af3_refold_scores.csv}` (batch skip is invalid across a re-batch) |
| 8 | 5_score gives **all-NaN** for co-design methods | Two disjoint namespaces: refold uses `<task>_<sample>` (matches top-level positions/scaffolds) but `1_filter/` uses `…_job_0_id_…` | `CODESIGN_METHODS={laproteina,pp1c_aa}` branch reads top-level `positions.jsonl` + `scaffolds/` |
| 9 | Proteina gen-only crash: `NameError df` then `FileNotFoundError 1n8z_motif_info.csv` | `df.to_csv` unconditional; `motif_task_name.split('_')[0]` assumed bare PDB id | Two source patches (guard + full task_name). 4 total patches must persist if re-cloned |
| 10 | 2mhs descriptors silently corrupted | extra unnamed pandas index column + positional `iloc[0]` returns row index | select **by column name**, raise if absent |
| 11 | SSE = P-SEA not DSSP, over-called coil | biotite `annotate_sse` under DSSP-named columns | vendored numpy Kabsch–Sander DSSP (`esp/_dssp.py`), verified 100 % vs mkdssp |
| 12 | Full-stretch 1n8z HER2 epitope refold **failed for BOTH RFdiff & Genie3** (0 designable, motifRMSD med ~8-9 Å) | genuine difficulty of the cys-rich loopy 41-res epitope; not a scoring bug (pos-control scored fine) | Advised **against** MSA-refold (breaks self-consistency); right use of context = antibody co-fold (functional test). On hold |
| 13 | Genie3 pre-MPNN motifRMSD ≈ 0 for ALL designs | Genie3 is conditioned on the motif → copies it in → non-discriminating pre-MPNN | Real fidelity test = post-MPNN AF3 refold; clash is the discriminating pre-filter for antibody work |
| 14 | Cross-check `laproteina` (idx) all-NaN | two positions files, different key naming; resolver preferred `1_filter/positions.jsonl` (raw gen keys) but matcher needs method-level keys | re-run with `--positions positions.jsonl` |
| 15 | batch 199 (uidx MPNN) never packed | pathological all-NaN batch hit the 30-retry quarantine cap | dropped (99.8 % complete); recovery = bump `seq 1 30`→`60`, resubmit `--array=199`. Cost = 62 designs, no motif lost |
| 16 | JupyterLab **clobbers** mount-side notebook edits | autosave writes stale in-memory state over on-disk edits | Always **Reload Notebook from Disk** before opening; keep a scratchpad backup before every injection; deploy via atomic `cp .tmp && json.load verify && mv` |
| 17 | Foldseek smoke gave bogus counts / faked a length shift | archives written in **task order**; `--limit N` head → 2–3 motifs; batch_0001 has 0 designable | `--limit` samples **evenly-spaced** archives |
| 18 | Design names not parseable / **collide across methods** | every generator names designs differently; genie3 & proteina both use `{task}_{N}` (1,810 collisions) | never parse the name — use `1_filter/tasks.json` + longest-prefix manifest match; always key on `method + "__" + design_name` |
| 19 | Mount instability | sshfs drops (`Device not configured`); `find .` hangs on a dead mount; macOS has **no `timeout`** (made `ls && echo OK || echo DOWN` falsely print DOWN) | test with plain `ls`; `-prune` any `find`; remount `diskutil unmount force eulermount; ./mount_euler.sh`; targeted `ls`, not broad `find`, over the mount |

---

## 8B. Success predictor & method recommender (decision layer, DONE + EXTERNALLY VALIDATED 2026-09-01)

> **Full write-up: [`docs/SUCCESS_PREDICTOR.md`](SUCCESS_PREDICTOR.md)** — model spec, fitted
> coefficients, ROI tables, CLI usage, integration notes, and the 2mhs validation.
> Fitted model + result CSVs: `docs/predictor_results/`.

Turns the benchmark into a data-driven rulebook. `notebooks/success_predictor.ipynb` (self-contained,
gen_env kernel) + `scripts/predict_motif.py` (stdlib CLI). Fit on the **baseline run** (5 methods) +
`laproteina_unindexed`; unit = per-(motif, method); outcome = 3-gate design-level designability as a
**binomial logistic regression** (the simplest model that explains the data — a plain logistic was not
beaten by fancier combos for design-success, [Garcia et al. 2026](https://onlinelibrary.wiley.com/doi/10.1002/pro.70453)).

Pipeline in the notebook (25 cells): assemble per-(motif,method) table → **incremental ladder M0…M4**
(McFadden pseudo-R² / AIC / LRT) → coefficient table (odds ratios) → **§2.5 predictive capability**
(grouped-5-fold CV **AUC** bar + pooled **calibration** curve + **predicted-vs-observed regression**
scatter per method with R²/ρ + **method-comparison matrices**: coefficient heatmap and
method×disjointedness success heatmap) → **per-feature nomogram-style log-odds contribution** → **ROI**
= GPU-h per designable design (`gpuh_per_design / p̂`) → gap-fill what-if + **depth-3 guideline decision
tree** (best-ROI method from motif features) → nearest-analog kNN → exports
`motif_library/analysis/success_model.json` → **§8 external application + Lisa-2mhs validation**.

**EXTERNAL VALIDATION — 2mhs / Mcl-1, held out, and it works.** The pad5 pool contains only 6 antigens
(her2, il1b, pdl1, tnfa, il13, il2) — **Mcl-1 is not among them**, so 2mhs is a genuine external test,
and Lisa ran ~1000 RFdiffusion trajectories per motif, giving a high-N ground truth.
Descriptors already exist at `motif_library/mcl1/mcl1_c_motif.csv` (104 motifs, `source_pdb = 2mhs.pdb`,
numbering matches Lisa's tables) — **no extraction re-run needed**. Ground truth = our own AF3 refold
`analysis/reports/af3_refold_2mhs_c_motif.csv`, aggregated under the *identical* 3-gate rule.
**94 motifs joined, median 999 designs each, 74,307 designs total:**

| metric | value |
|---|---|
| Spearman ρ (p̂ vs observed rate) | **0.652** (p = 1.1e-12) |
| Pearson r | 0.627 |
| AUC (motif has ≥1 designable) | 0.840 — but base rate 0.94, near-saturated, so ρ is the meaningful number |
| observed 3-gate rate / mean p̂ | 0.528 / 0.389 |

Predicted tertiles are monotonic (p̂ .137/.434/.596 → obs .176/.494/.663), and **the headline driver
reproduces almost exactly on unseen data**: 1 fragment p̂ .469 / obs .535 · 2 frag .133/.157 ·
3 frag .016/.018. The model **under-predicts by a near-constant 0.05–0.07** (2mhs motifs are more
contiguous — median 1 fragment vs 2 — and came from the older unpadded, non-interface-restricted
extraction, so they are genuinely easier) → use it as a **ranking/triage tool**, treat absolute p̂ as a
lower bound. Only the rfdiff arm is validatable, since that is the only method run on 2mhs.
⚠ `af3_refold_2mhs_c_motif.csv` is 243 MB and streams at ~0.25 MB/s over sshfs — the aggregation took
17 min; run it cluster-side or copy the file once.

**Applying to any other unseen set:** the model needs the four *descriptors* (motif_length,
disjointed_parts, dominant_sse, mean_rsa), **not** the raw motif-definition CSV. Generate them with the
same extractor — `python -m esp.run extract-motifs --config library/configs/<target>.yaml --out <LIB>
--archive` → `<target>_c_motif.csv` with descriptors (motif_num = surface residue) — then
`predict_motif.py --batch <target>_c_motif.csv` or notebook §8.

**Findings (validated on the real CSVs):**
- **Disjointedness dominates** — M1 (disjointed_parts alone) ≈ full-model pseudo-R² (0.12–0.16) for
  every method; length/interaction add ~nothing; SSE+RSA adds a little. `disjointed_parts` coef
  −1.3…−1.6 log-odds.
- **La-Proteina unindexed is the exception** — disjointedness coef only −0.32, pseudo-R² 0.02–0.05: it
  is largely *insensitive* to disjointedness (p̂ 0.19 on disjoint motifs vs Protpardelle 0.06).
- **CV AUC 0.811–0.866** (grouped 5-fold by antigen, motif-level ≥1 designable) for the five
  index-conditioned arms; uidx 0.614 — not a defect, there is simply little motif-feature signal left to
  predict because it succeeds fairly uniformly.
- **The shared AF3 refold dominates cost** (0.0444 GPU-h/design = 84–99% of the total; generation is
  0.0005–0.0085). **So the ROI ranking is essentially the accuracy ranking** — picking a "cheap"
  generator buys almost nothing. Real savings come from cutting MPNN sequences per design, not from the
  generator choice.
- **ROI rulebook (the split is on disjointedness):** *disjoint (>1 fragment)* → **La-Proteina unindexed,
  unambiguously** (0.41 GPU-h/success vs 4.0–14.3 for everything else, ~10×). *Contiguous* →
  genie3 0.17 / pp1c_bb 0.18 / uidx 0.20 / rfdiff 0.20 are **within ~15%, i.e. effectively tied** —
  choose on wall-clock or diversity, not ROI; avoid indexed La-Proteina and Proteina. ⚠ genie3's cost row
  covers only 60/314 tasks, so its rank-1 contiguous position is unverified.
- **Filling gaps is the largest single lever:** 3 fragments → 1 moves the best option from uidx p̂ 0.113
  (0.41 GPU-h/success) to genie3/pp1c_bb p̂ 0.32/0.24 (0.17–0.18) — **2.8× more likely, 2.4× cheaper**,
  and it changes which method to use.

CLI: `predict_motif.py --task her2_m128_base --pool <pool.csv>` or
`--length 9 --disjoint 3 --sse C --rsa 0.35` → per-method p̂, ROI, contribution breakdown,
recommendation, fill-gaps what-if. Caveats: `compute_costs.csv` has generation GPU-h only (flat
`AF3_SEC=20` refold constant, tunable; genie3 costed on 60 tasks only).

## 9. Open items / next steps

1. **Contig-scaling comparison** (run `lib_interface_pad5_contigscale`, already generated + scored):
   add the motif surface-**exposure** metric and compute Δ-designability vs the failing baseline —
   headline = how many of the 56 failed-all motifs are rescued, by which mode (asym/sym), on which
   method, and whether native linkers specifically help the disjointed ones.
2. **Indexed `laproteina_mpnn` RSA** (`lib_interface_pad5/laproteina_mpnn/motif_rsa.csv`) is the only
   missing RSA file — run `sbatch --export=ALL,RUN_DIR=$RUN,METHOD=laproteina_mpnn motif_rsa.sbatch`
   for the full 8-method exposure figure (§17).
3. **Re-run Proteina** through filter→…→score in both runs to lift the off-by-one under-count (#1).
4. **AA-composition analysis** (motif = fixed native vs scaffold = MPNN-designed) needs `STAGE=sse`
   re-run for `DATASET=lib` so `pred_sse.tsv` gains its `seq` column.
5. Move Part-E/F notebook rebuild helpers out of the session scratchpad into the repo if they'll be
   needed again (they are otherwise lost).

---

## 9B. Document map — what to use when writing up the benchmark

Written material that already exists, and what each is good for in the report:

| doc | scope | use it for |
|---|---|---|
| **`docs/MOTIF_SCAFFOLDING_BENCHMARK_HANDOFF.md`** (this file) | the whole benchmark | **the backbone of the write-up** — library construction, the 5 generators' setup/checkpoints/params, shared downstream, 3-gate definition, every analysis, results, the 19-row issues table |
| **`docs/SUCCESS_PREDICTOR.md`** | the decision layer | the "so what" chapter — model, coefficients, ROI rulebook, external validation on 2mhs |
| `docs/predictor_results/` | fitted model + result CSVs | numbers and figures for the predictor chapter; fully reproducible from the scripts inside |
| `epitope_scaffolding/pipeline/README.md` (133 l) | the `esp` pipeline | **Methods**: how motifs are extracted and how the RFdiffusion→MPNN→AF3 stages are configured and costed. This is the tooling chapter. |
| `motif_library_chimerax/full_deduped/README.md` (80 l) | library coverage surface maps | **Figure source**: colouring each of the 7 targets by how many library motifs cover each residue — the natural "what does the library actually sample" figure. Also records the dedup numbers (734 = 628 base + 106 filled, Jaccard ≥ 0.7, from 926 raw). |
| `lisa_pdbs/coverage_chimerax/README_success_maps.md` (69 l) | per-residue success maps | **Figure source**: painting a target surface with per-residue scaffolding success — the "where on the interface is designable" figure. Env-driven, works for any protein. |
| `slurm_reference.md` (99 l) | Euler/SLURM primer | appendix / reproducibility notes only; not report content |
| `CLAUDE.md` | repo orientation | pipeline stage table and data locations; useful as a checklist that the Methods section is complete |
| `holisticdb/README.md` | separate lab DB project | **not** part of this benchmark — exclude |

Two cluster-side docs were not re-read this session (the mount dropped) but are worth checking before
writing: `motif_library/PILOT_RUNBOOK.md` (the pilot protocol that preceded the full run) and
`motif_library/hardest_motifs_coverage.md` (the all-failure motif set that motivated contig-scaling).
There is also `bekessel/docs/binding_validation_report.md` — that belongs to the *binding-validation*
track, not the scaffolding benchmark; cite it only where the two meet.

**Suggested report skeleton from these:** Introduction/RQs (§1) → Methods: library (§2) + esp pipeline
README + generators (§4) + downstream and the 3-gate definition (§5) → Results: designability (§7) +
coverage/success surface figures (the two ChimeraX READMEs) → Decision layer (`SUCCESS_PREDICTOR.md`)
→ Limitations (§8 issues table + the cost caveats) → Outlook (§9).

---

## 10. Where everything lives

```
/cluster/work/wicky/bekessel/                        (= eulermount/bekessel/ locally)
├── gen_env/                       venv for all host-side scripts (source .../bin/activate)
├── epitope_scaffolding/pipeline/  esp package (motif.py, catalog.py, contig.py, _dssp.py,
│   ├── esp/                         af3_inputs.py, score.py, rmsd.py) — PYTHONPATH for esp.*
│   ├── library/configs/<tag>.yaml   the 7 extraction configs
│   └── scripts/stage_from_pool.py   shared stager
├── motif_library/
│   ├── motif_pool_dedup_j0.7.csv   734-motif library (+ motif_pool_interface_j0.7_pad5.csv)
│   ├── <protein>/motif_templates/  tar.zst archives (original_motif_NNN.pdb)
│   ├── scripts/{find_failed_all,scale_contigs,stage_contigscale}.py
│   ├── runs/lib_interface_pad5[/_uidx /_contigscale]/   the benchmark runs
│   │   ├── motifs/  manifest.csv  run.yaml
│   │   └── <method>/{configs,scaffolds,1_filter,mpnn,refold_inputs,af3,
│   │                 af3_refold_scores.csv, motif_rsa.csv, motif_placement.csv,
│   │                 motif_atom_crosscheck.csv}
│   └── foldseek/{pred_sse.tsv, clusters_tm050.tsv, designable_cifs.tar.zst}
├── software/
│   ├── rfdiff/rfdiff_generate.sbatch
│   ├── proteina/{scripts/*, checkpoints/, data/, RUN_ON_EULER.md}   (+ proteina.def fallback)
│   ├── genie3/{scripts/*, genie3.def, pretrained/, RUN_ON_EULER.md}
│   ├── protpardelle-1c/{scripts/*, protpardelle.def, model_params/, RUN_ON_EULER.md}
│   ├── la-proteina/{scripts/*, la-proteina.def, checkpoints_laproteina/}
│   ├── downstream/{1_filter.py, resolve_positions.py, mpnn_prep.py, 2_mpnn.sbatch,
│   │               3_af3_build.py, 4_af3.sbatch, 5_score.py, 5_score.sbatch}
│   └── esmfold2/                    generic ESMFold2 refold runner (secondary predictor)
├── scripts/{motif_atom_crosscheck.py, motif_rsa.py+.sbatch, motif_placement.py,
│            extract_structures.py, dssp_numpy.py, cif_backbone.py, designable_sse_extract.py,
│            predict_motif.py}
├── motif_library/analysis/success_model.json   (exported by success_predictor.ipynb)
├── notebooks/{motif_benchmark.ipynb, foldseek_cluster_analysis.ipynb, success_predictor.ipynb}
└── analysis/reports/{af3_refold_<pdb>_c_motif.csv, foldseek_tm050/}   Lisa's 3-protein set

containers/  RFdiffusion.sif · proteina.sif · openfold.sif (MPNN) · AF3 · foldseek.sif
params/RFParams/Base_ckpt.pt
```

Local mirror of notebooks/scripts under `~/Desktop_HD/SemesterProject/`; papers in `docs/`
(La-Proteina, Protpardelle-1c 2503.00710, the designable-library preprint 2026.02.16.706194,
2412.05788). Cluster-side repo-plan doc: `eulermount/bekessel/docs/MOTIF_BENCHMARK_REPO_PLAN.md`.

---

## 11. Operational constraints (must persist)

- **Never SSH to Euler.** Files are staged over the sshfs mount; the user runs every
  `sbatch`/generation on the login node; results are pulled over the mount.
- **`gen_env` is a venv, not conda.** Activate `source /cluster/work/wicky/bekessel/gen_env/bin/
  activate`, or call `/cluster/work/wicky/bekessel/gen_env/bin/python` directly (needed under
  `sbatch --wrap`, which runs `/bin/sh` where `source` is undefined). esp imports need
  `PYTHONPATH=/cluster/work/wicky/bekessel/epitope_scaffolding/pipeline`.
- **Paths live in config YAML**, not Python flags (flag hardcoded WICKY roots instead of
  parametrizing).
- **State the full path of every file created.**
- **Only write in `bekessel/`** (rest of the mount is read-only). Don't pile up subset CSVs — write
  supersets only, on approval; view filtered subsets in-notebook.
- **JupyterLab autosave clobbers mount edits** — Reload from Disk before running; back up before
  injecting cells; sshfs write-cache is unreliable for actively-run files, so give exact cluster-side
  edits.
- **Mount drops frequently** — test with plain `ls`, `-prune` any `find`, remount with
  `diskutil unmount force eulermount; ./mount_euler.sh`.
```
