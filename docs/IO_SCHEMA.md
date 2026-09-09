# I/O schema — what every stage and every model consumes and produces

Derived from the code and from real run artefacts, not from prose. Every column list below was read
off an actual file. `<run>` is a run directory, `<task>` is `{pdb_name}_m{motif_num}_{variant}`
(e.g. `her2_m568_base`) — the join key everywhere.

---

## 0. The contract in one picture

```
 target.pdb + target.yaml
        │  extract_motifs.py
        ▼
 <target>_sasa_resis.csv   <target>_c_motif.csv   motif_pdbs/*.tar.zst
        │  build_pool.py  (P5 — not yet written)
        ▼
 motifs/pool.csv ──────────┐
        │  stage_motif.py  │
        ▼                  │
 <run>/manifest.csv        │      ← THE contract. Written by the stager alone.
 <run>/motifs/<task>.pdb   │      ← one staged template per task
        │                  │
        │  ┌───────────────┘  (drivers may read the pool directly instead)
        ▼  ▼
 <run>/<method>/…  generation (per-method layout, §3)
        ▼
 1_filter → mpnn_prep → 2_mpnn → 3_af3_build → 4_af3 → 5_score   (identical for every method)
        ▼
 <run>/<method>/af3_refold_scores.csv
        │  aggregate_outcomes.py → fit_predictor.py
        ▼
 models/success_model.json  →  predict_motif.py
```

Three artefacts are the contract; everything else is derived:

| artefact | written by | read by |
|---|---|---|
| `<run>/motifs/<task>.pdb` | `stage_motif.py` | every generator · `mpnn_prep` · `1_filter` · `5_score` |
| `<run>/manifest.csv` | `stage_motif.py` **only** | `rfdiff_generate.sbatch` · `genie3_from_library.py` · `1_filter` |
| `<run>/<method>/af3_refold_scores.csv` | `5_score.py` | the analysis notebook · `aggregate_outcomes.py` |

---

## 1. Stage 0 — motif definition

### `scripts/motif/extract_motifs.py`

**In**

| what | form | notes |
|---|---|---|
| structure | `.pdb` / `.cif` | first model only; the antigen chain; a trailing 6×HIS tag is stripped |
| `target.yaml` | flat YAML | supplies every parameter below |

| parameter | default | meaning |
|---|---|---|
| `chain` | — | the antigen chain; motifs come from this chain only |
| `rsa_cutoff` | `0.20` | surface iff RSA > this; every surface residue becomes one motif centroid |
| `neighborhood_cutoff` | `7.0` | motif = centroid + residues within this many Å |
| `neighborhood_metric` | `ca` | `ca` or `cb` (Cβ, with Cα for Gly) |
| `gap_merge` | `4` | filled variant: merge fragments separated by ≤ this many residues |
| `emit_filled` | `true` | also emit the matched `filled` twin |

**Out**

`<target>_sasa_resis.csv` — one row per residue in the chain:

| column | type | meaning |
|---|---|---|
| `res_id` | int | author residue number |
| `res_name` / `aa` | str | three-letter / one-letter (`X` = non-standard) |
| `sasa` | float | solvent-accessible area, Å² |
| `rsa` | float | `sasa` ÷ Tien max ASA; `NaN` for non-standard |
| `surface` | 0/1 | `rsa > rsa_cutoff` — **this flag defines the centroid menu** |

`<target>_c_motif.csv` — one row per (centroid × variant):

| column | type | meaning |
|---|---|---|
| `pdb_name` | str | target tag |
| `motif_num` | int | **the centroid residue id** — `her2_m568` is built around residue 568 |
| `variant` | `base`/`filled` | `filled` = gaps ≤ `gap_merge` merged |
| `gap_merge` | int | populated on `filled` rows only |
| `source_pdb`, `chain` | str | provenance |
| `motif_residues` | space-separated ints | the motif |
| `motif_length` | int | residue count |
| `disjointed_parts` | int | **fragment count — the headline difficulty driver** |
| `dssp` | str | per-residue H/E/C, aligned to `motif_residues` |
| `dominant_sse` | H/E/C | most common class |
| `C_frac`,`E_frac`,`H_frac` | float | SSE composition |
| `rg` | float | radius of gyration of the Cα cloud, Å |
| `linearity`,`planarity` | float | from the coordinate covariance eigenvalues |
| `mean_cx` | float | convexity (Pintar); high = protruding |
| `mean_rsa` | float | mean exposure |

`motif_pdbs/c_motif{,_filled}.tar.zst` — one all-atom `original_motif_<motif_num>.pdb` per motif,
**original numbering and chain id** (contigs index against these).

> The four columns the predictor consumes are `motif_length`, `disjointed_parts`, `dominant_sse`,
> `mean_rsa`. A raw motif-definition CSV without descriptors cannot be scored.

### `scripts/motif/stage_motif.py`

**In** `pool.csv` (or any CSV with the `<target>_c_motif.csv` columns) + the motif template archives.
**Parameters** `--variants base,filled` · `--nsamples 100` · `--flank-n 2-10` · `--flank-c 35-75` ·
`--linker 1-5`.

**Out** `<run>/motifs/<task>.pdb`, and `<run>/manifest.csv`:

| column | type | meaning |
|---|---|---|
| `config_number` | int | **== the SLURM array index == the same motif in every method** |
| `task_name` | str | `{pdb_name}_m{motif_num}_{variant}` |
| `pdb_name`,`motif_num`,`variant`,`chain` | | provenance |
| `n_fragments` | int | number of motif fragments |
| `fragments` | `a-b;c-d` | fragment spans — **only the stager writes this column** |
| `contig` | str | e.g. `2-10/A148-149/1-5/A151-151/35-75` — RFdiffusion grammar |
| `motif_min_length`,`motif_max_length` | int | total scaffold-length bounds implied by the contig |
| `nsamples` | int | designs requested per task |
| `motif_pdb` | path | absolute path to the staged template |

---

## 2. Stages 2–5 — evaluation (identical for every method)

| stage | script | reads | writes |
|---|---|---|---|
| 2 | `1_filter.py` | `<method>/` designs, `manifest.csv`, `motifs/<task>.pdb` | `1_filter/motif_rmsd.csv`, `passed/`, `passed.txt`, `positions.jsonl`, `tasks.json` |
| 2.5 | `mpnn_prep.py` | `1_filter/{passed,positions.jsonl,tasks.json}` + templates | `mpnn/prepped/<design>.pdb`, `mpnn/fixed_positions.jsonl` |
| 3 | `2_mpnn.sbatch` | `mpnn/prepped/`, `mpnn/fixed_positions.jsonl` | `mpnn/seqs/<design>.fa` |
| 4 | `3_af3_build.py` | `mpnn/seqs/` (backbone) or `seqs/` (co-design) | `refold_inputs/b####.tar` |
| 5 | `4_af3.sbatch` | `refold_inputs/b####.tar` | `af3/archives/batch_####.tar.zst` |
| 6 | `5_score.py` | `af3/archives/`, `positions.jsonl`, `motifs/<task>.pdb` | `af3_refold_scores.csv` |

**`1_filter/motif_rmsd.csv`** — one row per generated design, **including failures** (this is the
correct denominator for a designability rate):

| column | meaning |
|---|---|
| `design` | design name (method-specific format — see §3) |
| `task` | resolved task |
| `motifRMSD` | min over contig-consistent placements, Cα, Å |
| `n_motif` | motif residues matched (**0 means the row is unusable**) |
| `pass` | `motifRMSD < CUTOFF` (default 1.0) |
| `note` | failure reason when present |

**`positions.jsonl`** — `{design: {"A": [1-based ranks]}}`. Design *positions*, not residue ids:
AF3 renumbers 1..L, so the motif can only be located by rank.

**`af3_refold_scores.csv`** — one row per refolded sequence:

| column | meaning |
|---|---|
| `prediction` | `<design>_seq###` |
| `task_name` | resolved task — emitted so nothing downstream re-derives it by regex |
| `design_name` | the backbone |
| `motifRMSD` | backbone N,CA,C at design positions vs native, Å |
| `scRMSD` | Cα, generated backbone vs its own refold, Å |
| `plddt_mean`,`ptm`,`iptm`,`ranking_score`,`has_clash`,`fraction_disordered` | AF3 confidence |
| `motif_ok`,`sc_ok` | gates 1 and 2 |
| `designable` | **all three gates** in current code; see the caveat below |
| `status` | `success` / `failed` — filter on this before aggregating |

> **Gate vintage.** The baseline run's CSVs (2026-08-18) predate the ranking-score gate and carry a
> **two-gate** `designable`; `5_score.py` has emitted **three** gates since 2026-08-26. Anything
> aggregating outcomes should re-derive from `motifRMSD` / `scRMSD` / `ranking_score` rather than
> trust the column. `aggregate_outcomes.py` does, and records the vintage it saw.

**Designability** — `motifRMSD ≤ 1.0` **and** `scRMSD ≤ 2.0` **and** `ranking_score ≥ 0.75`
(inclusive). Defined once, in `motifgo/score.py`. A *design* is designable iff any of its ≤8
sequences passes; a *motif* is designable iff any of its designs is.

---

## 3. The five generators — per-model I/O

Generation is the only per-method stage. All five take the same motif and the same length policy;
they differ in config format and in where they put things.

| | driver | reads | config written | designs land in | design name |
|---|---|---|---|---|---|
| **RFdiffusion** | *none* | `manifest.csv` row, by `config_number` | — | `<run>/rfdiff/scaffolds/<task>/design_<N>.pdb` (+ `.trb`) | `<task>_design_<N>` |
| **Proteina** | `proteina_from_library.py` | pool **or** `--manifest` | `<run>/proteina/configs/inf_<N>.yaml` | `<run>/proteina/scaffolds/inf_<N>/<task>_<N>.pdb` | `<task>_<N>` |
| **Genie3** | `genie3_from_library.py` | `manifest.csv` | `<run>/genie3/{problems/,motifs/,experiment.yaml}` | `<run>/genie3/output/<task>/pdbs/<task>_<N>.pdb` | `<task>_<N>` |
| **Protpardelle-1c** | `protpardelle_from_library.py` | pool **or** `--manifest` | `<run>/<method>/configs/inf_<N>.yaml` | `<run>/pp1c_bb/scaffolds/inf_<N>/<model-tag>/<task>/sample_<task>_<N>.pdb` | `<task>_sample_<task>_<N>` |
| **La-Proteina** | `la_proteina_from_library.py` | pool **or** `--manifest` | `<repo>/configs/inf_<runtag>_<N>.yaml` (symlink; real file in the run dir) | `<run>/laproteina/inference/…` → normalised to `scaffolds/` + `seqs/` | `<task>_<N>` |

**Shared driver flags:** `--pool | --manifest` · `--run-dir` · `--lib` · `--variants` · `--nsamples` ·
`--flank-n` · `--flank-c` · `--linker`.
**Shared sbatch env:** `RUN_DIR` · `SIF` · `REPO` (model repo) — plus `METHOD` (Protpardelle),
`RUNTAG` (La-Proteina), `CONFIG` (Genie3).

**Model selection.** Protpardelle `--model cc58` → method dir `pp1c_bb` (backbone; MPNN downstream);
`cc91` → `pp1c_aa` (all-atom co-design). La-Proteina `--variant idx|uidx` swaps the hydra base config,
which swaps the checkpoint (`LD4_motif_idx_aa` ↔ `LD6_motif_uidx_aa`) over the same autoencoder and
the same image. Use a separate run dir for uidx.

**Co-design branch.** La-Proteina skips stages 2–3 entirely. `la_proteina_extract_seqs.py` writes
`seqs/<design>.fa`, `scaffolds/<design>.pdb` and `positions.jsonl`, and the shared path resumes at
`3_af3_build`. For **uidx**, `unindexed_score.py` must run first — the model chooses where to place
the motif, so nominal positions are wrong until greedy relocalisation recovers them
(`positions_unindexed.jsonl`, then `5_score.py --positions … --preserve-order`).

> **Design names differ per method and one of them repeats the task** (`pp1c_bb`). Resolve a design
> to its task with the `task_name` column where present, else longest-prefix match against the pool.
> Token counting breaks on any protein tag that is not a single token. Protpardelle in the
> `contigscale` run emits `sample_<task>_<N>`, where the task is **not** a prefix at all — that case
> silently produced a 42,760-row `motif_rsa.csv` with `n_motif=0` in every row.

---

## 4. The predictor

| script | in | out |
|---|---|---|
| `aggregate_outcomes.py` | six `af3_refold_scores.csv` + pool | `outcomes.csv`: `task, method, run, method_dir, n_designs, n_designable, n_seqs, n_seqs_pass, csv_designable_gates` |
| `fit_predictor.py` | outcomes + pool + `compute_costs.csv` | `models/success_model.json` |
| `predict_motif.py` | model + four descriptors | per-method p̂, ROI, per-feature contributions, fill-gaps what-if |
| `validate_predictor.py` | model + a held-out protein's observed rates | ρ, tertiles, per-fragment breakdown |

**`success_model.json`** — the whole contract, consumable by anything that can do a dot product:

```jsonc
{
  "schema_version": 2,
  "cont_features":  ["disjointed_parts", "motif_length", "mean_rsa"],
  "final_features": ["disjointed_parts_z","motif_length_z","len_x_disj","H","E","mean_rsa_z"],
  "gates":  {"motif_rmsd_max": 1.0, "sc_rmsd_max": 2.0, "rank_min": 0.75},
  "mu": {...}, "sd": {...},              // standardisation, from the FITTING pool
  "refold_gpuh_per_fold": 0.00567,       // shared AF3 cost, per fold
  "methods": {
    "<method>": {
      "label": "...", "coef": {"const": ..., "<feature>": ...},
      "gen_gpuh_per_design": ...,        // generation only — the sole term that differs
      "seqs_per_design": 8.0,            // 8 backbone, 1 co-design -> refold cost scales with this
      "cost_tasks_covered": 314, "cost_tasks_total": 314,
      "design_rate": ..., "pseudo_r2_mcfadden": ...
    }
  }
}
```

`p̂ = sigmoid(const + Σ coefⱼ · xⱼ)`. Each `coefⱼ · xⱼ` is that feature's log-odds contribution —
for a linear model this *is* the nomogram decomposition, and equals SHAP.
`gpuh_per_success = (gen_gpuh_per_design + refold_gpuh_per_fold × seqs_per_design) / p̂`.

---

## 5. What a web app would need

Every step below is already a callable with a defined I/O contract, so a front end is wiring, not
new science:

| user action | call | needs |
|---|---|---|
| upload a target PDB | `extract_motifs.py --screen` | chain list + SSE composition, seconds, CPU |
| choose a chain, extract | `extract_motifs.py --target …` | ~1 min CPU → the two CSVs + templates |
| rank the motifs | `predict_motif.py --batch <target>_c_motif.csv` | milliseconds, stdlib only, no GPU |
| pick motifs → build a pool | subset of `<target>_c_motif.csv` | (a `build_pool.py` would formalise this) |
| stage a run | `stage_motif.py --pool … --run-dir …` | seconds |
| launch a method | `<method>_from_library.py` then `sbatch` | GPU; poll with `sacct -j <jobid>` |
| evaluate | the six `scripts/evaluate/` stages | GPU for AF3 only |
| show results | `af3_refold_scores.csv` + `1_filter/motif_rmsd.csv` | the rate needs **both** — passers alone overstate it |

Two things a front end must not get wrong: **the denominator** (use `1_filter/motif_rmsd.csv`, which
includes failures) and **the gate vintage** (re-derive the three gates; never trust `designable`).
