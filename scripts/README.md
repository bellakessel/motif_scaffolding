# `scripts/` — what runs at each step, and what it calls

Every executable in the pipeline lives under this directory. **No script here imports another
script.** The only shared code is the `motifgo` package at the repo top level; a script either
imports from `motifgo` or is entirely self-contained. That is the whole dependency story.

```
                     motifgo/          (library — imported, never executed)
                        ▲
                        │ imports
   motif/  ──►  generate/<method>/  ──►  evaluate/  ──►  analysis/
   stage 0          stage 1            stages 2-5        figures
                                            ▲
                                        extras/  (side analyses, contig rescue, costs)
```

## The pipeline, stage by stage

| # | stage | script | what it does | imports from `motifgo` |
|---|---|---|---|---|
| 0 | motif | `motif/motif_extract.py` | protein PDB + `target.yaml` → surface (SASA) → motif residue lists → descriptors → cross-library redundancy check | `motif_define` |
| 0 | motif | `motif/build_pool.py` *(P5)* | dedup + interface filter + padding → `motifs/pool.csv` | — |
| 0 | motif | `motif/stage_motif.py` | `pool.csv` → `<run>/manifest.csv` + staged motif PDBs. **The only manifest writer.** | `contig` |
| 1 | generate | `generate/rfdiff/rfdiff_generate.sbatch` | reads `manifest.csv` directly; no driver | — |
| 1 | generate | `generate/proteina/proteina_from_library.py` + `.sbatch` | per-motif configs → backbones | `contig` |
| 1 | generate | `generate/genie3/genie3_from_library.py` + `.sbatch` | manifest → MotifBench problem set + `experiment.yaml` | — |
| 1 | generate | `generate/protpardelle/protpardelle_from_library.py` + `.sbatch` | per-motif configs → backbones | `contig` |
| 1 | generate | `generate/la-proteina/la_proteina_from_library.py` + `.sbatch` | all-atom **co-design** (sequence + structure) | `contig` |
| 1′ | generate | `generate/la-proteina/la_proteina_extract_seqs.py` | co-design normaliser: writes `seqs/` + `scaffolds/` + `positions.jsonl` so the co-design branch rejoins the shared path at step 3 | — |
| 2 | evaluate | `evaluate/1_filter.py` + `resolve_filter.sbatch` | motif-geometry gate: localise the motif in each design, Kabsch, keep motifRMSD < cutoff | — |
| 2 | evaluate | `evaluate/mpnn_prep.py` + `.sbatch` | stamp native motif identities onto the backbone; emit MPNN fixed positions | — |
| 2 | evaluate | `evaluate/2_mpnn.sbatch` (+ `parse_chains.py`) | ProteinMPNN, 8 sequences per design, motif fixed | — |
| 3 | evaluate | `evaluate/3_af3_build.py` + `.sbatch` | one AF3 monomer-refold JSON per designed sequence | `af3_inputs` |
| 4 | evaluate | `evaluate/4_af3.sbatch` | AF3 inference, one array task per batch → `af3/archives/*.tar.zst` | — |
| 5 | evaluate | `evaluate/5_score.py` + `.sbatch` | designability: motifRMSD ≤ 1 ∧ scRMSD ≤ 2 ∧ ranking_score ≥ 0.75 | `score` |
| 5′ | evaluate | `evaluate/unindexed_score.py` + `.sbatch` | La-Proteina **uidx** only: greedy relocalisation of the motif the model placed itself | — |

**Branch:** co-design (La-Proteina) skips steps 2 entirely — `la_proteina_extract_seqs.py` replaces
`1_filter` → `mpnn_prep` → `2_mpnn`, then the shared path resumes at `3_af3_build`.

## Everything else

| dir | scripts | role |
|---|---|---|
| `extras/` | `motif_rsa.py` (imports `motif_define`), `scale_contigs.py`, `stage_contigscale.py`, `find_failed_all.py`, `compute_costs.py`, `extract_structures.py`, `make_example_bundle.py` | side analyses: motif burial ΔRSA, the contig-rescue sub-pipeline, sacct cost accounting, structure pulls |
| `analysis/` | `motif_benchmark.ipynb`, `figstyle.py`, `designable_sse_extract.py` (imports `dssp`), `cif_backbone.py`, `normalize_backbones.py`, `merge_refold.py`, `foldseek_*.sh` | figures and cross-method fold-space clustering |

## Only 8 scripts touch the library at all

`motif_extract.py` and `motif_rsa.py` → `motif_define` · `stage_motif.py`,
`proteina_from_library.py`, `protpardelle_from_library.py`, `la_proteina_from_library.py` → `contig` ·
`3_af3_build.py` → `af3_inputs` · `5_score.py` → `score` · (`designable_sse_extract.py` → `dssp`).

**Everything else is self-contained NumPy** — `1_filter.py`, `mpnn_prep.py`, `unindexed_score.py`,
`parse_chains.py`, `genie3_from_library.py`, `la_proteina_extract_seqs.py`, and all of
`extras/` except `motif_rsa.py`.

## Direction of every import

```
motif_define  ◄── motif_build ──►  contig,  dssp          af3_inputs      score
     ▲                  ▲            ▲        ▲                ▲            ▲
motif_extract    (stage 0 only)  stage_motif          3_af3_build   5_score
motif_rsa                        *_from_library (3)                     (designable_sse_extract → dssp)
```

Acyclic, one-way, and no module reaches outside the package by absolute path. Before the
consolidation, `score.py` used `sys.path.insert('/cluster/work/wicky/bekessel/scripts')` to reach an
untracked file for the designability thresholds — those now live in `motifgo/score.py` and nowhere else.
