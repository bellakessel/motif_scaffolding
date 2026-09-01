# `motif_benchmark/` — repo plan (v4, execution-ready)

Every v3 `REVIEW` / `NOTE` answered. Your annotated copies are kept as
`MOTIF_BENCHMARK_REPO_PLAN.v2-annotated.md` and `.v3-annotated.md`.
Shorthand `BEK = /cluster/work/wicky/bekessel`.

**Locked:** package **`motifgo`**, at the repo **top level** (§1) · repo at `BEK/motif_benchmark/`,
moving to Wicky GitLab when ready · example at **full depth, on disk, `.gitignore`d, no LFS, no tour** ·
the seven tasks · no `bin/mb` — plain `sbatch` · logs default to scratch, per-stage subdir ·
`motif.py`→**`motif_define.py`**, `catalog.py`→**`motif_build.py`** · `fab_chains` for her2 → `[A,B,C,D]` ·
notebook copied as-is.

**Gap merging (§0.2):** measured — `merge_gaps` alone is sufficient and spatially safe for gaps <= 5;
a spatial guard is only needed above ~6 and is stubbed for your future predictor. `filled_rows` stays.

---

## 0. Answers to your v3 notes

### 0.1 — N1: rename `motif.py` → `motif_define.py`, `catalog.py` → `motif_build.py`

**Yes, both names fit what the code does.** Checked function by function:

| new name | contains | verdict |
|---|---|---|
| `motif_define.py` (was `motif.py`, 240 L) | `surface_rsa`, `contacts_of`, `expand_neighborhood`, `select_residues`, `MAX_ASA`, `_load_chain`, `_rep_coords` | **defines what a motif *is*** — geometry only, writes nothing. Name is exact. |
| `motif_build.py` (was `catalog.py`, 193 L) | `sasa_rows`, `motif_rows`, `filled_rows`, `add_descriptors`, `write_motif_pdbs`, `write_csv` | **builds the artifacts** — every function emits a CSV row or a PDB. Name is exact. |

Dependency stays a clean one-way DAG: `motif_define` imports nothing from the package (leaf);
`motif_build` imports `motif_define`, `contig`, `dssp`; nothing imports `motif_build`.

`contig.py` keeps its name — a contig is a *scaffolding instruction*, not a motif definition, and
renaming it would blur the one boundary that is currently clear.

**Call sites to update (5, all one-liners):** `motif_build.py:20-21`, `motif_rsa.py:47`,
`motif_extract.py:33`, plus the two `from . import` lines. Not carried: `esp/run.py` (campaign only).

### 0.2 — Reducing disjointedness at motif generation: **yes, `merge_gaps` is enough** (measured)

You asked whether the current setup can reduce disjointedness when motifs are generated, using the
cKDTree spatial neighbours, for gaps of 2–3 residues. I measured it on all 163 disjoint base motifs.

**First, the mechanics, precisely.** `merge_gaps` is pure sequence-number arithmetic — the gap between
fragments `(a,b)` and `(c,d)` is `c - b - 1`, no coordinates read:

```python
merged = [runs[0]]
for c, d in runs[1:]:
    pa, pb = merged[-1]
    if c - pb - 1 <= gap_merge:  merged[-1] = (pa, d)   # fill; keep growing this fragment
    else:                        merged.append((c, d))
```

The cKDTree / `distance_matrix` work happens one step **earlier**, deciding which residues are in the
motif at all (`motif_define.contacts_of`, `expand_neighborhood`; `motif_build.motif_rows` = centroid →
Cα contacts ≤ 7 Å). Your "walk the neighbourhood tree" instinct is right about *chaining*, though: each
gap is measured from the **growing** fragment, so consecutive small gaps collapse in one pass —
`[(10,12),(16,18),(22,24)]` at `gap_merge=4` → `[(10,24)]`.

**Second — is sequence-only merging spatially safe?** This was the real question, and the answer is a
clean yes in your regime. For every inter-fragment gap in the pool (5,532 gap residues), minimum
heavy-atom distance from each gap residue to the motif:

| gap size | n residues | median | max | % beyond 7 Å |
|---|---|---|---|---|
| 1 | 20 | 1.33 Å | 1.34 Å | **0 %** |
| 2 | 36 | 1.33 Å | 1.36 Å | **0 %** |
| 3 | 36 | 1.33 Å | 3.79 Å | **0 %** |
| 4 | 20 | 2.04 Å | 4.30 Å | **0 %** |
| 5 | 35 | 3.07 Å | 6.89 Å | **0 %** |
| 6 | 78 | 3.46 Å | 7.73 Å | 5 % |
| 10 | 40 | 5.00 Å | 13.33 Å | 43 % |
| ≥ 16 | — | 9–21 Å | 42 Å | 50–92 % |

The 1.33 Å floor is the peptide C–N bond: a gap residue adjacent to a motif residue is covalently
bonded to it. **Below `gap_merge` ≈ 5 the sequence criterion and the spatial criterion agree
completely** — a short sequence gap cannot loop away in space. Divergence starts at 6 and becomes
serious from 10, where a long excursion leaves the neighbourhood entirely. So for the 2–3 residue
reduction you want, **no spatial gate is needed; `merge_gaps` alone is correct.**

**Third — how much disjointedness can it actually remove?** Of 163 disjoint base motifs:

| `gap_merge` | motifs improved | → fully contiguous | median residues added | max added |
|---|---|---|---|---|
| 1 | 19 (11.7 %) | 7 (4.3 %) | 0 | 2 |
| 2 | 34 (20.9 %) | 14 (8.6 %) | 0 | 4 |
| **3** | **44 (27.0 %)** | **18 (11.0 %)** | **0** | **6** |
| 4 | 49 (30.1 %) | 22 (13.5 %) | 0 | 6 |
| 5 | 56 (34.4 %) | 26 (16.0 %) | 0 | 6 |
| 8 | 82 (50.3 %) | 43 (26.4 %) | 1 | 16 |

At `gap_merge=3`: 18 motifs go 2→1 part, 16 go 3→2, 6 go 4→2, 1 goes 5→4. **94 of the 163 stay
unchanged** — their gap is genuinely long, and those are exactly the population your future predictor
has to rule on (and the only population where a spatial gate would matter).

**Validation:** running `merge_gaps(runs, 4)` over the pool reproduces the shipped `filled` rows
exactly — 49 improved, 22 fully contiguous, same transition matrix. The code and the data agree.

**What goes into the repo.** `gap_merge` becomes a first-class knob at motif generation
(`extract_motifs.py --gap-merge N`, and `gap_merge:` in `target.yaml`), not just a catalog-time
variant. Plus one thin, optional guard for the future predictor — ~15 lines reusing the cKDTree that
already exists:

```python
# motifgo/contig.py
def merge_gaps_spatial(runs, gap_merge, *, gap_coords=None, max_dist=None):
    """merge_gaps, but a merge is rejected if any gap residue lies farther than
    max_dist from the motif. A NO-OP below gap_merge~5 (measured: 0% of gaps exceed 7 A),
    so it costs nothing today and is the hook for the disjointedness predictor."""
```

**README section** (one mechanism, as you asked):

````markdown
### Reducing disjointedness: `gap_merge`

A motif grown around one surface centroid is often **disjoint** — several sequence stretches.
`gap_merge` is the knob that joins them:

```python
merge_gaps([(81,82), (91,91), (126,130), (155,155), (157,157)], gap_merge=4)
# -> [(81,82), (91,91), (126,130), (155,157)]      5 fragments -> 4
```

Two fragments merge when fewer than `gap_merge` residues separate them **in sequence numbering**; the
filled-in residues join the motif. Each gap is measured from the growing fragment, so a run of short
gaps collapses in one pass. This is safe without a distance check because a gap of <=5 residues is
always spatially adjacent to the motif (measured across the library: 0% farther than 7 A; the 1.33 A
floor is the peptide bond). Above ~6 that stops holding, and `merge_gaps_spatial` adds the guard.
The cost is fidelity: filled residues become part of the motif you scaffold, so it is no longer
exactly the epitope you measured.

*In the published benchmark:* `gap_merge=4` was applied at catalog time to emit a second `filled`
row beside each `base` motif -- 49 of the 314 tasks -- so base and filled could be scored head to head.

*Adjacent but different:* the contig **linker** (`1-5` by default) lets the generator invent residues
*between* fragments without adding them to the motif -- disjointedness is bridged, not reduced. That is
what every benchmark run actually scaffolded.
````

### 0.3 — README must not hardcode a personal environment

Agreed. Every README command uses a generic placeholder, and the repo carries no path to `gen_env`:

```bash
source <your-venv>/bin/activate          # e.g. python3 -m venv .venv && source .venv/bin/activate
pip install -e /path/to/motif_benchmark --no-deps
python -c "import motifgo; print(motifgo.__file__)"
```

`--no-deps` is the point: `numpy` / `biotite` / `scipy` are *declared* in `pyproject.toml` so the repo
is honest about what it needs, but not *resolved* at install — so pip cannot upgrade a dependency out
from under a working environment. Storage cost is a `.pth` file: **under 10 kB, no code copied**.
Zero-install fallback, equally supported: `export PYTHONPATH=/path/to/motif_benchmark`.

### 0.4 — N4: no tour, no LFS, files on disk

Adopted, simplified. `example/` sits at `motif_benchmark/example/` at **full depth** (~650 MB) and is
`.gitignore`d in full. Git carries only `example/README.md` (what the seven tasks show, how to rebuild)
and `scripts/extras/make_example_bundle.py`. Nothing else, no `tour/`, no LFS.

Measured sizes (not estimates): scaffolds **~11 MB/task** across all five methods → ~77 MB;
AF3 refolds **24.9 kB per prediction** × 3,300 predictions/task → ~82 MB/task → ~575 MB, tarred.
rfdiff's `schedules/` (7.7 MB/task of diffusion noise schedules — a runtime artifact, not a result) is
**not shipped**; that alone saves 54 MB.

### 0.5 — RESOLVED: `filled_rows` stays

You agreed to keep it. It is 25 lines and is just `merge_gaps` applied at catalog time; 49 of the 314
benchmark tasks are `filled`, and approved example task #4 (`il1b_m21_filled`) is the controlled pair.
`emit_filled: true` stays a `target.yaml` knob so a new protein can skip it.

**Its one-line role in the published benchmark** (goes in the README, per your note): *`gap_merge=4`
was applied at catalog time to emit a second `filled` row beside each `base` motif — 49 of the 314
tasks — so base and filled could be scored head to head.*

### 0.6 — N7 / Decision 1: visibility. **Top-level `motifgo/`, and a stage map.**

You want to trace what runs at each step. Two changes deliver that:

**(a) The package moves out of `src/` to the repo top level.** `motif_benchmark/motifgo/` sits *beside*
`scripts/`, not buried. It is still `pip install -e .`-able (flat layout) **and** works with
`PYTHONPATH=/path/to/motif_benchmark`. You lose nothing and it is immediately visible. This retires the
v3 deviation — there is no longer a `src/` to argue about.

**(b) The dependency surface is far thinner than it looked.** Of 24 candidate scripts, **only 8 import
the shared library at all**; the other 16 are self-contained NumPy. Full map, verified:

| stage | script | imports from `motifgo` |
|---|---|---|
| 0 motif | `scripts/motif/extract_motifs.py` | `motif_define`, `motif_build` |
| 0 motif | `scripts/motif/build_pool.py` | *(none — pandas/scipy only)* |
| 0 motif | `scripts/motif/stage_motif.py` | `contig` |
| 1 generate | `generate/proteina/proteina_from_library.py` | `contig` |
| 1 generate | `generate/protpardelle/protpardelle_from_library.py` | `contig` |
| 1 generate | `generate/la-proteina/la_proteina_from_library.py` | `contig` |
| 1 generate | `generate/genie3/genie3_from_library.py` | **none** |
| 1 generate | `generate/la-proteina/la_proteina_extract_seqs.py` | **none** |
| 2 evaluate | `evaluate/1_filter.py` | **none** |
| 2 evaluate | `evaluate/mpnn_prep.py`, `parse_chains.py` | **none** |
| 3 evaluate | `evaluate/3_af3_build.py` | `af3_inputs` |
| 5 evaluate | `evaluate/5_score.py` | `score` |
| 5 evaluate | `evaluate/unindexed_score.py` | **none** |
| extras | `extras/motif_rsa.py` | `motif_define` |
| extras | `scale_contigs.py`, `stage_contigscale.py`, `find_failed_all.py`, `compute_costs.py`, `extract_structures.py`, `merge_refold.py`, `normalize_backbones.py`, `cif_backbone.py` | **none** |
| analysis | `designable_sse_extract.py` | `dssp` (after F9) |

Nothing calls sideways between stage directories. `scripts/motif/README.md` and the top README carry
this table, so "which script does what at which step" is answerable without reading code.

**And to be explicit about your example:** `extract_motifs.py` is **not** called by `motif_build.py` —
it is the other way round. `extract_motifs.py` is the *executable*; `motif_build.py` is a *library
module* it imports. Executables never import each other.

### 0.7 — Tree note: "do we really need `refold` and `af3_inputs`? This is the repetitiveness."

**You were right that there is redundancy — but it is inside the modules, not between them.** Measured:

- **`af3_inputs.py` (80 L): keep as-is, no redundancy.** Four functions, no CLI, genuinely called by
  `3_af3_build.py` (`build_inputs`). Zero overlap with `score.py`.
- **`score.py` (215 L): ~90 lines are dead for the benchmark.** `5_score.py` uses exactly six names —
  `design_name_of`, `motif_rmsd_refold`, `sc_rmsd`, `rollup`, `write_csv`, `_ra`. It does **not** use
  `analyze`, `score_run`, `_init_worker`, `_process_archive`, `motif_res_from_trb` — those are the
  `esp.run` campaign scorer; `5_score.py` reimplements the parallel archive loop itself.
- **`af3_refold_analysis.py` (492 L): ~390 lines are dead.** `score.py` uses **seven** names:
  `superimpose`, `load_cif_structure`, `load_pdb_structure`, `mean_plddt`, `parse_af3_confidence`,
  `MOTIF_RMSD_MAX`, `SC_RMSD_MAX`. The other ~18 functions plus a full 60-line argparse CLI are the v1
  standalone scorer, superseded by `5_score.py`.

**So: merge, don't just copy.** `score.py` + the seven used names become **one** `motifgo/score.py`
of roughly 150 lines, with the two thresholds at the top (no separate `thresholds.py` — with one
module, "defined once" is automatic). Net: **707 lines → ~150**, and the hidden absolute-path import
into `BEK/scripts/` disappears.

Library after this: **6 modules, ~950 lines**, down from v3's 7 modules / 1,500 lines.

### 0.8 — F6: "is this solved once the structure is confirmed?"

**Yes, by P4.** Your reading is right: the starting point is a protein PDB + a `target.yaml`. Today
that path is broken because two different tools each do half the job with different output schemas
(`motif_extract.py` writes only a descriptors table; only `esp.run extract-motifs` writes the templates
`stage_motif.py` needs). P4 makes **one** `scripts/motif/extract_motifs.py` that takes
`--target proteins/<t>/target.yaml` and emits all four artifacts:

```
proteins/<t>/{<t>_sasa_resis.csv, <t>_c_motif.csv, motif_pdbs/}      + the redundancy check
```

Gate: re-running it on `il13` reproduces the 2026-08-05 artifacts byte-for-byte.

### 0.9 — Remaining decisions, as answered

**F7** — just change `fab_chains` to `[A,B,C,D]` for her2; no second key. Also fix all six dead
`complex_cif:` paths (§F7). **F1** — README caveat + `build_pool.py --jaccard` available; no re-run.
**Notebook** — copied as-is into `scripts/analysis/`, still pointed at the full runs.
**Repo** — `BEK/motif_benchmark/`, `git init` local; Wicky GitLab remote added by you when ready
(P6 leaves a clean first commit and a `.gitignore` that already excludes `example/`).

---

## 1. Tree (final)

```
BEK/motif_benchmark/
├── README.md                  # quickstart (generic venv) · task naming (F8) · gap merging (0.2)
│                              #   · stage→script map (0.6) · per-stage sbatch blocks with all knobs
│                              #   · "pool is not deduped" caveat (F1)
├── pyproject.toml             # package `motifgo`, flat layout;  pip install -e . --no-deps
├── site.yaml                  # cluster paths, container images, log root, runs root
├── .gitignore                 # example/**  (except its README)
│
├── motifgo/                   # ── the ONE library, top level & visible (0.6a) — 6 modules ──
│   ├── motif_define.py        #   was motif.py    — SASA, contacts, neighbourhood (leaf)
│   ├── motif_build.py         #   was catalog.py  — sasa csv → c_motif csv → motif PDBs
│   ├── contig.py              #   contig grammar: group_runs, merge_gaps, split_at_gaps, contig_string
│   ├── dssp.py                #   ONE DSSP — esp/_dssp.py ∪ scripts/dssp_numpy.py (F9)
│   ├── af3_inputs.py          #   AF3 monomer-refold JSON builder  (kept as-is, 0.7)
│   └── score.py               #   designability metric + the 7 used refold primitives (0.7)
│
├── proteins/<target>/         # ── where motifs come FROM ──
│   ├── target.yaml            #   pdb, chain, rsa 0.20, nbhd 7.0/ca, emit_filled, gap_merge 4,
│   │                          #   complex + fab_chains  (all 6 paths fixed; her2 → [A,B,C,D])
│   ├── <t>_sasa_resis.csv     #   all residues + `surface` flag
│   ├── <t>_c_motif.csv        #   one row per surface centroid (base + filled)
│   └── motif_pdbs/            #   extracted motif PDBs for this protein
│
├── motifs/                    # ── the scaffolding TASKS (flat, run-independent) ──
│   ├── pool.csv               #   the task list  (+ pool.provenance.json)
│   ├── interface_residues.json#   only for interface-restricted pools
│   └── pdbs/                  #   one <task>.pdb per pool row
│
├── scripts/                   # ── ALL pipeline executables; none import each other ──
│   ├── motif/                 #   Stage 0: target.yaml → motif PDBs → tasks
│   │   ├── extract_motifs.py  #     ONE entry point: sasa → c_motif → motif PDBs (F6)
│   │   ├── build_pool.py      #     dedup + interface + pad → pool.csv (F1, F7)
│   │   ├── stage_motif.py #     pool.csv → manifest.csv (the ONE writer, F2)
│   │   └── README.md          #     stage→script→module table
│   ├── generate/<method>/     #   Stage 1 ONLY: <method>_from_library.py + _generate.sbatch
│   │   ├── rfdiff/ proteina/ genie3/ protpardelle/
│   │   └── la-proteina/       #     + la_proteina_extract_seqs.py; idx/uidx = a FLAG, not a fork
│   ├── evaluate/              #   Stages 2–5: 1_filter, mpnn_prep, parse_chains, 2_mpnn,
│   │                          #     3_af3_build, 4_af3, 5_score, unindexed_score (+ sbatch)
│   ├── extras/                #   motif_rsa, scale_contigs, stage_contigscale, find_failed_all,
│   │                          #     compute_costs, extract_structures, make_example_bundle
│   └── analysis/              #   motif_benchmark.ipynb (as-is), figstyle, foldseek_*,
│                              #     normalize_backbones, merge_refold, designable_sse_extract, cif_backbone
│
├── containers/                # .def build recipes; .sif live in /cluster/work/wicky/containers
├── example/                   # full-depth worked example — GITIGNORED except README.md (0.4)
└── docs/                      # PILOT_RUNBOOK.md, UNIFIED_PIPELINE_WALKTHROUGH.md, this plan
```

`runs/` is absent by design — real runs stay at `motif_library/runs/` (704k inodes each), addressed by
`--run-dir`; `site.yaml` records `runs_root`.

**Copied, never moved.** `motif_library/`, `software/`, `scripts/`, `epitope_scaffolding/` stay
untouched and keep working throughout every phase.

**Not carried:** `proteina_{af3_build.py, af3.sbatch, fixed_positions.py, mpnn.sbatch, score.py}`
(08-07); `genie3_{af3_build.py, af3.sbatch, fixed_positions.py, mpnn.sbatch}` (07-24),
`genie3_{from_run.py, score.py}` (07-27), `genie3_mpnn_array.sbatch` (08-07);
`la_proteina_{af3_build.py, score.py}` (08-11), `la_proteina_af3.sbatch` (08-12);
`protpardelle-1c/scripts/{sample,train}.sbatch` (08-05, upstream training). Plus
`esp/{run,stages,config,compute,rmsd}.py` — 1,101 lines of campaign CLI no benchmark script calls —
and the ~480 dead lines identified in §0.7.

---

## 2. Containers

Four of seven images are already shared; three need one `cp` (**10.7 GB**):

| image | size | today | action |
|---|---|---|---|
| `RFdiffusion.sif` | 3.9 GB | `/cluster/work/wicky/containers/` | ✓ |
| `alphafold3.sif` | 5.3 GB | `/cluster/work/wicky/containers/` | ✓ |
| `proteina.sif` | 7.9 GB | `/cluster/work/wicky/containers/` | ✓ |
| `openfold.sif` (ProteinMPNN env) | 7.4 GB | `/cluster/work/wicky/containers/` | ✓ |
| `genie3.sif` | 3.4 GB | `software/genie3/` | **copy** |
| `la-proteina.sif` | 3.7 GB | `software/la-proteina/` | **copy** |
| `protpardelle.sif` | 3.6 GB | `software/protpardelle-1c/` | **copy** |
| `la-proteina-uidx/la-proteina.sif` | **65 B** | `software/la-proteina-uidx/` | symlink to the idx image — confirms uidx is a flag |

The five `.def` recipes (`genie3`, `la-proteina` ×2, `protpardelle`, `esmfold2`) go into
`containers/`. `site.yaml` names every image once; no sbatch carries a `software/`-relative path.

---

## 3. Logging

Per-stage default, portable across users via Slurm's `%u`:

```
/cluster/scratch/%u/motifgo_logs/{generate,filter,mpnn_prep,mpnn,af3_build,af3,score}/%x_%A_%a.out
```

Attribution needs no wrapper — name the job at submit:

```bash
mkdir -p /cluster/scratch/$USER/motifgo_logs/af3        # once: Slurm opens --output before the script runs
sbatch --job-name=pad5.rfdiff.af3 --array=1-391%92 \
       --export=ALL,RUN_DIR=$RUN,METHOD=rfdiff  scripts/evaluate/4_af3.sbatch
# -> .../motifgo_logs/af3/pad5.rfdiff.af3_11630940_17.out
```

Relocate with `sbatch -o /my/logs/%x_%A_%a.out -e … `. The one thing that cannot be engineered away:
`--output` is opened before the script body and expands only `%` patterns — **never environment
variables** — so the directory must exist at submit time.

The README carries one such block **per stage with every knob** (`RUN_DIR`, `METHOD`, `CUTOFF`,
`POSITIONS`, `PRESERVE`, `BATCH`, `PREDICTOR`, array spec, QOS relocation).

*Context:* logs are **6,000 of ~704,000 inodes (0.85 %)** on a full run. The inode cost is `mpnn/`
(~300k), `1_filter/passed/` (~150k symlinks), `scaffolds/` (~190k). `refold_inputs` is *already* a
scratch symlink; the same treatment on `mpnn/prepped` + `1_filter/passed` would free ~300k inodes per
run. Offered as a `site.yaml` knob, not built unless you ask.

---

## 4. The worked example — approved, full depth

| task | L | parts | SSE | variant | rfdiff | proteina | genie3 | pp1c_bb | laproteina |
|---|---|---|---|---|---|---|---|---|---|
| `il13_m103_base` | 9 | 1 | H | base | 724/800 | 294/800 | 728/800 | 552/800 | 85/100 |
| `il2_m101_base` | 6 | 1 | H | base | 621/800 | 185/800 | 573/800 | 561/800 | 80/100 |
| `il1b_m21_base` | 7 | **2** | E | base | 418/792 | 54/800 | 483/800 | 306/800 | 65/100 |
| `il1b_m21_filled` | 9 | **1** | E | **filled** | 513/800 | 190/800 | 624/800 | 584/800 | 84/100 |
| `pdl1_m113_base` | 11 | 3 | E | base | 276/696 | 28/800 | 405/800 | 165/800 | 39/100 |
| `her2_m568_base` | 15 | 4 | E | base | 2/792 | 1/800 | 4/800 | 2/800 | 0/100 |
| `tnfa_m27_base` | 9 | 4 | C | base | 0/176 | 0/800 | 28/800 | 3/800 | 0/100 |

Difficulty 724 → 0 designable; disjointedness 1 → 4; SSE H/E/C. `il1b_m21` base-vs-filled is the
controlled pair (see §0.5). Bare scaffolds + tarred refolds, no `schedules/`.

Build cost, unchanged: a task's refolds are scattered across all ~400–500 batch archives per method
(batches pack in seq-file order, not task order), so the bundle scans ~30 GB — one CPU job, once, not
on the login node.

---

## 5. Findings

**F1 — The benchmark pool was never deduplicated, despite its name.** Notebook cell G2 builds it from
**`motif_pool.csv`** (raw), interface+pad5 only; G3 saves it as `…_j0.7_pad5.csv`. **84 pairs at
Jaccard ≥ 0.7 survive, 5 are identical residue sets**; 61 of 314 are absent from
`motif_pool_deduped.csv`. Cross-method comparison unaffected. → README caveat + `--jaccard` flag.

**F2 — Two manifest writers, two schemas.** Live `manifest.csv` is `idx,config_name,task_name,…` with
**no `fragments` column** — written by `la_proteina_from_library.py`, not `stage_motif.py`.
`rfdiff_generate.sbatch` carries 20 lines of awk header-lookup to survive it.

**F3 — `esp.site` already is the portability loader** (`${VAR}`, hard error when unset, refuses to
guess a layout). Port it into `site.yaml`; do not invent a second mechanism.

**F4 — `design_name` has no common format.** rfdiff `<task>_design_<N>`; proteina/genie3/laproteina
`<task>_<N>`; pp1c_bb `<task>_sample_<task>_<N>` (**doubled**). No `task_name` column in
`af3_refold_scores.csv`, so every consumer re-derives the join by per-method regex. One column fixes it.

**F5 — 450 stale `inf_*.yaml` symlinks** in `software/la-proteina/configs/` across four runs →
`--clean-configs`.

**F6 — The "plug in a new protein" path is broken; P4 fixes it** (§0.8).

**F7 — her2 used BOTH Fabs, and every `complex_cif` path is dead.** `[A,B]`→21 residues,
`[C,D]`→29, **`[A,B,C,D]`→50 = `interface_residues.json` exactly**. Per your call, `fab_chains`
becomes `[A,B,C,D]`. Separately, **all six `complex_cif:` paths point at files that do not exist**
(`pdbs/8PWH.cif`, `pdbs/5L6Y.cif`, … absent; real files are `pdbs/*.pdb` and
`binding_validation/cifs/*.cif`) — the YAML complex block has never executed. Fixed in P5.

**F8 — `motif_num` is the SASA centroid.** All 265 base motifs: `motif_num` in its own residue list
265/265, present and `surface=1` in `<t>_sasa_resis.csv`, centroid RSA min 0.201 / median 0.417.
`her2_m568_base` = the motif grown around HER2 surface residue 568. Goes in the README.

**F9 — NEW: two copies of vendored DSSP.** `scripts/dssp_numpy.py` (173 L) and `esp/_dssp.py` (126 L)
are the same algorithm — identical `CONST_F`, `DEFAULT_CUTOFF=-0.5`, `DEFAULT_MARGIN=1.0`, same
`_unfold` / hydrogen-position / hbond-map / `assign` structure. They diverge only in extras:
`dssp_numpy` adds `assign_c3` + `selftest`, `_dssp` adds `_backbone_coord` + `sse_by_res`. Both are
live in different stages (`designable_sse_extract.py` uses one, `motif_build.py` the other). → merge
into `motifgo/dssp.py` carrying the union; repoint both consumers.

---

## 6. Phases

Nothing is deleted; both trees stay runnable throughout.

**P1 — Skeleton + `motifgo`.** Create the tree. Copy and rename `motif.py`→`motif_define.py`,
`catalog.py`→`motif_build.py`, `contig.py`, `af3_inputs.py`. Merge the two DSSP copies → `dssp.py`
(F9). Merge `score.py` + the 7 used `af3_refold_analysis` names → one `score.py`, dropping the ~480
dead lines (§0.7). `pyproject.toml`, `site.yaml`, `.gitignore`.
*Gate: `import motifgo` resolves to the new tree; the designability thresholds are defined in exactly
one place; `python -m pytest` on a smoke import passes.*

**P2 — Copy the pipeline, rewire the 8 import sites.** Populate
`scripts/{motif,generate,evaluate,extras,analysis}`. Drop the `importlib` + hardcoded-path blocks from
`3_af3_build.py` and `5_score.py`; collapse la-proteina idx/uidx to one directory + a flag; promote
`_THREE_TO_ONE` → `THREE_TO_ONE`. Write the stage→script→module table into `scripts/motif/README.md`
and the top README.
*Gate: re-run stage 5 on ONE existing archive; the CSV is byte-identical to the published one.*

**P3 — Containers + logs + README knobs.** Copy the three `.sif` (10.7 GB) to
`/cluster/work/wicky/containers/`; bring the five `.def` recipes in; point every sbatch at `site.yaml`.
Set the per-stage `--output` defaults with `%u`. Write the per-stage launch blocks. Add `task_name` to
`5_score.py` output (F4).
*Gate: `runs/smokes/proteina_smoke_1n8z_pc_n10` (10 designs) runs end-to-end from the new tree.*

**P4 — Stage 0 made whole (F6) + one manifest writer (F2).** One `extract_motifs.py`:
`target.yaml → sasa csv → c_motif csv → motif PDBs`, absorbing the old screen + cross-library
redundancy check. `stage_motif.py` becomes the sole manifest writer, restores `fragments`,
symlinks motif PDBs instead of copying 314 files per run; delete the awk hack from
`rfdiff_generate.sbatch`.
*Gate: re-running `extract_motifs.py` on `il13` reproduces the 2026-08-05 artifacts byte-for-byte.*

**P5 — `build_pool.py` (F1, F7).** Lift notebook cells G1–G3 into
`build_pool.py --jaccard --pad --interface-from-proteins --order {dedup-then-interface|interface-then-dedup}`.
Fix all six `complex:` paths; set her2 `fab_chains: [A,B,C,D]`.
*Gate: reproduce the current 314-row pool bit-for-bit under `--jaccard 0`, and report separately what a
real `--jaccard 0.7` pool would contain.*

**P6 — README + example + first commit.** README: quickstart, task naming (F8), gap merging (§0.2),
stage map (§0.6), per-stage launch blocks, the not-deduped caveat (F1). `make_example_bundle.py` (one
CPU job, ~30 GB scanned) → full-depth `example/` + its README. `--clean-configs` (F5). `git init`,
first commit, `.gitignore` already excluding `example/`. GitLab remote is yours to add.

---

## 7. Open items

**None.** Every question is answered; P1–P6 execute without further input. After P6, the agreed next
step is an **end-to-end pipeline test** on the smoke run.
