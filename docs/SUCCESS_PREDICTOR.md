# Motif-Scaffolding Success Predictor — how it works & how to integrate it

**Status:** built and validated. **Owner artefacts:**
- `/cluster/work/wicky/bekessel/notebooks/success_predictor.ipynb` (25 cells, the full build)
- `/cluster/work/wicky/bekessel/scripts/predict_motif.py` (stdlib CLI, no deps)
- model export: `/cluster/work/wicky/bekessel/motif_library/analysis/success_model.json`
  (not yet written on the cluster — a fitted copy lives locally, see §10)
- **local results + fitted model: `SemesterProject/docs/predictor_results/`** (§10)

Companion reference for the benchmark itself:
`SemesterProject/docs/MOTIF_SCAFFOLDING_BENCHMARK_HANDOFF.md` (§8B covers this predictor).

---

## 1. What problem it solves

The benchmark's point is to get **lab-testable designs for the least compute**. The predictor turns the
benchmark into a decision rule:

```
target interface → extract motif → 4 descriptors → per-method P(success) → ROI (GPU-h per success)
                                                        ↓
                                      "use method X; and if you can fill the gaps, do — here's the gain"
```

It answers three questions for a *motif that has never been run*:
1. **How likely** is each generator to produce a designable scaffold for it?
2. **Which feature** made it easy or hard (disjointedness? length? SSE? exposure?)
3. **Which method is cheapest per success**, not just most accurate?

## 2. The model (deliberately the simplest thing that fits)

Literature check: a plain logistic regression was not beaten by fancier models for de-novo design-success
prediction (Garcia et al., *Protein Science* 2026), and the standard way to show per-feature contribution
to a logistic prediction is a **nomogram**. So: logistic regression + nomogram-style decomposition +
a shallow decision tree as the at-a-glance rulebook.

**Unit of observation:** one row per `(motif, method)`.
**Outcome:** binomial — `n_designable` successes out of `n_designs` trials, where a design is designable
under the project's **3 gates**:

```
motifRMSD <= 1.0  AND  scRMSD <= 2.0  AND  ranking_score >= 0.75      (>= is inclusive; a large mass sits exactly at 0.75)
design designable  <=>  ANY of its <=8 MPNN sequences passes all three
motif  designable  <=>  ANY of its designs passes
```
(The `designable` column inside the score CSVs is only the *first two* gates — do not use it directly.)

**Model:** binomial logistic GLM, **fitted separately per method** (6 small models, so each generator gets
its own coefficients). This is "linear regression on the log-odds" and returns a calibrated probability.

**Features (4 descriptors, z-scored + 2 dummies + 1 interaction):**

| feature | why |
|---|---|
| `disjointed_parts` | number of backbone fragments — the headline driver |
| `motif_length` | residues in the motif |
| `motif_length x disjointed_parts` | the interaction (they are *not* independent, r≈0.58) |
| `dominant_sse` → `H`, `E` dummies | coil is the reference level |
| `mean_rsa` | mean relative solvent accessibility (how exposed the motif is) |

Standardisation constants (`mu`, `sd`) come from the **library pool**, and are stored in the JSON, so an
external motif is scaled identically to training data.

**Model ladder** (the notebook builds it incrementally and compares by McFadden pseudo-R², AIC and a
likelihood-ratio test) — `M0 intercept → M1 +disjoint → M2 +length → M3 +len*disj → M4 +SSE,RSA`.
The finding: **M1 alone already captures nearly all of it.** Disjointedness dominates; length and the
interaction add almost nothing; SSE+RSA add a little.

## 3. Fitted result (baseline run `lib_interface_pad5`, 314 motifs, 6 arms)

| method | overall design succ. | intercept | disjoint_z | length_z | len x disj | H | E | rsa_z | CV AUC |
|---|---|---|---|---|---|---|---|---|---|
| rfdiff | 0.211 | -2.55 | **-1.44** | -0.20 | 0.05 | 1.30 | 0.78 | -0.27 | 0.848 |
| proteina | 0.109 | -3.03 | **-1.35** | -0.14 | 0.02 | 0.99 | 0.68 | -0.17 | 0.841 |
| genie3 | 0.220 | -2.26 | **-1.47** | -0.16 | 0.14 | 1.11 | 1.00 | -0.19 | 0.811 |
| pp1c_bb | 0.162 | -2.63 | **-1.48** | -0.26 | 0.07 | 0.59 | 1.06 | -0.12 | 0.866 |
| laproteina (indexed) | 0.110 | -3.26 | **-1.75** | -0.13 | 0.12 | 0.48 | 0.94 | -0.09 | 0.828 |
| **laproteina unindexed** | 0.236 | -1.56 | **-0.36** | -0.35 | -0.01 | 0.64 | 0.76 | -0.13 | 0.614 |

CV AUC = grouped 5-fold (held-out antigen), motif-level ">=1 designable".

Read-off:
- Every index-conditioned method is punished ~1.4-1.75 log-odds per SD of disjointedness.
- **La-Proteina unindexed is the structural exception** — it places fragments freely, so it is largely
  *insensitive* to disjointedness (coef only -0.36). Its low AUC is not a defect: there is simply little
  motif-feature signal left to predict, because it succeeds fairly uniformly.
- CV AUC 0.81-0.87 for the index-conditioned methods → the model generalises to unseen motifs.

## 4. The three outputs the user actually consumes

**(a) Per-feature contribution (nomogram).** For a motif and method,
`contribution_j = beta_j * (x_j - mu_j)` in log-odds; the bars plus the intercept sum exactly to
`logit(p_hat)` (asserted to 1e-6 in the notebook). For a linear model this *is* SHAP — no extra machinery.

**(b) ROI = expected GPU-hours per designable design.**
```
gpuh_per_design  = generation_gpuh / (n_tasks * nsamples)  +  flat AF3-refold cost
gpuh_per_success = gpuh_per_design / p_hat            <-- rank methods by THIS, not by p_hat
P(>=1 designable in N samples) = 1 - (1 - p_hat)^N
```

**Cost basis (baseline run, nsamples=100, flat refold = 8 seqs x 20 s = 0.0444 GPU-h/design):**

| method | generation GPU-h/design | + flat refold | total GPU-h/design |
|---|---|---|---|
| rfdiff | 0.0062 | 0.0444 | 0.0506 |
| proteina | 0.0011 | 0.0444 | 0.0455 |
| genie3 | 0.0085 | 0.0444 | 0.0530 |
| pp1c_bb | 0.0005 | 0.0444 | 0.0449 |
| laproteina (indexed) | 0.0010 | 0.0444 | 0.0455 |
| laproteina unindexed | 0.0020 | 0.0444 | 0.0465 |

> **The single most important cost fact: the shared AF3 refold dominates.** It is 84-99% of the
> per-design cost, and it is *identical* for every method. Generation is nearly free by comparison
> (0.0005-0.0085 GPU-h/design, a 17x spread that moves total cost by <20%). **Consequence: the ROI
> ranking is essentially the accuracy ranking.** Choosing the "cheap" generator buys almost nothing;
> choosing the generator that actually succeeds on your motif buys everything. If you want real compute
> savings, cut the refold (fewer MPNN sequences per design), not the generator.

**ROI at a typical library motif** (median length 8, coil, median RSA 0.32):

*Contiguous (1 fragment)* — four methods are effectively tied:

| rank | method | p_hat | GPU-h per success |
|---|---|---|---|
| 1 | genie3 | 0.315 | **0.17** |
| 2 | pp1c_bb | 0.243 | **0.18** |
| 3 | laproteina unindexed | 0.235 | **0.20** |
| 4 | rfdiff | 0.254 | **0.20** |
| 5 | laproteina (indexed) | 0.182 | 0.25 |
| 6 | proteina | 0.160 | 0.28 |

*Disjoint (3 fragments)* — one method wins by an order of magnitude:

| rank | method | p_hat | GPU-h per success | P(>=1 in 100) |
|---|---|---|---|---|
| 1 | **laproteina unindexed** | **0.113** | **0.41** | 1.000 |
| 2 | genie3 | 0.013 | 4.05 | 0.732 |
| 3 | rfdiff | 0.010 | 4.95 | 0.642 |
| 4 | pp1c_bb | 0.009 | 5.12 | 0.586 |
| 5 | proteina | 0.007 | 6.30 | 0.515 |
| 6 | laproteina (indexed) | 0.003 | 14.28 | 0.273 |

**The rulebook, stated honestly:**
- **Disjoint motif (>1 fragment) → La-Proteina unindexed, unambiguously.** ~10x better ROI than anything
  else, because it is the only method that does not collapse on fragmented motifs. This is the one
  decision in the whole benchmark that is worth a lot of compute.
- **Contiguous motif → genie3 / pp1c_bb / uidx / rfdiff are within ~15% of each other.** Pick on
  secondary grounds (wall-clock, queue, familiarity, output diversity), not on ROI. Avoid the indexed
  La-Proteina and Proteina arms.
- Of the 314 library motifs: 124 contiguous, 190 disjoint.

> **Caveat that can flip the contiguous ranking:** genie3's `compute_costs.csv` row covers only **60 of
> 314 tasks** while its scores cover all 314. If its true cost is ~5x the recorded figure, its
> generation cost becomes ~0.045 GPU-h/design and its ROI drops from 0.17 to ~0.28 — last place. Treat
> genie3's rank-1 position as unverified until its cost row is fixed.

**(c) Fill-the-gaps what-if.** `base` and `filled` variants of the same patch carry different
`disjointed_parts`, so the model re-predicts automatically. For the typical motif above, merging
3 fragments into 1 moves the best available option from **uidx at p_hat 0.113 (0.41 GPU-h/success)** to
**genie3/pp1c_bb at p_hat 0.32/0.24 (0.17-0.18 GPU-h/success)** — a **2.8x lift in success probability
and 2.4x cheaper per success**, and it changes which method you should use. That is the quantified
"merge the gaps if the biology allows it" rule, and it is the largest single lever the benchmark found.

## 5. Notebook map (`success_predictor.ipynb`, 25 cells)

| section | what it does |
|---|---|
| §1 | assemble the `(motif, method)` modelling table from the 6 `af3_refold_scores.csv` + the motif pool + `compute_costs.csv` |
| §2 | incremental ladder M0..M4, pseudo-R²/AIC/LRT, coefficient table (odds ratios + CI + p) |
| **§2.5** | **grouped-5-fold CV AUC bar + calibration curve; predicted-vs-observed regression scatter per method; method-comparison matrices (coef heatmap, method x disjointedness success heatmap)** |
| §3 | per-feature nomogram contribution bars |
| §4 | ROI table + `P(>=1 in N)` |
| §5 | gap-fill what-if + depth-3 guideline decision tree (predicts the *best-ROI method* from motif features) |
| §6 | kNN nearest structural analog in z-scored descriptor space (empirical sanity check) |
| §7 | exports `motif_library/analysis/success_model.json` |
| **§8** | **external application: predict an unseen motif set, and validate against observed results** |

Cost lookup is keyed on `(run, method_subdir)` because `laproteina` and `laproteina_unindexed` share a
subdirectory name across two different runs.

The notebook prefers `statsmodels` GLM (for CIs/p-values) and falls back to
`sklearn.LogisticRegression(penalty=None)` on binomial-expanded rows with `sample_weight`; the
coefficients are identical to ~2 decimals.

## 6. The CLI (`predict_motif.py`)

Pure stdlib (`argparse, csv, json, math, os, sys`) — runs anywhere, no scientific stack needed. It reads
`success_model.json` and prints per-method `p_hat`, ROI, the contribution breakdown, the best-ROI
recommendation, and the fill-gaps what-if.

```bash
# one motif from the library pool
python scripts/predict_motif.py --task her2_m239_base --pool motif_library/motif_pool_interface_j0.7_pad5.csv

# one motif by hand
python scripts/predict_motif.py --length 12 --disjoint 2 --sse H --rsa 0.45

# a whole new set
python scripts/predict_motif.py --batch mcl1/mcl1_c_motif.csv --out preds.csv
```
`--batch` output columns: `motif_num, length, disjoint, sse, rsa, best_method, best_p_hat,
best_gpuh_per_success, p_<each method>`.

## 7. Applying it to an unseen motif set

**The raw motif-definition CSV is not enough.** `lbrandenburg/motif_scaffolding/2mhs_c_motif.csv` holds
only *surface residue + contacts*. The model needs the four **descriptors**
(`motif_length, disjointed_parts, dominant_sse, mean_rsa`). Generate them with the *same* `esp` extractor
the library used, so the feature definitions match exactly:

```bash
source /cluster/work/wicky/bekessel/gen_env/bin/activate      # gen_env is a VENV, not conda
export PYTHONPATH=/cluster/work/wicky/bekessel/epitope_scaffolding/pipeline
python -m esp.run extract-motifs --config library/configs/mcl1.yaml \
    --out /cluster/work/wicky/bekessel/motif_library --archive
```
→ writes `<lib>/mcl1/mcl1_c_motif.csv` with all descriptors; `motif_num` = the surface residue number.
Then feed it to notebook §8 (`EXT_FEATURES`) or to `predict_motif.py --batch`.

For 2mhs this has **already been run** — `motif_library/mcl1/mcl1_c_motif.csv` exists (104 motifs,
`source_pdb = 2mhs.pdb`, chain A) and its numbering matches Lisa's `2mhs_merged_og_motifs.csv` exactly.

## 8. Integration notes for the next session

- **Fit set is the baseline run only** (`runs/lib_interface_pad5` + `runs/lib_interface_pad5_uidx` for the
  unindexed arm). Do not mix in `contigscale` or `hardest_0pass` — different contig grammar, different
  cost basis.
- **`success_model.json` is the contract.** It stores `features`, `cont`, `mu`, `sd`, and per-method
  `{intercept, coef}`. Anything that can do a dot product and a sigmoid can consume it; that is why the
  CLI has no dependencies.
- **Re-fit trigger:** any change to the 3-gate thresholds, or adding a method/run, invalidates the JSON.
  Re-run notebook §1-§2 then §7.
- **Known data caveats:**
  - `compute_costs.csv` has generation GPU-h only; refold rows are mostly 0, so refold is modelled as one
    flat tunable constant (`AF3_SEC = 20 s/fold x 8 seqs`).
  - genie3's baseline cost row covers only 60 tasks (its scores cover all 314) → its GPU-h/task looks
    inflated. Flagged, not corrected.
  - Native sym/asym modes are longer scaffolds → use the `contigscale` `compute_costs.csv` for those.
- **Operational constraints that still apply:** never ssh to Euler (mount only, write only under
  `bekessel/`); JupyterLab autosave clobbers mount edits — *Reload from Disk* before running; the sshfs
  mount drops often (`diskutil unmount force eulermount; ./mount_euler.sh`).

---

## 9. External validation: 2mhs / Mcl-1 (held out) — RUN, and it works

**Why this is a fair test.** The training pool `motif_pool_interface_j0.7_pad5.csv` contains only
6 antigens — `her2, il1b, pdl1, tnfa, il13, il2`. **Mcl-1 (2mhs) is not among them**, so it is a genuine
external set. And Lisa ran RFdiffusion at ~1000 trajectories per motif, so the observed per-motif rate is
a high-N ground truth rather than a noisy estimate.

**Setup.** Model fit on the baseline run only. Predictions from `mcl1/mcl1_c_motif.csv` (104 motifs,
`source_pdb = 2mhs.pdb`). Ground truth from `bekessel/analysis/reports/af3_refold_2mhs_c_motif.csv`,
aggregated with the **identical 3-gate rule** (this is our own AF3 refold of her designs, so the outcome
definition matches the benchmark exactly). 94 motifs joined on `motif_num`; median 999 designs/motif;
**74,307 designs total**.

**Result:**

| metric | value |
|---|---|
| Spearman rho (predicted p_hat vs observed rate) | **0.652** (p = 1.1e-12) |
| Pearson r | 0.627 (p = 1.4e-11) |
| AUC, motif has >=1 designable | 0.840 (base rate 0.94 — near-saturated, so the rate correlation is the meaningful number) |
| observed overall 3-gate design rate | 0.528 |
| mean predicted p_rfdiff | 0.389 |

**Ranking is monotonic and calibration is close:**

| predicted tertile | n | mean predicted | mean observed | designs |
|---|---|---|---|---|
| low | 31 | 0.137 | 0.176 | 16,426 |
| mid | 32 | 0.434 | 0.494 | 26,888 |
| high | 31 | 0.596 | 0.663 | 30,993 |

**The headline driver reproduces almost exactly on unseen data:**

| disjointed_parts | n | mean predicted | mean observed |
|---|---|---|---|
| 1 | 73 | 0.469 | **0.535** |
| 2 | 17 | 0.133 | **0.157** |
| 3 | 4 | 0.016 | **0.018** |

**Domain check** — 2mhs sits inside the training feature ranges:
`motif_length` pool 4-21 (med 8) vs 2mhs 2-15 (med 8); `disjointed_parts` pool 1-5 (med 2) vs 2mhs 1-3
(med 1); `mean_rsa` pool 0.10-0.60 (med 0.32) vs 2mhs 0.13-0.70 (med 0.32). A handful of 2mhs motifs are
shorter (2-3 res) than anything in the pool — mild extrapolation at the low end.

**Interpretation / caveats.**
- The model **ranks** unseen motifs well (rho 0.65) and its *absolute* probabilities are conservative by a
  roughly constant ~0.05-0.07 — it systematically under-predicts here. Expected: 2mhs motifs are on
  average more contiguous (median 1 fragment vs 2 in the pool) and were extracted with Lisa's older,
  non-interface-restricted, unpadded protocol, which yields easier motifs than the pad5 library.
- Use it as a **ranking / triage tool** first (which motifs to spend GPU on, and which method to spend it
  on), and treat the absolute p_hat as a lower bound for easier extraction modes.
- Only the **rfdiff** arm can be validated this way, because rfdiff is the only method Lisa ran on 2mhs.

**Reproduce:** the scratch scripts used are `fit_and_test.py` (fit + predict) and `validate_2mhs.py`
(join + statistics); their logic is exactly notebook §1-2 and §8. Per-motif output was written to
`pred_vs_obs.csv` (predicted p_hat for all 6 methods + observed rate per motif). Nothing was written to
the cluster; regenerate on demand.

> Practical note: `af3_refold_2mhs_c_motif.csv` is 243 MB and streams at roughly 0.25 MB/s over the sshfs
> mount — the one-pass `awk` aggregation took ~17 minutes. Run that aggregation cluster-side, or copy the
> file locally once, rather than reading it repeatedly.


---

## 10. Saved results (local only — nothing was written to the cluster)

`SemesterProject/docs/predictor_results/`

| file | what it is |
|---|---|
| `success_model.json` | **the fitted model** — `features`, `cont`, `mu`, `sd`, and per-method `{intercept, coef}`. This is the whole contract; `predict_motif.py` consumes it directly. |
| `mcl1_2mhs_predictions.csv` | predicted `p_hat` for all 6 methods on the 104 held-out Mcl-1/2mhs motifs, with their descriptors |
| `2mhs_pred_vs_observed.csv` | the validation join — 94 motifs, predicted vs observed rate, per motif |
| `2mhs_observed_3gate.csv` | observed ground truth: `motif_num, n_designs, n_designable, n_seqs, n_seqs_pass` (3-gate, from 74,307 designs) |
| `agg_<method>.csv` | per-task `n_designs, n_designable` for each of the 6 benchmark arms — the training outcome table |
| `fit_and_test.py` | fits the 6 per-method models from the `agg_*` tables + the pool, writes `success_model.json`, predicts on Mcl-1 |
| `validate_2mhs.py` | joins predictions to ground truth, computes rho / AUC / tertile + disjointedness breakdowns |
| `results_dump.py` | regenerates the coefficient and ROI tables in §3 and §4 |
| `agg.awk` | the one-pass 3-gate aggregator (`task` = first 3 tokens of `design_name`) |

**To reproduce from scratch** (all local, mount read-only):
```bash
for m in rfdiff proteina genie3 pp1c_bb laproteina; do
  awk -F, -f agg.awk <RUN>/lib_interface_pad5/$m/af3_refold_scores.csv > agg_$m.csv
done
awk -F, -f agg.awk <RUN>/lib_interface_pad5_uidx/laproteina/af3_refold_scores.csv > agg_laproteina_unindexed.csv
python3 fit_and_test.py && python3 validate_2mhs.py && python3 results_dump.py
```

**Not done on purpose:** `success_model.json` was *not* copied to
`motif_library/analysis/` on the cluster. Do that (or run notebook §7) when the CLI needs to run
cluster-side.
