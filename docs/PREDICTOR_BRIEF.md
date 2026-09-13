# The success predictor — model brief and operating guide

Two parts. **Part 1** is self-contained and meant to be pasted into another agent's context.
**Part 2** is how you run it and read the output.

---

# Part 1 · Brief (paste this)

```text
MOTIF-SCAFFOLDING SUCCESS PREDICTOR — MODEL BRIEF
=================================================

PURPOSE
Given a protein motif that has never been scaffolded, predict, for each of six generator
arms, the probability that a single generated design will be "designable", and from that the
expected GPU-hours per designable design. Used to triage motifs and choose a method before
spending compute. It is advisory: no pipeline stage depends on it.

UNIT OF OBSERVATION
One row per (motif, method). Outcome is binomial: n_designable successes out of n_designs
trials.

OUTCOME DEFINITION (three gates, all inclusive, all required)
  a SEQUENCE passes  iff  motifRMSD <= 1.0 A          # refold backbone N,CA,C at the design
                    AND   scRMSD    <= 2.0 A          #   positions, vs the native motif
                    AND   ranking_score >= 0.75       # AlphaFold3 confidence
  a DESIGN is designable  iff ANY of its sequences passes   (8 for backbone arms, 1 co-design)
  a MOTIF  is designable  iff ANY of its designs is designable
Gates are always re-derived from the raw columns. Do NOT read the `designable` column of
af3_refold_scores.csv: the baseline CSVs predate the third gate and carry a TWO-gate value.

FEATURES — four motif descriptors, six design-matrix columns
  continuous, z-scored with the FITTING POOL's mu/sd (stored in the model, so an unseen motif
  is scaled identically):
    disjointed_parts   number of backbone fragments      mu=1.8248  sd=0.8256
    motif_length       residues in the motif             mu=7.9459  sd=2.4896
    mean_rsa           mean relative solvent accessibility  mu=0.3320  sd=0.1073
  derived:
    len_x_disj = motif_length_z * disjointed_parts_z     (they are correlated, r~0.58)
  categorical: dominant_sse -> dummies H, E   (coil C is the reference level)

  x = [disjointed_parts_z, motif_length_z, len_x_disj, H, E, mean_rsa_z]

MODEL
Binomial logistic GLM (logit link), fitted SEPARATELY PER METHOD — six independent small
models, so each generator gets its own coefficients. Fitted with statsmodels
  sm.GLM(np.column_stack([n_success, n_failure]), [1|X], family=Binomial()).fit()
i.e. maximise  sum_i [ y_i log p_i + (n_i - y_i) log(1 - p_i) ],  p_i = sigmoid(b0 + b·x_i).
No regularisation. (sklearn LogisticRegression(penalty=None) on binomial-expanded rows with
sample_weight is the fallback and gives the same coefficients to ~2 dp.)

  logit(p) = b0 + b1*disj_z + b2*len_z + b3*len_x_disj + b4*H + b5*E + b6*rsa_z
  p_hat    = 1 / (1 + exp(-logit))

Rationale for the simplest model: a plain logistic was not beaten by more complex models for
de-novo design-success prediction (Garcia et al., Protein Science 2026).

PER-FEATURE ATTRIBUTION
  contribution_j = coef_j * x_j        (log-odds)
  intercept + sum_j contribution_j = logit(p_hat)   exactly
For a linear model this IS the nomogram decomposition, and equals SHAP. No extra machinery.

FITTED COEFFICIENTS  (log-odds; baseline run, 314 motifs x 6 arms)
  method                const   disj_z   len_z  len*disj      H      E   rsa_z   rate  pR2
  rfdiff               -2.554   -1.445  -0.198     0.051  1.302  0.779  -0.274  0.211 .181
  proteina             -3.027   -1.346  -0.138     0.025  0.996  0.684  -0.165  0.109 .154
  genie3               -2.262   -1.468  -0.157     0.139  1.112  1.000  -0.186  0.220 .200
  pp1c_bb              -2.631   -1.484  -0.259     0.070  0.589  1.059  -0.123  0.162 .168
  laproteina           -3.268   -1.754  -0.133     0.119  0.483  0.946  -0.088  0.110 .175
  laproteina_unindexed -1.556   -0.365  -0.354    -0.010  0.643  0.764  -0.133  0.236 .047
  (rate = observed design-level success; pR2 = McFadden pseudo-R^2)

HEADLINE RESULT
Disjointedness dominates. Every index-conditioned arm loses 1.35-1.75 log-odds per SD of
disjointed_parts; length, the interaction and exposure add little. La-Proteina UNINDEXED is
the structural exception at -0.365: it places the motif itself rather than being told where,
so fragment count barely hurts it. Its low pseudo-R^2 (0.047) is not a defect -- it succeeds
fairly uniformly, so there is little motif-feature signal left to predict.

COST MODEL
Only GENERATION cost differs between methods. The AF3 refold is the same shared job, priced
per FOLD and multiplied by each arm's own sequences-per-design (8 backbone, 1 co-design):
  gpuh_per_design  = gen_gpuh_per_design + refold_gpuh_per_fold * seqs_per_design
  gpuh_per_success = gpuh_per_design / p_hat
  P(>=1 designable in N designs) = 1 - (1 - p_hat)^N
  refold_gpuh_per_fold = 0.005672   (~20 s/fold, from the one measured refold row)
  gen_gpuh_per_design: rfdiff 0.00616 · proteina 0.00105 · genie3 0.00852 ·
                       pp1c_bb 0.00048 · laproteina 0.00101 · uidx 0.00202
Pricing the refold per DESIGN instead (one flat constant for all) overcharges the co-design
arms 8x. Because the refold term dominates for backbone arms, their ROI ranking is close to
their accuracy ranking -- a "cheap" generator buys almost nothing.

TRAINING DATA
Fit set is the BASELINE run only: runs/lib_interface_pad5 (5 arms) + runs/lib_interface_pad5_uidx
(the unindexed arm). 314 tasks, 100 designs requested each. Do NOT mix in contigscale or
hardest_0pass: different contig grammar, different cost basis.
Pool: motif_pool_interface_j0.7_pad5.csv (7 antigens; her2, pdl1, il13, il1b, il2, tnfa).

EXTERNAL VALIDATION — Mcl-1 / 2mhs, genuinely held out (not among the 7 fit antigens)
Predictions from motif descriptors ALONE; ground truth from an AF3 refold of ~1000 RFdiffusion
trajectories per motif, aggregated under the identical three gates.
  78 motifs joined, 65,606 designs
  Spearman rho 0.599 · Pearson r 0.641
  predicted tertiles  0.133 / 0.470 / 0.603  ->  observed  0.211 / 0.592 / 0.714  (monotonic)
  by fragment count:  1 frag  pred 0.504 obs 0.633
                      2 frag  pred 0.131 obs 0.165
                      3 frag  pred 0.016 obs 0.018
The model UNDER-predicts by roughly a constant offset (2mhs motifs are more contiguous and came
from an older unpadded extraction, so they are genuinely easier). Use it to RANK and TRIAGE;
read absolute p_hat as a lower bound. Only the rfdiff arm is validatable this way, since that
is the only method run on 2mhs.

KNOWN CAVEATS
- genie3's cost row covers 60 of 314 tasks while its scores cover all 314, so its
  gen_gpuh_per_design is inflated ~5x and any ROI rank it wins is UNVERIFIED. fit_predictor.py
  warns on this automatically.
- The 314-task pool carries "j0.7" in its filename but was never deduplicated: 84 motif pairs
  at Jaccard >= 0.7 remain, 5 of them identical residue sets. Cross-method comparison is
  unaffected (all arms got the same 314); aggregate rates are weighted by that redundancy.
- Re-fit trigger: any change to the three gate thresholds, or adding a method or run,
  invalidates the model. Re-run aggregate_outcomes.py then fit_predictor.py.

MODEL FILE CONTRACT  (models/success_model.json, ~6 kB, consumable by anything that can do a
dot product and a sigmoid)
  schema_version, cont_features, final_features, gates{motif_rmsd_max,sc_rmsd_max,rank_min},
  mu{}, sd{}, refold_gpuh_per_fold,
  methods{ <arm>: { label, coef{const, <feature>...}, gen_gpuh_per_design, seqs_per_design,
                    design_rate, pseudo_r2_mcfadden, cost_tasks_covered, cost_tasks_total } },
  meta{ fitted, outcomes, pool, costs[], nsamples, backend }

CODE
  motifgo/predictor.py            the contract: descriptors, gates, featurise, predict, model I/O
  scripts/predict/aggregate_outcomes.py   score CSVs -> (task, method, n_designs, n_designable)
  scripts/predict/fit_predictor.py        outcomes + pool + costs -> success_model.json
  scripts/predict/predict_motif.py        the CLI (pure stdlib, no scientific stack needed)
  scripts/predict/validate_predictor.py   held-out check against observed rates
```

---

# Part 2 · Running it

## Predict one motif

```bash
cd /cluster/work/wicky/bekessel/motif_benchmark
source /cluster/work/wicky/bekessel/gen_env/bin/activate     # or any python3 — the CLI is stdlib only

# by hand
python scripts/predict/predict_motif.py --length 11 --disjoint 3 --sse E --rsa 0.201

# a motif already in the pool
python scripts/predict/predict_motif.py --task pdl1_m113_base \
    --pool /cluster/work/wicky/bekessel/motif_library/motif_pool_interface_j0.7_pad5.csv \
    --contrib-method rfdiff

# a whole new target, straight from the extractor's output
python scripts/predict/predict_motif.py --batch proteins/<target>/<target>_c_motif.csv --out preds.csv
```

`--batch` needs the four descriptors, so feed it a `<target>_c_motif.csv` produced by
`scripts/motif/extract_motifs.py`. A raw motif residue list has no descriptors and cannot be scored.

## How the numbers come together

Worked end to end for `pdl1_m113_base` — length 11, **3 fragments**, sheet, RSA 0.201, on the
RFdiffusion arm.

**1 · Standardise, using the fitting pool's constants**

| descriptor | raw | mu | sd | z |
|---|---|---|---|---|
| `disjointed_parts` | 3 | 1.8248 | 0.8256 | **+1.423** |
| `motif_length` | 11 | 7.9459 | 2.4896 | **+1.227** |
| `mean_rsa` | 0.201 | 0.3320 | 0.1073 | **−1.221** |

then `len_x_disj = 1.227 × 1.423 = 1.746`, and `H = 0`, `E = 1` (coil is the reference level).

**2 · Multiply by the arm's coefficients — this sum *is* the nomogram**

| term | coef | x | contribution |
|---|---|---|---|
| intercept | −2.554 | — | **−2.554** |
| `disjointed_parts_z` | −1.445 | 1.423 | **−2.056** ← the motif's problem |
| `motif_length_z` | −0.198 | 1.227 | −0.243 |
| `len_x_disj` | 0.051 | 1.746 | +0.089 |
| `H` | 1.302 | 0.000 | 0.000 |
| `E` | 0.779 | 1.000 | +0.779 |
| `mean_rsa_z` | −0.274 | −1.221 | +0.335 |
| | | **logit** | **−3.651** |

**3 · Squash to a probability**

```
p_hat = 1 / (1 + exp(3.651)) = 0.0253
```

So ~2.5% of RFdiffusion designs for this motif will be designable. Being 1.4 SD more fragmented
than the average library motif costs it **−2.06 log-odds**, more than the intercept itself —
read straight off the contribution column.

**4 · Turn it into compute**

```
cost/design  = gen 0.00616 + refold (0.005672 × 8 folds) 0.04537 = 0.05153 GPU-h
cost/success = 0.05153 / 0.0253                                   = 2.04 GPU-h
P(≥1 in 100) = 1 − (1 − 0.0253)^100                               = 0.92
```

**5 · What the CLI prints**

```
motif pdl1_m113_base: length=11 disjoint_parts=3 sse=E mean_rsa=0.201

=== motif as given (disjoint_parts=3) ===
method               p̂  GPU-h/success
La-Proteina (uidx)  0.168           0.05     ← 6.6× more likely, 40× cheaper per success
La-Proteina (idx)   0.009           0.72
Genie3              0.044           1.22
Protpardelle-1c     0.023           1.96
RFdiffusion         0.025           2.04
Proteina            0.015           3.09
  -> best ROI: La-Proteina (uidx)

=== what-if: gaps filled (disjoint_parts=1) ===
La-Proteina (uidx)  0.335           0.02
Genie3              0.517           0.10
RFdiffusion         0.425           0.12
  ...
fill-gaps effect on best-ROI p̂: 0.168 -> 0.316  (Δ +0.149)
```

Three decisions fall out of that block:

- **Which method.** On a 3-fragment motif the unindexed arm is 6.6× more likely to succeed than
  RFdiffusion and ~40× cheaper per success. That gap only exists because it is the one model that
  places the motif itself.
- **Whether to merge the gaps.** If the biology tolerates adding a few residues, this motif goes
  from p̂ 0.17 to 0.32 — and the best method changes. That is the largest single lever in the
  benchmark.
- **Whether to run it at all.** `P(≥1 in 100) = 0.92` says 100 designs is enough here. Where that
  number is low, spend the GPU elsewhere.

## Re-fitting from raw data

Everything is re-derivable from the cluster; nothing is hand-maintained.

```bash
REPO=/cluster/work/wicky/bekessel/motif_benchmark
RUNS=/cluster/work/wicky/bekessel/motif_library/runs
POOL=/cluster/work/wicky/bekessel/motif_library/motif_pool_interface_j0.7_pad5.csv

# 1. six score CSVs -> per-(motif, method) successes, three gates re-derived  (~20 s)
python $REPO/scripts/predict/aggregate_outcomes.py \
    --runs-root $RUNS --pool $POOL --out $REPO/models/outcomes.csv

# 2. fit six GLMs + attach the cost basis  (seconds)
python $REPO/scripts/predict/fit_predictor.py \
    --outcomes $REPO/models/outcomes.csv --pool $POOL \
    --cost lib_interface_pad5:$RUNS/lib_interface_pad5/compute_costs.csv \
    --cost lib_interface_pad5_uidx:$RUNS/lib_interface_pad5_uidx/compute_costs.csv \
    --out $REPO/models/success_model.json --report
```

`--report` prints the coefficient table, the generation-cost table, and worked ROI for a typical
contiguous and a typical disjoint motif. It also **warns when an arm's cost row covers fewer tasks
than its score table** — which it does for genie3 (60 of 314).

Sanity check after any re-fit: `disjointed_parts_z` should sit near **−1.4 to −1.8** for the five
index-conditioned arms and near **−0.37** for `laproteina_unindexed`. That contrast is the result
the whole method recommendation rests on.

## Held-out validation

```bash
# aggregate the ground truth cluster-side: 244 MB, ~25 s here, ~17 min over an sshfs mount
python $REPO/scripts/predict/validate_predictor.py aggregate \
    --scores /cluster/work/wicky/bekessel/analysis/reports/af3_refold_2mhs_c_motif.csv \
    --out $REPO/models/2mhs_observed_3gate.csv

python $REPO/scripts/predict/validate_predictor.py report \
    --features /cluster/work/wicky/bekessel/motif_library/mcl1/mcl1_c_motif.csv \
    --observed $REPO/models/2mhs_observed_3gate.csv \
    --model $REPO/models/success_model.json --method rfdiff \
    --out $REPO/models/2mhs_pred_vs_observed.csv
```

Expect ρ ≈ 0.60 and the fragment-count breakdown reproducing (1 frag 0.50/0.63 · 2 frag 0.13/0.17 ·
3 frag 0.016/0.018). The same two commands work for any other protein that has been run and refolded.
