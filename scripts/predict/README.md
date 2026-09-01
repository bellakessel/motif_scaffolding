# `scripts/predict/` — the success predictor (an addition, not a gate)

Nothing in the pipeline depends on this. It reads finished runs and advises on motifs that have not
been run yet: how likely each generator is to succeed, which feature made the motif hard, and which
method is cheapest per success.

```
 af3_refold_scores.csv x6 ──► aggregate_outcomes.py ──► outcomes.csv
 pool.csv (4 descriptors) ──┐                              │
 compute_costs.csv (gen)  ──┴──► fit_predictor.py ─────────┴──► models/success_model.json
                                                                     │
                          predict_motif.py  ◄───────────────────────┘  (a NEW motif, no compute spent)
                          validate_predictor.py ◄──────────────────┘  (a held-out protein)
```

| script | in | out |
|---|---|---|
| `aggregate_outcomes.py` | six `af3_refold_scores.csv` + the pool | `(task, method, n_designs, n_designable)` |
| `fit_predictor.py` | outcomes + pool + generation costs | `models/success_model.json` (+ `--report`) |
| `predict_motif.py` | the model + a motif's four descriptors | per-method p̂, ROI, contributions, fill-gaps what-if |
| `validate_predictor.py` | the model + a held-out protein's observed rates | ρ, tertiles, per-fragment breakdown |

`motifgo/predictor.py` holds the two definitions everything else shares — the **four descriptors** and
the **three-gate outcome** — so the fit, the CLI and the notebook cannot drift apart.

## Two things this deliberately does not do

**It never reads the CSV's `designable` column.** The baseline CSVs (2026-08-18) predate the
ranking-score gate and carry a **two-gate** column; `5_score.py` has emitted three gates since
2026-08-26. Re-deriving from `motifRMSD` / `scRMSD` / `ranking_score` is correct for both vintages, and
which vintage each file used is recorded in `outcomes.csv` (`csv_designable_gates`) rather than
rediscovered. Verified on the baseline: proteina's table is 3-gate, the other five are 2-gate.

**It does not parse the task out of `design_name` by token count.** Design names differ per method —
rfdiff `<task>_design_<N>`, proteina/genie3/laproteina `<task>_<N>`, pp1c_bb `<task>_sample_<task>_<N>`
(the task repeated). Taking "the first three underscore tokens" is right for today's names by luck and
wrong for any protein tag that is not one token. It uses the `task_name` column `5_score.py` now emits,
falling back to longest-prefix matching against the pool for older CSVs.

## Reproducing the fit from raw cluster data

```bash
REPO=<repo>; RUNS=<runs>; POOL=<pool.csv>

python $REPO/scripts/predict/aggregate_outcomes.py \
    --runs-root $RUNS --pool $POOL --out $REPO/models/outcomes.csv

python $REPO/scripts/predict/fit_predictor.py \
    --outcomes $REPO/models/outcomes.csv --pool $POOL \
    --cost lib_interface_pad5:$RUNS/lib_interface_pad5/compute_costs.csv \
    --cost lib_interface_pad5_uidx:$RUNS/lib_interface_pad5_uidx/compute_costs.csv \
    --out $REPO/models/success_model.json --report
```

Fit set is the **baseline run only** (`lib_interface_pad5` + `_uidx`). Do not mix in `contigscale` or
`hardest_0pass` — different contig grammar, different cost basis. Re-fit whenever a gate threshold
changes or a method/run is added.

## Cost: generation only, refold priced per fold

Only **generation** differs between methods. The AF3 refold is the identical shared job, so it is one
constant **per fold** multiplied by each method's own `seqs_per_design` — 8 for a backbone method,
**1 for a co-design method**, measured from the data, not assumed. `--refold-gpuh-per-fold` defaults to
0.00567 GPU-h (≈20 s), derived from the one measured refold row in the baseline costs.

`fit_predictor.py` warns when a method's cost row covers fewer tasks than its score table. It does for
**genie3** — 60 tasks costed, 314 scored, so its GPU-h/design is inflated ~5× and any ROI rank it wins
is unverified.

## Predicting a motif

```bash
python predict_motif.py --task her2_m128_base --pool <pool.csv> --contrib-method rfdiff
python predict_motif.py --length 8 --disjoint 3 --sse C --rsa 0.32
python predict_motif.py --batch <target>_c_motif.csv --out preds.csv
```

The batch input is a `<target>_c_motif.csv` from `scripts/motif/extract_motifs.py` — the four
descriptors must come from the same extractor the library used, or the features are not comparable.
Override the model with `--model` or `MOTIFGO_MODEL`.

## Held-out validation

```bash
# cluster-side: the 2mhs table is 244 MB (25 s here; ~17 min over an sshfs mount)
python validate_predictor.py aggregate --scores <af3_refold_2mhs_c_motif.csv> --out observed.csv
python validate_predictor.py report --features <mcl1_c_motif.csv> --observed observed.csv \
       --model models/success_model.json --method rfdiff --out pred_vs_obs.csv
```

Predictions use **only the motif descriptors** — nothing about the designs — so this measures the
actual claim: can motif features alone rank motifs before compute is spent.
