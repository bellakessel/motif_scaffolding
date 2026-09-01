"""predictor.py — the success-predictor contract: descriptors, outcome rule, model I/O.

The benchmark's point is lab-testable designs for the least compute. This module turns the finished
runs into a decision rule: for a motif nobody has run yet, how likely is each generator to produce a
designable scaffold, which feature made it hard, and which method is cheapest per success.

WHY THIS IS A MODULE AND NOT THREE SCRIPTS. Two definitions decide every number downstream — what a
motif's FEATURES are, and what counts as a SUCCESS. Before this file they lived in three places at
once (the notebook, an awk one-liner, and the CLI's own `design_vector`), which is the same
duplication pattern that produced four copies of `stage_motif_pdb`. Defined here, imported everywhere.

    THE OUTCOME (three gates, project-wide)
        a SEQUENCE passes  iff  motifRMSD <= 1.0  AND  scRMSD <= 2.0  AND  ranking_score >= 0.75
        a DESIGN is designable  iff  ANY of its <=8 MPNN sequences passes
        a MOTIF  is designable  iff  ANY of its designs is designable

    Always re-derived from the raw columns, never read from the CSV's `designable` column: the
    baseline run's CSVs (2026-08-18) predate the third gate and carry a TWO-gate column, while
    5_score.py has emitted three gates since 2026-08-26. `outcome_gates()` reports which vintage a
    file is, so the distinction is visible instead of silent.

    THE FEATURES (four descriptors, from <target>_c_motif.csv or the pool)
        disjointed_parts   number of backbone fragments  <- the headline driver
        motif_length       residues in the motif
        dominant_sse       H / E / C   (coil is the reference level)
        mean_rsa           mean relative solvent accessibility

    Design matrix: 3 z-scored continuous + 2 SSE dummies + 1 interaction. Standardisation constants
    come from the fitting pool and travel inside the model JSON, so an unseen motif is scaled exactly
    as the training data was.

    THE MODEL  binomial logistic GLM, fitted SEPARATELY PER METHOD (one small model per generator).
    A plain logistic was not beaten by fancier models for de-novo design-success prediction
    (Garcia et al., Protein Science 2026), and for a linear model the per-feature log-odds
    contribution beta_j * (x_j - mu_j) IS the nomogram decomposition — and equals SHAP exactly.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

# ── the three gates ──────────────────────────────────────────────────────────────────────────
MOTIF_RMSD_MAX = 1.0
SC_RMSD_MAX = 2.0
RANK_MIN = 0.75

# ── the design matrix ────────────────────────────────────────────────────────────────────────
CONT_FEATURES = ["disjointed_parts", "motif_length", "mean_rsa"]
FINAL_FEATURES = ["disjointed_parts_z", "motif_length_z", "len_x_disj", "H", "E", "mean_rsa_z"]
SCHEMA_VERSION = 2


def seq_passes(motif_rmsd, sc_rmsd, ranking_score) -> bool:
    """The three gates, on one refolded sequence. A missing ranking_score (ESMFold2) is NOT gated."""
    try:
        m, s = float(motif_rmsd), float(sc_rmsd)
    except (TypeError, ValueError):
        return False
    if m != m or s != s:                                  # NaN
        return False
    if not (m <= MOTIF_RMSD_MAX and s <= SC_RMSD_MAX):
        return False
    try:
        r = float(ranking_score)
    except (TypeError, ValueError):
        return True                                       # absent -> gate not applied
    return True if r != r else r >= RANK_MIN


def outcome_gates(row: dict) -> Optional[int]:
    """How many gates the CSV's own `designable` column implements for this row: 2, 3, or None.

    Diagnostic only — nothing fits on the column itself. Used to stamp a run's vintage into the
    aggregate so the 2-gate/3-gate distinction is recorded rather than rediscovered."""
    try:
        d, mo, so = int(row["designable"]), int(row["motif_ok"]), int(row["sc_ok"])
        rs = float(row["ranking_score"]) if row.get("ranking_score") else None
    except (KeyError, TypeError, ValueError):
        return None
    g2 = int(bool(mo and so))
    g3 = int(bool(g2 and (rs is None or rs != rs or rs >= RANK_MIN)))
    if g2 != g3:
        return 2 if d == g2 else (3 if d == g3 else None)
    return None                                           # this row cannot tell them apart


# ── features ─────────────────────────────────────────────────────────────────────────────────
def descriptors(row: dict) -> Optional[dict]:
    """Pull the four descriptors out of a pool / c_motif row. None if any is unusable."""
    try:
        d = {"disjointed_parts": float(row["disjointed_parts"]),   # d = the four descriptors
             "motif_length": float(row["motif_length"]),
             "mean_rsa": float(row["mean_rsa"]),
             "dominant_sse": str(row["dominant_sse"]).strip().upper()[:1] or "C"}
    except (KeyError, TypeError, ValueError):
        return None
    if any(v != v for v in (d["disjointed_parts"], d["motif_length"], d["mean_rsa"])):
        return None
    return d


def featurise(length, disjoint, sse, rsa, mu: Dict[str, float], sd: Dict[str, float]) -> Dict[str, float]:
    """Descriptors -> the 6-column design row, z-scored with the FITTING pool's mu/sd."""
    z = {c: (v - mu[c]) / sd[c] for c, v in
         [("disjointed_parts", float(disjoint)), ("motif_length", float(length)), ("mean_rsa", float(rsa))]}
    s = str(sse).strip().upper()[:1]                     # s = first letter of the DSSP class: H, E or C
    return {"disjointed_parts_z": z["disjointed_parts"],
            "motif_length_z": z["motif_length"],
            "len_x_disj": z["motif_length"] * z["disjointed_parts"],
            "H": 1.0 if s == "H" else 0.0,
            "E": 1.0 if s == "E" else 0.0,
            "mean_rsa_z": z["mean_rsa"]}


def standardisation(rows: List[dict]) -> tuple:
    """(mu, sd) over the fitting pool. sd falls back to 1.0 for a degenerate column."""
    mu, sd = {}, {}                                      # per-feature mean and standard deviation
    for c in CONT_FEATURES:
        v = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        n = len(v)
        m = sum(v) / n
        var = sum((x - m) ** 2 for x in v) / (n - 1) if n > 1 else 0.0
        mu[c], sd[c] = m, (math.sqrt(var) or 1.0)
    return mu, sd


# ── scoring ──────────────────────────────────────────────────────────────────────────────────
def predict_one(x: Dict[str, float], spec: dict) -> tuple:
    """(p_hat, {feature: log-odds contribution}) for one method. Contributions + intercept == logit."""
    coef = spec["coef"]
    contrib = {f: coef.get(f, 0.0) * x[f] for f in FINAL_FEATURES}
    logit = coef.get("const", 0.0) + sum(contrib.values())
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit)))), contrib


def score_all(x: Dict[str, float], model: dict) -> Dict[str, dict]:
    """Every method: p_hat, per-feature contributions, and cost per designable design.

    COST. `gen_gpuh_per_design` is the only GENERATION term, and it is what differs between
    methods. The AF3 refold is the same shared job, priced as one constant PER FOLD and multiplied
    by the method's own `seqs_per_design`: a backbone method refolds 8 MPNN sequences per design,
    a co-design method refolds 1. Pricing the refold per DESIGN instead -- one flat constant for
    everyone -- overcharges the co-design arms 8x, which is what a flat 0.0444 GPU-h/design basis
    did. The refold term still dominates for backbone methods (~0.045 vs 0.0005-0.0085 GPU-h), so
    their total-cost ranking stays close to their accuracy ranking; the co-design arms are
    genuinely cheaper per design because they skip MPNN entirely.
    """
    per_fold = model.get("refold_gpuh_per_fold", 0.0)
    out = {}
    for meth, spec in model["methods"].items():
        p, contrib = predict_one(x, spec)
        gen = spec.get("gen_gpuh_per_design") or 0.0
        refold = per_fold * (spec.get("seqs_per_design") or 1.0)
        total = gen + refold
        out[meth] = {"p_hat": p, "contrib": contrib, "label": spec.get("label", meth),
                     "gen_gpuh_per_design": gen, "refold_gpuh_per_design": refold,
                     "gpuh_per_design": total,
                     "gen_gpuh_per_success": gen / max(p, 1e-9),
                     "gpuh_per_success": total / max(p, 1e-9),
                     "cost_tasks_covered": spec.get("cost_tasks_covered"),
                     "cost_tasks_total": spec.get("cost_tasks_total")}
    return out


def p_at_least_one(p_hat: float, n: int) -> float:
    """P(>= 1 designable design in n samples)."""
    return 1.0 - (1.0 - p_hat) ** n


# ── model I/O — this schema IS the contract predict_motif.py reads ───────────────────────────
def save_model(path: str, *, mu, sd, methods: Dict[str, dict], refold_gpuh_per_fold: float,
               meta: Optional[dict] = None) -> dict:
    model = {"schema_version": SCHEMA_VERSION,
             "cont_features": CONT_FEATURES,
             "final_features": FINAL_FEATURES,
             "gates": {"motif_rmsd_max": MOTIF_RMSD_MAX, "sc_rmsd_max": SC_RMSD_MAX,
                       "rank_min": RANK_MIN},
             "mu": mu, "sd": sd,
             "refold_gpuh_per_fold": refold_gpuh_per_fold,
             "methods": methods,
             "meta": meta or {}}
    with open(path, "w") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
    return model


def load_model(path: str) -> dict:
    with open(path) as fh:
        m = json.load(fh)
    for k in ("mu", "sd", "methods", "final_features"):
        if k not in m:
            raise ValueError(f"{path}: not a success model — missing '{k}'")
    missing = [c for c in CONT_FEATURES if c not in m["mu"] or c not in m["sd"]]
    if missing:
        raise ValueError(f"{path}: standardisation missing for {missing}")
    if m.get("final_features") != FINAL_FEATURES:
        raise ValueError(f"{path}: feature set {m.get('final_features')} != {FINAL_FEATURES}")
    for meth, spec in m["methods"].items():
        if "coef" not in spec:
            raise ValueError(f"{path}: method '{meth}' has no coef block")
    return m
