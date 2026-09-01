#!/usr/bin/env python3
"""
Uniform backbone normalisation for the cross-method Foldseek clustering of the
motif_library scaffolds (5 generators: rfdiff, proteina, pp1c_bb, laproteina, genie3).

WHY THIS EXISTS
---------------
The five generators write mutually incompatible PDBs (measured on
runs/lib_interface_pad5/{method}/1_filter/passed/):

    rfdiff      N,CA,C,O      GLY scaffold + native motif residues
    pp1c_bb     N,CA,C,O      GLY scaffold + native motif residues
    laproteina  N,CA,C,O + sc full designed sequence
    proteina    CA only       all ALA
    genie3      CA only       UNK + native motif (stray sidechain atoms, no N/C)

Two consequences, both fatal to a naive `foldseek createdb` over the raw files:

  1. Foldseek's 3Di alphabet is derived from the local N-CA-C frame (it needs
     those three atoms to place each residue's virtual centre). The two CA-only
     generators cannot be embedded at all, and createdb does NOT error -- it
     drops or garbage-encodes them, silently turning a 5-way comparison into a
     3-way one plus two piles of noise.

  2. Residue naming is *perfectly* correlated with the generator (poly-Gly /
     all-ALA / all-UNK / real sequence). Foldseek's default score blends 3Di
     with an amino-acid substitution term, so the methods would separate on
     their naming convention rather than on their fold. Note that
     `--min-seq-id 0.0` does NOT fix this: it drops the sequence-identity
     *filter*, not the AA term in the score.

So every structure is put through the SAME transform:

    keep the CA trace only  ->  rebuild N, C, O from the CA frame  ->  all residues GLY

Reducing the three full-backbone methods to CA as well is deliberate. It costs
some per-structure fidelity, but it guarantees that no generator gains an edge
from having happened to write real N/C or a real sequence -- and the CA trace
already determines the fold. When the question is "do the methods differ",
uniformity of treatment beats per-structure fidelity.

HOW THE REBUILD WORKS
---------------------
At each residue an orthonormal frame is built from the neighbouring CA positions
(bisector / plane-normal / cross). N, C and O are then placed at a fixed offset
expressed in that frame. The offsets are NOT magic constants: they are fit by
least squares on the real backbones that rfdiff/pp1c_bb/laproteina do provide.

A single global offset per atom only reaches ~0.45 A accuracy, because the true
N/C position in the CA frame depends on the local conformation. So, BBQ-style
(Gront et al. 2007), the offsets are stored in a lookup table binned by the CA
pseudo-angle theta (CA_i-1, CA_i, CA_i+1) and the CA pseudo-dihedral tau
(CA_i-1..CA_i+2), with graceful fallback: (theta,tau) bin -> theta-only bin ->
global mean. `selfcheck` reports the resulting error, including the error on the
virtual centre that 3Di is actually computed from.

USAGE
-----
    # 1) fit the offsets on the full-backbone methods
    python normalize_backbones.py fit \
        --lists RUN/rfdiff/1_filter/passed.txt RUN/pp1c_bb/1_filter/passed.txt \
                RUN/laproteina/1_filter/passed.txt \
        --out $SCRATCH/bb_coeffs.json --nsample 400

    # 2) how good is the rebuild? (held-out structures, reports virtual-centre error)
    python normalize_backbones.py selfcheck \
        --lists ... --coeffs $SCRATCH/bb_coeffs.json --nsample 200

    # 3) normalise one method into the shared flat output dir
    python normalize_backbones.py apply \
        --list RUN/proteina/1_filter/passed.txt --method proteina \
        --coeffs $SCRATCH/bb_coeffs.json --out-dir $SCRATCH/bb_norm \
        --stats $SCRATCH/bb_lengths.tsv --workers 16

Output files are named  {method}__{design}.pdb  so that Foldseek's structure ID
carries the generator and the design name; `design` matches the `design` column
of {method}/1_filter/motif_rmsd.csv, which joins to `task` and thence to the
run's manifest.csv (motif_num / variant / n_fragments / contig).

Driver: scripts/foldseek_backbones.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BB = ("N", "CA", "C", "O")          # written, in this order, for every residue
REBUILT = ("N", "C", "O")           # CA is kept as-is; these three are rebuilt
KINDS = ("first", "mid", "last")    # frame flavour -> its own lookup table

THETA_MIN, THETA_W, N_THETA = 60.0, 10.0, 9      # pseudo-angle bins, degrees
TAU_W, N_TAU = 15.0, 24                          # pseudo-dihedral bins, degrees
MIN_BIN = 25                                     # samples needed to trust a bin


# ── PDB I/O (minimal, deliberately dependency-free) ───────────────────────────
def parse_pdb(path):
    """-> (ordered residue keys, {key: {atom_name: xyz}}).

    First model only, first altloc only, first occurrence of a duplicated atom
    name wins. The residue key includes the chain so a stray multi-chain file
    cannot silently merge residues.
    """
    order, res = [], {}
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            if line[16] not in (" ", "A"):          # altloc
                continue
            key = (line[21], line[22:27])           # chain + resseq/icode
            if key not in res:
                res[key] = {}
                order.append(key)
            d = res[key]
            name = line[12:16].strip()
            if name not in d:
                d[name] = np.array((float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54])))
    return order, res


def write_pdb(path, xyz):
    """xyz: (n_res, 4, 3) in BB order. Everything is GLY, chain A, renumbered 1..n."""
    out, serial = [], 1
    for i in range(len(xyz)):
        for j, name in enumerate(BB):
            x, y, z = xyz[i, j]
            out.append(
                "ATOM  "
                f"{serial:5d}"
                " "
                f" {name:<3s}"
                " "
                "GLY"
                " A"
                f"{i + 1:4d}"
                "    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                "  1.00  0.00"
                "          "
                f"{name[0]:>2s}\n"
            )
            serial += 1
    out.append("TER\nEND\n")
    with open(path, "w") as fh:
        fh.writelines(out)


# ── CA-frame geometry ─────────────────────────────────────────────────────────
def _frame(o, p, q):
    """Right-handed orthonormal frame at CA `o` from two neighbour CAs `p`, `q`.

    e1 = bisector of (p-o, q-o), e3 = normal of the CA plane, e2 = e3 x e1.
    Rows of the returned 3x3 are the basis vectors, so the local coords of a
    point x are  F @ (x - o), and the inverse map is  o + F.T @ c.
    """
    v1, v2 = p - o, q - o
    e1 = v1 + v2
    n1 = np.linalg.norm(e1)
    if n1 < 1e-6:                       # antiparallel neighbours -> use v1
        e1 = v1.copy()
        n1 = np.linalg.norm(e1)
    e1 = e1 / n1
    e3 = np.cross(v1, v2)
    n3 = np.linalg.norm(e3)
    if n3 < 1e-6:                       # collinear CAs -> any normal to e1
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(float(tmp @ e1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e3 = np.cross(e1, tmp)
        n3 = np.linalg.norm(e3)
    e3 = e3 / n3
    return np.stack([e1, np.cross(e3, e1), e3])


def _angle(a, b, c):
    """Angle at vertex b, in degrees."""
    v1, v2 = a - b, c - b
    cosv = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral p0-p1-p2-p3, in degrees."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - (b0 @ b1n) * b1n
    w = b2 - (b2 @ b1n) * b1n
    return float(np.degrees(np.arctan2(float(np.cross(b1n, v) @ w), float(v @ w))))


def virtual_centre(N, CA, C):
    """The virtual CB that Foldseek's 3Di descriptor is built from (trRosetta form)."""
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def kind_of(i, n):
    return "first" if i == 0 else ("last" if i == n - 1 else "mid")


def frames_and_bins(ca):
    """-> (F, kinds, theta_bin, tau_bin) for every residue of a CA trace (n>=3)."""
    n = len(ca)
    F = np.empty((n, 3, 3))
    tb = np.zeros(n, dtype=int)
    db = np.full(n, -1, dtype=int)

    F[0] = _frame(ca[0], ca[1], ca[2])
    F[n - 1] = _frame(ca[n - 1], ca[n - 2], ca[n - 3])
    for i in range(1, n - 1):
        F[i] = _frame(ca[i], ca[i - 1], ca[i + 1])

    def theta_bin(t):
        return int(np.clip((t - THETA_MIN) // THETA_W, 0, N_THETA - 1))

    tb[0] = theta_bin(_angle(ca[0], ca[1], ca[2]))
    tb[n - 1] = theta_bin(_angle(ca[n - 1], ca[n - 2], ca[n - 3]))
    for i in range(1, n - 1):
        tb[i] = theta_bin(_angle(ca[i - 1], ca[i], ca[i + 1]))
        if i + 2 < n:
            tau = _dihedral(ca[i - 1], ca[i], ca[i + 1], ca[i + 2])
            db[i] = int(np.clip((tau + 180.0) // TAU_W, 0, N_TAU - 1))

    kinds = [kind_of(i, n) for i in range(n)]
    return F, kinds, tb, db


def ca_trace(path):
    order, res = parse_pdb(path)
    ca = [res[k]["CA"] for k in order if "CA" in res[k]]
    return np.asarray(ca) if ca else np.empty((0, 3))


def full_backbone(path):
    """-> (ca, {atom: (n,3)}) if EVERY residue has a complete N/CA/C/O, else None."""
    order, res = parse_pdb(path)
    if len(order) < 4 or not all(all(a in res[k] for a in BB) for k in order):
        return None
    ca = np.asarray([res[k]["CA"] for k in order])
    true = {a: np.asarray([res[k][a] for k in order]) for a in REBUILT}
    return ca, true


# ── offset lookup ─────────────────────────────────────────────────────────────
def lookup(coeffs, kind, tb, db, atom):
    """(theta,tau) bin -> theta-only bin -> global mean."""
    tbl = coeffs[kind]
    if db >= 0:
        hit = tbl["full"].get(f"{tb}_{db}")
        if hit is not None:
            return hit[atom]
    hit = tbl["theta"].get(str(tb))
    if hit is not None:
        return hit[atom]
    return tbl["global"][atom]


def _as_arrays(coeffs_json):
    """JSON lists -> np arrays, once per process."""
    out = {}
    for kind in KINDS:
        t = coeffs_json[kind]
        out[kind] = {
            "full": {k: {a: np.asarray(v[a]) for a in REBUILT}
                     for k, v in t["full"].items()},
            "theta": {k: {a: np.asarray(v[a]) for a in REBUILT}
                      for k, v in t["theta"].items()},
            "global": {a: np.asarray(t["global"][a]) for a in REBUILT},
        }
    return out


def rebuild(ca, coeffs):
    """CA trace -> (n,4,3) N/CA/C/O."""
    n = len(ca)
    F, kinds, tb, db = frames_and_bins(ca)
    xyz = np.empty((n, 4, 3))
    for i in range(n):
        Fi = F[i].T
        xyz[i, 0] = ca[i] + Fi @ lookup(coeffs, kinds[i], tb[i], db[i], "N")
        xyz[i, 1] = ca[i]
        xyz[i, 2] = ca[i] + Fi @ lookup(coeffs, kinds[i], tb[i], db[i], "C")
        xyz[i, 3] = ca[i] + Fi @ lookup(coeffs, kinds[i], tb[i], db[i], "O")
    return xyz


# ── fit ───────────────────────────────────────────────────────────────────────
def _sample(lists, nsample, seed, skip=0):
    paths = []
    for lst in lists:
        with open(lst) as fh:
            got = [l.strip() for l in fh if l.strip()]
        random.Random(seed).shuffle(got)
        paths += got[skip: skip + nsample]
    return paths


def cmd_fit(args):
    paths = _sample(args.lists, args.nsample, args.seed)
    print(f"fitting on up to {len(paths)} structures from {len(args.lists)} list(s)")

    # per kind: samples keyed by (theta,tau), by theta, and pooled
    acc = {k: {"full": defaultdict(list), "theta": defaultdict(list), "global": []}
           for k in KINDS}
    used = 0
    for p in paths:
        got = full_backbone(p)
        if got is None:
            continue
        used += 1
        ca, true = got
        F, kinds, tb, db = frames_and_bins(ca)
        for i in range(len(ca)):
            local = np.stack([F[i] @ (true[a][i] - ca[i]) for a in REBUILT])
            k = kinds[i]
            acc[k]["global"].append(local)
            acc[k]["theta"][str(tb[i])].append(local)
            if db[i] >= 0:
                acc[k]["full"][f"{tb[i]}_{db[i]}"].append(local)

    if used == 0:
        sys.exit("ERROR: no structure with a complete N/CA/C/O backbone in --lists")

    def mean_of(samples):
        M = np.asarray(samples).mean(axis=0)          # (3 atoms, 3 coords)
        return {a: M[j].tolist() for j, a in enumerate(REBUILT)}

    coeffs, nbins = {}, {}
    for k in KINDS:
        coeffs[k] = {
            "global": mean_of(acc[k]["global"]),
            "theta": {b: mean_of(v) for b, v in acc[k]["theta"].items()
                      if len(v) >= MIN_BIN},
            "full": {b: mean_of(v) for b, v in acc[k]["full"].items()
                     if len(v) >= MIN_BIN},
        }
        nbins[k] = (len(coeffs[k]["full"]), len(coeffs[k]["theta"]),
                    len(acc[k]["global"]))

    with open(args.out, "w") as fh:
        json.dump({**coeffs,
                   "_meta": {"n_structures_fit": used, "lists": args.lists,
                             "nsample": args.nsample, "seed": args.seed,
                             "min_bin": MIN_BIN}}, fh, indent=2)

    print(f"\nfit on {used} full-backbone structures -> {args.out}")
    print(f"{'frame':>6} {'(theta,tau) bins':>17} {'theta bins':>11} {'residues':>10}")
    for k in KINDS:
        f, t, m = nbins[k]
        print(f"{k:>6} {f:>17} {t:>11} {m:>10}")
    print("\nRun `selfcheck` to see the reconstruction error this table achieves.")


# ── selfcheck ─────────────────────────────────────────────────────────────────
def cmd_selfcheck(args):
    with open(args.coeffs) as fh:
        coeffs = _as_arrays(json.load(fh))
    # skip past the structures used for the fit so this is genuinely held out
    paths = _sample(args.lists, args.nsample, args.seed, skip=args.skip)
    print(f"held-out check on up to {len(paths)} structures (skip={args.skip})")

    err = {a: [] for a in REBUILT}
    err["virtual_centre"] = []
    ang = []
    used = 0
    for p in paths:
        got = full_backbone(p)
        if got is None:
            continue
        used += 1
        ca, true = got
        xyz = rebuild(ca, coeffs)
        for j, a in enumerate(REBUILT):
            k = {"N": 0, "C": 2, "O": 3}[a]
            err[a] += list(np.linalg.norm(xyz[:, k] - true[a], axis=1))
        for i in range(len(ca)):
            vt = virtual_centre(true["N"][i], ca[i], true["C"][i])
            vr = virtual_centre(xyz[i, 0], ca[i], xyz[i, 2])
            err["virtual_centre"].append(float(np.linalg.norm(vr - vt)))
            u = vt - ca[i]
            v = vr - ca[i]
            cosv = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            ang.append(float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))))

    if used == 0:
        sys.exit("ERROR: no held-out full-backbone structure (raise --nsample/lower --skip)")

    print(f"\nreconstruction error on {used} structures "
          f"({len(err['N'])} residues), distance in A:")
    print(f"{'quantity':>16} {'mean':>7} {'median':>7} {'p90':>7} {'max':>7}")
    for a in list(REBUILT) + ["virtual_centre"]:
        e = np.asarray(err[a])
        print(f"{a:>16} {e.mean():7.3f} {np.median(e):7.3f} "
              f"{np.percentile(e, 90):7.3f} {e.max():7.3f}")
    A = np.asarray(ang)
    print(f"\nvirtual-centre direction error (deg): mean {A.mean():.2f}  "
          f"median {np.median(A):.2f}  p90 {np.percentile(A, 90):.2f}")
    print("\nvirtual_centre is what Foldseek's 3Di descriptor is computed from, so it\n"
          "is the number that matters -- N/C/O errors only matter through it.")


# ── apply ─────────────────────────────────────────────────────────────────────
_COEFFS = None      # per-worker cache, populated by the pool initializer


def _init_worker(coeffs_path):
    global _COEFFS
    with open(coeffs_path) as fh:
        _COEFFS = _as_arrays(json.load(fh))


def _apply_one(job):
    src, dst = job
    try:
        ca = ca_trace(src)
    except Exception as exc:                      # unreadable / malformed file
        return (src, 0, f"parse: {exc}")
    n = len(ca)
    if n < 4:
        return (src, 0, f"only {n} CA (need >=4)")
    try:
        xyz = rebuild(ca, _COEFFS)
    except Exception as exc:
        return (src, 0, f"rebuild: {exc}")
    if not np.isfinite(xyz).all():
        return (src, 0, "non-finite coordinates after rebuild")
    try:
        write_pdb(dst, xyz)
    except Exception as exc:
        return (src, 0, f"write: {exc}")
    return (src, n, None)


def _design_of(path):
    b = os.path.basename(path)
    return b[:-4] if b.lower().endswith(".pdb") else b


def cmd_apply(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.list) as fh:
        srcs = [l.strip() for l in fh if l.strip()]
    if args.limit:
        srcs = srcs[: args.limit]

    jobs = [(s, os.path.join(args.out_dir, f"{args.method}__{_design_of(s)}.pdb"))
            for s in srcs]

    # a collision here would silently merge two designs into one Foldseek ID
    if len({d for _, d in jobs}) != len(jobs):
        sys.exit(f"ERROR: {args.method}: duplicate output basenames -- "
                 "Foldseek IDs must be unique across the whole DB")

    ok, lengths, fails = 0, [], []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.coeffs,)) as ex:
        for src, n, e in ex.map(_apply_one, jobs, chunksize=256):
            if e:
                fails.append((src, e))
            else:
                ok += 1
                lengths.append((_design_of(src), n))

    if args.stats:
        with open(args.stats, "a") as fh:
            for design, n in lengths:
                fh.write(f"{args.method}\t{design}\t{n}\n")

    L = np.asarray([n for _, n in lengths]) if lengths else np.array([0])
    print(f"[{args.method}] in={len(jobs)} ok={ok} failed={len(fails)}  "
          f"n_res: min={L.min()} p25={int(np.percentile(L, 25))} "
          f"median={int(np.median(L))} p75={int(np.percentile(L, 75))} "
          f"max={L.max()} mean={L.mean():.1f}")
    for src, e in fails[:20]:
        print(f"  FAIL {src}: {e}", file=sys.stderr)
    if len(fails) > 20:
        print(f"  ... and {len(fails) - 20} more failures", file=sys.stderr)
    if fails:
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit N/C/O offsets on real full backbones")
    f.add_argument("--lists", nargs="+", required=True,
                   help="passed.txt files of the FULL-BACKBONE methods")
    f.add_argument("--out", required=True, help="coefficients JSON to write")
    f.add_argument("--nsample", type=int, default=400, help="structures per list")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("selfcheck", help="held-out reconstruction error")
    s.add_argument("--lists", nargs="+", required=True)
    s.add_argument("--coeffs", required=True)
    s.add_argument("--nsample", type=int, default=200)
    s.add_argument("--skip", type=int, default=400,
                   help="skip the first N per list (the ones used for the fit)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_selfcheck)

    a = sub.add_parser("apply", help="normalise one method into the shared flat dir")
    a.add_argument("--list", required=True, help="passed.txt of this method")
    a.add_argument("--method", required=True, help="method tag used as ID prefix")
    a.add_argument("--coeffs", required=True, help="coefficients JSON from `fit`")
    a.add_argument("--out-dir", required=True, help="shared flat output dir")
    a.add_argument("--stats", help="TSV to append method/design/n_res to")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--limit", type=int, help="only the first N (smoke test)")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
