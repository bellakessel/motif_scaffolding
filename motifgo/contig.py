"""contig.py — turn a motif residue list into an RFdiffusion contig string.

The three user-facing "scaffolding approaches" are a single `mode` knob over the same
residue list:

  as_is           natural consecutive runs -> one fragment each, joined by `/{linker}/`
  reduce_disjoint  merge runs whose sequence gap <= gap_merge (fill the gap) -> fewer fragments
  full_stretch     one fragment spanning min..max of the motif (e.g. HER2 C557-607)

Output looks like  [2-10/C557-561/1-5/C570-573/.../35-75]  with contigmap.length=70-110.

Pure-python (no biotite / no PDB read), so `build-contig` runs anywhere for verification.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional

Run = Tuple[int, int]  # inclusive (start, end)


def group_runs(residues: List[int]) -> List[Run]:
    """Group a sorted, de-duplicated residue list into consecutive (start, end) runs."""
    res = sorted(set(int(r) for r in residues))          # res = the motif residues, sorted and unique
    if not res:
        return []
    runs: List[Run] = []                                 # runs = (first, last) of each unbroken stretch
    start = prev = res[0]                                # start = stretch opener; prev = last id accepted
    for r in res[1:]:
        if r == prev + 1:
            prev = r
        else:
            runs.append((start, prev))
            start = prev = r
    runs.append((start, prev))
    return runs


def merge_gaps(runs: List[Run], gap_merge: int) -> List[Run]:
    """Merge adjacent runs separated by a sequence gap of <= gap_merge residues.

    gap between (a,b) and (c,d) is  c - b - 1  (number of residues strictly between).
    Merging fills that gap so the fragment becomes continuous (c-b-1 <= gap_merge).
    """
    if not runs:
        return []
    merged = [runs[0]]
    for c, d in runs[1:]:                                # (c, d) = the next fragment being considered
        pa, pb = merged[-1]                              # (pa, pb) = the fragment we are currently GROWING
        if c - pb - 1 <= gap_merge:                      # residues strictly between pb and c
            merged[-1] = (pa, d)  # fill the gap
        else:
            merged.append((c, d))
    return merged


def merge_gaps_spatial(runs: List[Run], gap_merge: int, *,
                       coords_of=None, max_dist: float | None = None) -> List[Run]:
    """`merge_gaps` with an optional SPATIAL guard: a merge is rejected if any residue in the
    gap lies farther than `max_dist` (A) from the residues already in the motif.

    Why this is optional, and why it is a no-op today. `merge_gaps` compares residue NUMBERS, not
    coordinates. Measured across the whole motif library (5,532 inter-fragment gap residues), the
    minimum heavy-atom distance from a gap residue to the motif is:

        gap <= 5 residues :  median 1.33-3.07 A, max 6.89 A,  0.0% farther than 7 A
        gap    6 residues :  max 7.73 A,                      5.1% farther than 7 A
        gap   10 residues :  max 13.33 A,                    42.5% farther than 7 A

    The 1.33 A floor is the peptide C-N bond -- a gap residue adjacent to a motif residue is
    covalently bonded to it. So below gap_merge ~5 the sequence criterion and the spatial criterion
    agree completely, and plain `merge_gaps` is already correct. Above ~6 a long sequence excursion
    can loop away in space, and that is where this guard earns its keep -- and where a
    disjointedness predictor would plug in.

    `coords_of(res_id) -> (n,3) array` supplies heavy-atom coordinates; pass
    `motifgo.motif_define`-derived coordinates, or omit both arguments for plain `merge_gaps`.
    """
    if coords_of is None or max_dist is None:
        return merge_gaps(runs, gap_merge)
    import numpy as np
    if not runs:
        return []
    merged = [runs[0]]
    motif_xyz = [coords_of(r) for a, b in [runs[0]] for r in range(a, b + 1)]
    for c, d in runs[1:]:
        pa, pb = merged[-1]
        gap = list(range(pb + 1, c))
        ok = (c - pb - 1) <= gap_merge
        if ok and gap:
            ref = np.vstack([x for x in motif_xyz if x is not None and len(x)])
            for g in gap:
                gx = coords_of(g)
                if gx is None or len(gx) == 0:
                    ok = False           # no coordinates -> cannot vouch for it -> do not merge
                    break
                if np.sqrt(((gx[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min() > max_dist:
                    ok = False
                    break
        if ok:
            merged[-1] = (pa, d)
            motif_xyz.extend(coords_of(r) for r in range(pb + 1, d + 1))
        else:
            merged.append((c, d))
            motif_xyz.extend(coords_of(r) for r in range(c, d + 1))
    return merged


def build_fragments(
    residues: List[int],
    mode: str,
    gap_merge: int = 4,
    full_span: Optional[Run] = None,
) -> List[Run]:
    """Return the list of (start, end) motif fragments for the chosen mode."""
    runs = group_runs(residues)
    if not runs:
        raise ValueError("no motif residues to build a contig from")
    if mode == "as_is":
        return runs
    if mode == "reduce_disjoint":
        return merge_gaps(runs, gap_merge)
    if mode == "full_stretch":
        if full_span is not None:
            return [tuple(int(x) for x in full_span)]  # explicit span, e.g. HER2 (557, 607)
        return [(runs[0][0], runs[-1][1])]
    raise ValueError(f"unknown mode {mode!r} (as_is | reduce_disjoint | full_stretch)")


def split_at_gaps(fragments: List[Run], present: set) -> List[Run]:
    """Split each fragment so it contains only residues present in the PDB — RFdiffusion needs
    coordinates for EVERY residue in a contig range, so a fragment spanning a crystallographic gap
    (e.g. HER2 C557-607 across the missing 581-590 loop) must break into present runs (C557-580,
    C591-607). A no-op when no fragment spans a gap."""
    out: List[Run] = []
    for a, b in fragments:
        keep = [r for r in range(a, b + 1) if r in present]   # keep = the span minus any missing density
        out.extend(group_runs(keep))
    return out


def parse_range(tok: str):
    """'2-10' -> (2, 10);  '5' -> (5, 5).  A contig token is always a range."""
    if "-" in tok:
        a, b = tok.split("-", 1)
        return int(a), int(b)
    return int(tok), int(tok)


def length_bounds(fragments: List[Run], flank_n: str, flank_c: str, linker: str):
    """Total scaffold-length bounds implied by a contig: fixed motif residues + the range tokens.

        min = motif_fixed + flank_n.lo + flank_c.lo + linker.lo * (n_fragments - 1)
        max = likewise with .hi

    Lives here, not in each generator's driver: four copies of this used to exist, and a mismatch
    between any of them and the stager silently changes what a method is asked to build.
    """
    motif_fixed = sum(b - a + 1 for a, b in fragments)   # residues the generator may NOT change
    (nlo, nhi), (clo, chi), (llo, lhi) = parse_range(flank_n), parse_range(flank_c), parse_range(linker)
    #  n/c/l = N-flank, C-flank, linker;  lo/hi = the low and high end of each range token
    nlink = max(0, len(fragments) - 1)                   # nlink = linkers needed to join the fragments
    return (motif_fixed + nlo + clo + llo * nlink,
            motif_fixed + nhi + chi + lhi * nlink)


def contig_string(
    chain: str,
    fragments: List[Run],
    flank_n: str = "2-10",
    flank_c: str = "35-75",
    linker: str = "1-5",
) -> str:
    """Assemble the RFdiffusion contig string from motif fragments.

    A single-residue fragment MUST be written as a range (C579-579); RFdiffusion's
    expand_sampled_mask does int(subcon.split('-')[1]) and IndexErrors on a bare 'C579'.
    """
    frags = [f"{chain}{a}-{b}" for a, b in fragments]  # a==b -> "C579-579" (range form)
    body = f"/{linker}/".join(frags)                   # body = fragments joined by de-novo linkers
    return f"{flank_n}/{body}/{flank_c}"


def disjointedness(fragments: List[Run]) -> int:
    """Number of inter-fragment breaks (0 = single continuous fragment)."""
    return max(0, len(fragments) - 1)


def make_manifest(
    *,
    target: str,
    chain: str,
    mode: str,
    source: str,
    centroids: List[int],
    selected_residues: List[int],
    fragments: List[Run],
    contig: str,
    length_range: str,
    gap_merge: int,
    neighborhood: Dict,
    rsa_cutoff: float | None = None,
    rsa_by_res: Dict[int, float] | None = None,
) -> Dict:
    """Everything a downstream stage or a human needs to reproduce this contig."""
    # residues that appear in the contig fragments but were NOT in the motif = filler that
    # reduce_disjoint/full_stretch adds to bridge gaps (the fidelity cost of being less disjoint).
    motif_set = set(int(r) for r in selected_residues)   # what the motif definition actually selected
    in_fragments = {r for a, b in fragments for r in range(a, b + 1)}   # everything the contig now spans
    filled = sorted(in_fragments - motif_set)            # filled = residues added purely to bridge gaps
    return {
        "target": target,
        "chain": chain,
        "mode": mode,
        "source": source,                       # sasa | explicit | partner
        "neighborhood": neighborhood,           # {kind: contact|radius, cutoff/radius, metric}
        "rsa_cutoff": rsa_cutoff,               # only for source == sasa
        "gap_merge": gap_merge if mode == "reduce_disjoint" else None,
        "centroids": list(centroids),           # surface centroids (sasa) or seed residues
        "n_centroids": len(centroids),
        "selected_residues": list(selected_residues),
        "n_motif_residues": len(selected_residues),
        "filled_residues": filled,              # non-motif residues pulled in to bridge fragments
        "n_filled": len(filled),
        "fragments": [{"start": a, "end": b, "length": b - a + 1} for a, b in fragments],
        "n_fragments": len(fragments),
        "disjointedness": disjointedness(fragments),
        "motif_length": sum(b - a + 1 for a, b in fragments),
        "contig": contig,
        "length_range": length_range,
        # centroid RSA (rounded) for reference; empty unless sasa
        "centroid_rsa": {str(c): round(rsa_by_res.get(c, 0.0), 3) for c in centroids} if rsa_by_res else {},
    }
