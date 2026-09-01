#!/usr/bin/env python3
"""dssp.py — DSSP secondary-structure assignment, pure NumPy, no dependencies beyond NumPy.

WHY VENDORED. This is pydssp 0.9.1's `pydssp_numpy` backend (Shintaro Minami, MIT licence,
https://github.com/ShintaroMinami/PyDSSP), reproduced here with two mechanical changes:

  1. `import torch` is gone. pydssp's package __init__ imports torch unconditionally, even for
     the NumPy code path, so `pip install pydssp` into gen_env would pull ~2 GB of torch to
     reach 113 lines of NumPy. gen_env has no torch.
  2. `einops.repeat` / `einops.rearrange` are replaced by plain NumPy broadcasting, indexing and
     `swapaxes`. gen_env has no einops either. The einops `repeat`s in `get_hbond_map` only
     materialised arrays that the very next line broadcast against each other anyway, so they
     are simply dropped in favour of `[:, :, None, :]` / `[:, None, :, :]` indexing.

The scientific content -- the electrostatic H-bond energy (Kabsch & Sander 1983), the turn/helix
priority rules and the bridge/ladder logic -- is UNCHANGED. `selftest` below asserts this module
is bit-identical to upstream pydssp on real structures; it was run against pydssp 0.9.1 before
this file was committed. If you ever touch the maths here, re-run it.

Only the 3-state alphabet is provided ('-' coil, 'H' helix, 'E' strand), because pydssp 0.9.1
itself offers only c3 -- there is no 8-state DSSP output to be had from this implementation.

    from motifgo.dssp import assign_c3
    sse = assign_c3(coord)      # coord (L, 4, 3) float, atom order N, CA, C, O
"""
from __future__ import annotations

import numpy as np

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0
C3_ALPHABET = np.array(['-', 'H', 'E'])


def _unfold(a: np.ndarray, window: int, axis: int):
    idx = np.arange(window)[:, None] + np.arange(a.shape[axis] - window + 1)[None, :]
    unfolded = np.take(a, idx, axis=axis)
    return np.moveaxis(unfolded, axis - 1, -1)


def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape) == 3) or (len(org_shape) == 4), \
        "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = coord[None, ...] if len(org_shape) == 3 else coord      # einops: repeat('... -> b ...', b=1)
    return coord, org_shape


def _get_hydrogen_atom_position(coord: np.ndarray) -> np.ndarray:
    # A little bit lazy (but should be OK) definition of H position here.
    vec_cn = coord[:, 1:, 0] - coord[:, :-1, 2]
    vec_cn = vec_cn / np.linalg.norm(vec_cn, axis=-1, keepdims=True)
    vec_can = coord[:, 1:, 0] - coord[:, 1:, 1]
    vec_can = vec_can / np.linalg.norm(vec_can, axis=-1, keepdims=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / np.linalg.norm(vec_nh, axis=-1, keepdims=True)
    return coord[:, 1:, 0] + 1.01 * vec_nh


def get_hbond_map(coord: np.ndarray, donor_mask=None, cutoff: float = DEFAULT_CUTOFF,
                  margin: float = DEFAULT_MARGIN, return_e: bool = False) -> np.ndarray:
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    assert (a == 4) or (a == 5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:, 1:, 4] if a == 5 else _get_hydrogen_atom_position(coord)

    # distance matrix. einops repeat()s dropped: index m on axis -3, index n on axis -2, then
    # let NumPy broadcast -- identical result, no (b, l-1, l-1, 3) temporaries.
    nmap = coord[:, 1:, 0][:, :, None, :]        # (b, m, 1, c)
    hmap = h[:, :, None, :]                      # (b, m, 1, c)
    cmap = coord[:, 0:-1, 2][:, None, :, :]      # (b, 1, n, c)
    omap = coord[:, 0:-1, 3][:, None, :, :]      # (b, 1, n, c)
    d_on = np.linalg.norm(omap - nmap, axis=-1)
    d_ch = np.linalg.norm(cmap - hmap, axis=-1)
    d_oh = np.linalg.norm(omap - hmap, axis=-1)
    d_cn = np.linalg.norm(cmap - nmap, axis=-1)

    # electrostatic interaction energy
    e = np.pad(CONST_Q1Q2 * (1. / d_on + 1. / d_ch - 1. / d_oh - 1. / d_cn) * CONST_F,
               [[0, 0], [1, 0], [0, 1]])
    if return_e:
        return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~np.eye(l, dtype=bool)
    local_mask *= ~np.diag(np.ones(l - 1, dtype=bool), k=-1)
    local_mask *= ~np.diag(np.ones(l - 2, dtype=bool), k=-2)
    # mask for donor H absence (Proline). einops made this (l1, l2) constant in l2, then (b, l1, l2).
    dm = np.array(donor_mask).astype(float) if donor_mask is not None else np.ones(l, dtype=float)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = np.clip(cutoff - margin - e, a_min=-margin, a_max=margin)
    hbond_map = (np.sin(hbond_map / margin * np.pi / 2) + 1.) / 2
    hbond_map = hbond_map * local_mask[None, :, :]
    hbond_map = hbond_map * dm[None, :, None]
    hbond_map = np.squeeze(hbond_map, axis=0) if len(org_shape) == 3 else hbond_map
    return hbond_map


def assign(coord: np.ndarray, donor_mask=None) -> np.ndarray:
    """One-hot [loop, helix, strand] per residue."""
    coord, org_shape = _check_input(coord)
    hbmap = get_hbond_map(coord, donor_mask=donor_mask)
    hbmap = np.swapaxes(hbmap, -1, -2)          # convert into "i:C=O, j:N-H" form
    # identify turn 3, 4, 5
    turn3 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=3) > 0.
    turn4 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=4) > 0.
    turn5 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = np.pad(turn3[:, :-1] * turn3[:, 1:], [[0, 0], [1, 3]])
    h4 = np.pad(turn4[:, :-1] * turn4[:, 1:], [[0, 0], [1, 4]])
    h5 = np.pad(turn5[:, :-1] * turn5[:, 1:], [[0, 0], [1, 5]])
    # helix4 first
    helix4 = h4 + np.roll(h4, 1, 1) + np.roll(h4, 2, 1) + np.roll(h4, 3, 1)
    h3 = h3 * ~np.roll(helix4, -1, 1) * ~helix4     # helix4 is higher prioritized
    h5 = h5 * ~np.roll(helix4, -1, 1) * ~helix4     # helix4 is higher prioritized
    helix3 = h3 + np.roll(h3, 1, 1) + np.roll(h3, 2, 1)
    helix5 = h5 + np.roll(h5, 1, 1) + np.roll(h5, 2, 1) + np.roll(h5, 3, 1) + np.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0.
    unfoldmap_rev = np.swapaxes(unfoldmap, 1, 2)
    p_bridge = ((unfoldmap[:, :, :, 0, 1] * unfoldmap_rev[:, :, :, 1, 2])
                + (unfoldmap_rev[:, :, :, 0, 1] * unfoldmap[:, :, :, 1, 2]))
    p_bridge = np.pad(p_bridge, [[0, 0], [1, 1], [1, 1]])
    a_bridge = ((unfoldmap[:, :, :, 1, 1] * unfoldmap_rev[:, :, :, 1, 1])
                + (unfoldmap[:, :, :, 0, 2] * unfoldmap_rev[:, :, :, 0, 2]))
    a_bridge = np.pad(a_bridge, [[0, 0], [1, 1], [1, 1]])
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # H, E, L of C3
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    onehot = np.stack([loop, helix, strand], axis=-1)
    onehot = onehot[0] if len(org_shape) == 3 else onehot
    return onehot


def assign_c3(coord: np.ndarray, donor_mask=None) -> str:
    """(L, 4, 3) N/CA/C/O coordinates -> a length-L string over {'-', 'H', 'E'}."""
    onehot = assign(coord, donor_mask=donor_mask)
    return "".join(C3_ALPHABET[np.argmax(onehot, axis=-1)])


def selftest(paths, verbose=True) -> bool:
    """Assert this module reproduces upstream pydssp EXACTLY on real structures.

    Run wherever pydssp (and therefore torch) happens to be installed -- it is a development
    check, not a runtime dependency:  python dssp_numpy.py selftest <file.cif> [...]
    """
    import pydssp                                          # noqa: F401  (dev-only)
    from scripts.analysis.cif_backbone import backbone_coords  # dev-only helper
    n_res = n_ok = 0
    for p in paths:
        coord, _ = backbone_coords(p)
        mine = assign_c3(coord)
        theirs = "".join(pydssp.assign(coord, out_type="c3"))
        n_res += len(theirs)
        n_ok += sum(a == b for a, b in zip(mine, theirs))
        if mine != theirs and verbose:
            print(f"MISMATCH {p}\n  mine   {mine}\n  pydssp {theirs}")
    ok = (n_ok == n_res)
    if verbose:
        print(f"selftest: {len(paths)} structures, {n_res} residues, "
              f"{'IDENTICAL' if ok else f'{n_res - n_ok} DIFFER'}")
    return ok




# ---------------------------------------------------------------------------
# biotite bridge (absorbed from esp/_dssp.py — the second copy of this algorithm
# that used to live in the pipeline; see plan F9). These two functions are what
# motif_build._sse_by_res calls; everything above is the vendored pydssp maths.
# ---------------------------------------------------------------------------

def _backbone_coord(arr):
    """(res_ids, (L,4,3) N,CA,C,O per residue in chain order). Missing atom -> CA-filled (local only)."""
    import biotite.structure as struc
    starts = struc.get_residue_starts(arr)
    res_ids = arr.res_id[starts]
    coord = np.zeros((len(res_ids), 4, 3), dtype=float)
    for k, rid in enumerate(res_ids):
        r = arr[arr.res_id == rid]
        ca = r[r.atom_name == "CA"]
        cac = ca.coord[0] if len(ca) else r.coord.mean(0)
        for a_i, nm in enumerate(("N", "CA", "C", "O")):
            sel = r[r.atom_name == nm]
            coord[k, a_i] = sel.coord[0] if len(sel) else cac
    return res_ids, coord


def sse_by_res(arr):
    """Per-residue 3-state DSSP {res_id: 'H'|'E'|'C'} for a single-chain biotite AtomArray."""
    res_ids, coord = _backbone_coord(arr)
    if len(res_ids) < 6:                       # too short for turns/bridges
        return {int(r): "C" for r in res_ids}
    labels = np.array(["C", "H", "E"])[assign(coord).argmax(-1)]
    return {int(r): str(s) for r, s in zip(res_ids, labels)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "selftest":
        raise SystemExit(0 if selftest(sys.argv[2:]) else 1)
    print(__doc__)
