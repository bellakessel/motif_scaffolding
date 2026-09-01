#!/usr/bin/env python3
"""parse_chains.py — drop-in for ProteinMPNN's helper_scripts/parse_multiple_chains.py, but O(files)
I/O instead of O(files x 352).

The stock parser re-opens and re-reads every PDB ONCE PER CHAIN LETTER in a 352-entry alphabet
(init 52 + extra 300). At library scale (thousands of single-chain designs on a networked FS) that is
millions of redundant file reads and the parse step appears to hang. Here we read each file's lines ONCE,
then probe chains over the in-memory lines — identical JSONL schema (name / num_of_chains / seq /
seq_chain_<L> / coords_chain_<L>{N,CA,C,O or CA}), just ~350x fewer reads.

    python parse_chains.py --input_path <dir_of_pdbs> --output_path <pdbs.jsonl> [--ca_only]
"""
import argparse


def main(args):
    import numpy as np
    import os, json, glob

    ca_only = args.ca_only
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']
    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}

    def N_to_AA(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    def parse_lines(lines, atoms, chain):
        """SAME logic as parse_PDB_biounits, but over pre-read lines (no file open here)."""
        xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
        for line in lines:
            if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")
            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12:12 + 4].strip()
                    resi = line[17:17 + 3]
                    resn = line[22:22 + 5].strip()
                    x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]
                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1
                    if resn < min_resn:
                        min_resn = resn
                    if resn > max_resn:
                        max_resn = resn
                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi
                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])
        seq_, xyz_ = [], []
        try:
            for resn in range(int(min_resn), int(max_resn) + 1):
                if resn in seq:
                    for k in sorted(seq[resn]):
                        seq_.append(aa_3_N.get(seq[resn][k], 20))
                else:
                    seq_.append(20)
                if resn in xyz:
                    for k in sorted(xyz[resn]):
                        for atom in atoms:
                            if atom in xyz[resn][k]:
                                xyz_.append(xyz[resn][k][atom])
                            else:
                                xyz_.append(np.full(3, np.nan))
                else:
                    for atom in atoms:
                        xyz_.append(np.full(3, np.nan))
            return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
        except TypeError:
            return 'no_chain', 'no_chain'

    folder = args.input_path
    if folder[-1] != '/':
        folder += '/'
    init_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    extra_alphabet = [str(i) for i in range(300)]
    chain_alphabet = init_alphabet + extra_alphabet
    sidechain_atoms = ['CA'] if ca_only else ['N', 'CA', 'C', 'O']

    pdb_dict_list = []
    for biounit in glob.glob(folder + '*.pdb'):
        with open(biounit, "rb") as fh:                       # <-- read ONCE
            lines = [ln.decode("utf-8", "ignore").rstrip() for ln in fh]
        present = {ln[21:22] for ln in lines if ln[:4] == "ATOM"} \
                | {ln[21:22] for ln in lines if ln[:6] == "HETATM" and ln[17:20] == "MSE"}
        my_dict, s, concat_seq = {}, 0, ''
        for letter in chain_alphabet:
            if letter not in present:                         # skip absent chains (no re-read, no work)
                continue
            xyz, seq = parse_lines(lines, sidechain_atoms, letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                cc = {}
                if ca_only:
                    cc['CA_chain_' + letter] = xyz.tolist()
                else:
                    cc['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    cc['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    cc['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    cc['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = cc
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s < len(chain_alphabet):
            pdb_dict_list.append(my_dict)

    with open(args.output_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_path", type=str, help="folder with pdb files")
    ap.add_argument("--output_path", type=str, help="where to save the parsed .jsonl")
    ap.add_argument("--ca_only", action="store_true", default=False, help="parse a Cα-only backbone")
    main(ap.parse_args())
