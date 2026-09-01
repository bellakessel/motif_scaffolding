"""motifgo — the shared library of the motif-scaffolding benchmark.

Six modules, one import path. Every pipeline script under `scripts/` imports from here; nothing
here imports from `scripts/`, and no module reaches outside the package by absolute path.

    motif_define   what a motif IS      — SASA surface, Ca/Cb contacts, neighbourhood expansion
    motif_build    what a motif BECOMES — sasa csv -> c_motif csv -> motif PDBs (imports motif_define)
    contig         the scaffolding instruction — group_runs, merge_gaps, contig_string, split_at_gaps
    dssp           vendored pure-numpy DSSP (3-state H/E/C)
    af3_inputs     AF3 monomer-refold JSON builder
    score          designability metric: motifRMSD <= 1.0 A AND scRMSD <= 2.0 A

Dependency direction is one-way and acyclic:
    motif_define <- motif_build -> contig, dssp        af3_inputs, score  (independent)
"""
__version__ = "0.1.0"
__all__ = ["motif_define", "motif_build", "contig", "dssp", "af3_inputs", "score"]
