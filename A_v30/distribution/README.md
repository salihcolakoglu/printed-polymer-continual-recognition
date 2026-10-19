# Complete A30 supplement

Read [`A30_TRANSPORT_INDEX.json`](A30_TRANSPORT_INDEX.json) for the exact package
identity and SHA-256 checksums. The full supplement contains 2,386 files. Download
both independently extractable archives:

- `A30_Supplement_S1_v1_part01.zip` — 24,827,382 bytes.
- `A30_Supplement_S1_v1_part02.zip` — 22,518,972 bytes.

Check each archive's size and SHA-256 against the index. Extract both into the
same new empty destination; both use the common `A30_Supplement_S1_v1/` prefix.
Their members are disjoint and collectively reproduce the entire supplement,
including all data, saved states/predictions, Simulink models/traces, protocols,
figures/numerical sources, reproduction tools and notices. Verify the reconstructed
`MANIFEST_SHA256.json` before running saved-state utilities.

The complete `A30_Supplement_S1_v1.zip` attached to release v30.0 is an alternative
download containing the same members. It is 47,346,332 bytes; its SHA-256 is also in
the transport index. The readable code under the parent folder is an inspection
selection and needs the full supplement's inputs/layout. Fresh learning is a
separate reviewed operation. Proprietary MATLAB/Simulink software is not included.

Tracked archive parts ensure the tagged source tree contains the whole package.
Actual Zenodo archival has been verified: [version DOI 10.5281/zenodo.23474663](https://doi.org/10.5281/zenodo.23474663). The archived source tree contains both parts and the exact index. All 2,386 sealed supplement members match. This later main-only guide update leaves the v30.0 tag and archived scientific files unchanged.

Original code: MIT. Authored scientific artifacts: CC BY 4.0 within the authors' rights; external terms retained.
