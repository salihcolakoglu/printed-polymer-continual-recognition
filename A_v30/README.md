# A30 computational materials

This folder exposes the MATLAB/Python source for inspection. Files under
`03_MATLAB_Kodlari/` and `tools/` preserve their names, subpaths and exact bytes
from the complete A30 supplement. This readable selection is not a standalone
dataset/model installation. Its evaluation scripts require the full supplement's
inputs, saved states, protocols and directory layout.

The full scientific payload is distributed under [`distribution/`](distribution/).
Read its `A30_TRANSPORT_INDEX.json` and checksums, download all independently extractable
ZIP parts and extract them into the same empty directory. Every part uses the
common `A30_Supplement_S1_v1/` prefix. Use that reconstructed directory's README,
dependency list, scientific manifest and model index before running a checker.
The complete ZIP may also be attached to GitHub release v30.0 for convenience.

The tracked ZIP parts ensure the tagged source tree contains the complete
supplement, rather than depending solely on an extra release asset being included
by a DOI archive. Actual Zenodo archival and the assigned version DOI must still
be verified. The all-versions DOI is10.5281/zenodo.23289832; the historical v27 DOI
10.5281/zenodo.23293866 does not identify this new group-heldout/model-parity bank.

The new inference verification covers47,502source-record decisions across2,610
logical populations, with the accepted learning states unchanged. It establishes
numerical implementation parity. Source averages and repeated computational views
are not independent specimens or new physical validation.

Original code is MIT under the repository's LICENSE_CODE.txt. Authored scientific
artifacts are CC BY4.0 within the authors' rights under LICENSE_SCOPE.json.
Retain METHOD_PROVENANCE.txt and both third-party notices here and in the full
supplement. MATLAB/Simulink proprietary software is not included.
