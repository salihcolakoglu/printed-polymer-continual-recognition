# Printed-polymer continual recognition — A30 materials

By **Salih Çolakoğlu and Mustafa Onat**, Marmara University.

Browse the [source and package guide](A_v30/README.md). The complete scientific
supplement has **2,386 files** and is distributed as [two independently extractable
ZIP parts](A_v30/distribution/A30_TRANSPORT_INDEX.json). Download both parts and
extract them into the same empty directory. The transport index supplies byte
sizes and SHA-256 hashes; the reconstructed scientific manifest identifies every
member. The full ZIP attached to release v30.0 is an alternative download of
the same accepted package.

This computational release preserves the original fixed-allocation comparisons
and supplies the five-fold group-heldout task-incremental and class-incremental
comparisons, together with saved states, predictions and plotting data. The
comparisons include simple references, Replay, EWC, LwF and Walsh-negotiated
configurations. The package retains the methods, random initializations,
ties, nonselections and distinct evaluation protocols.

## Simulink inference and reproducibility

Five saved Simulink inference banks reproduce the accepted MATLAB source-record
decisions without retraining. Validation covers **47,502 decisions** across
**2,610 logical evaluation populations**, including matched and simple-reference
states, eligible masks, decisions, ties and native coverage/error points. The
largest observed absolute numerical difference was **9.77 × 10⁻¹⁵**. The saved
models were cold-reopened and their semantic state read back for comparison.
This establishes implementation parity for these stored inputs and states.

The complete supplement contains the scientific protocols, source records,
learning code, model states, predictions, Simulink models/traces, figures and
their numerical sources, file-level licence scope, and reproducibility evidence.
Use its manifest and model index to identify the exact files. If distribution
uses several independently extractable ZIP parts, the transport index lists
every part and its SHA-256; extract all parts into the same empty directory.

Saved-state evaluation and model-inference checking do not require new learning.
Fresh training is a separate operation with reviewed source access, dependencies,
resource limits and fresh output locations. Consult the supplied reproduction
instructions for the exact checked entrypoints. MATLAB and Simulink are proprietary
dependencies and are not redistributed.

## Data and interpretation

The cohort contains **104 published average records in 26 source-condition groups**.
Classifier inputs are tensile modulus and Shore D hardness. The averages do not
identify independent specimens. All records had been inspected before the
exploratory group-heldout study; this is not a new blind test. Task views and
random initializations are repeated computations. Numerical implementation
parity supplies no new physical validation or algorithmic accuracy result.

## Versions and citation

The exact GitHub release is **v30.0**. Its version-specific Zenodo DOI is assigned
when the linked release is archived; use the actual DOI displayed by Zenodo once
available. Until then, use the release URL and `CITATION.cff`. No old version DOI
identifies the new package.

The [all-versions DOI](https://doi.org/10.5281/zenodo.23289832) identifies the archive
family. The [v27.0 DOI](https://doi.org/10.5281/zenodo.23293866) preserves the original
fixed-allocation materials and excludes these new group-heldout results and A30
inference validation. The [v23.0 DOI](https://doi.org/10.5281/zenodo.23289833) preserves
its earlier reference-comparison component. Old tags and archives remain unchanged.
This computational release does not assert journal publication of the manuscript.

## Reuse and attribution

Original software code is **MIT**. The authors' original scientific figures,
documentation, computational outcomes, learned states and authored model artifacts
are **CC BY 4.0 within the authors' rights**. `LICENSE_SCOPE.json` and the supplement
define the separate file groups. Original code embedded in authored Simulink
artifacts retains its MIT terms. Source material and external notices retain
their original conditions. No MathWorks software licence is granted by this release.

Retain the [original dataset](https://doi.org/10.17632/zd6td6svd6.2),
[source article](https://doi.org/10.3390/mi17070859), method provenance and third-party
notices. The negotiated-representations source is credited as a preprint, separately
from journal sources. Academic software citation is requested without adding a
restriction to MIT. Unpublished manuscripts, private reviews and correspondence
are excluded from this computational release.
