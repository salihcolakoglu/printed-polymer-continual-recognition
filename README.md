# A31 computational materials

The A31 source-average analysis and reproducibility materials are in [A_v31](A_v31/), with independently extractable files and instructions in [the GitHub distribution](A_v31/github_web/). The A31 version-specific archive is [Zenodo v31.0](https://doi.org/10.5281/zenodo.23720507). The earlier v30 description follows.

---

# Printed-polymer continual recognition — A30 materials

By **Salih Çolakoğlu and Mustafa Onat**, Marmara University.

Browse the [source and package guide](A_v30/README.md). Download the [complete supplement ZIP](https://github.com/salihcolakoglu/printed-polymer-continual-recognition/releases/download/v30.0/A30_Supplement_S1_v1.zip), or download [part 1](https://github.com/salihcolakoglu/printed-polymer-continual-recognition/raw/refs/tags/v30.0/A_v30/distribution/A30_Supplement_S1_v1_part01.zip) and [part 2](https://github.com/salihcolakoglu/printed-polymer-continual-recognition/raw/refs/tags/v30.0/A_v30/distribution/A30_Supplement_S1_v1_part02.zip) with the [transport index](https://github.com/salihcolakoglu/printed-polymer-continual-recognition/blob/v30.0/A_v30/distribution/A30_TRANSPORT_INDEX.json). Extract both independently extractable parts into the same empty directory; do not concatenate them. Both routes contain the same accepted 2,386-file scientific package.

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

The exact [GitHub release is v30.0](https://github.com/salihcolakoglu/printed-polymer-continual-recognition/releases/tag/v30.0). Cite its verified version-specific archive DOI: **[10.5281/zenodo.23474663](https://doi.org/10.5281/zenodo.23474663)**. The immutable release tag identifies commit `cb88f13f26a7ea68562db4f815e88037720142b2`. This later main-branch documentation update adds the assigned DOI without changing that tag or the archived scientific files.

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
