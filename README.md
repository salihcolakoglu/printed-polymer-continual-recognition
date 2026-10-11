# Printed-polymer continual recognition — A27 materials

By **Salih Çolakoğlu and Mustafa Onat**, Marmara University.

The complete **v27.0 computational package supporting manuscript A27** is in [`A_v27/`](A_v27/README.md). It combines reference recognizers, ordinary and Walsh code learning, negotiated targets, replay, class-incremental comparisons, saved models, two Simulink inference demonstrations, and eight manuscript figures with their numerical sources.

Start with the [package guide](A_v27/README.md), [figure map](A_v27/FIGURE_MAP.csv), [reproduction instructions](A_v27/REPRODUCIBILITY.md), and [data attribution](A_v27/DATA_SOURCES.json).

The dataset contains 104 published average records in 26 source-condition groups. The two classifier inputs are tensile modulus and Shore D hardness. Both development partitions and all five initializations remain visible. Simulink applies recorded models; MATLAB performed the learning. These are exploratory source-record comparisons.

## Version and citation

Select GitHub release **v27.0** to identify these exact files. Its version-specific Zenodo DOI is assigned when the linked release is archived; use that verified DOI once available. `CITATION.cff` supplies the authors and software citation metadata.

The [v23.0 archive](https://doi.org/10.5281/zenodo.23289833) preserves the earlier reference-comparison component. Its DOI does not identify the expanded A27 package. Root `03_Hesaplama_Eki/`, `figures/` and `FILE_MANIFEST.json` remain historical v23 materials. The current package has its own manifest in `A_v27/FILE_MANIFEST.json`. `RELEASE_MAP.json` distinguishes the versions.

## Reuse and attribution

Original code is **MIT**. Attributed data derivatives and the authors' figures, documentation, results, learned weights and model artifacts are **CC BY 4.0 within the authors' rights**. [LICENSE_SCOPE.json](A_v27/LICENSE_SCOPE.json) specifies the file groups. Third-party and MathWorks rights remain with their owners. Academic citation is requested without adding a restriction to MIT.

Retain the [original dataset](https://doi.org/10.17632/zd6td6svd6.2), [source article](https://doi.org/10.3390/mi17070859) and [method attribution](A_v27/METHOD_PROVENANCE.md) when reusing the materials.
