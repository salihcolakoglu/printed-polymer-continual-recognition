# A_v23.0 computational materials

This release accompanies **Continual Recognition of Printed Polymer Contexts from Published Average Mechanical Records**, by Salih Çolakoğlu and Mustafa Onat. It brings the original MATLAB analysis, recorded outputs and four figure source tables together for inspection and reuse. The manuscript now presents the contributions first and explains the evaluation setting in the Methods and Discussion.

## Contents

- `recorded_package_v2/` preserves the 27 original computational-supplement files byte for byte.
- `DATA_SOURCES.json` links the source article, dataset version and extracted cohort.
- `DATA_DICTIONARY.md` explains the records, condition groups, decision views and optimization seeds.
- `REPRODUCIBILITY.md` explains the analysis entry point and use of the saved outputs.
- `CITATION.cff` provides authorship and the source citations.
- `release_manifest_v1.json` records the current member hashes.
- The supplementary ZIP includes `figures/`, with four source CSVs, plotting and formatting code, and the final PNG/PDF/SVG figures. In the working folder these are in `02_Sekiller/`.

## Recorded analysis

The package includes 11 original MATLAB source files, the fixed protocol, 104 extracted average records in 26 condition groups, fixed training and development allocations, normalization parameters, 90 completed training-stage entries, 14,508 decision rows, 837 task-stage summaries and saved model/optimizer states. The final reporting population comprises 24 task-conditioned views of 16 averages in four groups. This organization links each computation to its source record.

The MATLAB analysis compares linear and small neural classifiers with and without replay, joint access to seen tasks, and a nearest-response reference. The four figures show the workflow, observed mechanical properties, decision counts and paired old-task gains. The results retain every optimization seed and both development populations. This v23 revision reuses the completed A experiment and the v22 figures.

## Source and licences

The original dataset is available in Mendeley Data version 2: https://doi.org/10.17632/zd6td6svd6.2 . The source article is https://doi.org/10.3390/mi17070859 . The fixed cohort is an attributed extraction from that dataset; the original files remain available through the authoritative source.

Original analysis and plotting/helper code is distributed under the MIT License. The historical scope of the 11 analysis files is listed in `recorded_package_v2/license_scope_v1.json`. Source-derived measurements and figure source data retain Creative Commons Attribution 4.0 terms. Keep the source authors, version, DOI and extraction description with reused data.

## Release location

These are local submission and sharing files. A public repository URL or archival DOI can be added when an actual deposit is made. The original frozen files retain their historical local paths as provenance; `REPRODUCIBILITY.md` describes how to use the code in a separate working directory.
