# Printed-polymer continual recognition — version 32 materials

By **Salih Çolakoğlu and Mustafa Onat**, Marmara University.

This study compares Continual Learning with stored-record and classical reference methods across two printed-polymer measurement archives. Whole-condition-group evaluation, explicit historical-data access and matched two- versus four-input experiments connect recognition accuracy to the information available to each procedure.

## Download and citation

The complete computational package and the reviewed supplementary methods are available in the **[Zenodo v32.0 record](https://doi.org/10.5281/zenodo.23819158)**:

- [Complete code, data, saved models, results and figures](https://zenodo.org/records/23819158/files/A32_v32.0_computational_materials.zip?download=1)
- [Supplementary methods and results — PDF](https://zenodo.org/records/23819158/files/Supplementary_Methods_and_Results_v32_r8.pdf?download=1) and [editable Word](https://zenodo.org/records/23819158/files/Supplementary_Methods_and_Results_v32_r8.docx?download=1)
- [File-level SHA-256 manifest](https://zenodo.org/records/23819158/files/A32_v32.0_MANIFEST_SHA256.csv?download=1) and [archive checksums](https://zenodo.org/records/23819158/files/A32_v32.0_CHECKSUMS.txt?download=1)

Archive: `A32_v32.0_computational_materials.zip`, **558,395,936 bytes**.  
SHA-256: `e56eacb944fd73dfa553ef474bf0ce8b08ddcbb185de7ae4af919b75a5eb2f68`

Cite exact version DOI **10.5281/zenodo.23819158**, together with the original data and method sources. The software archive is separate from journal publication of the manuscript. [A32 package guide](A_v32/README.md) describes entry points and earlier versions.

## Comparisons and implementation

Task Incremental Learning (Task-IL) and Class Incremental Learning (Class-IL) are evaluated separately. Comparisons include nearest-record and nearest-centroid rules, three/five-nearest-neighbour rules, Linear/Quadratic Discriminant Analysis, streaming LDA, linear and Cross-Entropy networks, Replay, Elastic Weight Consolidation and Learning without Forgetting. Uncoded, ordinary-coded and Walsh-coded negotiated learning use five constant Negotiation Rates and symmetric taskwise increasing/decreasing profiles. The rate is fixed within each task.

The package contains 104 primary condition means and a separate 27-mean PLA archive. The latter combines tabulated tensile modulus with Shore D hardness estimated in two separate graph readings. Models are trained within each archive's own groups; digitization scenarios and printer/source differences are documented.

The 53 new Simulink files and five historical model files retain their separate evidence. Fixed-weight inference, exact source-record mappings and same-computer relocated readback have explicitly defined scopes. Some stricter cross-library optimizer-trajectory comparisons exceed their recorded numerical criterion; those diagnostics are retained separately from saved-model inference. These are numerical implementation checks on condition means, rather than new physical specimen tests.

## Reuse and reproducibility

Extract to a short Windows path such as `C:\A32`, open `START_HERE.md`, and run `python verify_tables.py` with Python 3.10 or newer for the portable table/arithmetic check. MATLAB/Simulink is required for the respective model pathways. Historical run receipts can retain original-machine paths; complete retraining on another computer requires environment/path adaptation.

Original authored code is **MIT**. Author-created scientific documentation, figures, model/result artifacts and derived tables are **CC BY 4.0 within the authors' rights**, with file-specific scope and original third-party notices retained. Consult `LICENSE_SCOPE.json` inside this version's package. Proprietary runtimes, third-party full-paper libraries and private correspondence are excluded.
