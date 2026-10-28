# A31 computational materials

This directory supports the A31 manuscript on recognition of printed-polymer contexts from published average mechanical records. It contains the A31 computational extension, its detailed supplement, and the later Simulink verification addendum.

## Extract the A31 extension

Download all 18 files, `A31_S2_Web_part01.zip` through `A31_S2_Web_part18.zip` listed in `PARTS_MANIFEST.json`. Extract each ZIP independently into the **same empty directory**. Do not concatenate the ZIP files. Their union reconstructs the `A31_v4/` directory and all 505 archived files. Check `A31_v4/MANIFEST_v4.json` and `A31_v4/PUBLIC_PATH_REDACTION_v1.json` after extraction. `PARTS_MANIFEST.json` records the hash of each downloadable part and the hash and path of every member.

`A31_Simulink_Verification_Addendum_v1.zip` contains the later selected desktop MATLAB/Simulink verification route. It should be read together with the earlier A31 extension: the earlier two-population issue in its historical compiled NL4 route is retained as provenance; the later addendum documents the separately checked desktop route and its MATLAB helper. MATLAB/Simulink is separately licensed and not distributed.

The previously published v30 S1 package, including the original two-input benchmark, is available at [Zenodo v30](https://doi.org/10.5281/zenodo.23474663) and as the [complete v30 GitHub release asset](https://github.com/salihcolakoglu/printed-polymer-continual-recognition/releases/download/v30.0/A30_Supplement_S1_v1.zip). The source measurements are [Mendeley Data version 2](https://doi.org/10.17632/zd6td6svd6.2). The [Zenodo v31.0 record](https://doi.org/10.5281/zenodo.23720507) archives the A31 computational materials. The supplement v17 files in this directory are preparatory snapshots; consult the Zenodo record for the latest supplementary document included there.

Authored code uses MIT. Authored scientific materials are CC BY 4.0 within the authors' rights. Third-party material retains its own notices and terms. See `LICENSE_CODE.txt`, `LICENSE_DATA_FIGURES_DOCS_A31.txt`, both `THIRD_PARTY` notices, `METHOD_PROVENANCE.txt`, and the scope files within the extracted package. Two historical launchers had user-specific Python executable paths removed; some other historical scripts and provenance records retain original workspace paths and are not guaranteed to run unmodified.
