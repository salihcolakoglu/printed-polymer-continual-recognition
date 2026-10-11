# A v23 — published-average mechanical-record recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23289833.svg)](https://doi.org/10.5281/zenodo.23289833)

This computational release accompanies **Continual Recognition of Printed Polymer Contexts from Published Average Mechanical Records**, by **Salih Çolakoğlu and Mustafa Onat**, Marmara University. It brings together the completed MATLAB analysis, the exact source-average cohort, saved learning results, and four reusable figures with their source tables.

The study examines how linear and small neural classifiers learn three known-pair recognition tasks in sequence, with and without a small memory of earlier records. It makes the data allocation, decision counts and variability across five numerical starts visible. The published average records support an exploratory source-context study; they are not independently identified printed specimens. All records had been inspected previously.

## Read the results first

1. Read `03_Hesaplama_Eki/DATA_DICTIONARY.md` and `DATA_SOURCES.json` for the source, variables and fixed allocations.
2. View the four figures in `figures/`. PNG is convenient for viewing, PDF/SVG for scalable graphics, and the four `Figure_*_plot_data.csv` files expose the plotted numbers. Captions and production status are in `figures/README_figures.txt`.
3. Inspect recorded decisions and summaries in `03_Hesaplama_Eki/recorded_package_v2/source_results_v2/`. CSV and JSON can be read without retraining. `models_and_optimizer_states_v1.mat` contains saved MATLAB states.
4. Read `03_Hesaplama_Eki/REPRODUCIBILITY.md` before running code.

The cohort contains **104 averages in 26 processing-condition groups**, allocated as 72 training, 16 first-development and 16 reporting-development records. The final reporting result has 24 task-conditioned views of 16 averages in four groups. Views, numerical starts and optimization steps are not independent material specimens. The only classifier measurements are tensile modulus and Shore D hardness. The eligible pair is supplied to every method.

## Folder layout

- `03_Hesaplama_Eki/` preserves the complete A v23 computational-materials folder unchanged. Its nested `recorded_package_v2/` contains the original completed analysis and recorded outputs.
- `figures/` preserves all A v23 figure files, four source CSVs and the MATLAB/Python plotting helpers unchanged. Code names containing `v22` identify the original figure-production version reused by A v23.
- `CITATION.cff` records authorship and the source article/data citations.
- `LICENSE_CODE.txt`, `LICENSE_DATA_FIGURES_DOCS.txt` and `LICENSE_SCOPE.json` define separate software and scientific-material terms. The top-level scope describes this public package; narrower historical notices remain unchanged as provenance.
- `FILE_MANIFEST.json` lists exact copies and new release documentation.

## Optional reproduction

The recorded analysis used **MATLAB R2026a Update 5**. The original entry point is `03_Hesaplama_Eki/recorded_package_v2/run_r04_source_experiment_v2.m`. It finds utilities and input files relative to its own location and deliberately refuses to overwrite an existing `source_results_v2` folder.

For fresh training, use a separate working directory with copies of the analysis code, utility directory, staged inputs, protocol and required historical review metadata. Keep the recorded output folder intact elsewhere. This release does not claim a newly tested one-command public reproduction wrapper. Historical JSON records retain the original `C:\Springer ...` working paths and original launch receipt: those paths identify the original computer's audit references, not folders readers must possess. The original machine-specific Python launcher is not a public entry point. Do not delete the review metadata while expecting the unchanged analysis runner to find it.

For figures, the simplest route is the included CSV files and `figures/redraw_A_v22_figures.m`: add its folder to the MATLAB path and call `redraw_A_v22_figures('a_new_empty_output_folder')`. The script reads the CSVs beside itself and preserves existing image outputs. This command is a usage example; preparing this release did not rerun it.

The optional Python `generate_A_v22_figures.py` requires `--source-zip` with the **original computational-supplement v2 ZIP member layout**, not the whole manuscript delivery ZIP or this public ZIP. Its data-only branch was previously used; its matplotlib drawing branch was not executed in the recorded production environment. `finalize_A_v22_vector_outputs.py` is an advanced formatting helper for prior MATLAB vector output and requires Pillow and pypdf. No new neural training, Simulink simulation, or physical experiment was performed to prepare this release.

## Sources and licences

**Original dataset:** Aktepe, E.; Ergün, U. *FDM 3D Printing Dataset: Printer Type, Materials, Process Parameters, and Tensile, Hardness and Roughness Test Results*. Mendeley Data, **version 2** (2026), https://doi.org/10.17632/zd6td6svd6.2 .

**Associated source article:** Aktepe, E.; Ergün, U. *HMQ-ES-Stack-GBR: A Hybrid Ensemble Learning Model for Mechanical and Physical Quality Prediction in FDM 3D Printing*. Micromachines 2026, 17, 859. https://doi.org/10.3390/mi17070859 .

The fixed cohort is an attributed extraction; source originals are available from the authoritative dataset record and were not changed. **Original analysis and plotting/helper code is MIT licensed. Source-derived measurements and the authors' original figures, scientific documentation, computational outcomes and saved model files are CC BY 4.0.** Preserve the relevant authors, citations, version and descriptions of extraction or changes when reusing them. MIT does not relicense the original source dataset or proprietary MATLAB software. No MATLAB runtime/toolbox is distributed.

There is no executed Simulink experiment in A. The workflow figure is a diagram of the completed computation. No B-research material, manuscript, cover letter, private research journal or account information is included here.

## How to cite

If this software contributes to your research, please cite the versioned release below. **This is a voluntary academic request for MIT-licensed code and adds no condition to the MIT licence.** Attribution remains required under CC BY 4.0 for the applicable data, figures, scientific documentation, computational outcomes and saved model files. Retain the original dataset and source-article citations above when reusing their measurements or derived materials.

Çolakoğlu, S.; Onat, M. (2026). *Computational materials for Continual Recognition of Printed Polymer Contexts from Published Average Mechanical Records* (A_v23.0) [Computer software and computational materials]. Zenodo. https://doi.org/10.5281/zenodo.23289833

Use the **version-specific DOI** above to identify the archived v23.0 materials used in an analysis. The **concept DOI**, https://doi.org/10.5281/zenodo.23289832, groups the archive's versions; it does not replace the version-specific identifier when reporting the exact version used. `CITATION.cff` supplies the software citation. This citation describes the computational release; it does not assert that the associated manuscript has been published.

## Release identity

The scientific software/material version remains **A_v23.0**, published as GitHub tag **v23.0** and archived under https://doi.org/10.5281/zenodo.23289833. Repository: https://github.com/salihcolakoglu/printed-polymer-continual-recognition . The repository's original sharing package is **A_v23_Public_Release_v2**, based on manuscript **A v23**.

The main-branch README, top-level citation and corresponding file-manifest pins receive a documentation-only update identified locally as **A_v23_GitHub_Metadata_Update_v1**. It does not replace the v23.0 tag/archive or create a new scientific result or manuscript version. Code, measurements, figures, results, model states and licence files remain unchanged.

Nested citation/review files and `LICENSE_SCOPE.json` retain their original package metadata as historical provenance, including an initially unassigned archive-DOI field. Use the top-level citation and the DOI links here for the actual archived release. The top-level manifest excludes its own hash; a separate local preparation receipt records that hash. The local preparation guide and receipt are not repository upload files.
