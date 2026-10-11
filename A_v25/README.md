# Computational materials for Continual Recognition of Printed Polymer Contexts from Published Average Mechanical Records

Prepared version: **v25.0**, supporting manuscript A v25. Authors: **Salih Çolakoğlu and Mustafa Onat**, Marmara University.

This package brings together the study's MATLAB learning code, attributed source-average records, saved model states, learning results, Simulink inference models, and eight figures with their plotting data. It supports inspection and reuse of comparisons between ordinary and Walsh codes, negotiated learning, small replay memories, and joint access to seen tasks.

**Release status:** these are review materials prepared for computational release v25.0. The v25.0 release and its version-specific DOI remain pending. A public review branch may make the files inspectable before that archive is created. The [existing GitHub repository](https://github.com/salihcolakoglu/printed-polymer-continual-recognition) currently provides the earlier release; [DOI 10.5281/zenodo.23289833](https://doi.org/10.5281/zenodo.23289833) identifies v23.0 files only. These existing links identify the repository and its earlier archive. Cite a verified review commit to identify the expanded files until the new version-specific archive is available.

## Start with the saved results

1. Read `DATA_DICTIONARY.md` and `DATA_SOURCES.json` for the two measurements, class contexts, and fixed condition groups.
2. Read `07_Sonuclar/learning_run_v1/summary_v1/summary_by_seed.csv` and `summary_ranges.csv`. Stage-wise acquisition and retention are in `stage_metrics.csv`.
3. View the four coded-learning figures in `09_Sekiller/learning_v5/` and the four reference-comparison figures in `figures/`. PNG, PDF, and SVG files are supplied. The CSV files and saved summary tables contain the plotted numbers.
4. Follow `REPRODUCIBILITY.md` to open the saved Simulink example or prepare a separate reproduction copy.

The input cohort contains **104 published averages in 26 processing-condition groups**. Only tensile modulus in MPa and Shore D hardness enter the learners. Task-incremental learning receives the relevant candidate pair. Class-incremental learning chooses among all classes introduced so far. Final task-incremental decisions contain 24 views of 16 averages; class-incremental decisions contain 16 averages. Both development partitions remain separately reported, and all five numerical initializations are retained.

The recorded MATLAB implementation contains 165 training stages and 99,000 updates. The supplied Simulink comparisons apply 30 reference models to 720 decision views and 55 coded or class-incremental models to 1,040 views. Simulink applies saved weights; training took place in MATLAB. Simulation ticks indicate record order. Published averages and numerical initializations describe the computational study's observation units; physical specimen independence is outside this dataset's documented scope.

## Folder map

| Folder or file | Purpose |
|---|---|
| `01_Kaynaklar/` | Data attribution and method provenance |
| `02_Veriler/` | Exact source-average cohort used in learning |
| `03_Hesaplama_Eki/recorded_package_v2/` | Reference comparison's preserved code, data, states, and decisions |
| `03_MATLAB_Kodlari/` | Coded learning, class-incremental controls, summaries, plotting, and model construction |
| `04_Simulink_Modelleri/` | Two saved inference demonstrations |
| `05_Ogrenilmis_Modeller/` | Saved final-stage inference banks |
| `06_Protokoller/` | Exact computational rules used by the accepted runs |
| `07_Sonuclar/` | Saved learning and inference results |
| `08_Inceleme/` | The single historical launch document required by the preserved training driver |
| `09_Sekiller/learning_v5/` and `figures/` | Eight figures and numerical plotting sources |
| `FILE_MANIFEST.json` | SHA256 file inventory for this candidate |

Historical code and protocol filenames retain their original version identifiers so they continue to match the saved results. Pre-run status fields in copied protocols are dated snapshots; the preserved launch document and recorded outputs identify the accepted computational settings. They describe components of this study, not an assertion that an earlier manuscript was journal-published. Manuscript drafts, private preparation logs, failed export attempts, caches, and obsolete figure/model variants are excluded from this public candidate. Their research history remains in the local archive.

## Attribution and reuse

The data originate from Aktepe and Ergün's [Mendeley Data v2](https://doi.org/10.17632/zd6td6svd6.2), with the associated [Micromachines article](https://doi.org/10.3390/mi17070859). Retain those source credits when using the derived cohort.

Negotiated target construction is a source-informed MATLAB adaptation of Korhan and Oner's work and [author repository](https://github.com/nurikorhan/Negotiated-Representations-for-Continual-Learning). `METHOD_PROVENANCE.md` explains the implementation choices. `THIRD_PARTY_CODE_NOTICE.txt` retains the source repository's MIT notice. The method's preprint is identified as a preprint rather than a journal article.

Original code is **MIT**. Data derivatives, figures, explanatory documents, calculated results, learned weights, and authored model artifacts are **CC BY 4.0 within the authors' rights**. See `LICENSE_SCOPE.json` for the file groups. Third-party and MathWorks components retain their own terms. Academic citation of this software is requested; it adds no restriction to the MIT license. Cite the exact public version once published and retain the original data and method citations.
