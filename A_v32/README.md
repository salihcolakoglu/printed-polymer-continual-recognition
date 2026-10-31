# A32 / version 32.0 package guide

The complete computational package is [Zenodo version v32.0](https://doi.org/10.5281/zenodo.23819158). Download the ZIP and the adjacent checksum/manifest files from that record. The release supports matched two/four-input and Task/Class Incremental Learning comparisons, a separate graph-derived PLA archive, and documented fixed-weight implementation checks.

1. Extract the ZIP to a short folder, then read `START_HERE.md` and `README.md`.
2. The `supplement/` folder contains the reviewed supplementary Word/PDF document.
3. The `tables/` folder and `verify_tables.py` provide a portable check of the 64 + 28 main-table cells and the 174 paired comparisons. The verifier does not retrain models.
4. `source/` retains the source data, code, saved states and research receipts. `simulink_addendum/` contains new model banks and implementation evidence. The scope CSV explains each pathway and the historical overlaps.
5. File-specific licence notices identify authored MIT code, CC BY 4.0 scientific artifacts and attributed third-party sources.

The previous [A31 directory](../A_v31/), [A30 directory](../A_v30/) and older release tags remain as historical components. Use the exact archive DOI for the version used. These software versions do not assert publication or acceptance of the manuscript.
