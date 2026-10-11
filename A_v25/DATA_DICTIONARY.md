# Used data and decision units

One CSV row represents a published average for a source-defined material, printer, and processing context. The cohort contains 26 common condition groups and four source contexts: PLA+, ABS, PETG, and PLA-CF.

| Column | Role |
|---|---|
| `E_MPa` | Tensile modulus in MPa; classifier input 1 |
| `Hardness_ShoreD` | Shore D hardness; classifier input 2 |
| `label` | Context code: 1 PLA+, 2 ABS, 3 PETG, 4 PLA-CF; training target/evaluation truth |
| `partition` | Fixed TRAIN, FIT_DEVELOPMENT, or REPORTING_DEVELOPMENT allocation |
| `condition_group`, `group_order`, `native_key_id` | Grouping and prescribed replay ordering; trace fields |
| `row_id`, `source_excel_row` | Trace to the published workbook |

The allocation is 72 TRAIN averages in 18 groups, 16 FIT averages in four groups, and 16 REPORT averages in four groups. Trace identifiers, filenames, labels, and process settings are excluded from the learner's measurement input. Only the prescribed pair or introduced-class eligibility is supplied as a decision control.

The normalizer uses the first-stage 36 TRAIN averages and remains fixed. The cohort and its development results had been inspected before these comparisons. The supplied records contain published averages; specimen-level repeat counts, covariance, and independent production batches are not documented. No missing physical values or synthetic measurement noise were introduced.

Task-incremental learning follows PLA+/ABS, PLA+/PETG, then PLA+/PLA-CF. A PLA+ average can appear in more than one pair-conditioned view. Final evaluation has 24 views of 16 unique averages per development partition. Class-incremental learning introduces PLA+/ABS, then PETG, then PLA-CF; final evaluation has 16 unique averages per partition. Computational seeds 101, 211, 307, 401, and 503 describe initialization sensitivity.
