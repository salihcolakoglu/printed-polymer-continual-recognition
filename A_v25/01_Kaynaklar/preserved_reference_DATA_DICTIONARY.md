# Staged data dictionary

One row is one published averaged record in one source-defined material/printer/process context. It is not an independently identified printed specimen. Physical repeat counts, covariance, build and batch independence are unknown. All 104 records had been inspected before this exploratory learning study.

| Column | Meaning and role |
|---|---|
| row_id | Traceable identifier linking the published workbook sheet/context and original Excel row; audit only. |
| condition_group | Common processing-condition group identifier; keeps the fixed allocation together; audit only. |
| group_order | Fixed source-group ordering used for the prescribed small replay memory; not a model input. |
| native_key_id | Identifier of matched native processing controls; audit only. |
| partition | TRAIN, FIT_DEVELOPMENT or REPORTING_DEVELOPMENT; fixed before this run. |
| label | 1 PLA+, 2 ABS, 3 PETG, 4 PLA-CF. These are source contexts, not chemistry-only or complete physical profiles. |
| E_MPa | Published average tensile modulus, in MPa. First model input. |
| Hardness_ShoreD | Published average Shore D indentation hardness. Second model input. |
| source_excel_row | Row number in the original source workbook; audit only. |

The cohort contains 26 groups × 4 contexts = 104 averages. TRAIN contains 18 groups/72 rows; FIT_DEVELOPMENT and REPORTING_DEVELOPMENT each contain 4 groups/16 rows. A group is not asserted to be an independent manufacturing batch.

Tasks are fixed known-pair decisions: PLA+ vs ABS, then PLA+ vs PETG, then PLA+ vs PLA-CF. The eligible pair is supplied to all methods. A PLA+ source average may appear in several task views, so counts of views exceed counts of unique averages. Only E_MPa and Hardness_ShoreD enter the classifier. IDs, names, process settings and partitions are excluded from its input.

The normalizer is fitted using 36 original averages from the first task's TRAIN allocation; it is then held fixed. No missing-value imputation, synthetic measurement noise or physical parameter distribution was added. The source-native processing controls and extraction rule are recorded in the historical protocol and trace identifiers.

Optimization starts [101, 211, 307, 401, 503] describe numerical sensitivity; they are not specimen repeats. Decision scores are uncalibrated comparison scores. The score-gap threshold 0.15 is a fixed engineering rule, not a measured physical confidence interval.
