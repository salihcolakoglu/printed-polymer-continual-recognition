# Inspecting and reproducing the analysis

## Explore the recorded result

Start with `recorded_package_v2/prospective_protocol_v1.json`, `staged_input_manifest_v1.json` and `staged_source_records_v1.csv`. The saved decisions are in `source_results_v2/decision_rows_v1.csv`; summaries, stage records and normalization parameters are in `task_stage_summaries_v1.json`, `training_ledger_v1.json` and `normalizer_v1.json`. MATLAB model and optimizer states are in `models_and_optimizer_states_v1.mat`. The historical execution receipt identifies MATLAB R2026a Update 5 and the original elapsed time.

## Analysis entry point

`run_r04_source_experiment_v2.m` derives its working folder from the function location, adds `matlab_utility`, and reads the staged CSV, protocol and review receipt using relative paths. Ten utility MATLAB files accompany it. The neural network, weighted pair loss and Adam updates use MATLAB matrix operations directly. MATLAB R2026a Update 5 is the recorded execution environment; other release/toolbox combinations remain to be evaluated.

For a fresh reproduction, make a separate working directory containing copies of the code, utilities, staged inputs, protocol and review metadata. Keep the archived `source_results_v2` separately for comparison. The entry point writes a new `source_results_v2` beside its code and checks that this output folder does not already exist. This preserves the recorded outputs while allowing a new run to be compared with them.

## Historical launch record and checks

`review/launch_acceptance_v2.json` documents the original local launch, including absolute audit paths and a local hash-guarded Python launcher. The core MATLAB code and relative input paths are supplied; the original machine-specific launcher is retained as provenance rather than a portable entry point. A newly tested one-command public reproduction wrapper remains a useful software extension.

`matlab_utility/run_r04nn_math_checks.m` checks the mathematical implementation in a fresh output location. These implementation checks and the recorded learning outcomes support inspection of the computation. The v23 editorial revision uses the existing A outputs without retraining.

## Interpreting and redrawing the outputs

The analysis concerns previously inspected published averages, supplied candidate pairs and fixed condition groups. Decision views and optimization seeds describe repeated computations on those average records. The plotting tables and MATLAB renderer retain every seed and all nonselection/failure denominators. Subsequent specimen-level validation can build on this record-level reference using newly held-out, traceable printed specimens.
