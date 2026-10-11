# Inspect and reproduce the computational study

## Environment and saved outputs

Recorded execution used MATLAB and Simulink R2026a Update 5. MATLAB supplied the learning calculations; Simulink supplied saved-model inference. The CSV summary calculations use Python's standard library. The summary script's complete command-line path also generates figures and requires Matplotlib for that stage. Reading the CSV and JSON tables requires no MATLAB license. The package does not distribute MATLAB, Simulink, or their toolboxes.

The copied data, code, states, and results retain the exact accepted source bytes. Preparing this candidate performed no new training or Simulink run. Numerical acceptance of the recorded code does not establish runtime compatibility on a different computer or MATLAB release.

Start with `07_Sonuclar/learning_run_v1/summary_v1/` and `figures/Figure_*_plot_data.csv`. Full stage states are in `checkpoints_v1.json` and `models_and_optimizer_states_v1.mat`. Decision-level outputs, all initializations, and both development partitions are retained.

## Open a saved Simulink demonstration

Set MATLAB's current folder to the package root and run:

```matlab
addpath(genpath('03_MATLAB_Kodlari'));
open_system('04_Simulink_Modelleri/a24_A_inference_v1.slx');
sim('a24_A_inference_v1');
```

The saved example is task-incremental, ordinary code without negotiation, seed 101, stage three. It was fixed as an example rather than selected for its performance. `baseline_A_inference_v5.slx` provides the saved reference classifier demonstration. Final-stage model banks are in `05_Ogrenilmis_Modeller/`.

Models take two measurements plus the declared eligibility controls. Ground-truth labels and record identities are kept outside the predictor. Simulink ticks are record indices rather than physical time. These diagrams demonstrate the implemented recognizers; they do not model polymer mechanics or train online.

## Recompute in a fresh working copy

The preserved producers protect accepted output folders. Keep this release unchanged and create an independent working directory. For coded and class-incremental learning, copy `02_Veriler/`, `03_MATLAB_Kodlari/`, `06_Protokoller/`, and the single `08_Inceleme/A24_launch_acceptance_v1.json` file into that directory. Do not copy `07_Sonuclar/` initially. From the fresh root:

```matlab
addpath(genpath('03_MATLAB_Kodlari'));
run_a24_learning_v1;
```

The historical launch card records the accepted original settings; its status is not a claim of independent review of your new machine or modifications. The driver records all 165 stages and 99,000 updates. The original runtime was approximately 68 seconds on the recorded local machine; this is an observation, not a runtime promise.

Summarize the newly produced decisions with:

```text
python 03_MATLAB_Kodlari/summarize_a24_results_v1.py --root PATH_TO_FRESH_COPY
```

The command writes CSV summaries before it imports Matplotlib for plotting. Without Matplotlib, it can therefore leave the completed CSV files and then exit with an import error; this is not a successful full CLI run. Matplotlib was unavailable in the recorded environment. Accepted publication plots were generated from the saved CSV summaries by MATLAB `a24_plot_summaries_v4`. For that MATLAB route, use the saved or newly produced CSV summaries and call the plotting function with the fresh root, with no existing `09_Sekiller/learning_v5/` directory. The package supplies accepted plots for inspection without either plotting rerun.

For the reference comparison, copy `03_Hesaplama_Eki/recorded_package_v2/` to a separate working directory without its `source_results_v2/` folder. Its MATLAB utility, protocol, data, and historical review file must remain alongside the driver. Add that directory to the MATLAB path and call `run_r04_source_experiment_v2`.

To reconstruct all inference comparisons, use a fresh working copy with learning results available and without the two target SLX files or final-stage model banks. The accepted code refuses to replace those files. Call `a24_run_inference_parity_v1(learningDir, newOutputDir)` for 55 models. Call `baseline_run_inference_parity_v5(originalPublicRoot, newOutputDir)` for 30 reference models; `originalPublicRoot` must contain `03_Hesaplama_Eki/recorded_package_v2/`, supplied in this package. Both functions create their model artifacts. These are optional new computations and are not required to inspect the saved results.

## Recreate the manuscript titles without learning again

The accepted clean-title exports are supplied in `09_Sekiller/learning_editorial_v1/`. To regenerate them, use a fresh copy of this package with that output directory absent. Keep the saved summary CSV files in `07_Sonuclar/learning_run_v1/summary_v1/`, set MATLAB's current folder to the fresh package root, then run:

```matlab
addpath('03_MATLAB_Kodlari');
report = a25_plot_editorial_titles_v1(pwd);
```

This helper reads the package's own summaries, writes four figures in PNG/PDF/SVG, and refuses to overwrite an existing output folder. It does not call a learning driver or modify a numerical table. The public adaptation changes only the source-directory expression from the private manuscript producer. Historical A24/A25 filenames preserve the component provenance.

## Integrity and attribution

`FILE_MANIFEST.json` lists every candidate file's size and SHA256. Dataset attribution, modifications, and rights are in `DATA_SOURCES.json` and `LICENSE_SCOPE.json`. Preserve the source data and method citations when reusing the files. Use a verified review-commit link while inspecting these materials. The exact v27.0 archive DOI should be added to citation metadata only after the release is publicly archived and verified.
