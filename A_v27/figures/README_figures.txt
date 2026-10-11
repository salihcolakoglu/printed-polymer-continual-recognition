A v23 figure materials

The four numerical figures and source CSVs are unchanged from A_v22.0_2026_10_11. No plotting run, training or simulation was repeated for this language revision. The MATLAB renderer was executed in v22; Python was used for data-only extraction and vector formatting. The Python matplotlib drawing branch was not executed. Original code is MIT; source-derived data retain CC BY4.0.

Source dataset: https://doi.org/10.17632/zd6td6svd6.2
Associated source article: https://doi.org/10.3390/mi17070859

Captions used in A v23

Figure 1. Workflow of the completed computational recognition experiment. The 104 published average records belong to 26 condition groups; the three fixed allocations contain 72/16/16 records in 18/4/4 groups. Tensile modulus and Shore D hardness enter the classifier, and each task supplies its eligible pair. Task-conditioned views and optimization seeds describe repeated computations on the retained source averages. The workflow connects the source records, sequential learning and complete decision accounting.

Figure 2. Tensile modulus and Shore D hardness of the 104 retained published averages. Colors identify source-defined contexts, and marker shapes identify training, first-development and reporting-development allocations. Each point represents an average source record. The plot describes the previously inspected cohort and makes the observed feature distributions and fixed allocations accessible for interpretation.

Figure 3. Final reporting-development decision counts for every fixed optimization seed. MLP denotes the multilayer perceptron. Each learned-method point summarizes 24 task-conditioned decision views from 16 unique averages in four groups; the deterministic nearest method is shown once. Selected correctness follows the fixed score-gap rule, while forced correctness always chooses the larger eligible score. Nonselection records a withheld answer. The points describe optimization sensitivity on the same retained source population.

Figure 4. Paired final old-task forced-correct gain from replay minus learning without replay. Each cell uses the same 16 decision views from tasks 1 and 2 within one allocation and optimization seed. MLP denotes the multilayer perceptron. Positive values favor replay, zero indicates equality and negative values favor learning without replay. The figure presents the variable first-development MLP gains alongside uniformly positive reporting-development gains as descriptive comparisons on the retained source averages.
