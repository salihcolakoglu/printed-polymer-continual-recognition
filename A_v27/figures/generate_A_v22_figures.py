# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Salih Colakoglu and Mustafa Onat
"""Redraw four A article figures from the frozen, already completed results.

No model training, MAT-file loading, Simulink execution, data resplitting,
seed selection, or new scientific experiment occurs in this program.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import median
import zipfile

# Plotting dependencies are imported only when Python is used to render.
# --data-only permits the same fixed science check and CSV extraction before
# the MATLAB renderer, without requiring matplotlib in the local runtime.


SEEDS = [101, 211, 307, 401, 503]
METHODS = [
    ("linear", "sequential_no_replay", "Linear: no replay"),
    ("linear", "sequential_replay", "Linear: replay"),
    ("linear", "joint_seen_reference", "Linear: joint seen"),
    ("mlp_tanh", "sequential_no_replay", "MLP: no replay"),
    ("mlp_tanh", "sequential_replay", "MLP: replay"),
    ("mlp_tanh", "joint_seen_reference", "MLP: joint seen"),
    ("nearest", "seen_reference", "Nearest seen reference"),
]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#595959"]
FIELDS = ["selected_correct", "selected_wrong", "nonselection", "failure", "forced_argmax_correct"]


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def savefig(fig, out, stem):
    for extension in ["png", "pdf", "svg"]:
        destination = out / f"{stem}.{extension}"
        if destination.exists():
            raise FileExistsError(destination)
        fig.savefig(destination, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagram(out):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, text, face="#F3F6F8", size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                   linewidth=.9, edgecolor="#43505C", facecolor=face))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "lw": 1.05, "color": "#43505C"})

    box(.22, .86, .56, .10, "Published mechanical measurements\n104 averages in 26 processing-condition groups", "#E7F0F5")
    for x, name, rows, groups in [(.035, "Training", 72, 18), (.355, "First development", 16, 4), (.675, "Reporting development", 16, 4)]:
        box(x, .68, .29, .105, f"{name}\n{rows} averages / {groups} groups", size=8.7)
        arrow(.50, .855, x+.145, .79)
    box(.10, .49, .80, .11, "Two observed inputs only\nTensile modulus (MPa) and indentation hardness (Shore D)", "#E7F0F5")
    for x in [.18, .50, .82]:
        arrow(x, .67, x, .605)
    box(.10, .315, .80, .105, "Known candidate pair supplied at every stage\nPLA+ vs ABS  →  PLA+ vs PETG  →  PLA+ vs PLA-CF", size=9)
    arrow(.50, .485, .50, .425)
    box(.035, .135, .445, .105, "Sequential classifiers\nWithout replay / limited replay", face="#F9F1E7", size=8.8)
    box(.52, .135, .445, .105, "Reference comparisons\nJoint access to seen tasks / nearest seen response", face="#F9F1E7", size=8.5)
    arrow(.30, .31, .255, .245)
    arrow(.70, .31, .745, .245)
    ax.text(.50, .067, "Every decision counted: correct / wrong / nonselection / failure", ha="center", va="center", fontsize=8.7)
    ax.text(.50, .018, "Completed source-record experiment; no new physical simulation", ha="center", va="center", fontsize=8, color="#43505C")
    fig.subplots_adjust(left=.02, right=.98, top=.99, bottom=.01)
    savefig(fig, out, "Figure_1_experiment_workflow")


def scatter(out, source):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    labels = {1: "PLA+", 2: "ABS", 3: "PETG", 4: "PLA-CF"}
    partitions = [("TRAIN", "o", "Training"), ("FIT_DEVELOPMENT", "s", "First development"),
                  ("REPORTING_DEVELOPMENT", "^", "Reporting development")]
    for label in labels:
        for partition, marker, _ in partitions:
            group = [row for row in source if row["label"] == label and row["partition"] == partition]
            ax.scatter([row["E_MPa"] for row in group], [row["Hardness_ShoreD"] for row in group],
                       marker=marker, s=36, facecolor=COLORS[label-1], edgecolor="white", linewidth=.5,
                       alpha=.86, zorder=3)
    ax.set(xlabel="Tensile modulus (MPa)", ylabel="Indentation hardness (Shore D)")
    ax.grid(alpha=.20, linewidth=.55)
    handles1 = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS[k-1],
                       markeredgecolor="none", label=v, markersize=6) for k, v in labels.items()]
    first = ax.legend(handles=handles1, title="Source context", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.add_artist(first)
    handles2 = [Line2D([], [], marker=marker, linestyle="none", markerfacecolor="#777777", markeredgecolor="none",
                       label=label, markersize=6) for _, marker, label in partitions]
    ax.legend(handles=handles2, title="Fixed population", loc="lower left", bbox_to_anchor=(1.02, .0), frameon=False)
    fig.subplots_adjust(left=.11, right=.70, bottom=.14, top=.97)
    savefig(fig, out, "Figure_2_mechanical_records")


def final_counts(out, final):
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 4.5), sharey=True)
    offsets = dict(zip(SEEDS, [-.16, -.08, 0, .08, .16]))
    panels = [("selected_correct", "(a) Correct selected decisions"),
              ("forced_argmax_correct", "(b) Correct forced decisions"),
              ("nonselection", "(c) Nonselections")]
    for ax, (field, title) in zip(axes, panels):
        for idx, (_, _, name) in enumerate(METHODS):
            rows = [row for row in final if row["method"] == name]
            for row in rows:
                if row["seed"] == 0:
                    ax.scatter(row[field], idx, s=37, marker="D", color="black", zorder=4)
                else:
                    j = SEEDS.index(row["seed"])
                    ax.scatter(row[field], idx+offsets[row["seed"]], s=29, facecolor=COLORS[j],
                               edgecolor="white", linewidth=.35, zorder=4)
        ax.set_title(title, fontsize=9, pad=10)
        ax.set(xlim=(-.7, 24.7), xticks=range(0, 25, 4), xlabel="Count out of 24 task views")
        ax.set_yticks(range(7), [method[2] for method in METHODS])
        ax.set_ylim(6.55, -.55)
        ax.grid(axis="x", alpha=.25, linewidth=.6)
        for y in [2.5, 5.5]:
            ax.axhline(y, color="#BBBBBB", lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
    legend = [Line2D([], [], marker="o", linestyle="none", color=COLORS[j], label=str(seed), markersize=5)
              for j, seed in enumerate(SEEDS)]
    legend.append(Line2D([], [], marker="D", linestyle="none", color="black", label="Deterministic nearest", markersize=5))
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, title="Optimization seed / deterministic reference",
               bbox_to_anchor=(.59, -.005), fontsize=7.7, title_fontsize=8)
    fig.subplots_adjust(left=.225, right=.99, bottom=.23, top=.88, wspace=.17)
    savefig(fig, out, "Figure_3_final_reporting_decisions")


def gain_heatmap(out, gains):
    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    rownames = ["Linear / first development", "Linear / reporting development",
                "MLP / first development", "MLP / reporting development"]
    matrix = np.array([[next(row["forced_correct_gain"] for row in gains if row["row"] == name and row["seed"] == seed)
                        for seed in SEEDS] for name in rownames])
    image = ax.imshow(matrix, cmap="RdBu", vmin=-4, vmax=4, aspect="auto")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{matrix[i,j]:+d}", ha="center", va="center", fontsize=10,
                    color="white" if abs(matrix[i,j]) >= 3 else "#222222")
    ax.set_xticks(range(5), [str(seed) for seed in SEEDS])
    ax.set_yticks(range(4), rownames)
    ax.set_xlabel("Optimization seed")
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Final old-task gain: replay minus no replay", fontsize=10, pad=12)
    cb = fig.colorbar(image, ax=ax, fraction=.047, pad=.04, ticks=range(-4, 5, 2))
    cb.set_label("Change in correct forced decisions\n(out of 16 old-task views)", fontsize=8)
    fig.subplots_adjust(left=.33, right=.89, top=.80, bottom=.20)
    savefig(fig, out, "Figure_4_paired_old_task_gain")


def run(source_zip: Path, out: Path, data_only=False):
    out.mkdir(parents=True, exist_ok=True)
    if not data_only:
        global plt, Line2D, FancyBboxPatch, np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
                             "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.transparent": False})
    with zipfile.ZipFile(source_zip) as archive:
        src_bytes = archive.read("staged_source_records_v1.csv")
        sum_bytes = archive.read("source_results_v2/task_stage_summaries_v1.json")
        source = list(csv.DictReader(io.StringIO(src_bytes.decode("utf-8-sig"))))
        summaries = json.loads(sum_bytes)
    for row in source:
        row["label"] = int(row["label"])
        row["E_MPa"] = float(row["E_MPa"])
        row["Hardness_ShoreD"] = float(row["Hardness_ShoreD"])
    checks = []

    def check(name, condition, observed=None, expected=None):
        checks.append({"name": name, "pass": bool(condition), "observed": observed, "expected": expected})
        if not condition:
            raise AssertionError(f"{name}: {observed!r} != {expected!r}")

    check("source averages", len(source) == 104, len(source), 104)
    check("source condition groups", len({row["condition_group"] for row in source}) == 26)
    check("all four labels in every group", all({row["label"] for row in source if row["condition_group"] == group} == {1,2,3,4}
                                                for group in {row["condition_group"] for row in source}))
    for partition, nrows, ngroups in [("TRAIN",72,18),("FIT_DEVELOPMENT",16,4),("REPORTING_DEVELOPMENT",16,4)]:
        selected = [row for row in source if row["partition"] == partition]
        check(partition+" rows/groups", len(selected) == nrows and len({row["condition_group"] for row in selected}) == ngroups)
    group_parts = {}
    for row in source:
        group_parts.setdefault(row["condition_group"], set()).add(row["partition"])
    check("group partitions do not overlap", all(len(parts) == 1 for parts in group_parts.values()))

    final = []
    for architecture, arm, name in METHODS:
        for seed in ([0] if architecture == "nearest" else SEEDS):
            cells = [row for row in summaries if row["architecture"] == architecture and row["arm"] == arm
                     and row["seed"] == seed and row["stage"] == 3 and row["partition"] == "REPORTING_DEVELOPMENT"]
            check(f"final three task cells {name} {seed}", len(cells) == 3 and {row["task"] for row in cells} == {1,2,3})
            check(f"fixed cell population {name} {seed}", all(row["decision_views"] == 8 for row in cells))
            total = {field: sum(row[field] for row in cells) for field in FIELDS}
            check(f"four outcomes sum to24 {name} {seed}", sum(total[field] for field in FIELDS[:4]) == 24)
            final.append({"method":name,"architecture":architecture,"arm":arm,"seed":seed,
                          "stage":3,"partition":"REPORTING_DEVELOPMENT","decision_views":24,**total})
    expected_table1 = {
        "Linear: no replay": [[19,19,19],[1,1,2],[3,4,4],[0,0,0],[21,21,21]],
        "Linear: replay": [[20,21,21],[0,0,1],[3,3,3],[0,0,0],[22,23,23]],
        "Linear: joint seen": [[18,19,19],[1,1,1],[4,4,5],[0,0,0],[21,21,21]],
        "MLP: no replay": [[19,21,22],[2,3,5],[0,0,1],[0,0,0],[19,21,22]],
        "MLP: replay": [[23,24,24],[0,0,1],[0,0,0],[0,0,0],[23,24,24]],
        "MLP: joint seen": [[24,24,24],[0,0,0],[0,0,0],[0,0,0],[24,24,24]],
        "Nearest seen reference": [[19,19,19],[0,0,0],[5,5,5],[0,0,0],[24,24,24]],
    }
    for _, _, name in METHODS:
        records = [row for row in final if row["method"] == name]
        for field, expected in zip(FIELDS, expected_table1[name]):
            values = [row[field] for row in records]
            actual = [min(values), median(values), max(values)]
            check(f"Table1 {name} {field}", actual == expected, actual, expected)

    gains = []
    expected_table2 = {("linear","FIT_DEVELOPMENT"):[2,2,0,1,0],
                       ("linear","REPORTING_DEVELOPMENT"):[1,2,2,2,2],
                       ("mlp_tanh","FIT_DEVELOPMENT"):[1,0,-1,0,-1],
                       ("mlp_tanh","REPORTING_DEVELOPMENT"):[3,4,2,4,2]}
    for architecture, model in [("linear","Linear"),("mlp_tanh","MLP")]:
        for partition, part_name in [("FIT_DEVELOPMENT","first development"),("REPORTING_DEVELOPMENT","reporting development")]:
            for seed, expected in zip(SEEDS, expected_table2[architecture,partition]):
                totals={}
                for arm in ["sequential_no_replay","sequential_replay"]:
                    cells=[row for row in summaries if row["architecture"]==architecture and row["arm"]==arm
                           and row["seed"]==seed and row["stage"]==3 and row["partition"]==partition and row["task"] in [1,2]]
                    check(f"old task population {architecture} {arm} {partition} {seed}", len(cells)==2 and sum(row["decision_views"] for row in cells)==16)
                    totals[arm]=sum(row["forced_argmax_correct"] for row in cells)
                gain=totals["sequential_replay"]-totals["sequential_no_replay"]
                check(f"Table2 {architecture} {partition} {seed}", gain==expected, gain, expected)
                gains.append({"row":f"{model} / {part_name}","architecture":architecture,"partition":partition,"seed":seed,
                              "stage":3,"tasks":"1+2","decision_views":16,"no_replay_forced_correct":totals["sequential_no_replay"],
                              "replay_forced_correct":totals["sequential_replay"],"forced_correct_gain":gain})

    write_csv(out/"Figure_2_plot_data.csv",source)
    write_csv(out/"Figure_3_plot_data.csv",final)
    write_csv(out/"Figure_4_plot_data.csv",gains)
    write_csv(out/"Figure_1_plot_data.csv",[
        {"population":"All source averages","average_records":104,"condition_groups":26},
        {"population":"TRAIN","average_records":72,"condition_groups":18},
        {"population":"FIT_DEVELOPMENT","average_records":16,"condition_groups":4},
        {"population":"REPORTING_DEVELOPMENT","average_records":16,"condition_groups":4}])
    if not data_only:
        diagram(out);scatter(out,source);final_counts(out,final);gain_heatmap(out,gains)
    report={"status":"PASS_SAVED_RESULT_FIGURES_ONLY","checks":checks,"check_count":len(checks),"failures":0,
            "source_zip":str(source_zip),"source_zip_sha256":hashlib.sha256(source_zip.read_bytes()).hexdigest(),
            "source_members_sha256":{"staged_source_records_v1.csv":hashlib.sha256(src_bytes).hexdigest(),
                                      "source_results_v2/task_stage_summaries_v1.json":hashlib.sha256(sum_bytes).hexdigest()},
            "new_training_runs":0,"MAT_files_read":0,"Simulink_runs":0,"new_physical_measurements":0,
            "new_blind_tests":0,"seeds_not_specimens":True,"source_averages_not_independent_specimens":True,
            "MLP_definition":"Two-input multilayer perceptron with four tanh hidden units and four outputs",
            "MATLAB_redraw_status":"PLANNED_SEPARATE_RENDER","python_redraw_status":"DATA_CHECK_ONLY" if data_only else "EXECUTED"}
    with (out/"figure_science_check_v1.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({"status":report["status"],"checks":len(checks),"failures":0,
                      "figures_rendered_by_python":0 if data_only else 4,
                      "formats":["600dpi PNG","vector PDF","SVG"],"new_training":0,"MAT_read":0}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-zip",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--data-only",action="store_true")
    args=parser.parse_args()
    run(args.source_zip,args.out,args.data_only)
