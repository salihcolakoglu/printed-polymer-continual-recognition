"""Summarize one saved A24 run; never train or read MATLAB model payloads.

REPORT and FIT remain separate. TIL primary metrics include acquired tasks
only; future-task diagnostics have their own stage-table rows. All five
seeds are retained. Percentages, ranges and means are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SEEDS = (101, 211, 307, 401, 503)
CODE_METHODS = (
    "onehot_no_negotiation", "onehot_negotiated",
    "walsh_no_negotiation", "walsh_negotiated",
)
CE_METHODS = ("sequential_no_replay", "sequential_replay", "joint_seen_reference")
STREAMS = ("TaskIncremental", "ClassIncremental")
METHODS = {STREAMS[0]: CODE_METHODS, STREAMS[1]: CODE_METHODS + CE_METHODS}
LABELS = {
    "onehot_no_negotiation": "One-hot\nno negotiation",
    "onehot_negotiated": "One-hot\nnegotiated",
    "walsh_no_negotiation": "Walsh\nno negotiation",
    "walsh_negotiated": "Walsh\nnegotiated",
    "sequential_no_replay": "CE\nno replay",
    "sequential_replay": "CE\nreplay",
    "joint_seen_reference": "CE joint-seen\nreference",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty input CSV: {path}")
    return rows


def integer(value) -> int:
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"Expected an integer, received {value!r}")
    return int(number)


def boolean(value) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Expected a Boolean, received {value!r}")


def partition_name(value: str) -> str:
    text = value.strip().upper()
    if text.startswith("REPORT"):
        return "REPORT"
    if text.startswith("FIT"):
        return "FIT"
    return text


def load_inputs(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    decision_path = run_dir / "decisions_v1.csv"
    ledger_path = run_dir / "training_ledger_v1.csv"
    raw = read_csv(decision_path)
    ledger = read_csv(ledger_path)
    required = {"category", "method", "seed", "stage", "evaluation_task",
                "acquisition_unit", "partition", "row_id", "condition_group",
                "true_label", "forced_label", "correct", "exact_tie"}
    if not required <= raw[0].keys():
        raise ValueError(f"Missing decision fields: {sorted(required - raw[0].keys())}")
    if len(raw) != 17550 or len(ledger) != 165:
        raise ValueError("Expected one complete frozen run: 17550 decisions / 165 stage entries.")
    seen_keys = set()
    for row in raw:
        for key in ("seed", "stage", "evaluation_task", "acquisition_unit",
                    "true_label", "forced_label"):
            row[key] = integer(row[key])
        for key in ("correct", "exact_tie"):
            row[key] = boolean(row[key])
        row["partition"] = partition_name(row["partition"])
        category, method = row["category"], row["method"]
        if category not in METHODS or method not in METHODS[category]:
            raise ValueError("Unexpected stream or method in saved decisions.")
        if row["seed"] not in SEEDS or row["stage"] not in (1, 2, 3):
            raise ValueError("Unexpected seed/stage in saved decisions.")
        if row["correct"] != (row["true_label"] == row["forced_label"]):
            raise ValueError("Saved correctness disagrees with saved true/forced labels.")
        if category == STREAMS[0]:
            if row["evaluation_task"] not in (1, 2, 3):
                raise ValueError("TIL decisions need an evaluation task in 1..3.")
            if row["acquisition_unit"] != row["evaluation_task"]:
                raise ValueError("TIL acquisition unit must equal its evaluation task.")
        else:
            if row["evaluation_task"] != 0:
                raise ValueError("CIL evaluation must not use a supplied query task pair.")
            if row["acquisition_unit"] != max(1, row["true_label"] - 1):
                raise ValueError("CIL unit mapping must be {1,2}->1, 3->2, 4->3.")
            if row["true_label"] > row["stage"] + 1:
                raise ValueError("A future true class entered the CIL evaluation cohort.")
        key = tuple(row[k] for k in ("category", "method", "seed", "stage",
                                    "evaluation_task", "partition", "row_id"))
        if key in seen_keys:
            raise ValueError("Duplicate decision within the same query population.")
        seen_keys.add(key)
    ledger_keys = set()
    for row in ledger:
        for key in ("seed", "stage", "updates", "parameter_count", "parameter_bytes",
                    "optimizer_m_v_bytes", "codebook_bytes", "update_gradient_examples",
                    "teacher_forward_examples"):
            row[key] = integer(row[key])
        key = tuple(row[k] for k in ("category", "method", "seed", "stage"))
        if key in ledger_keys:
            raise ValueError("Duplicate training-ledger checkpoint.")
        ledger_keys.add(key)
    expected = {(stream, method, seed, stage)
                for stream in STREAMS for method in METHODS[stream]
                for seed in SEEDS for stage in (1, 2, 3)}
    if ledger_keys != expected or sum(r["updates"] for r in ledger) != 99000:
        raise ValueError("The ledger does not cover all declared methods, seeds and stages.")
    provenance = {
        "input_decisions": str(decision_path.resolve()),
        "input_ledger": str(ledger_path.resolve()),
        "decisions_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "decision_rows": len(raw), "training_stage_entries": len(ledger),
        "training_updates": sum(r["updates"] for r in ledger),
        "seeds": list(SEEDS), "new_training_performed_by_summarizer": 0,
        "MAT_or_model_payload_read": False,
        "original_A_reference_loaded": False,
    }
    return raw, ledger, provenance


def metric(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["true_label"]].append(row["correct"])
    return {
        "correct": correct, "n_views": n,
        "n_unique_records": len({r["row_id"] for r in rows}),
        "n_condition_groups": len({r["condition_group"] for r in rows}),
        "accuracy_pct": 100 * correct / n if n else None,
        "class_balanced_accuracy_pct": 100 * mean(mean(v) for v in by_label.values()) if n else None,
        "exact_ties": sum(r["exact_tie"] for r in rows),
    }


def table_outputs(decisions: list[dict], ledger: list[dict]) -> tuple[list, list, list]:
    grouped = defaultdict(list)
    ledger_groups = defaultdict(list)
    for row in decisions:
        if row["partition"] in ("REPORT", "FIT"):
            key = tuple(row[k] for k in ("category", "method", "seed", "partition", "stage"))
            grouped[key].append(row)
    for row in ledger:
        ledger_groups[(row["category"], row["method"], row["seed"])].append(row)
    stage_rows, summaries = [], []
    stage_lookup = {}
    for stream in STREAMS:
        for method in METHODS[stream]:
            for seed in SEEDS:
                for partition in ("REPORT", "FIT"):
                    unit_metrics = {}
                    for stage in (1, 2, 3):
                        key = stream, method, seed, partition, stage
                        cohort = grouped.get(key, [])
                        primary = [r for r in cohort if r["acquisition_unit"] <= stage]
                        expected_n = 8 * stage if stream == STREAMS[0] else 4 * (stage + 1)
                        if len(primary) != expected_n:
                            raise ValueError(f"Wrong primary stage denominator at {key}: {len(primary)} != {expected_n}")
                        scopes = [
                            ("seen_all", 0, primary, True),
                            ("old_units", 0, [r for r in primary if r["acquisition_unit"] < stage], True),
                            ("current_unit", stage, [r for r in primary if r["acquisition_unit"] == stage], True),
                        ]
                        for unit in range(1, stage + 1):
                            rows = [r for r in primary if r["acquisition_unit"] == unit]
                            expected_unit_n = 8 if stream == STREAMS[0] or unit == 1 else 4
                            if len(rows) != expected_unit_n:
                                raise ValueError("Unexpected acquisition-unit denominator.")
                            if len({r["row_id"] for r in rows}) != expected_unit_n:
                                raise ValueError("A unit cohort contains duplicate source records.")
                            scopes.append(("acquired_unit", unit, rows, True))
                            unit_metrics[stage, unit] = metric(rows)
                        if stream == STREAMS[0]:
                            for unit in range(stage + 1, 4):
                                rows = [r for r in cohort if r["evaluation_task"] == unit]
                                if len(rows) != 8:
                                    raise ValueError("Unexpected future-task diagnostic denominator.")
                                scopes.append(("future_task_diagnostic", unit, rows, False))
                        for scope, unit, rows, is_primary in scopes:
                            out = {"stream": stream, "method": method, "seed": seed,
                                   "partition": partition, "stage": stage, "scope": scope,
                                   "acquisition_unit": unit, "eligible_for_primary": is_primary,
                                   **metric(rows)}
                            stage_rows.append(out)
                            stage_lookup[(stream, method, seed, partition, stage, scope, unit)] = out
                    final_rows = [r for r in grouped[stream, method, seed, partition, 3]
                                  if r["acquisition_unit"] <= 3]
                    final = metric(final_rows)
                    if final["n_unique_records"] != 16 or final["n_condition_groups"] != 4:
                        raise ValueError("Final cohort must contain 16 unique averages in four groups.")
                    old = metric([r for r in final_rows if r["acquisition_unit"] < 3])
                    costs = ledger_groups[stream, method, seed]
                    summary = {
                        "stream": stream, "method": method, "seed": seed, "partition": partition,
                        "final_all_correct": final["correct"], "final_all_n_views": final["n_views"],
                        "final_unique_records": final["n_unique_records"],
                        "final_condition_groups": final["n_condition_groups"],
                        "final_all_accuracy_pct": final["accuracy_pct"],
                        "final_all_class_balanced_accuracy_pct": final["class_balanced_accuracy_pct"],
                        "final_all_exact_ties": final["exact_ties"],
                        "final_old_correct": old["correct"], "final_old_n_views": old["n_views"],
                        "final_old_accuracy_pct": old["accuracy_pct"],
                        "training_stage_entries": len(costs),
                        "training_updates": sum(r["updates"] for r in costs),
                        "update_gradient_examples": sum(r["update_gradient_examples"] for r in costs),
                        "teacher_forward_examples": sum(r["teacher_forward_examples"] for r in costs),
                        "parameter_count": costs[-1]["parameter_count"],
                        "parameter_bytes": costs[-1]["parameter_bytes"],
                        "optimizer_m_v_bytes": costs[-1]["optimizer_m_v_bytes"],
                        "codebook_bytes": costs[-1]["codebook_bytes"],
                    }
                    old_acq_correct = old_n = 0
                    weighted_final_drop = 0.0
                    for unit in (1, 2, 3):
                        acq = unit_metrics[unit, unit]
                        end = unit_metrics[3, unit]
                        prior = [unit_metrics[s, unit]["accuracy_pct"] for s in range(unit, 3)]
                        observed = [unit_metrics[s, unit]["accuracy_pct"] for s in range(unit, 4)]
                        signed_final_drop = max(prior) - end["accuracy_pct"] if prior else None
                        max_observed_drop = max([0.0] + [observed[i] - observed[j]
                                                       for i in range(len(observed))
                                                       for j in range(i + 1, len(observed))])
                        prefix = f"unit{unit}_"
                        summary.update({
                            prefix + "n_views": end["n_views"],
                            prefix + "initial_acquisition_accuracy_pct": acq["accuracy_pct"],
                            prefix + "final_accuracy_pct": end["accuracy_pct"],
                            prefix + "final_retention_delta_pp": end["accuracy_pct"] - acq["accuracy_pct"],
                            prefix + "final_max_prior_postacquisition_drop_pp": signed_final_drop,
                            prefix + "maximum_observed_postacquisition_drop_pp": max_observed_drop,
                        })
                        if unit < 3:
                            old_acq_correct += acq["correct"]
                            old_n += acq["n_views"]
                            weighted_final_drop += end["n_views"] * signed_final_drop
                    old_reference = 100 * old_acq_correct / old_n
                    summary["final_old_acquisition_reference_pct"] = old_reference
                    summary["final_old_retention_delta_pp"] = old["accuracy_pct"] - old_reference
                    summary["final_old_weighted_max_prior_postacquisition_drop_pp"] = weighted_final_drop / old_n
                    summaries.append(summary)
    range_rows = []
    by_method = defaultdict(list)
    for row in summaries:
        by_method[row["stream"], row["method"], row["partition"]].append(row)
    science_fields = [k for k in summaries[0] if k.startswith("final_") or k.startswith("unit")]
    for (stream, method, partition), rows in by_method.items():
        if sorted(r["seed"] for r in rows) != list(SEEDS):
            raise ValueError("A summary group lost a declared seed.")
        for field in science_fields:
            valid = [(r["seed"], r[field]) for r in rows if r[field] is not None]
            values = [v for _, v in valid]
            range_rows.append({
                "stream": stream, "method": method, "partition": partition, "metric": field,
                "declared_seed_count": 5, "defined_seed_count": len(valid),
                "minimum": min(values) if values else None,
                "descriptive_mean": mean(values) if values else None,
                "maximum": max(values) if values else None,
                "all_seed_values": ";".join(f"{s}:{v:.12g}" for s, v in valid),
            })
    return summaries, range_rows, stage_rows


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figures(summaries: list[dict], stages: list[dict], figures: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = dict(zip(CODE_METHODS + CE_METHODS,
                      ("#0072B2", "#56B4E9", "#D55E00", "#E69F00",
                       "#009E73", "#CC79A7", "#555555")))
    offsets = dict(zip(SEEDS, (-.18, -.09, 0, .09, .18)))
    generated = []

    def save(fig, stem):
        for suffix in ("png", "pdf", "svg"):
            path = figures / f"{stem}.{suffix}"
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, dpi=300 if suffix == "png" else None,
                        bbox_inches="tight", facecolor="white")
            generated.append(str(path.resolve()))
        plt.close(fig)

    for partition in ("REPORT", "FIT"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), sharey=True)
        for ax, stream in zip(axes, STREAMS):
            for x, method in enumerate(METHODS[stream]):
                rows = [r for r in summaries if r["stream"] == stream and
                        r["method"] == method and r["partition"] == partition]
                values = [r["final_all_accuracy_pct"] for r in rows]
                for row in rows:
                    ax.scatter(x + offsets[row["seed"]], row["final_all_accuracy_pct"],
                               color=colors[method], s=42, edgecolor="white", linewidth=.5, zorder=3)
                ax.plot([x-.23, x+.23], [mean(values)]*2, color="black", linewidth=2, zorder=4)
            ax.set_xticks(range(len(METHODS[stream])))
            ax.set_xticklabels([LABELS[m] for m in METHODS[stream]], rotation=28, ha="right")
            cohort = ("24 pair-conditioned views; 16 unique averages; 4 groups"
                      if stream == STREAMS[0] else "16 unique averages; 4 groups")
            title = "Known-pair task-incremental" if stream == STREAMS[0] else "Seen-class incremental"
            ax.set_title(f"{title}\nFinal denominator: {cohort}", pad=12)
            ax.set_ylim(-3, 105)
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Forced correct / evaluated views (%)")
        fig.suptitle(f"A24 final forced-decision accuracy — {partition}", fontsize=15, y=1.02)
        fig.text(.5, -.10,
                 "Points: all five seeds (101, 211, 307, 401, 503); black line: descriptive mean. "
                 "The two streams use different information and populations.\n"
                 "Previously inspected published averages; no independent specimen validation. "
                 "Code and CE heads differ in size/loss; matched code controls isolate the codebook package.",
                 ha="center", fontsize=9)
        fig.tight_layout()
        save(fig, f"A24_final_accuracy_{partition}_v1")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, sharex=True)
        for col, stream in enumerate(STREAMS):
            for row_index, scope in enumerate(("current_unit", "old_units")):
                ax = axes[row_index, col]
                for method in METHODS[stream]:
                    x_values, y_values = [], []
                    for stage in (1, 2, 3):
                        rows = [r for r in stages if r["stream"] == stream and
                                r["method"] == method and r["partition"] == partition and
                                r["stage"] == stage and r["scope"] == scope and r["n_views"] > 0]
                        if not rows:
                            continue
                        values = [r["accuracy_pct"] for r in rows]
                        x_values.append(stage); y_values.append(mean(values))
                        for r in rows:
                            ax.scatter(stage + offsets[r["seed"]] * .22, r["accuracy_pct"],
                                       color=colors[method], alpha=.28, s=16)
                    ax.plot(x_values, y_values, "o-", color=colors[method], linewidth=1.8,
                            markersize=4, label=LABELS[method].replace("\n", " "))
                stream_title = "Known-pair task-incremental" if stream == STREAMS[0] else "Seen-class incremental"
                scope_title = "Current acquisition unit" if scope == "current_unit" else "Previously acquired units"
                if stream == STREAMS[0]:
                    denominator = "views per stage: 8 / 8 / 8" if scope == "current_unit" else "views per stage: 0 / 8 / 16"
                else:
                    denominator = "unique averages per stage: 8 / 4 / 4" if scope == "current_unit" else "unique averages per stage: 0 / 8 / 12"
                ax.set_title(f"{stream_title}: {scope_title}\n{denominator}")
                ax.set_xticks((1, 2, 3)); ax.set_xlim(.85, 3.15); ax.set_ylim(-3, 105)
                ax.grid(alpha=.22)
                if col == 0:
                    ax.set_ylabel("Forced accuracy (%)")
                if row_index == 1:
                    ax.set_xlabel("Learning stage (not physical time)")
        handles = [Line2D([0], [0], color=colors[m], marker="o", linewidth=1.8,
                          label=LABELS[m].replace("\n", " ")) for m in CODE_METHODS + CE_METHODS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.03), ncol=3, fontsize=10)
        fig.suptitle(f"A24 acquisition and retention across stages — {partition}", fontsize=15)
        fig.text(.5, -.08,
                 "Lines: descriptive means; faint points: every seed. Future task diagnostics are excluded. "
                 "Unit 1 contains classes 1 and 2 in the seen-class stream.\n"
                 "Known-pair task views can reuse the same source average; final denominators are 24 views versus 16 unique averages.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, .96))
        save(fig, f"A24_acquisition_retention_{partition}_v1")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = root / "07_Sonuclar" / "learning_run_v1"
    output = run_dir / "summary_v1"
    figures = root / "09_Sekiller" / "learning_v1"
    if output.exists() or figures.exists():
        raise FileExistsError("Summary/figure output exists; inspect it instead of silently rerunning.")
    decisions, ledger, provenance = load_inputs(run_dir)
    summaries, ranges, stages = table_outputs(decisions, ledger)
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    write_table(output / "summary_by_seed.csv", summaries)
    write_table(output / "summary_ranges.csv", ranges)
    write_table(output / "stage_metrics.csv", stages)
    generated = create_figures(summaries, stages, figures)
    provenance.update({
        "summary_rows": len(summaries), "range_rows": len(ranges), "stage_metric_rows": len(stages),
        "figure_files": generated, "REPORT_FIT_pooled": False,
        "TIL_primary_rule": "evaluation_task <= stage; acquired task views only",
        "TIL_final_REPORT_and_FIT": "24 task views / 16 unique source averages / 4 groups",
        "CIL_final_REPORT_and_FIT": "16 unique source averages / 4 groups",
        "unit_mapping": "TIL: task1/2/3; CIL: classes1&2/unit1, class3/unit2, class4/unit3",
        "retention_delta_pp": "Final unit accuracy minus its own acquisition-stage accuracy",
        "final_max_prior_postacquisition_drop_pp":
            "Maximum accuracy among acquired earlier stages minus final unit accuracy; signed, negative means final exceeds all prior values; no prior stage -> null",
        "maximum_observed_postacquisition_drop_pp":
            "Largest positive earlier-minus-later decline over stages at/after unit acquisition; zero if none",
        "old_acquisition_reference": "Unit-specific acquisition accuracies weighted by their fixed evaluated view counts",
        "cost_note": "Training costs repeat on REPORT and FIT summary rows; count once per stream/method/seed, never sum across partitions",
        "inference": "No p-values, confidence intervals, independent-specimen claims, or physical-time curves",
        "new_training_or_model_fit": 0,
    })
    with (output / "summary_provenance_v1.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "COMPLETE", "summary_directory": str(output),
                      "figure_directory": str(figures), "summary_rows": len(summaries),
                      "stage_metric_rows": len(stages), "figure_files": len(generated)}))


if __name__ == "__main__":
    main()
