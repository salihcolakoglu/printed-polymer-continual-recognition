"""MIT. Verify portable saved Simulink payloads without MATLAB or training.

Checks raw CSV, frozen banks, all47502 saved35-channel rows and2610 exact
native-gap populations. Does not execute SLX or independently decode mxarray.
NumPy is the only non-standard Python dependency. Outputs go outside package.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from a30_audit_numeric_core import Audit, independent_valid_row, load, loadmat, pin, verify_saved_slx

def source_vector(r, family, kind, raw, norm, mid):
    """Convert authoritative old output schema into declared35-channel form."""
    z = (np.asarray(raw) - np.asarray(norm["mean"])) / np.asarray(norm["sd"])
    v = np.full(35, np.nan)
    v[:2] = z
    eligible = np.asarray(r["eligible"], dtype=int) - 1
    if family == "matched":
        logits = np.asarray(r["logits"], dtype=float)
        out = np.asarray(r["outputs"], dtype=float)
        v[2:2 + len(logits)] = logits
        v[10:10 + len(out)] = out
        v[18 + eligible] = np.asarray(r["scores"], dtype=float)
        v[22 + eligible] = np.asarray(r["distances"], dtype=float)
        label = r["label"]
        valid = bool(r["valid"]) and not bool(r["failure"])
    else:
        v[2:6] = np.asarray(r["logits"], dtype=float)
        v[18:22] = np.asarray(r["scores"], dtype=float)
        v[22:26] = np.asarray(r["distances"], dtype=float)
        if kind == 1:
            v[10:14] = np.asarray(r["scores"], dtype=float)
        label = r["prediction"]
        valid = r["status"] == "valid"
    v[26:30] = [label, r["gap"], int(r["tie"]), int(valid)]
    v[30:34] = [int(c in r["eligible"]) for c in range(1, 5)]
    v[34] = mid
    return v


def gap_blocks(gaps, valid):
    levels = np.sort(np.unique(gaps[valid]))[::-1]
    return levels, [tuple(np.flatnonzero(valid & (gaps == g)).tolist()) for g in levels]


def native_counts(labels, truth, valid, blocks):
    n = len(labels)
    correct = valid & (labels == truth)
    answered = np.zeros(n, dtype=bool)
    cells = []
    for block in [()] + blocks:
        if block:
            answered[np.asarray(block, dtype=int)] = True
        c = int(np.sum(answered & correct))
        w = int(np.sum(answered & ~correct))
        abstain = int(np.sum(valid & ~answered))
        fail = int(np.sum(~valid))
        count = c + w
        cells.append([c, w, abstain, fail, n, count, count / n, w / count if count else None])
    return cells


def verify_trace(a, run, cfg):
    doc = load(run / "actual_trace.json")
    actual = np.asarray(doc["values"], dtype=float)
    n = cfg["expected_rows"]
    a.check("trace:shape", actual.shape == (n, 35))
    a.array("trace:time", doc["time"], np.arange(n), exact=True)
    mat = loadmat(run / "actual_trace.mat")
    a.array("trace:MAT_JSON", mat["trace"], actual, exact=True)
    a.array("trace:MAT_time", np.ravel(mat["time"]), np.arange(n), exact=True)
    sem = load(run / "saved_model_semantic_readback.json")
    raw = mat["trace_raw"]
    a.check("trace:raw_shape_receipt", list(raw.shape) == list(np.ravel(sem["raw_trace_size"]).astype(int)))
    axes = sem["trace_axes"]
    if axes == "samples_by_channels" and raw.shape == (n, 35):
        remapped = raw
    elif axes == "channels_singleton_samples_permute_312" and raw.shape == (35, 1, n):
        remapped = np.transpose(raw, (2, 0, 1)).reshape(n, 35)
    elif axes == "singleton_channels_samples_permute_321" and raw.shape == (1, 35, n):
        remapped = np.transpose(raw, (2, 1, 0)).reshape(n, 35)
    else:
        a.check("trace:explicit_axes", False, {"shape": list(raw.shape), "axis": axes})
        raise ValueError("Unknown trace axes; do not silently reshape")
    a.array("trace:exact_axes_conversion", remapped, actual, exact=True)
    return actual


def relative_file(root, value):
    rel = Path(value)
    assert not rel.is_absolute() and ":" not in str(value)
    resolved = (root / rel).resolve(strict=True)
    resolved.relative_to(root)
    return resolved


def public_pin_matches(root, card):
    p = relative_file(root, card["path"])
    info = pin(p)
    return info["bytes"] == card["bytes"] and info["sha256"] == card["sha256"]


def colmajor(matrix):
    return [row[j] for j in range(len(matrix[0])) for row in matrix]


def check_projection(a, fold, bank, mapping, source, baseline):
    fits = {j["job_id"]: j for j in source["unique_fits"]}
    stages = {j["id"]: j for j in source["logical_stages"]}
    linear = {j["job_id"]: j for j in baseline["linear_jobs"]}
    deterministic = {j["state_id"]: j for j in baseline["deterministic_states"]}
    a.check(f"f{fold}:522bankentries", len(mapping) == len(bank["theta"]) == len(bank["meta"]) == 522)
    a.check(f"f{fold}:bankIDs", [m["model_id"] for m in mapping] == list(range(1, 523)))
    a.check(f"f{fold}:bankonly_declared_fields", set(bank) == {"schema", "fold", "theta", "meta", "mean", "sd", "refs", "books"})
    norm = next(n for n in baseline["normalizers"] if n["fold"] == fold)
    a.array(f"f{fold}:saved_mean", bank["mean"], norm["mean"], exact=True)
    a.array(f"f{fold}:saved_sd", bank["sd"], norm["sd"], exact=True)
    a.check(f"f{fold}:source_normalizers", source["normalizers"] == baseline["normalizers"])
    a.array(f"f{fold}:pinned_codebooks", bank["books"], source["codebooks"]["OH8"] + source["codebooks"]["WH8"], exact=True)
    refs = [[0.0, 0.0, 0.0] for _ in range(252)]
    for s in range(1, 4):
        nn = deterministic[f"D_f{fold}_s{s}"]["nn"]
        a.check(f"f{fold}:NN{s}capacity", len(nn["X"]) == len(nn["y"]) == len(nn["row_ids"]) <= 84)
        for i, (z, y) in enumerate(zip(nn["X"], nn["y"])):
            refs[(s - 1) * 84 + i] = z + [y]
    a.array(f"f{fold}:allNNrefs_and_padding", bank["refs"], refs, exact=True)
    keys = {}
    for row in mapping:
        mid = row["model_id"]
        a.check(f"f{fold}:model{mid}:fold", row["fold"] == fold)
        if row["family"] == "matched":
            st = stages[row["logical_stage_id"]]
            fit = fits[st["job_id"]]
            d = fit["D"]
            book = 0 if d == 4 else 2 if row["arm"].startswith("WH8") else 1
            params = fit["final_state"]["theta"]
            meta = [2 if d == 4 else 3, d, book, 0, 0]
            a.check(f"f{fold}:model{mid}:logical_identity", all(row[k] == st[k] for k in ["seed", "arm", "stream", "stage", "job_id"]))
            key = ("matched", st["id"])
        else:
            a.check(f"f{fold}:model{mid}:baseline_family", row["family"] == "baseline")
            key = ("baseline", row["stream"], row["stage"], row["method"], row["seed"])
            if row["method"].startswith("LIN_"):
                aliases = [r for r in baseline["aliases"] if all(r[k] == row[k] for k in ["fold", "stream", "stage", "method", "seed"])]
                a.check(f"f{fold}:model{mid}:linear_alias", len(aliases) == 1 and aliases[0]["job_id"] == row["state_id"])
                state = linear[row["state_id"]]["end_state"]
                params = colmajor(state["W"]) + state["b"]
                meta = [1, 4, 0, 0, 0]
            else:
                state = deterministic[row["state_id"]]
                a.check(f"f{fold}:model{mid}:deterministic_identity", state["stage"] == row["stage"] and state["fold"] == fold and row["seed"] is None)
                kind = 4 if row["method"] == "PROTO" else 5
                params = colmajor(state["proto"]["sumS"]) + state["proto"]["countn"] if kind == 4 else []
                meta = [kind, 0, 0, row["stage"], len(state["nn"]["X"])]
        a.array(f"f{fold}:model{mid}:exact_theta_padding", bank["theta"][mid - 1], params + [0.0] * (52 - len(params)), exact=True)
        a.array(f"f{fold}:model{mid}:exact_metadata", bank["meta"][mid - 1], meta, exact=True)
        a.check(f"f{fold}:model{mid}:unique_logical_key", key not in keys)
        keys[key] = mid
    return norm, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replay-root", type=Path, help="Optional external MATLAB replay outputs, containing fold_1..fold_5")
    parser.add_argument("--out", type=Path, required=True, help="New report outside the package")
    parser.add_argument("--fold", type=int, action="append", help="Optional1..5 subset for a bounded fresh replay check; defaultall5")
    parser.add_argument("--staging-map-only", action="store_true", help="Explicit unsealed-staging mode; does not certify whole-package manifest integrity")
    args = parser.parse_args()
    root = args.package_root.resolve(strict=True)
    output = args.out.resolve()
    assert not output.exists(), "Preserve earlier reports."
    assert not output.is_relative_to(root), "Write verification output outside sealed package."
    selected = args.fold or list(range(1, 6))
    assert len(selected) == len(set(selected)) and all(f in range(1, 6) for f in selected)
    start = time.monotonic()
    a = Audit(2e-11, 2e-11)
    map_path = root / "A30_SIMULINK_PUBLIC_MAP_v1.json"
    public = load(map_path)
    a.check("public_map_schema", public["schema"] == "A30_PORTABLE_SIMULINK_INPUT_MAP_V1")
    a.check("five_ordered_folds", [f["fold"] for f in public["folds"]] == list(range(1, 6)))
    required_source_roles = {"source_csv", "baseline_result"} | {f"matched_fold_{f}" for f in range(1, 6)}
    a.check("exact_source_roles", isinstance(public["source_pins"], dict) and set(public["source_pins"]) == required_source_roles)
    for card in public["code_pins"] + list(public["source_pins"].values()):
        a.check("public_input_pin:" + card["path"], public_pin_matches(root, card))
    manifest_path = root / "MANIFEST_SHA256.json"
    if not args.staging_map_only:
        manifest = load(manifest_path)
        listed = {r["path"] for r in manifest["files"]}
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != manifest_path}
        a.check("manifest_exact_membership", listed == actual_files and len(listed) == len(manifest["files"]))
        for card in manifest["files"]:
            a.check("manifest:" + card["path"], public_pin_matches(root, card))
    source_csv = relative_file(root, "02_Veriler/A_source_average_records_v1.csv")
    a.check("original_rawCSV_identity", pin(source_csv)["sha256"] == "6d3902c59e696a53cd36e7bb9a42df1bac1fd9216737e34bfc90bb53b07c70a4")
    with source_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    raw = {r["row_id"]: [float(r["E_MPa"]), float(r["Hardness_ShoreD"])] for r in rows}
    a.check("104unique_source_rows", len(raw) == len(rows) == 104)
    baseline_rel = "07_Sonuclar/BASELINE_RUN_02/results/a29_baseline_result_v1.json"
    required_sources = {"02_Veriler/A_source_average_records_v1.csv", baseline_rel}
    required_sources.update(f"07_Sonuclar/CL_SOURCE_FOLD_0{f}_RUN_01/a29cl_source_fold{f}_v1.json" for f in range(1, 6))
    a.check("all_source_inputs_pinned", required_sources <= {p["path"] for p in public["source_pins"].values()})
    baseline = load(relative_file(root, baseline_rel))
    totalrows = totalpop = 0
    summaries = []
    for entry in public["folds"]:
        fold = entry["fold"]
        if fold not in selected:
            continue
        for field in ["bank", "inputs", "model_map", "reference_predictions", "slx", "trace_json", "trace_mat", "semantic"]:
            a.check(f"f{fold}:pin:" + field, public_pin_matches(root, entry[field]))
        bank_path = relative_file(root, entry["bank"]["path"])
        bank = load(bank_path)
        mapping = load(relative_file(root, entry["model_map"]["path"]))
        reference = load(relative_file(root, entry["reference_predictions"]["path"]))
        source = load(relative_file(root, f"07_Sonuclar/CL_SOURCE_FOLD_0{fold}_RUN_01/a29cl_source_fold{fold}_v1.json"))
        norm, logical_keys = check_projection(a, fold, bank, mapping, source, baseline)
        with relative_file(root, entry["inputs"]["path"]).open(encoding="utf-8-sig", newline="") as f:
            inputs = np.asarray([[float(v) for v in r] for r in csv.reader(f)])
        oldrows = [("matched", i, p) for i, p in enumerate(source["predictions"])] + [("baseline", i, p) for i, p in enumerate(baseline["predictions"]) if p["fold"] == fold]
        run = args.replay_root.resolve(strict=True) / f"fold_{fold}" if args.replay_root else bank_path.parent
        sem = load(run / "saved_model_semantic_readback.json")
        slx = list(run.glob(f"a30_inference_f{fold}_*.slx"))
        a.check(f"f{fold}:one_actual_SLX", len(slx) == 1)
        cfg = {"mode": sem["mode"], "fold": fold, "expected_rows": entry["rows"], "model_path": str(slx[0]), "code_dir": str(root / "03_MATLAB_Kodlari/simulink_parity_v1")}
        actual = verify_trace(a, run, cfg)
        saved = verify_saved_slx(a, run, cfg, bank, Path(cfg["code_dir"]) / "a30_inference_chart_v1.m")
        a.check(f"f{fold}:full_rows", len(actual) == len(inputs) == len(reference) == len(oldrows) == entry["rows"] == (10962 if fold == 1 else 9135))
        numpy_bank = dict(bank)
        for k in ["theta", "meta", "refs", "books", "mean", "sd"]:
            numpy_bank[k] = np.asarray(bank[k], dtype=float)
        mids, labels, gaps, truths, valid = [], [], [], [], []
        for i, ((family, index, old), inp, got, exported) in enumerate(zip(oldrows, inputs, actual, reference)):
            if i % 1000 == 0:
                assert time.monotonic() - start <= 180, "Finite180s verification budget exceeded"
            key = ("matched", old["logical_stage_id"]) if family == "matched" else ("baseline", old["stream"], old["stage"], old["method"], old["seed"])
            mid = logical_keys[key]
            prefix = f"f{fold}:row{i + 1}"
            a.check(prefix + ":exact_source_gold", exported == {"family": family, "source_prediction_index": index, "model_id": mid, "record": old})
            a.array(prefix + ":raw_input_selector_mask", inp, raw[old["parentrow"]] + [mid] + [int(c in old["eligible"]) for c in range(1, 5)], exact=True)
            expected = source_vector(old, family, int(bank["meta"][mid - 1][0]), raw[old["parentrow"]], norm, mid)
            a.array(prefix + ":source35", got, expected)
            a.array(prefix + ":exact_discrete_selector_mask", got[[26, 28, 29, 30, 31, 32, 33, 34]], expected[[26, 28, 29, 30, 31, 32, 33, 34]], exact=True)
            with np.errstate(all="ignore"):
                independent = independent_valid_row(numpy_bank, inp)
            a.array(prefix + ":independent_matrix35", got, independent)
            a.array(prefix + ":independent_discrete", got[[26, 28, 29]], independent[[26, 28, 29]], exact=True)
            if family == "matched":
                a.array(prefix + ":recorded_Z", expected[:2], old["Z"])
            mids.append(mid)
            labels.append(expected[26])
            gaps.append(old["gap"])
            valid.append(expected[29] == 1)
            truths.append(old["truth"])
        mids, labels, gaps, valid, truths = np.asarray(mids), np.asarray(labels), np.asarray(gaps), np.asarray(valid, dtype=bool), np.asarray(truths)
        a.check(f"f{fold}:all522logical_populations", set(mids) == set(range(1, 523)))
        actual_valid = (actual[:, 29] == 1) & np.isfinite(actual[:, 26]) & np.isfinite(actual[:, 27])
        for mid in range(1, 523):
            ii = np.flatnonzero(mids == mid)
            la, ba = gap_blocks(actual[ii, 27], actual_valid[ii])
            ls, bs = gap_blocks(gaps[ii], valid[ii])
            a.check(f"f{fold}:population{mid}:exact_native_block_partition_order", ba == bs)
            a.array(f"f{fold}:population{mid}:native_thresholds", la, ls)
            ca = native_counts(actual[ii, 26], truths[ii], actual_valid[ii], ba)
            cs = native_counts(labels[ii], truths[ii], valid[ii], bs)
            a.check(f"f{fold}:population{mid}:all_CWAFN_cells", ca == cs)
            a.check(f"f{fold}:population{mid}:undefined_error_null", ca[0][-1] is None and cs[0][-1] is None)
        summaries.append({"fold": fold, "rows": len(actual), "logical_populations": 522, "unexpected_failures": int(np.sum(~actual_valid)), "trace": pin(run / "actual_trace.json"), "saved_model": saved})
        totalrows += len(actual)
        totalpop += 522
    expected_rows = sum(10962 if f == 1 else 9135 for f in selected)
    a.check("complete_requested_decisions_populations", totalrows == expected_rows and totalpop == 522 * len(selected))
    if set(selected) == set(range(1, 6)):
        a.check("complete47502_decisions2610_populations", totalrows == 47502 and totalpop == 2610)
    report = {"schema": "A30_PORTABLE_SAVED_SIMULINK_AUDIT_V1", "status": "HOLD" if a.failures else "PASS_FOR_STAGING_MAP_AND_SAVED_PARITY" if args.staging_map_only else "PASS", "created_utc": datetime.now(timezone.utc).isoformat(), "checks": a.checks, "failure_count": len(a.failures), "failures": a.failures, "folds_requested": selected, "all_five_complete": set(selected) == set(range(1, 6)), "logical_decisions": totalrows, "logical_populations": totalpop, "maximum_absolute_difference": a.maximum_absolute_difference, "elapsed_seconds": time.monotonic() - start, "whole_package_manifest_verified": not args.staging_map_only, "public_map": pin(map_path), "checker": pin(Path(__file__)), "numeric_core": pin(Path(__file__).with_name("a30_audit_numeric_core.py")), "folds": summaries, "scope": "Saved-payload inference/missingness/decisions/coverage parity and savedSLX XML/cold-runtime-readback consistency. Does not execute MATLAB/Simulink or independently decode binary modelWorkspace.mxarray. No training, physical validation or algorithmic performance claim."}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if len(encoded.encode("utf-8")) > 33554432:
        report["failure_report_truncated_under_32MiB_cap"] = True
        report["failures"] = report["failures"][:100]
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    assert len(encoded.encode("utf-8")) <= 33554432
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": a.checks, "failures": len(a.failures), "rows": totalrows, "populations": totalpop, "report": str(output)}, ensure_ascii=False))
    return 1 if a.failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())


