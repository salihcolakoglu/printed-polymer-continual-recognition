"""Independent saved Simulink audit. Imports no producer or MATLAB kernel.

MIT. Portable independent numeric/MAT5/XML helpers; no CLI side effects.
All numerical comparisons occur only after an admitted run has saved outputs.
SLX .mxarray is hashed, not falsely claimed independently semantically decoded.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import struct
import zlib


def loadmat(path):
    """Bounded independent MAT-v5 little-endian real-double numeric reader.

    Supports exactly MATLAB -v7 trace/time arrays; rejects other classes.
    MATLAB miCOMPRESSED top-level elements are not eight-byte padded.
    """
    data = Path(path).read_bytes()
    assert len(data) >= 128 and data[126:128] == b"IM", "MAT5 little endian only"
    arrays = {}
    def element(blob, pos, top=False):
        first, second = struct.unpack_from("<II", blob, pos)
        if first >> 16:
            size, typ = first >> 16, first & 0xffff
            assert size <= 4
            return typ, blob[pos + 4:pos + 4 + size], pos + 8
        typ, size = first, second
        end = pos + 8 + size
        assert end <= len(blob)
        nxt = end if top and typ == 15 else end + (-size) % 8
        return typ, blob[pos + 8:end], nxt
    def matrix(payload):
        pieces, p = [], 0
        while p < len(payload):
            t, v, p = element(payload, p)
            pieces.append((t, v))
        assert len(pieces) == 4, "Only noncomplex numeric arrays"
        (ft, flags), (dt, dimensions), (nt, name), (vt, values) = pieces
        assert ft == 6 and len(flags) == 8
        flag = struct.unpack_from("<I", flags)[0]
        assert flag & 0xff == 6 and not flag & 0xa00, "Real double only"
        assert dt == 5 and nt in [1, 2]
        dims = struct.unpack("<" + "i" * (len(dimensions) // 4), dimensions)
        assert all(x >= 0 for x in dims)
        key = name.decode("ascii")
        assert key not in arrays and key in ["trace", "trace_raw", "time"]
        assert vt == 9 or key == "time", "Trace arrays require binary64 storage"
        types = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}
        assert vt in types
        values = np.frombuffer(values, dtype=types[vt]).astype(float)
        assert values.size == math.prod(dims)
        arrays[key] = values.reshape(dims, order="F").copy()
    pos = 128
    while pos < len(data):
        typ, payload, pos = element(data, pos, top=True)
        if typ == 15:
            d = zlib.decompressobj()
            plain = d.decompress(payload, 67108865)
            assert len(plain) <= 67108864 and not d.unconsumed_tail and d.eof
            t, v, end = element(plain, 0)
            assert t == 14 and end == len(plain)
            matrix(v)
        else:
            assert typ == 14
            matrix(payload)
    assert set(arrays) == {"trace", "trace_raw", "time"}
    return arrays



def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def pin(p):
    p = Path(p)
    b = p.read_bytes()
    return {"path": str(p.resolve()), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


class Audit:
    def __init__(self, atol=2e-11, rtol=2e-11):
        self.atol, self.rtol = atol, rtol
        self.checks = 0
        self.failures = []
        self.maximum_absolute_difference = 0.0

    def check(self, name, condition, detail=None):
        self.checks += 1
        if not bool(condition):
            self.failures.append({"check": name, "detail": detail})

    def array(self, name, actual, expected, exact=False):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.check(name + ":shape", a.shape == b.shape, [list(a.shape), list(b.shape)])
        if a.shape != b.shape:
            return
        self.check(name + ":missingness", np.array_equal(np.isnan(a), np.isnan(b)))
        self.check(name + ":infinity", np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(np.isneginf(a), np.isneginf(b)))
        sel = np.isfinite(b)
        if np.any(sel):
            delta = np.abs(a[sel] - b[sel])
            if np.all(np.isfinite(delta)):
                self.maximum_absolute_difference = max(self.maximum_absolute_difference, float(np.max(delta)))
            ok = np.array_equal(a[sel], b[sel]) if exact else np.all(delta <= self.atol + self.rtol * np.abs(b[sel]))
            self.check(name + (":exact" if exact else ":numeric"), ok)


def sigmoid(a):
    out = np.empty_like(a)
    p = a >= 0
    out[p] = 1 / (1 + np.exp(-a[p]))
    e = np.exp(a[~p])
    out[~p] = e / (1 + e)
    return out


def independent_valid_row(bank, u):
    """NumPy matrix formulation of five declared inference families, not chart loops."""
    theta = np.asarray(bank["theta"], dtype=float)
    meta = np.asarray(bank["meta"], dtype=float)
    refs = np.asarray(bank["refs"], dtype=float)
    books = np.asarray(bank["books"], dtype=float)
    u = np.asarray(u, dtype=float)
    mid = int(u[2]) - 1
    kind, width, book, snapshot, count = meta[mid].astype(int)
    p = theta[mid]
    z = (u[:2] - np.asarray(bank["mean"])) / np.asarray(bank["sd"])
    eligible = np.flatnonzero(u[3:] == 1)
    y = np.full(35, np.nan)
    y[:2] = z
    logits, outputs = np.full(8, np.nan), np.full(8, np.nan)
    scores, distances = np.full(4, np.nan), np.full(4, np.nan)
    if kind == 1:
        logits[eligible] = (z @ p[:8].reshape(2, 4, order="F") + p[8:12])[eligible]
    elif kind in [2, 3]:
        h = np.tanh(z @ p[:8].reshape(2, 4, order="F") + p[8:12])
        logits[:width] = h @ p[12:12 + 4 * width].reshape(4, width, order="F") + p[12 + 4 * width:12 + 5 * width]
    if kind in [1, 2]:
        v = logits[eligible] - np.max(logits[eligible])
        expv = np.exp(v)
        prob = np.exp(v - np.log(np.sum(expv))) if kind == 2 else expv / np.sum(expv)
        scores[eligible] = prob
        outputs[eligible] = prob
    elif kind == 3:
        outputs = sigmoid(logits)
        c = books[(book - 1) * 4:book * 4]
        distances[eligible] = np.mean((c[eligible] - outputs) ** 2, axis=1)
        scores[eligible] = -distances[eligible]
    elif kind == 4:
        centers = p[:8].reshape(4, 2, order="F") / p[8:12, None]
        distances[eligible] = np.sum((z - centers[eligible]) ** 2, axis=1)
        scores[eligible] = -distances[eligible]
    elif kind == 5:
        r = refs[(snapshot - 1) * 84:(snapshot - 1) * 84 + count]
        for c in eligible:
            distances[c] = np.min(np.sum((r[r[:, 2] == c + 1, :2] - z) ** 2, axis=1))
        scores[eligible] = -distances[eligible]
    else:
        raise ValueError("Not a valid fixture family")
    best = np.max(scores[eligible])
    winners = eligible[scores[eligible] == best]
    ordered = np.sort(scores[eligible])[::-1]
    y[2:10], y[10:18], y[18:22], y[22:26] = logits, outputs, scores, distances
    y[26:30] = [int(np.min(winners)) + 1, ordered[0] - ordered[1], int(len(winners) > 1), 1]
    y[30:34], y[34] = u[3:], u[2]
    return y


def properties(e):
    return {p.attrib.get("Name"): p.text for p in e.findall("P")}


def verify_saved_slx(audit, run_dir, cfg, bank, chart_path):
    sem = load(run_dir / "saved_model_semantic_readback.json")
    audit.check("cold_reopened", sem["cold_reopened"] is True)
    audit.check("semantic_context", sem["mode"] == cfg["mode"] and sem["fold"] == cfg["fold"])
    for key, saved_key in [("theta", "theta"), ("meta", "meta"), ("mean", "mean"), ("sd", "sd"), ("refs", "refs"), ("books", "books")]:
        audit.array("cold_bank:" + key, sem[saved_key], bank[key], exact=True)
    script = chart_path.read_text(encoding="utf-8-sig").replace(
        "function y = a30_inference_chart_v1(u,theta,meta,mu,sd,refs,books)",
        "function y = fcn(u,theta,meta,mu,sd,refs,books)", 1)
    normalized = lambda s: s.replace("\r\n", "\n").rstrip("\n")
    audit.check("cold_chart_source", normalized(sem["chart_script"]) == normalized(script))
    model = Path(cfg["model_path"])
    audit.check("semantic_model_filename", Path(sem["model_path"].replace("\\", "/")).name == model.name)
    audit.check("model_scope", model.parent.resolve() == run_dir.resolve())
    with zipfile.ZipFile(model) as z:
        audit.check("SLX_CRC", z.testzip() is None)
        scripts = []
        for name in z.namelist():
            if name.startswith("simulink/stateflow/") and name.endswith(".xml"):
                r = ET.fromstring(z.read(name))
                scripts.extend(x.text or "" for x in r.iter("P") if x.attrib.get("Name") == "script")
        audit.check("SLX_one_exact_embedded_chart", len(scripts) == 1 and normalized(scripts[0]) == normalized(script))
        root = ET.fromstring(z.read("simulink/systems/system_root.xml"))
        blocks = {b.attrib["Name"]: b for b in root.findall("Block")}
        expected = {"Raw records and eligibility", "Seven channels", "Normalize and infer", "Full numeric trace", "theta", "meta", "mu", "sd", "refs", "books"}
        audit.check("SLX_exact_blocks", set(blocks) == expected)
        if set(blocks) == expected:
            ids = {n: b.attrib["SID"] for n, b in blocks.items()}
            for n in ["theta", "meta", "mu", "sd", "refs", "books"]:
                audit.check("SLX_const:" + n, blocks[n].attrib["BlockType"] == "Constant" and properties(blocks[n]).get("Value") == n)
            src = properties(blocks["Raw records and eligibility"])
            audit.check("SLX_raw_source", src.get("VariableName") == "queries" and src.get("Interpolate") == "off" and src.get("SampleTime") == "1")
            spec = properties(blocks["Seven channels"])
            audit.check("SLX_input_spec", spec.get("Dimensions") == "7" and spec.get("OutDataTypeStr") == "double")
            sink = properties(blocks["Full numeric trace"])
            audit.check("SLX_trace_sink", sink.get("VariableName") == "trace" and sink.get("SampleTime") == "1")
            target = ids["Normalize and infer"]
            edges = {(properties(e).get("Src"), properties(e).get("Dst")) for e in root.findall("Line")}
            need = {(ids["Raw records and eligibility"] + "#out:1", ids["Seven channels"] + "#in:1"), (ids["Seven channels"] + "#out:1", target + "#in:1"), (target + "#out:1", ids["Full numeric trace"] + "#in:1")}
            need.update((ids[n] + "#out:1", target + "#in:" + str(i + 2)) for i, n in enumerate(["theta", "meta", "mu", "sd", "refs", "books"]))
            audit.check("SLX_exact_wiring", edges == need)
        conf = ET.fromstring(z.read("simulink/configSet0.xml"))
        cp = {e.attrib.get("Name"): e.text for e in conf.iter("P")}
        audit.check("SLX_fixed_step", cp.get("FixedStep") == "1" and "FixedStepDiscrete" in cp.values() and cp.get("StartTime") == "0")
        workspace_parts = [n for n in z.namelist() if "modelWorkspace" in n]
        audit.check("SLX_workspace_present", len(workspace_parts) == 1)
        workspace = {"part": workspace_parts[0], "bytes": z.getinfo(workspace_parts[0]).file_size, "sha256": hashlib.sha256(z.read(workspace_parts[0])).hexdigest()} if workspace_parts else None
    return {"SLX_pin": pin(model), "semantic_pin": pin(run_dir / "saved_model_semantic_readback.json"), "workspace_binary_pin": workspace, "workspace_review_boundary": "Actual cold-reopened MATLAB semantic readback independently compared with pinned bank; binary mxarray content hashed only, no independent mxarray decoder."}

