#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CHF-PT demo server: the five released checkpoints behind a web form.

    python server.py                        # local link + a temporary public link
    python server.py --tunnel-name chfpt    # local link + a permanent public link
    python server.py --no-tunnel            # local network only
    python server.py --port 8080 --device cpu

No argument is required: run the file as it is (the green Run button of an
editor included) and it serves on the first free port from 5000 up, so another
demo already holding 5000 on this machine is not in the way. CHFPT_PORT,
CHFPT_HOST, CHFPT_DEVICE and CHFPT_NO_TUNNEL set the same things from the
environment, for a launch configuration that passes no arguments.

What it serves
    /                    the interface
    GET  /api/schema     the input schema the interface is built from
    POST /api/derive     saturation properties, derived groups and anchors only
    POST /api/predict    the four delivered outputs: CHF, the two conformal
                         bounds and the novelty score, plus every intermediate
    POST /api/sweep      the same prediction along a sweep of one input
    POST /api/map        CHF over a plane of two operating variables
    POST /api/attribution  how much the prediction moves when one token is withheld
    POST /api/design     size one variable for a margin against the predicted limit
    POST /api/batch      a whole table of operating points in one call
    GET  /api/example    a real record of one section, drawn at random
    GET  /api/health     liveness and what is loaded

Everything the model needs travels inside the checkpoints and ``assets/``; the
32,271-record database is not needed at run time and is not shipped here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file

try:                                        # the WSGI server the demo prefers
    from waitress import serve as _waitress_serve
except ModuleNotFoundError:                 # an environment without it still runs
    _waitress_serve = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chfpt_core import build_model, load_neural_checkpoint          # noqa: E402
import physics                                                      # noqa: E402
from physics import DeriveError, FluidProperties, derive_record, record_to_frame   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chfpt")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["JSON_SORT_KEYS"] = False

def _default_model_dir() -> Path:
    """Weights live in ``models/`` when the demo is shipped on its own, and in
    ``../model/chfpt`` inside the release repository. ``CHFPT_MODEL_DIR`` overrides both."""
    env = os.environ.get("CHFPT_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    local = HERE / "models"
    if any(local.glob("chfpt_seed*.pt")):
        return local
    shared = HERE.parent / "model" / "chfpt"
    return shared if any(shared.glob("chfpt_seed*.pt")) else local


MODEL_DIR = _default_model_dir()
ASSETS = HERE / "assets"
INDEX = HERE / "index.html"
DEFAULT_PORT = int(os.environ.get("CHFPT_PORT", 5000))
MAX_SWEEP = 96
MAX_BATCH = 2000          # rows accepted in one batch file
INFER_CHUNK = 512         # rows per forward pass, so a large batch cannot spike memory


# ===========================================================================
# 1. Everything that is loaded once
# ===========================================================================
class Ensemble:
    """The five seeds, the frozen preprocessors and the calibration constants."""

    def __init__(self, device: torch.device):
        self.device = device
        self.lock = threading.Lock()
        self.props = FluidProperties(ASSETS / "fluid_props.json")
        with open(ASSETS / "schema.json", "r", encoding="utf-8") as fh:
            self.schema = json.load(fh)
        with open(ASSETS / "examples.json", "r", encoding="utf-8") as fh:
            self.examples = json.load(fh)

        self.entries = []
        for path in sorted(MODEL_DIR.glob("chfpt_seed*.pt")):
            cfg, pp, state, ck = load_neural_checkpoint(path, map_location=device)
            model = build_model(cfg, pp).to(device).eval()
            model.load_state_dict(state)
            n = sum(p.numel() for p in model.parameters())
            self.entries.append({"seed": int(ck["seed"]), "model": model, "pp": pp,
                                 "params": n, "file": path.name})
            logger.info(f"loaded {path.name}  seed {int(ck['seed'])}  {n/1e6:.2f} M parameters")
        if not self.entries:
            raise SystemExit(f"no checkpoints found in {MODEL_DIR}")

        c = np.load(ASSETS / "calibration.npz")
        order = [int(s) for s in c["seeds"]]
        idx = [order.index(e["seed"]) for e in self.entries]
        self.alphas = [float(a) for a in c["alphas"]]
        self.Q_ens = {round(float(a), 4): float(q) for a, q in zip(c["alphas"], c["Q_ens"])}
        self.coverage = {round(float(a), 4): float(v) for a, v in zip(c["alphas"], c["coverage_ens"])}
        # the novelty score inverts a 384 x 384 covariance, so it is kept in double
        self.z_mean = torch.tensor(c["z_mean"][idx], dtype=torch.float64, device=device)
        self.z_prec = torch.tensor(c["z_prec"][idx], dtype=torch.float64, device=device)
        self.ood_pct = {int(k): float(v) for k, v in zip(c["ood_pct_levels"], c["ood_pct"])}
        logger.info(f"calibration: {len(self.entries)} seeds | 90% interval Q="
                    f"{self.Q_ens[0.1]:.4f} | novelty p95={self.ood_pct[95]:.1f}")

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def _forward(self, frame):
        """Run every seed on one frame, in chunks. Arrays of shape (seeds, rows)."""
        mus, q05, q95, ood = [], [], [], []
        n = len(frame)
        for k, e in enumerate(self.entries):
            enc = e["pp"].encode(frame)
            num_val = np.concatenate([enc["num_val"], enc["anc_val"]], 1)
            num_mask = np.concatenate([enc["num_mask"], enc["anc_mask"]], 1)
            cat = enc["cat_code"]
            a, b, c, d_ = [], [], [], []
            for s0 in range(0, n, INFER_CHUNK):
                sl = slice(s0, min(s0 + INFER_CHUNK, n))
                out = e["model"](torch.tensor(num_val[sl], device=self.device),
                                 torch.tensor(num_mask[sl], device=self.device),
                                 torch.tensor(cat[sl], device=self.device))
                a.append(out["mu"].float().cpu().numpy())
                b.append(out["q05"].float().cpu().numpy())
                c.append(out["q95"].float().cpu().numpy())
                dz = out["z"].double() - self.z_mean[k]
                d_.append(torch.sqrt(torch.clamp((dz @ self.z_prec[k] * dz).sum(-1),
                                                 min=0.0)).cpu().numpy())
            mus.append(np.concatenate(a)); q05.append(np.concatenate(b))
            q95.append(np.concatenate(c)); ood.append(np.concatenate(d_))
        return (np.array(mus), np.array(q05), np.array(q95), np.array(ood))

    def predict(self, records, alpha: float = 0.10, anchor_override: float = None):
        """The delivered outputs for a list of records, as plain python floats.

        ``anchor_override`` keeps the residual base fixed while the records
        differ, which is what isolates the effect of a single token.
        """
        frame = record_to_frame(records)
        anchor = (frame["anchor_zuber_kW_m2"].to_numpy(dtype=np.float64)
                  if anchor_override is None
                  else np.full(len(records), float(anchor_override)))
        la = np.log(anchor)
        with self.lock:
            t0 = time.perf_counter()
            mu, q05, q95, ood = self._forward(frame)
            dt = (time.perf_counter() - t0) * 1e3

        per_seed = np.exp(mu + la)                       # (seeds, rows)
        y_hat = per_seed.mean(0)
        Q = self.Q_ens[round(alpha, 4)]
        lower = np.exp(q05.mean(0) + la - Q)
        upper = np.exp(q95.mean(0) + la + Q)
        return {"y_hat": y_hat, "per_seed": per_seed, "lower": lower, "upper": upper,
                "ood": ood.mean(0), "anchor": anchor, "ms": dt}

    def novelty_level(self, score: float) -> dict:
        p = self.ood_pct
        if score <= p[90]:
            lab, key = "typical of the corpus", "in"
        elif score <= p[99]:
            lab, key = "unusual, still represented", "edge"
        else:
            lab, key = "outside the corpus", "out"
        return {"score": float(score), "label": lab, "level": key,
                "percentiles": {str(k): v for k, v in p.items()}}


ENS: Ensemble = None       # set in main()


# ===========================================================================
# 2. Request handling
# ===========================================================================
def _payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise DeriveError("expected a JSON object")
    return data


def _alpha(data) -> float:
    a = data.get("alpha", 0.10)
    try:
        a = round(float(a), 4)
    except (TypeError, ValueError):
        raise DeriveError("alpha must be a number")
    if a not in ENS.Q_ens:
        raise DeriveError(f"alpha must be one of {sorted(ENS.Q_ens)}")
    return a


def _num(v: float) -> str:
    """Readable in a sentence: thousands separated, four significant digits."""
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def _outside_corpus(payload: dict, record: dict) -> list:
    """Which inputs leave the range this test section was pre-trained on.

    Leaving it is allowed and the model still answers; the interface only says
    so, because a prediction there is an extrapolation.
    """
    geo = next((g for g in ENS.schema["geometries"]
                if g["key"] == payload.get("geometry_type")), None)
    if geo is None:
        return []
    out = []
    checks = [("P_kPa", "pressure", "kPa", 1.0),
              ("G_kg_m2s", "mass flux", "kg m-2 s-1", 1.0),
              ("DT_sub_K", "subcooling", "K", 1.0),
              ("heated_length_m", "heated length", "m", 1.0),
              ("char_length_m", "channel scale", "mm", 1e3)]
    key_of = {"char_length_m": "char_length_mm"}
    for key, name, unit, mul in checks:
        r = geo["ranges"].get(key_of.get(key, key)) or {}
        v = record.get(key)
        if v is None or not r:
            continue
        v = v * mul
        if v < r["min"] or v > r["max"]:
            # the numbers travel unformatted so the interface can show them in
            # whatever unit the reader has selected
            out.append({"kind": "range", "field": key_of.get(key, key), "name": name,
                        "value": float(v), "unit": unit,
                        "min": float(r["min"]), "max": float(r["max"])})
    if payload.get("custom_fluid"):
        out.append({"kind": "note", "field": "fluid",
                    "text": "a fluid defined by its own properties; the corpus pairs this section "
                            "with " + ", ".join(geo["fluids"])})
    elif payload.get("fluid") not in geo["fluids"]:
        out.append({"kind": "note", "field": "fluid",
                    "text": f"{payload.get('fluid')} in this section; the corpus pairs it with "
                            + ", ".join(geo["fluids"])})
    return out


@app.route("/")
def index():
    return send_file(INDEX)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "device": str(ENS.device),
                    "seeds": [e["seed"] for e in ENS.entries],
                    "parameters_M": round(ENS.entries[0]["params"] / 1e6, 2),
                    "alphas": ENS.alphas})


@app.route("/api/schema", methods=["GET"])
def schema():
    s = dict(ENS.schema)
    s["runtime"] = {"device": str(ENS.device), "seeds": [e["seed"] for e in ENS.entries],
                    "parameters_M": round(ENS.entries[0]["params"] / 1e6, 2)}
    return jsonify(s)


@app.route("/api/example", methods=["GET"])
def example():
    """A real record of one test section, drawn at random from the corpus.

    The pool covers all four subsets and every entry says which one it came
    from, because a training record is one the checkpoints have already seen:
    comparing the prediction with the measurement only means something on the
    validation, calibration and test subsets.
    """
    geom = request.args.get("geometry") or ""
    pool = ENS.examples.get(geom)
    if not pool:
        return jsonify({"ok": False, "error": f"no examples for {geom!r}"}), 400
    avoid = request.args.get("avoid")
    choices = [e for e in pool if str(e["row_id"]) != str(avoid)] or pool
    return jsonify({"ok": True, "example": random.choice(choices), "pool": len(pool)})


@app.route("/api/derive", methods=["POST"])
def derive():
    """Everything the form implies, without running the network."""
    try:
        data = _payload()
        record, report = derive_record(data, ENS.props)
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    report["outside"] = _outside_corpus(data, record)
    return jsonify({"ok": True, "derived": report,
                    "record": {k: v for k, v in record.items() if v is not None}})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = _payload()
        alpha = _alpha(data)
        record, report = derive_record(data, ENS.props)
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:                                   # noqa: BLE001
        logger.exception("derive failed")
        return jsonify({"ok": False, "error": f"could not build the record: {e}"}), 400

    try:
        p = ENS.predict([record], alpha)
    except Exception as e:                                   # noqa: BLE001
        logger.exception("inference failed")
        return jsonify({"ok": False, "error": f"inference failed: {e}"}), 500

    y = float(p["y_hat"][0])
    per_seed = [{"seed": e["seed"], "chf_kW_m2": float(v)}
                for e, v in zip(ENS.entries, p["per_seed"][:, 0])]
    spread = float(np.std(p["per_seed"][:, 0]) / y * 100) if y > 0 else 0.0
    report["outside"] = _outside_corpus(data, record)

    out = {
        "ok": True,
        "chf_kW_m2": y,
        "interval": {"alpha": alpha, "confidence": round((1 - alpha) * 100),
                     "lower_kW_m2": float(p["lower"][0]), "upper_kW_m2": float(p["upper"][0]),
                     "empirical_coverage": ENS.coverage[alpha]},
        "seeds": {"per_seed": per_seed, "spread_pct": spread},
        "novelty": ENS.novelty_level(float(p["ood"][0])),
        "derived": report,
        "record": {k: v for k, v in record.items() if v is not None},
        "anchor_ratio": y / float(p["anchor"][0]),
        "timing_ms": round(p["ms"], 1),
    }
    q_op = physics._f(data.get("q_op_kW_m2"))
    if q_op and q_op > 0:
        out["margin"] = {"q_op_kW_m2": q_op, "DNBR": y / q_op,
                         "DNBR_lower": float(p["lower"][0]) / q_op}
    return jsonify(out)


def _apply_variable(payload: dict, var: str, x: float) -> dict:
    """One point of a sweep: the payload with ``var`` set to ``x``."""
    one = dict(payload)
    if var == "char_length_mm":
        geo = dict(one.get("geometry") or {})
        src = physics.CHAR_LENGTH_FROM.get(one.get("geometry_type"))
        if src:
            geo[src] = float(x)
            one["geometry"] = geo
        one["char_length_mm"] = float(x)
    else:
        one[var] = float(x)
        if var == "DT_sub_K":
            one["subcooling_mode"] = "DT_sub"
    return one


def _axis(data: dict, key: str, known: dict):
    """Read a sweep axis: variable name, bounds, number of points, log or not."""
    var = str(data.get(key) or "")
    if var not in known:
        raise DeriveError(f"{key} must be one of {sorted(known)}")
    lo, hi = physics._f(data.get(key + "_min")), physics._f(data.get(key + "_max"))
    if lo is None or hi is None or not (hi > lo):
        raise DeriveError(f"{key}_min and {key}_max are required and must increase")
    n = int(data.get(key + "_n") or 32)
    log = bool(known[var]["log"]) and lo > 0
    return var, lo, hi, max(4, min(64, n)), log


def _grid(lo, hi, n, log):
    return (np.exp(np.linspace(math.log(lo), math.log(hi), n)) if log
            else np.linspace(lo, hi, n))


@app.route("/api/map", methods=["POST"])
def operating_map():
    """CHF over a plane of two operating variables: the map an engineer reads."""
    try:
        data = _payload()
        alpha = _alpha(data)
        known = {v["key"]: v for v in ENS.schema["sweep_variables"]}
        xv, xlo, xhi, nx, xlog = _axis(data, "x", known)
        yv, ylo, yhi, ny, ylog = _axis(data, "y", known)
        if xv == yv:
            raise DeriveError("the two axes must use different variables")
        if nx * ny > 2000:
            raise DeriveError("the grid is limited to 2,000 points")
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    xs, ys = _grid(xlo, xhi, nx, xlog), _grid(ylo, yhi, ny, ylog)
    base = {k: v for k, v in data.items()
            if k not in ("x", "x_min", "x_max", "x_n", "y", "y_min", "y_max", "y_n")}
    records, holes = [], []
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            try:
                rec, _ = derive_record(_apply_variable(_apply_variable(base, yv, yy), xv, xx),
                                       ENS.props)
                records.append(rec)
            except DeriveError:
                holes.append((j, i))
    if not records:
        return jsonify({"ok": False, "error": "no valid record on this grid"}), 400

    p = ENS.predict(records, alpha)
    z, k = [], 0
    hole = set(holes)
    for j in range(len(ys)):
        row = []
        for i in range(len(xs)):
            if (j, i) in hole:
                row.append(None)
            else:
                row.append(float(p["y_hat"][k])); k += 1
        z.append(row)
    return jsonify({"ok": True,
                    "x": {"key": xv, "label": known[xv]["label"], "unit": known[xv]["unit"],
                          "log": xlog, "values": [float(v) for v in xs]},
                    "y": {"key": yv, "label": known[yv]["label"], "unit": known[yv]["unit"],
                          "log": ylog, "values": [float(v) for v in ys]},
                    "chf_kW_m2": z, "n": len(records)})


@app.route("/api/attribution", methods=["POST"])
def attribution():
    """How much the prediction moves when one token is withheld.

    Each defined feature is removed in turn, as for a record that never
    measured it, and the model is run again on the same residual
    base, so the change is the marginal weight the network puts on that token.
    The layer is deliberately over-complete (the density ratio, for one, is
    implied by two other tokens), so these are marginal effects, not an
    additive decomposition.
    """
    try:
        data = _payload()
        record, _ = derive_record(data, ENS.props)
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    base_anchor = float(record["anchor_zuber_kW_m2"])
    defined = [k for k, v in record.items() if v is not None]
    variants = [record] + [{**record, k: None} for k in defined]
    p = ENS.predict(variants, 0.10, anchor_override=base_anchor)
    y = p["y_hat"]
    base = float(y[0])
    rows = [{"feature": k, "chf_without_kW_m2": float(v),
             "delta_pct": float((v - base) / base * 100)}
            for k, v in zip(defined, y[1:])]
    rows.sort(key=lambda r: -abs(r["delta_pct"]))
    return jsonify({"ok": True, "chf_kW_m2": base, "n_features": len(defined),
                    "features": rows})


def _cross(xs, ys, i, j, target, log):
    """Where between grid points i and j the curve crosses ``target``."""
    y0, y1 = ys[i], ys[j]
    if y1 == y0:
        return float(xs[j])
    t = (target - y0) / (y1 - y0)
    t = min(max(t, 0.0), 1.0)
    if log and xs[i] > 0 and xs[j] > 0:
        return float(math.exp(math.log(xs[i]) + t * (math.log(xs[j]) - math.log(xs[i]))))
    return float(xs[i] + t * (xs[j] - xs[i]))


@app.route("/api/design", methods=["POST"])
def design():
    """Size one variable for a margin against the predicted limit.

    The margin is taken on the **lower** conformal bound, not on the point
    prediction: q_LCB / q_op is what a designer can defend. The curve is scanned
    rather than assumed monotone, and the answer is the boundary of the safe
    region nearest to the operating point, so a non-monotone response cannot
    produce a wrong single number.
    """
    try:
        data = _payload()
        alpha = _alpha(data)
        q_op = physics._f(data.get("q_op_kW_m2"))
        if q_op is None or q_op <= 0:
            raise DeriveError("an operating heat flux is required to size for a margin")
        sf = physics._f(data.get("safety_factor")) or 1.3
        if sf <= 0:
            raise DeriveError("the safety factor must be positive")
        known = {v["key"]: v for v in ENS.schema["sweep_variables"]}
        var, lo, hi, n, log = _axis(data, "x", known)
        base_rec, _ = derive_record(data, ENS.props)
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # where the operating point sits on this axis
    cur = (float(base_rec["char_length_m"]) * 1e3 if var == "char_length_mm"
           else physics._f(base_rec.get(var)))

    xs = _grid(lo, hi, max(n, 24), log)
    records, kept = [base_rec], []
    for x in xs:
        try:
            rec, _ = derive_record(_apply_variable(data, var, float(x)), ENS.props)
        except DeriveError:
            continue
        records.append(rec)
        kept.append(float(x))
    if len(records) < 3:
        return jsonify({"ok": False, "error": "the range produced no usable record"}), 400

    p = ENS.predict(records, alpha)
    base_lower = float(p["lower"][0])
    lower = p["lower"][1:]
    chf = p["y_hat"][1:]
    margin = [float(v) / q_op for v in lower]
    ok = [m >= sf for m in margin]

    ans = {"satisfied": None, "required": None, "direction": None,
           "safe_from": None, "safe_to": None}
    if any(ok):
        i_cur = min(range(len(kept)), key=lambda i: abs(kept[i] - (cur if cur is not None
                                                                  else kept[0])))
        if ok[i_cur]:
            ans["satisfied"] = True
            j = i_cur
            while j > 0 and ok[j - 1]:
                j -= 1
            ans["safe_from"] = (None if j == 0 else
                                _cross(kept, margin, j - 1, j, sf, log))
            j = i_cur
            while j < len(ok) - 1 and ok[j + 1]:
                j += 1
            ans["safe_to"] = (None if j == len(ok) - 1 else
                              _cross(kept, margin, j, j + 1, sf, log))
        else:
            ans["satisfied"] = False
            up = next((i for i in range(i_cur + 1, len(ok)) if ok[i]), None)
            dn = next((i for i in range(i_cur - 1, -1, -1) if ok[i]), None)
            pick, direction = None, None
            if up is not None and (dn is None or (up - i_cur) <= (i_cur - dn)):
                pick, direction = _cross(kept, margin, up - 1, up, sf, log), "increase"
            elif dn is not None:
                pick, direction = _cross(kept, margin, dn + 1, dn, sf, log), "decrease"
            ans["required"], ans["direction"] = pick, direction
    else:
        ans["satisfied"] = False

    return jsonify({"ok": True, "variable": var, "label": known[var]["label"],
                    "unit": known[var]["unit"], "log": log, "current": cur,
                    "q_op_kW_m2": q_op, "safety_factor": sf,
                    "allowable_q_op_kW_m2": base_lower / sf,
                    "margin_now": base_lower / q_op,
                    "x": kept, "margin": margin,
                    "chf_kW_m2": [float(v) for v in chf],
                    "lower_kW_m2": [float(v) for v in lower],
                    **ans})


def _merge_row(base: dict, row: dict):
    """One row of a batch file on top of the form values.

    A column the file does not carry keeps the value on the left, so a study
    that varies two quantities needs only those two columns. Naming a different
    test section drops the geometry of the form, since its dimensions would
    otherwise leak in as tokens the new section does not have.
    """
    out = {k: v for k, v in base.items()
           if k not in ("rows", "base", "alpha", "variable", "min", "max", "n")}
    given = {k: v for k, v in row.items()
             if v is not None and str(v).strip() != ""}
    changed = ("geometry_type" in given
               and str(given["geometry_type"]) != str(base.get("geometry_type")))
    geo = {} if changed else dict(base.get("geometry") or {})
    custom = dict(base.get("custom_fluid") or {})
    unknown = []
    for k, v in given.items():
        if k in physics.GEOMETRY_KEYS or k in physics.GEOMETRY_CATS:
            geo[k] = v
        elif k.startswith("custom_"):
            custom[k[len("custom_"):]] = v
        elif k in physics.COMMON_KEYS:
            out[k] = v
        elif k not in ("id", "row_id", "label", "name", "note"):
            unknown.append(k)
    out["geometry"] = geo
    if custom and (base.get("custom_fluid") or any(k.startswith("custom_") for k in given)):
        out["custom_fluid"] = custom
    if "subcooling_mode" not in given:
        t, d = "T_in_C" in given, "DT_sub_K" in given
        out["subcooling_mode"] = "both" if (t and d) else ("T_in" if t else
                                                           ("DT_sub" if d else
                                                            base.get("subcooling_mode", "DT_sub")))
    return out, unknown


@app.route("/api/batch", methods=["POST"])
def batch():
    """Predict a whole table of operating points in one call."""
    try:
        data = _payload()
        alpha = _alpha(data)
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            raise DeriveError("send the table as 'rows': a list of objects")
        if len(rows) > MAX_BATCH:
            raise DeriveError(f"at most {MAX_BATCH:,} rows in one call, {len(rows):,} sent")
        base = data.get("base") or {}
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    results = [None] * len(rows)
    records, index, unknown = [], [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            results[i] = {"ok": False, "error": "not an object"}
            continue
        merged, unk = _merge_row(base, row)
        unknown.update(unk)
        try:
            rec, rep = derive_record(merged, ENS.props)
        except DeriveError as e:
            results[i] = {"ok": False, "error": str(e)}
            continue
        outside = _outside_corpus(merged, rec)
        records.append(rec)
        index.append((i, merged, rec, outside))

    if records:
        p = ENS.predict(records, alpha)
        for k, (i, merged, rec, outside) in enumerate(index):
            y = float(p["y_hat"][k])
            r = {"ok": True, "chf_kW_m2": y,
                 "lower_kW_m2": float(p["lower"][k]), "upper_kW_m2": float(p["upper"][k]),
                 "ood_score": float(p["ood"][k]),
                 "ood_level": ENS.novelty_level(float(p["ood"][k]))["level"],
                 "anchor_zuber_kW_m2": float(rec["anchor_zuber_kW_m2"]),
                 "anchor_bowe_kW_m2": rec["anchor_bowe_kW_m2"],
                 "anchor_bowring_kW_m2": rec["anchor_bowring_kW_m2"],
                 "outside_corpus": len(outside),
                 "outside": "; ".join(o.get("name", o.get("text", "")) for o in outside)}
            q_op = physics._f(merged.get("q_op_kW_m2"))
            if q_op and q_op > 0:
                r["DNBR"] = y / q_op
                r["DNBR_lower"] = float(p["lower"][k]) / q_op
            results[i] = r

    n_ok = sum(1 for r in results if r and r.get("ok"))
    return jsonify({"ok": True, "results": results, "n": len(rows), "n_ok": n_ok,
                    "n_failed": len(rows) - n_ok,
                    "ignored_columns": sorted(unknown), "confidence": round((1 - alpha) * 100)})


@app.route("/api/sweep", methods=["POST"])
def sweep():
    """One input varied over a range, everything else held fixed."""
    try:
        data = _payload()
        alpha = _alpha(data)
        var = str(data.get("variable") or "")
        known = {v["key"]: v for v in ENS.schema["sweep_variables"]}
        if var not in known:
            raise DeriveError(f"variable must be one of {sorted(known)}")
        lo, hi = physics._f(data.get("min")), physics._f(data.get("max"))
        n = int(data.get("n") or 40)
        if lo is None or hi is None or not (hi > lo):
            raise DeriveError("min and max are required and max must exceed min")
        n = max(4, min(MAX_SWEEP, n))
        log = bool(known[var]["log"]) and lo > 0
        xs = (np.exp(np.linspace(np.log(lo), np.log(hi), n)) if log
              else np.linspace(lo, hi, n))
    except DeriveError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    base = dict(data)
    for k in ("variable", "min", "max", "n"):
        base.pop(k, None)
    records, kept = [], []
    for x in xs:
        try:
            rec, _ = derive_record(_apply_variable(base, var, float(x)), ENS.props)
        except DeriveError:
            continue
        records.append(rec)
        kept.append(float(x))
    if not records:
        return jsonify({"ok": False, "error": "the sweep produced no valid record"}), 400

    p = ENS.predict(records, alpha)
    return jsonify({"ok": True, "variable": var, "log": bool(known[var]["log"]),
                    "unit": known[var]["unit"], "label": known[var]["label"],
                    "x": kept,
                    "chf_kW_m2": [float(v) for v in p["y_hat"]],
                    "lower_kW_m2": [float(v) for v in p["lower"]],
                    "upper_kW_m2": [float(v) for v in p["upper"]],
                    "zuber_kW_m2": [float(v) for v in p["anchor"]],
                    "novelty": [float(v) for v in p["ood"]],
                    "confidence": round((1 - alpha) * 100)})


# ===========================================================================
# 3. Public link
# ===========================================================================
def lan_address() -> str:
    """The address of this machine on its own network, for the local link."""
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.settimeout(0.2)
        sk.connect(("8.8.8.8", 80))                    # no packet is sent
        ip = sk.getsockname()[0]
        sk.close()
        return ip
    except Exception:                                  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:                              # noqa: BLE001
            return ""


def port_is_free(host: str, port: int) -> bool:
    """Whether this process could bind ``port`` right now."""
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sk.bind((host if host != "0.0.0.0" else "", port))
        sk.close()
        return True
    except OSError:
        return False


def pick_port(host: str, wanted: int, tries: int = 20) -> int:
    """The first free port from ``wanted`` up.

    This machine also runs other demos, so the default is regularly taken by
    one of them; moving over is better than refusing to start. The tunnel is
    opened afterwards, on whichever port was actually taken.
    """
    for port in range(wanted, wanted + tries):
        if port_is_free(host, port):
            if port != wanted:
                logger.warning(f"port {wanted} is already in use on this machine - "
                               f"serving on {port} instead")
            return port
    raise SystemExit(f"no free port between {wanted} and {wanted + tries - 1}; "
                     f"pass --port, or set CHFPT_PORT")


def resolve_device(requested: str = None) -> torch.device:
    """cuda when it is really usable, cpu otherwise.

    The card is shared with other jobs, so a request for cuda that cannot be
    initialised falls back rather than killing the demo at import time.
    """
    if requested:
        dev = torch.device(requested)
        if dev.type != "cuda":
            return dev
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        return torch.device("cpu")
    try:
        torch.zeros(1, device=dev)
        free, total = torch.cuda.mem_get_info(dev)
        logger.info(f"gpu memory free: {free/2**30:.1f} of {total/2**30:.1f} GiB "
                    f"(the card is shared with any other job on it)")
        return dev
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"{dev} is not usable ({e}) - falling back to cpu")
        return torch.device("cpu")


def run_server(host: str, port: int, threads: int) -> None:
    """waitress when it is installed, flask's own server as the fallback."""
    if _waitress_serve is not None:
        _waitress_serve(app, host=host, port=port, threads=threads)
        return
    logger.warning("waitress is not installed in this environment - using flask's built-in "
                   "server, which is fine for a demo but slower. 'pip install waitress' "
                   "to use the proper one.")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


class Links:
    """Prints the two addresses of the demo once both are known.

    The local one works for anyone on this network and lasts as long as the
    process. The public one is a Cloudflare tunnel: a quick tunnel is anonymous
    and its address changes at every start, while a named tunnel keeps the same
    hostname for good (see --tunnel-name in the README).
    """

    def __init__(self, port: int, permanent_url: str = None):
        self.port = port
        self.permanent = permanent_url
        self.public = permanent_url
        self.kind = "permanent" if permanent_url else None
        self.printed = False
        self.lock = threading.Lock()

    def set_public(self, url: str, kind: str) -> None:
        with self.lock:
            if self.permanent:                          # a named tunnel wins
                return
            self.public, self.kind = url, kind
        self.show()

    def show(self) -> None:
        with self.lock:
            if self.printed:
                return
            self.printed = True
        ip = lan_address()
        pad = 10
        print("\n" + "=" * 74)
        print("  " + "Local".ljust(pad) + f"http://localhost:{self.port}")
        if ip:
            print("  " + "".ljust(pad) + f"http://{ip}:{self.port}"
                  + "   (anyone on this network)")
        if self.public:
            note = ("(permanent, same address at every start)" if self.kind == "permanent"
                    else "(temporary, a new address at every start)")
            print("  " + "Public".ljust(pad) + f"{self.public}   {note}")
        else:
            print("  " + "Public".ljust(pad) + "not published; see --tunnel-name for a "
                                              "permanent address")
        print("=" * 74 + "\n", flush=True)


def start_cloudflared(links: Links, port: int, name: str = None, token: str = None) -> None:
    """Quick tunnel by default; a named tunnel when one is configured."""
    if token:
        cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}",
               "run", "--token", token]
        kind = "permanent"
    elif name:
        cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "run", name]
        kind = "permanent"
    else:
        cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]
        kind = "temporary"
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        logger.info(f"starting the {kind} Cloudflare tunnel on port {port} ...")
    except FileNotFoundError:
        logger.warning("'cloudflared' is not on PATH - serving locally only. Install it from "
                       "https://github.com/cloudflare/cloudflared to publish a link.")
        links.show()
        return
    except Exception as e:                                   # noqa: BLE001
        logger.error(f"could not start the tunnel: {e}")
        links.show()
        return

    def reader(pipe):
        for line in iter(pipe.readline, ""):
            m = re.search(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com", line)
            if m:
                links.set_public(m.group(0), "temporary")
            if "Registered tunnel connection" in line or "Connection registered" in line:
                links.show()                             # a named tunnel is up

    # cloudflared prints its banner on stderr, but not every build does
    for pipe in (proc.stderr, proc.stdout):
        threading.Thread(target=reader, args=(pipe,), daemon=True).start()
    # never leave the user without the local address if the tunnel is slow
    threading.Timer(20.0, links.show).start()


def main() -> None:
    global ENS
    ap = argparse.ArgumentParser(description="Serve the released CHF-PT checkpoints.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="first port to try; the next free one is used if it is taken "
                         f"(default {DEFAULT_PORT}, or CHFPT_PORT)")
    ap.add_argument("--host", default=os.environ.get("CHFPT_HOST", "0.0.0.0"))
    ap.add_argument("--device", default=os.environ.get("CHFPT_DEVICE"),
                    help="cuda, cuda:1, cpu (default: cuda if present)")
    ap.add_argument("--no-tunnel", action="store_true",
                    default=os.environ.get("CHFPT_NO_TUNNEL", "").lower()
                    in ("1", "true", "yes"),
                    help="do not publish a public link (or set CHFPT_NO_TUNNEL=1)")
    ap.add_argument("--tunnel-name", default=os.environ.get("CHFPT_TUNNEL_NAME"),
                    help="run this named Cloudflare tunnel instead of an anonymous quick tunnel, "
                         "which keeps the same public address at every start")
    ap.add_argument("--tunnel-token", default=os.environ.get("CHFPT_TUNNEL_TOKEN"),
                    help="token of a Cloudflare tunnel created in the Zero Trust dashboard")
    ap.add_argument("--public-url", default=os.environ.get("CHFPT_PUBLIC_URL"),
                    help="the hostname a named tunnel is routed to, so it can be printed")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
    logger.info(f"compute device: {device}")
    logger.info(f"base directory: {HERE}")

    for p in (MODEL_DIR, ASSETS / "schema.json", ASSETS / "fluid_props.json",
              ASSETS / "calibration.npz", ASSETS / "examples.json", INDEX):
        if not p.exists():
            raise SystemExit(f"missing {p}\nRun 'python prepare_assets.py' first.")

    ENS = Ensemble(device)
    d = ENS.schema["dataset"]
    logger.info(f"pre-training corpus: {d['records']:,} records, {d['domains']} fluid-geometry "
                f"domains, {len(ENS.schema['geometries'])} geometries")
    # chosen after the checkpoints are loaded, so the tunnel is opened on a port
    # that is still free a moment later
    port = pick_port(args.host, args.port)
    links = Links(port, args.public_url)
    if args.no_tunnel:
        links.show()
    else:
        start_cloudflared(links, port, args.tunnel_name, args.tunnel_token)
    logger.info(f"serving on http://{args.host}:{port}")
    try:
        run_server(args.host, port, args.threads)
    except KeyboardInterrupt:
        logger.info("stopped")


if __name__ == "__main__":
    main()
