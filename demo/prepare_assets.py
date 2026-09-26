#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the three files in ``assets/`` that let the demo run without the database.

The server needs three things that the released checkpoints do not carry: the
saturation properties of the seven fluids, the conformal correction and the
out-of-distribution reference, and the input schema the interface is built
from. All three are derived here, once, from this repository (``data/`` and
``model/chfpt``), and after that the demo folder is self-contained and portable.

    python prepare_assets.py                 # build everything
    python prepare_assets.py --verify        # rebuild nothing, check the physics
    python prepare_assets.py --release PATH  # a release tree somewhere else

``--verify`` is the important one: it reconstructs held-out records of the
released database from their primitive inputs alone, exactly as the web form
does, and compares every derived feature and every prediction with the released
values, and lists every column in which they differ.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chfpt_core import (ANCHOR_COLS, ANCHOR_DEGENERATE, CAT_COLS, LEAKY_CELLS, NUM_COLS,
                        Preprocessor, TAG_COL, build_model, load_neural_checkpoint)   # noqa: E402
import physics                                                                        # noqa: E402

# The released checkpoints store two anchor columns under earlier names; they
# are mapped onto the current names when they are loaded.
ANCHOR_ALIASES = {
    "anchor_kandlikar_kW_m2": "anchor_bowe_kW_m2",
    "Bo\u2013We flow-scaling feature": "anchor_bowe_kW_m2",
    "anchor_flow_kW_m2": "anchor_zuber_scaled_kW_m2",
}


def canonical_anchor_names(names):
    """Map stored anchor names, as column labels or schema entries, to the current ones."""
    return [ANCHOR_ALIASES.get(str(n).strip(), n) for n in names]

from physics import P_CRIT_KPA, POOL_LIKE, TUBE_CLASS                                 # noqa: E402

ASSETS = HERE / "assets"
MODELS = (HERE / "models") if any((HERE / "models").glob("chfpt_seed*.pt")) \
    else (HERE.parent / "model" / "chfpt")
DEFAULT_RELEASE = HERE.parent
ALPHAS = [0.20, 0.10, 0.05]
OOD_REF_N = 8000


# ===========================================================================
# The database, loaded exactly as test/test_chfpt.py loads it
# ===========================================================================
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="CHF_data")
    df.columns = canonical_anchor_names(df.columns)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), other=np.nan)
    df[ANCHOR_DEGENERATE] = df["anchor_zuber_kW_m2"].to_numpy()
    for tag, cols in LEAKY_CELLS.items():                 # --leakage_policy mask
        m = df[TAG_COL] == tag
        for c in cols:
            df.loc[m, c] = np.nan
    df["_log_anchor"] = np.log(df["anchor_zuber_kW_m2"].to_numpy())
    return df.reset_index(drop=True)


def load_models(device):
    out = []
    for p in sorted(MODELS.glob("chfpt_seed*.pt")):
        cfg, pp, state, ck = load_neural_checkpoint(p, map_location=device)
        model = build_model(cfg, pp).to(device).eval()
        model.load_state_dict(state)
        out.append({"seed": int(ck["seed"]), "path": p, "model": model, "pp": pp})
    if not out:
        raise SystemExit(f"no checkpoints in {MODELS}")
    return out


@torch.no_grad()
def forward(entry, frame: pd.DataFrame, device, keys=("mu", "q05", "q95", "log_sigma"),
            batch: int = 2048):
    pp, model = entry["pp"], entry["model"]
    enc = pp.encode(frame.reset_index(drop=True))
    num_val = torch.tensor(np.concatenate([enc["num_val"], enc["anc_val"]], 1), device=device)
    num_mask = torch.tensor(np.concatenate([enc["num_mask"], enc["anc_mask"]], 1), device=device)
    cat = torch.tensor(enc["cat_code"], device=device)
    acc = {k: [] for k in keys}
    for s in range(0, len(frame), batch):
        sl = slice(s, min(s + batch, len(frame)))
        out = model(num_val[sl], num_mask[sl], cat[sl])
        for k in keys:
            acc[k].append(out[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


# ===========================================================================
# 1. Saturation property tables
# ===========================================================================
COOLPROP_NAME = {"water": "Water", "R123": "R123", "R134a": "R134a",
                 "FC-72": "n-Perfluorohexane", "nPFH": "n-Perfluorohexane"}
# FC-72 and nPFH: the compilation took rho, h_fg and cp from the CoolProp
# n-perfluorohexane equation of state and sigma, mu and k from the 3M
# datasheet, which is why these three are constants for both fluids.
FIXED_TRANSPORT = {"FC-72": {"sigma": 0.010, "mu_l": 0.00064, "k_l": 0.057},
                   "nPFH": {"sigma": 0.010, "mu_l": 0.00064, "k_l": 0.057}}
# FC-77 and PF-5052 have one manufacturer datasheet point each.
CONSTANT_FLUIDS = {
    "FC-77": {"Tsat_C": 97.0, "rho_l": 1780.0, "rho_v": 13.4, "hfg": 89.0,
              "sigma": 0.008, "mu_l": 0.0014, "cp_l": 1.05, "k_l": 0.057,
              "P_valid_kPa": [80.0, 130.0]},
    "PF-5052": {"Tsat_C": 50.0, "rho_l": 1700.0, "rho_v": 2.3, "hfg": 105.0,
                "sigma": 0.013, "mu_l": 0.00064, "cp_l": 1.05, "k_l": 0.06,
                "P_valid_kPa": [80.0, 130.0]},
}
GRID = {"water": (50.0, 21500.0, 3000), "R123": (60.0, 3200.0, 1200),
        "R134a": (60.0, 3600.0, 1200), "FC-72": (30.0, 1600.0, 800),
        "nPFH": (30.0, 1600.0, 800)}


def build_fluid_props(df: pd.DataFrame) -> dict:
    from CoolProp.CoolProp import PropsSI

    tables = {}
    for fluid, (lo, hi, n) in GRID.items():
        name = COOLPROP_NAME[fluid]
        grid = np.exp(np.linspace(math.log(lo), math.log(hi), n))
        cols = {k: [] for k in ("Tsat_C", "rho_l", "rho_v", "hfg", "sigma", "mu_l", "cp_l", "k_l")}
        for p in grid:
            P = float(p) * 1e3
            cols["Tsat_C"].append(PropsSI("T", "P", P, "Q", 0, name) - 273.15)
            cols["rho_l"].append(PropsSI("D", "P", P, "Q", 0, name))
            cols["rho_v"].append(PropsSI("D", "P", P, "Q", 1, name))
            cols["hfg"].append((PropsSI("H", "P", P, "Q", 1, name)
                                - PropsSI("H", "P", P, "Q", 0, name)) / 1e3)
            cols["cp_l"].append(PropsSI("C", "P", P, "Q", 0, name) / 1e3)
            fixed = FIXED_TRANSPORT.get(fluid)
            cols["sigma"].append(fixed["sigma"] if fixed else PropsSI("I", "P", P, "Q", 0, name))
            cols["mu_l"].append(fixed["mu_l"] if fixed else PropsSI("V", "P", P, "Q", 0, name))
            cols["k_l"].append(fixed["k_l"] if fixed else PropsSI("L", "P", P, "Q", 0, name))
        tables[fluid] = {"kind": "table", "source": f"CoolProp {name}"
                         + (" + 3M datasheet transport properties" if fluid in FIXED_TRANSPORT else ""),
                         "P_crit_kPa": P_CRIT_KPA[fluid],
                         "P_kPa": [float(f"{v:.10g}") for v in grid],
                         **{k: [float(f"{v:.10g}") for v in vals] for k, vals in cols.items()}}
    for fluid, vals in CONSTANT_FLUIDS.items():
        tables[fluid] = {"kind": "constant", "source": "3M datasheet, single point at 1 atm",
                         "P_crit_kPa": P_CRIT_KPA[fluid], **vals}

    ASSETS.mkdir(exist_ok=True)
    with open(ASSETS / "fluid_props.json", "w", encoding="utf-8") as fh:
        json.dump(tables, fh, separators=(",", ":"))
    size = (ASSETS / "fluid_props.json").stat().st_size / 1e6
    print(f"[assets] fluid_props.json  {size:.2f} MB, {len(tables)} fluids")

    # agreement with the properties frozen in the released database
    fp = physics.FluidProperties(ASSETS / "fluid_props.json")
    worst = {}
    for fluid, g in df.groupby("fluid"):
        u = g.drop_duplicates("P_kPa")
        e = []
        for _, r in u.iterrows():
            s = fp.saturation(fluid, r.P_kPa)
            for key, col in [("rho_l", "P_rho_l"), ("rho_v", "P_rho_v"), ("hfg", "P_hfg"),
                             ("sigma", "P_sigma"), ("mu_l", "P_mu_l"), ("cp_l", "P_cp_l"),
                             ("k_l", "P_k_l")]:
                e.append(abs(s[key] - r[col]) / abs(r[col]))
        worst[fluid] = max(e)
    print("           max relative deviation from the released columns: "
          + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    return tables


# ===========================================================================
# 2. Conformal correction and out-of-distribution reference
# ===========================================================================
def build_calibration(df: pd.DataFrame, device) -> dict:
    models = load_models(device)
    cal = df[df["split"] == "cal"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    train = df[df["split"] == "train"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    ref = train.iloc[rng.choice(len(train), OOD_REF_N, replace=False)].reset_index(drop=True)

    la_cal = cal["_log_anchor"].to_numpy()
    y_cal = np.log(cal["CHF_kW_m2"].to_numpy())
    la_test = test["_log_anchor"].to_numpy()
    y_test = test["CHF_kW_m2"].to_numpy()

    out = {"seeds": [], "alphas": ALPHAS}
    q05_c, q95_c, q05_t, q95_t, mu_t, z_t = [], [], [], [], [], []
    for e in models:
        pc = forward(e, cal, device, keys=("q05", "q95"))
        pt = forward(e, test, device, keys=("mu", "q05", "q95"))
        pr = forward(e, ref, device, keys=("z",))
        z_ref = pr["z"].astype(np.float64)
        mu_z = z_ref.mean(0)
        cov = np.cov(z_ref, rowvar=False) + 1e-3 * np.eye(z_ref.shape[1])
        prec = np.linalg.pinv(cov)
        q05_c.append(pc["q05"]); q95_c.append(pc["q95"])
        q05_t.append(pt["q05"]); q95_t.append(pt["q95"]); mu_t.append(pt["mu"])
        z_t.append(forward(e, test, device, keys=("z",))["z"].astype(np.float64))
        seed = {"seed": e["seed"], "z_mean": mu_z, "z_prec": prec, "Q": {}}
        for a in ALPHAS:
            sc = np.maximum(pc["q05"] + la_cal - y_cal, y_cal - (pc["q95"] + la_cal))
            k = min(1.0, math.ceil((len(sc) + 1) * (1 - a)) / len(sc))
            seed["Q"][a] = float(np.quantile(sc, k, method="higher"))
        out["seeds"].append(seed)

    # the ensemble: seed-averaged log quantiles, calibrated on the same split
    q05_ce, q95_ce = np.mean(q05_c, 0), np.mean(q95_c, 0)
    q05_te, q95_te = np.mean(q05_t, 0), np.mean(q95_t, 0)
    y_hat_seeds = np.array([np.exp(m + la_test) for m in mu_t])
    y_hat = y_hat_seeds.mean(0)
    sc = np.maximum(q05_ce + la_cal - y_cal, y_cal - (q95_ce + la_cal))
    ens = {"Q": {}, "coverage": {}, "rel_width": {}}
    for a in ALPHAS:
        k = min(1.0, math.ceil((len(sc) + 1) * (1 - a)) / len(sc))
        Q = float(np.quantile(sc, k, method="higher"))
        lo, hi = np.exp(q05_te + la_test - Q), np.exp(q95_te + la_test + Q)
        ens["Q"][a] = Q
        ens["coverage"][a] = float(np.mean((y_test >= lo) & (y_test <= hi)))
        ens["rel_width"][a] = float(np.median((hi - lo) / y_test))

    # OOD scores of the held-out records, for the scale the interface shows
    def maha(z, mu_z, prec):
        d = z - mu_z
        return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))

    s_test = np.mean([maha(z_t[i], out["seeds"][i]["z_mean"], out["seeds"][i]["z_prec"])
                      for i in range(len(models))], 0)
    pct = {int(p): float(np.percentile(s_test, p)) for p in (50, 75, 90, 95, 99)}

    mape = float(np.mean(np.abs(y_hat - y_test) / y_test) * 100)
    dom = pd.DataFrame({"f": test["fluid"], "g": test["geometry_type"],
                        "e": np.abs(y_hat - y_test) / y_test * 100})
    macro = float(dom.groupby(["f", "g"])["e"].mean().mean())

    np.savez_compressed(
        ASSETS / "calibration.npz",
        seeds=np.array([s["seed"] for s in out["seeds"]], dtype=np.int64),
        alphas=np.array(ALPHAS, dtype=np.float64),
        Q_seed=np.array([[s["Q"][a] for a in ALPHAS] for s in out["seeds"]], dtype=np.float64),
        Q_ens=np.array([ens["Q"][a] for a in ALPHAS], dtype=np.float64),
        coverage_ens=np.array([ens["coverage"][a] for a in ALPHAS], dtype=np.float64),
        rel_width_ens=np.array([ens["rel_width"][a] for a in ALPHAS], dtype=np.float64),
        z_mean=np.array([s["z_mean"] for s in out["seeds"]], dtype=np.float64),
        z_prec=np.array([s["z_prec"] for s in out["seeds"]], dtype=np.float64),
        ood_pct_levels=np.array([50, 75, 90, 95, 99], dtype=np.int64),
        ood_pct=np.array([pct[p] for p in (50, 75, 90, 95, 99)], dtype=np.float64),
        n_cal=np.int64(len(cal)), n_test=np.int64(len(test)),
        ens_test_mape=np.float64(mape), ens_test_macro_mape=np.float64(macro),
    )
    print(f"[assets] calibration.npz  {len(models)} seeds | "
          f"ensemble test micro-MAPE {mape:.2f}%  macro {macro:.2f}%")
    for a in ALPHAS:
        print(f"           {(1-a)*100:.0f}% interval: Q={ens['Q'][a]:.4f}  "
              f"empirical coverage on {len(test):,} held-out records = {ens['coverage'][a]:.3f}  "
              f"median relative width {ens['rel_width'][a]*100:.0f}%")
    print(f"           novelty score on held-out records: "
          + " ".join(f"p{p}={pct[p]:.1f}" for p in (50, 90, 95, 99)))
    return {"pct": pct, "ens": ens}


# ===========================================================================
# 3. The input schema the interface is generated from
# ===========================================================================
# label, one-line description, and the group the interface lists it under. The
# six fusion mock-ups share a group so that they read as one family rather than
# as five tubes and something else.
CHANNELS = "Channels"
SURFACES = "Enhanced and sprayed surfaces"
FUSION = "Fusion high-heat-flux mock-ups"
GEOM_LABEL = {
    "tube": ("Vertical tube", "Uniformly heated round tube, the bulk of the corpus.", CHANNELS),
    "annulus": ("Annulus", "Internally heated annular channel.", CHANNELS),
    "rod_bundle": ("Rod bundle", "Multi-rod fuel-assembly section.", CHANNELS),
    "plate": ("Parallel plate", "Narrow rectangular plate channel.", CHANNELS),
    "rectangular_channel": ("Rectangular channel",
                            "NASA FBCE flow-boiling cell, the microgravity records.", CHANNELS),
    "helical_coil": ("Helical coil", "Helically coiled tube, horizontal axis.", CHANNELS),
    "helical_minichannel": ("Helical minichannel", "Coiled minichannel, R134a.", CHANNELS),
    "pin_fin_surface": ("Pin-fin surface",
                        "Micro-pin-fin enhanced surface in a stagnant pool.", SURFACES),
    "spray_flat_surface": ("Spray-cooled surface",
                           "Flat surface under an impinging spray.", SURFACES),
    "Smooth_Tube": ("Smooth tube", "Fusion divertor mock-up, smooth bore.", FUSION),
    "Screw_Tube": ("Screw-threaded tube", "Fusion divertor mock-up, screw-threaded bore.", FUSION),
    "Internal_Fin_Tube": ("Internally finned tube",
                          "Fusion divertor mock-up, internally finned.", FUSION),
    "Ext_Fin_Smooth": ("Externally finned tube", "Externally finned smooth tube.", FUSION),
    "Ext_Fin_Swirl": ("Externally finned tube with swirl",
                      "Externally finned tube with a swirl insert.", FUSION),
    "Hypervapotron": ("Hypervapotron", "Slotted high-heat-flux element.", FUSION),
}
GROUP_ORDER = [CHANNELS, SURFACES, FUSION]
ORIENT_LABEL = {
    "vertical_upflow": "Vertical upflow", "horizontal_upflow": "Horizontal flow",
    "horizontal_helical": "Horizontal helical", "pool_boiling": "Stagnant pool",
    "spray_impingement": "Spray impingement", "vertical_upflow_1g": "Vertical upflow (1 g reference)",
    "microgravity_ISS": "Microgravity (ISS)",
}
FIELD_META = {
    "tube_Di_mm": ("Inner diameter D", "mm", "Sets the channel scale of every dimensionless group."),
    "annulus_Dh_mm": ("Hydraulic diameter D_h", "mm", "Channel scale of the annulus."),
    "annulus_De_mm": ("Heated equivalent diameter D_e", "mm", "Reported separately; not the channel scale."),
    "plate_Dh_mm": ("Plate channel width", "mm", "Geometry token of the plate section."),
    "rect_Dh_mm": ("Hydraulic diameter D_h", "mm", "Channel scale of the rectangular cell."),
    "rod_Drod_mm": ("Rod diameter", "mm", "Also the channel scale used for the bundle."),
    "rod_pitch_mm": ("Rod pitch", "mm", "Reported for 7.5% of the bundle records; leave empty if unknown."),
    "helical_Dtube_mm": ("Tube inner diameter", "mm", "Channel scale of the coil."),
    "helical_Do_mm": ("Tube outer diameter", "mm", ""),
    "helical_Dcoil_mm": ("Coil diameter", "mm", "With the tube diameter it sets the Dean number."),
    "helical_pitch_mm": ("Helix pitch", "mm", ""),
    "fin_width_um": ("Fin width", "um", ""),
    "fin_height_um": ("Fin height", "um", ""),
    "fin_spacing_um": ("Fin spacing", "um", ""),
    "fin_coverage": ("Fin coverage", "-", "Fraction of the surface carrying fins."),
    "fin_porosity": ("Array porosity", "-", ""),
    "fin_roughness": ("Relative roughness", "-", ""),
    "fin_MBL_total_um": ("Modulated boiling layer", "um", "Total thickness of the fin structure."),
    "spray_angle_deg": ("Spray cone angle", "deg", ""),
    "spray_Tsat_C": ("Spray saturation temperature", "degC", ""),
    "spray_DTsub_C": ("Spray subcooling", "degC", ""),
}
GEOM_FIELDS = {
    **{g: ["tube_Di_mm"] for g in TUBE_CLASS},
    "annulus": ["annulus_Dh_mm", "annulus_De_mm"],
    "plate": ["plate_Dh_mm"],
    "rectangular_channel": ["rect_Dh_mm"],
    "rod_bundle": ["rod_Drod_mm", "rod_pitch_mm"],
    "helical_coil": ["helical_Dtube_mm", "helical_Do_mm", "helical_Dcoil_mm", "helical_pitch_mm"],
    "helical_minichannel": ["helical_Dtube_mm"],
    "pin_fin_surface": ["fin_width_um", "fin_height_um", "fin_spacing_um", "fin_coverage",
                        "fin_porosity", "fin_roughness", "fin_MBL_total_um"],
    "spray_flat_surface": ["spray_angle_deg", "spray_Tsat_C", "spray_DTsub_C"],
}
GEOM_CATS = {"pin_fin_surface": ["fin_shape", "fin_array", "surface_material"]}
CAT_LABEL = {"fin_shape": "Fin cross-section", "fin_array": "Array arrangement",
             "surface_material": "Surface material"}
OPTIONAL_FIELDS = {"rod_pitch_mm", "annulus_De_mm", "helical_Do_mm", "helical_pitch_mm"}


def _rng(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
            "p1": float(s.quantile(0.01)), "p99": float(s.quantile(0.99))}


def _record_example(r: pd.Series, geom: str) -> dict:
    """One database row as the form would have been filled in to produce it."""
    g = {k: (None if pd.isna(r.get(k)) else float(r[k])) for k in GEOM_FIELDS.get(geom, [])}
    for c in GEOM_CATS.get(geom, []):
        g[c] = None if pd.isna(r.get(c)) else str(r[c])
    payload = {
        "fluid": str(r["fluid"]), "geometry_type": geom, "orientation": str(r["orientation"]),
        "P_kPa": float(r["P_kPa"]), "subcooling_mode": "both",
        "DT_sub_K": float(r["DT_sub_K"]), "T_in_C": float(r["T_in_C"]),
        "G_kg_m2s": None if (geom in POOL_LIKE or pd.isna(r["G_kg_m2s"])) else float(r["G_kg_m2s"]),
        "heated_length_m": None if (geom in POOL_LIKE or pd.isna(r["heated_length_m"]))
                           else float(r["heated_length_m"]),
        "geometry": g,
    }
    if geom == "plate":
        payload["char_length_mm"] = float(r["char_length_m"]) * 1e3
    return {"row_id": int(r["row_id"]), "split": str(r["split"]),
            "label": f"{r['fluid']} / {GEOM_LABEL[geom][0]}",
            "measured_CHF_kW_m2": float(r["CHF_kW_m2"]),
            "reference": str(r.get("Reference", ""))[:120], "payload": payload}


def _example(sub: pd.DataFrame, geom: str) -> dict:
    """The record the page opens with: held out, near the median CHF of the section."""
    s = sub[sub["split"] == "test"]
    if s.empty:
        s = sub
    i = (s["CHF_kW_m2"] - s["CHF_kW_m2"].median()).abs().idxmin()
    return _record_example(s.loc[i], geom)


def build_examples(df: pd.DataFrame, per_split: int = 15) -> dict:
    """A pool of real records the page can draw from, all four splits included.

    A training record is one the checkpoints have seen, so the pool carries the
    split of every entry and the interface says which it is: comparing a
    prediction with a measurement only means something on the three held-out
    subsets.
    """
    rng = np.random.default_rng(11)
    pool, n = {}, 0
    for geom, sub in df.groupby("geometry_type"):
        picks = []
        for split in ("test", "val", "cal", "train"):
            g = sub[sub["split"] == split]
            if g.empty:
                continue
            take = g if len(g) <= per_split else g.iloc[
                rng.choice(len(g), per_split, replace=False)]
            picks += [_record_example(r, geom) for _, r in take.iterrows()]
        pool[geom] = picks
        n += len(picks)
    with open(ASSETS / "examples.json", "w", encoding="utf-8") as fh:
        json.dump(pool, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"[assets] examples.json    {n:,} records over {len(pool)} sections, "
          f"{(ASSETS / 'examples.json').stat().st_size / 1e3:.0f} kB")
    return pool


def build_schema(df: pd.DataFrame, calib: dict) -> dict:
    train = df[df["split"] == "train"]
    fp = physics.FluidProperties(ASSETS / "fluid_props.json")

    fluids = []
    for f in ["water", "R123", "R134a", "FC-72", "FC-77", "PF-5052", "nPFH"]:
        lo, hi = fp.pressure_range(f)
        sub = df[df["fluid"] == f]
        fluids.append({"key": f, "P_crit_kPa": P_CRIT_KPA[f],
                       "P_table_kPa": [lo, hi],
                       "P_corpus_kPa": [float(sub["P_kPa"].min()), float(sub["P_kPa"].max())],
                       "n_records": int(len(sub)),
                       "source": fp.tables[f]["source"]})

    geometries = []
    for g in sorted(df["geometry_type"].unique(), key=lambda x: -len(df[df["geometry_type"] == x])):
        sub = df[df["geometry_type"] == g]
        tsub = train[train["geometry_type"] == g]
        label, note, grp = GEOM_LABEL[g]
        fields = []
        for k in GEOM_FIELDS.get(g, []):
            lab, unit, help_ = FIELD_META[k]
            fields.append({"key": k, "label": lab, "unit": unit, "help": help_,
                           "optional": k in OPTIONAL_FIELDS, "range": _rng(tsub[k])})
        cats = []
        for c in GEOM_CATS.get(g, []):
            levels = sorted(sub[c].dropna().astype(str).unique().tolist())
            cats.append({"key": c, "label": CAT_LABEL[c], "levels": levels,
                         "optional": True})
        geometries.append({
            "key": g, "label": label, "note": note, "group": grp,
            "n_records": int(len(sub)),
            "flow": g not in POOL_LIKE,
            "char_length_from": physics.CHAR_LENGTH_FROM.get(g),
            "char_length_input": g == "plate",
            "orientations": [{"key": o, "label": ORIENT_LABEL[o]}
                             for o in sorted(sub["orientation"].unique())],
            "fluids": sorted(sub["fluid"].unique().tolist()),
            "fields": fields, "categoricals": cats,
            "ranges": {"P_kPa": _rng(tsub["P_kPa"]), "G_kg_m2s": _rng(tsub["G_kg_m2s"]),
                       "DT_sub_K": _rng(tsub["DT_sub_K"]), "T_in_C": _rng(tsub["T_in_C"]),
                       "heated_length_m": _rng(tsub["heated_length_m"]),
                       "char_length_mm": _rng(tsub["char_length_m"] * 1e3),
                       "CHF_kW_m2": _rng(sub["CHF_kW_m2"])},
            "example": _example(sub, g),
        })

    schema = {
        "geometry_groups": GROUP_ORDER,
        "dataset": {
            "name": "CHF_dataset_v1.1", "records": int(len(df)),
            "domains": int(df.groupby(["fluid", "geometry_type"]).ngroups),
            "fluids": int(df["fluid"].nunique()), "geometries": int(df["geometry_type"].nunique()),
            "CHF_range_kW_m2": [float(df["CHF_kW_m2"].min()), float(df["CHF_kW_m2"].max())],
            "split": {k: int((df["split"] == k).sum()) for k in ("train", "val", "cal", "test")},
        },
        "model": {
            "name": "CHF-PT", "seeds": 5, "parameters_M": 17.81,
            "target": "ln(CHF) - ln(q_Zuber)",
            "test_micro_mape": calib["ens"]["mape"], "test_macro_mape": calib["ens"]["macro"],
            "coverage": {str(a): calib["ens"]["coverage"][a] for a in ALPHAS},
            "rel_width": {str(a): calib["ens"]["rel_width"][a] for a in ALPHAS},
        },
        "ood": {"percentiles": {str(k): v for k, v in calib["pct"].items()}},
        "alphas": ALPHAS,
        "fluids": fluids,
        "geometries": geometries,
        "sweep_variables": [
            {"key": "P_kPa", "label": "Pressure", "unit": "kPa", "log": True},
            {"key": "G_kg_m2s", "label": "Mass flux", "unit": "kg m-2 s-1", "log": False},
            {"key": "DT_sub_K", "label": "Inlet subcooling", "unit": "K", "log": False},
            {"key": "char_length_mm", "label": "Channel scale", "unit": "mm", "log": True},
            {"key": "heated_length_m", "label": "Heated length", "unit": "m", "log": True},
        ],
    }
    with open(ASSETS / "schema.json", "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=1, ensure_ascii=False)
    print(f"[assets] schema.json      {len(geometries)} geometries, {len(fluids)} fluids, "
          f"{(ASSETS / 'schema.json').stat().st_size / 1e3:.0f} kB")
    return schema


# ===========================================================================
# 4. Verification: rebuild released records from their primitive inputs
# ===========================================================================
def verify(df: pd.DataFrame, device) -> None:
    """Rebuild every held-out record from its primitive inputs and compare.

    This is the demo's correctness statement: what the web form assembles from
    a pressure, a temperature, a mass flux and a diameter must be the record the
    released evaluation script reads straight out of the database.
    """
    fp = physics.FluidProperties(ASSETS / "fluid_props.json")
    test = df[df["split"] == "test"].reset_index(drop=True)
    rows, recs, skipped = [], [], 0
    for i in range(len(test)):
        r = test.loc[i]
        g = r["geometry_type"]
        if pd.isna(r["heated_length_m"]) and g not in POOL_LIKE:
            skipped += 1                      # heated length masked by the leakage policy
            continue
        payload = _example(test.loc[[i]], g)["payload"]
        try:
            rec, _ = physics.derive_record(payload, fp)
        except physics.DeriveError as e:
            print(f"  ! row {int(r['row_id'])} ({g}): {e}")
            continue
        rows.append(r)
        recs.append(rec)
    frame_ref = pd.DataFrame(rows).reset_index(drop=True)
    frame_new = physics.record_to_frame(recs)

    print(f"\n=== verify: {len(recs):,} of the {len(test):,} held-out records rebuilt from their "
          f"primitive inputs ({skipped} skipped, heated length masked by the leakage policy) ===")
    worst = []
    for c in NUM_COLS + ANCHOR_COLS:
        a = pd.to_numeric(frame_ref[c], errors="coerce").to_numpy(dtype=np.float64)
        b = frame_new[c].to_numpy(dtype=np.float64)
        both = np.isfinite(a) & np.isfinite(b)
        mism = int((np.isfinite(a) != np.isfinite(b)).sum())
        scale = np.maximum(np.abs(a[both]), np.median(np.abs(a[both])) if both.any() else 1.0)
        rel = np.abs(a[both] - b[both]) / np.maximum(scale, 1e-30)
        worst.append((c, float(rel.max()) if rel.size else 0.0, mism))
    worst.sort(key=lambda t: -t[1])
    clean = [w for w in worst if w[1] <= 1e-3 and w[2] == 0]
    bad = [w for w in worst if w not in clean]
    print(f"  {len(clean)} of {len(worst)} numeric columns agree to better than 1e-3 "
          f"(worst of them {clean[0][1]:.1e}; the tables were sampled from a newer CoolProp "
          f"than the compilation used)")
    for c, m, mism in bad:
        a = pd.to_numeric(frame_ref[c], errors="coerce").to_numpy(dtype=np.float64)
        b = frame_new[c].to_numpy(dtype=np.float64)
        both = np.isfinite(a) & np.isfinite(b)
        n_mat = int(np.nansum(np.where(both, np.abs(a - b) / np.abs(a), np.nan) > 0.01))
        print(f"  ! {c:24s} max deviation {m:.3e} | {n_mat} records differ by more than 1% "
              f"| {mism} presence mismatches")
    if bad:
        print("    Both are evidence tokens. The demo evaluates both correlations from the inputs\n"
              "    the user supplies, while the released columns keep the values of the compiled\n"
              "    database; demo/README.md lists where the two differ.")
    for c in CAT_COLS:
        a = frame_ref[c].astype("object").where(frame_ref[c].notna(), None)
        diff = int(sum(1 for x, y in zip(a, frame_new[c]) if (x or None) != (y or None)))
        print(f"  categorical {c:18s} {'identical' if diff == 0 else str(diff) + ' mismatches'}")

    models = load_models(device)
    la = np.log(frame_new["anchor_zuber_kW_m2"].to_numpy())
    la_ref = frame_ref["_log_anchor"].to_numpy()
    p_new = np.mean([np.exp(forward(e, frame_new, device, keys=("mu",))["mu"] + la) for e in models], 0)
    p_ref = np.mean([np.exp(forward(e, frame_ref, device, keys=("mu",))["mu"] + la_ref) for e in models], 0)
    d = np.abs(p_new - p_ref) / p_ref
    y = frame_ref["CHF_kW_m2"].to_numpy(dtype=np.float64)
    same = float(np.mean(d < 1e-3) * 100)
    print(f"\n  prediction from the form vs from the database row: identical to 0.1% on "
          f"{same:.1f}% of the records | median {np.median(d):.1e} | max {d.max():.2e}")
    print(f"  ensemble MAPE against the measurement: {np.mean(np.abs(p_new - y) / y) * 100:.2f}% "
          f"from the form, {np.mean(np.abs(p_ref - y) / y) * 100:.2f}% from the database rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--verify", action="store_true", help="only check, build nothing")
    ap.add_argument("--schema-only", action="store_true",
                    help="rebuild assets/schema.json only, reusing the existing calibration")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = args.release / "data" / "CHF_dataset_v1.1.xlsx"
    if not data.exists():
        raise SystemExit(f"released database not found: {data}\nPass --release <path>.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== CHF-PT demo assets | release {args.release} | device {device} ===")
    df = load_dataset(data)
    print(f"[data] {len(df):,} records, leakage policy 'mask' (as in the released scripts)")

    if args.schema_only:
        z = np.load(ASSETS / "calibration.npz")
        c = {"pct": {int(k): float(v) for k, v in zip(z["ood_pct_levels"], z["ood_pct"])},
             "ens": {"coverage": {a: float(v) for a, v in zip(z["alphas"], z["coverage_ens"])},
                     "rel_width": {a: float(v) for a, v in zip(z["alphas"], z["rel_width_ens"])},
                     "mape": float(z["ens_test_mape"]), "macro": float(z["ens_test_macro_mape"])}}
        build_examples(df)
        build_schema(df, c)
        return
    if not args.verify:
        ASSETS.mkdir(exist_ok=True)
        build_fluid_props(df)
        build_examples(df)
        c = build_calibration(df, device)
        c["ens"]["mape"] = float(np.load(ASSETS / "calibration.npz")["ens_test_mape"])
        c["ens"]["macro"] = float(np.load(ASSETS / "calibration.npz")["ens_test_macro_mape"])
        build_schema(df, c)
    verify(df, device)


if __name__ == "__main__":
    main()
