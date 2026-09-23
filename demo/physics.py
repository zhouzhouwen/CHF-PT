#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""From a handful of measured quantities to the full CHF-PT input record.

A user of the demo types only what an experimenter actually reads off the rig:
the fluid, the system pressure, the inlet temperature or subcooling, the mass
flux, the heated length and the two or three numbers that describe the test
section. Everything else that the model consumes — eight saturation properties,
eleven dimensionless or derived groups and three closed-form CHF correlations —
is reconstructed here with the *same* formulas that built the released
database, so a record assembled from the form is indistinguishable from the
corresponding row of ``CHF_dataset_v1.1.xlsx``.

Every formula below was verified against the released file
(``prepare_assets.py --verify``); the agreement is at machine precision.

Saturation properties come from ``assets/fluid_props.json``, a table sampled
from CoolProp (and, for FC-77 and PF-5052, from the manufacturer data used when
the database was compiled). The table is shipped so that a deployment needs
neither CoolProp nor the 25 MB database file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# 1. Constants of the compiled database
# ===========================================================================

# Reference gravity. Every gravity-dependent length or group in the database is
# evaluated here, including for the twelve ISS records, exactly as the Zuber
# anchor is; the real gravity level enters the model only through the three
# explicit gravity features.
GRAV_REF = 9.81

# Critical pressures used to define P_red when the database was built. For the
# perfluorocarbons these are manufacturer values and differ from CoolProp's, so
# they are pinned here rather than recomputed.
P_CRIT_KPA = {
    "water": 22064.0, "R123": 3662.0, "R134a": 4059.0,
    "FC-72": 1830.0, "FC-77": 1620.0, "PF-5052": 2050.0, "nPFH": 1865.0,
}

# Geometry families that share one channel-scale rule.
TUBE_CLASS = {"tube", "Smooth_Tube", "Screw_Tube", "Internal_Fin_Tube",
              "Ext_Fin_Smooth", "Ext_Fin_Swirl", "Hypervapotron"}
# Configurations without a channel mass flux: the flow block emits no token and
# the channel scale becomes the capillary length (as in the released file).
POOL_LIKE = {"pin_fin_surface", "spray_flat_surface"}
# Where the Bowring correlation is evaluated in the released file.
BOWRING_GEOMS = TUBE_CLASS | {"rod_bundle"}

# char_length_m per geometry: the geometry field it is copied from, in mm.
CHAR_LENGTH_FROM = {
    **{g: "tube_Di_mm" for g in TUBE_CLASS},
    "annulus": "annulus_Dh_mm",
    "rectangular_channel": "rect_Dh_mm",
    "rod_bundle": "rod_Drod_mm",
    "helical_coil": "helical_Dtube_mm",
    "helical_minichannel": "helical_Dtube_mm",
    # "plate" has no rule: its 120 mm plate_Dh_mm token and its 15 mm channel
    # scale are different quantities, so the scale is asked for directly.
}

# Heated length is duplicated into a geometry token by these geometries.
HEATED_MIRROR = {"rod_bundle": "rod_Lh_mm", "helical_coil": "helical_Lh_mm"}

# The geometry-specific layer, as the form and a batch file may address it.
GEOMETRY_KEYS = ("tube_Di_mm", "annulus_De_mm", "annulus_Dh_mm", "plate_Dh_mm", "rect_Dh_mm",
                 "rod_Drod_mm", "rod_pitch_mm", "rod_Lh_mm",
                 "helical_Dtube_mm", "helical_Do_mm", "helical_Dcoil_mm", "helical_pitch_mm",
                 "helical_Lh_mm", "helical_Dean",
                 "fin_width_um", "fin_height_um", "fin_spacing_um", "fin_coverage",
                 "fin_porosity", "fin_roughness", "fin_MBL_total_um",
                 "spray_angle_deg", "spray_Tsat_C", "spray_DTsub_C")
GEOMETRY_CATS = ("fin_shape", "fin_array", "surface_material")
# The quantities a record carries outside the geometry layer.
COMMON_KEYS = ("fluid", "geometry_type", "orientation", "P_kPa", "T_in_C", "DT_sub_K",
               "G_kg_m2s", "heated_length_m", "char_length_mm", "subcooling_mode",
               "q_op_kW_m2")


# ===========================================================================
# 2. Saturation properties
# ===========================================================================


class FluidProperties:
    """Saturation table lookup, linear in ln(P), one table per fluid."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as fh:
            self.tables: Dict[str, dict] = json.load(fh)
        self.keys = ("Tsat_C", "rho_l", "rho_v", "hfg", "sigma", "mu_l", "cp_l", "k_l")
        for name, t in self.tables.items():
            if t.get("kind") == "table":
                t["_lnP"] = np.log(np.asarray(t["P_kPa"], dtype=np.float64))
                for k in self.keys:
                    t["_" + k] = np.asarray(t[k], dtype=np.float64)
                    if k != "Tsat_C":
                        t["_ln_" + k] = np.log(t["_" + k])

    @property
    def fluids(self) -> List[str]:
        return list(self.tables)

    def pressure_range(self, fluid: str) -> Tuple[float, float]:
        t = self.tables[fluid]
        if t.get("kind") == "table":
            return float(t["P_kPa"][0]), float(t["P_kPa"][-1])
        return float(t["P_valid_kPa"][0]), float(t["P_valid_kPa"][1])

    def saturation(self, fluid: str, P_kPa: float) -> Dict[str, float]:
        """The eight saturation quantities the schema needs, at pressure P.

        A constant-property fluid (FC-77, PF-5052, for which the compilation
        used a single manufacturer datasheet point) returns that point for any
        pressure inside its declared validity window.
        """
        if fluid not in self.tables:
            raise KeyError(f"unknown fluid: {fluid}")
        t = self.tables[fluid]
        if t.get("kind") == "constant":
            out = {k: float(t[k]) for k in self.keys}
        else:
            lnp = math.log(max(float(P_kPa), 1e-6))
            lnp = float(np.clip(lnp, t["_lnP"][0], t["_lnP"][-1]))
            # every quantity but the temperature is interpolated in log-log,
            # where the saturation curves are close to straight
            out = {k: (float(np.interp(lnp, t["_lnP"], t["_" + k])) if k == "Tsat_C"
                       else float(np.exp(np.interp(lnp, t["_lnP"], t["_ln_" + k]))))
                   for k in self.keys}
        out["P_crit_kPa"] = float(P_CRIT_KPA[fluid])
        return out


# ===========================================================================
# 3. The three physics anchors
# ===========================================================================


def zuber_kW_m2(rho_l: float, rho_v: float, hfg_kJkg: float, sigma: float,
                g: float = GRAV_REF) -> Optional[float]:
    """Zuber (1959) hydrodynamic-instability limit, K = pi/24 ~ 0.131.

    The base of the CHF-PT target: the network regresses ln(CHF) - ln(q_Zuber).
    Always evaluated at the reference gravity, so it is defined for every
    record including the microgravity ones.
    """
    if not all(np.isfinite([rho_l, rho_v, hfg_kJkg, sigma])):
        return None
    if rho_l <= rho_v or rho_v <= 0 or sigma <= 0 or hfg_kJkg <= 0:
        return None
    q_W = 0.131 * (hfg_kJkg * 1e3) * math.sqrt(rho_v) * (g * sigma * (rho_l - rho_v)) ** 0.25
    return q_W * 1e-3


def kandlikar_kW_m2(G: Optional[float], We_l: Optional[float], rho_ratio: float,
                    hfg_kJkg: float) -> Optional[float]:
    """Bo-We flow-scaling token, Equation (A1) of the paper.

        Bo_CHF = 0.62 (rho_v/rho_l)^0.1 We_l^-0.4,   q = Bo_CHF G h_fg

    Needs a channel mass flux, so it emits no token for pool boiling and
    sprays — the same absence that removes G, Re and We from the common layer.
    """
    if G is None or We_l is None or not np.isfinite(G) or not np.isfinite(We_l):
        return None
    if G <= 0 or We_l <= 0 or rho_ratio <= 0 or hfg_kJkg <= 0:
        return None
    Bo = 0.62 * (1.0 / rho_ratio) ** 0.1 * We_l ** -0.4
    return Bo * G * hfg_kJkg


# --- water saturation polynomial used by the Bowring correlation -----------
# Verbatim from src/water_props.py, the routine the database was built with.
# Bowring is evaluated with this h_fg (not with the CoolProp value) so that the
# anchor reproduces the released column exactly.
def _polyval3(L: float, c3: float, c2: float, c1: float, c0: float) -> float:
    return ((c3 * L + c2) * L + c1) * L + c0


def _bowring_hfg_kJkg(P_kPa: float) -> float:
    if P_kPa <= 0:
        return float("nan")
    L = math.log10(max(P_kPa, 1e-3))
    return max(_polyval3(L, -490.2738, 4155.1687, -11714.804, 13008.2267), 1.0)


def _bowring_F(P_MPa: float) -> Tuple[float, float, float, float]:
    """Pressure functions F1..F4 of Bowring (1972), Table 1."""
    P_R = P_MPa / 6.895
    if P_R <= 0:
        return (float("nan"),) * 4
    if P_R < 1.0:
        F1 = ((P_R ** 18.942) * math.exp(20.89 * (1 - P_R)) + 0.917) / 1.917
        F2 = ((P_R ** 1.316) * math.exp(2.444 * (1 - P_R)) + 0.309) / 1.309
        F3 = ((P_R ** 17.023) * math.exp(16.658 * (1 - P_R)) + 0.667) / 1.667
    else:
        F1 = (P_R ** -0.368) * math.exp(0.648 * (1 - P_R))
        F2 = (P_R ** -0.448) * math.exp(0.245 * (1 - P_R))
        F3 = P_R ** 0.219
    return F1, F2, F3, F3 * P_R ** 1.649


def bowring_kW_m2(P_kPa: float, G: Optional[float], D_m: Optional[float],
                  L_m: Optional[float], DH_in_kJkg: Optional[float]) -> Optional[float]:
    """Bowring (1972) water-tube correlation, the closed-form proxy for the
    Groeneveld look-up table. Valid for water between 0.2 and 19 MPa with a
    non-zero mass flux; outside that envelope it emits no token."""
    if G is None or D_m is None or L_m is None:
        return None
    if not all(np.isfinite([P_kPa, G, D_m, L_m])):
        return None
    P_MPa = P_kPa * 1e-3
    if not (0.2 <= P_MPa <= 19.0) or G <= 0 or D_m <= 0 or L_m <= 0:
        return None
    F1, F2, F3, F4 = _bowring_F(P_MPa)
    if not all(np.isfinite([F1, F2, F3, F4])):
        return None
    hfg = _bowring_hfg_kJkg(P_kPa)
    if not np.isfinite(hfg):
        return None
    hfg_SI = hfg * 1e3
    X_in = -DH_in_kJkg / hfg if (DH_in_kJkg is not None and np.isfinite(DH_in_kJkg)) else 0.0
    n = 2.0 - 0.5 * (P_MPa / 6.895)
    A = 2.317 * (hfg_SI * D_m * G / 4.0) * F1 / (1.0 + 0.0143 * F2 * math.sqrt(D_m) * G)
    B = D_m * G / 4.0
    C = 0.077 * F3 * D_m * G / (1.0 + 0.347 * F4 * (G / 1356.0) ** n)
    q_W = (A - B * hfg_SI * X_in) / (C + L_m)
    if not np.isfinite(q_W) or q_W <= 0:
        return None
    return q_W * 1e-3


# ===========================================================================
# 4. One form submission -> one complete record
# ===========================================================================


def _f(value) -> Optional[float]:
    """Empty string, None and NaN all mean 'this record does not define it'."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# The nine numbers that define a fluid for this schema. A user who supplies all
# of them can predict for a fluid the model has never seen: the categorical
# identity token is then simply not emitted, and the network works from the
# properties, the family and the geometry alone.
CUSTOM_KEYS = ("T_sat_C", "rho_l", "rho_v", "hfg", "sigma", "mu_l", "cp_l", "k_l", "P_crit_kPa")


def _num(v: float) -> str:
    """Readable inside a sentence: thousands separated, four significant digits."""
    return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.4g}"


class DeriveError(ValueError):
    """The submission cannot be turned into a record (bad or missing input)."""


def derive_record(payload: dict, props: FluidProperties) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build the CHF-PT input record from what the form collected.

    Returns ``(record, report)``. ``record`` holds every schema column the model
    reads, with ``None`` wherever the configuration defines no value, so the
    tokenizer simply emits no token. ``report`` groups the same numbers for
    display, together with the provenance of each one and any warning raised
    while deriving it.
    """
    warn: List[str] = []
    notes: List[str] = []

    fluid = str(payload.get("fluid") or "").strip()
    geom = str(payload.get("geometry_type") or "").strip()
    orient = str(payload.get("orientation") or "").strip()
    custom = payload.get("custom_fluid") or None
    if not custom and fluid not in P_CRIT_KPA:
        raise DeriveError(f"unknown fluid: {fluid!r}. Supply custom_fluid to define your own.")
    if custom and not fluid:
        fluid = "custom"
    if not geom:
        raise DeriveError("geometry_type is required")
    if not orient:
        raise DeriveError("orientation is required")

    P = _f(payload.get("P_kPa"))
    if P is None or P <= 0:
        raise DeriveError("pressure P_kPa must be a positive number")

    # --- 1. saturation state -----------------------------------------------
    if custom:
        vals = {k: _f(custom.get(k)) for k in CUSTOM_KEYS}
        missing = [k for k, v in vals.items() if v is None]
        if missing:
            raise DeriveError("a custom fluid needs every property: missing " + ", ".join(missing))
        if not (vals["rho_l"] > vals["rho_v"] > 0):
            raise DeriveError("a custom fluid needs rho_l > rho_v > 0")
        for k in ("hfg", "sigma", "mu_l", "cp_l", "k_l", "P_crit_kPa"):
            if vals[k] <= 0:
                raise DeriveError(f"a custom fluid needs {k} > 0")
        s = {"Tsat_C": vals["T_sat_C"], "P_crit_kPa": vals["P_crit_kPa"],
             **{k: vals[k] for k in ("rho_l", "rho_v", "hfg", "sigma", "mu_l", "cp_l", "k_l")}}
        if fluid not in P_CRIT_KPA:
            notes.append("the fluid identity token is not emitted: the network sees the "
                         "properties, the fluid family and the geometry, not a name it knows")
        notes.append("the properties are the ones you entered and do not follow the pressure")
    else:
        p_lo, p_hi = props.pressure_range(fluid)
        if not (p_lo <= P <= p_hi):
            warn.append(f"the pressure {_num(P)} kPa is outside the tabulated saturation range "
                        f"of {fluid} ({_num(p_lo)} to {_num(p_hi)} kPa); the properties are "
                        f"clamped to the nearest end of the table")
        s = props.saturation(fluid, P)
    rho_l, rho_v = s["rho_l"], s["rho_v"]
    hfg, sigma, mu_l, cp_l, k_l = s["hfg"], s["sigma"], s["mu_l"], s["cp_l"], s["k_l"]
    Tsat = s["Tsat_C"]
    P_red = P / s["P_crit_kPa"]
    rho_ratio = rho_l / rho_v
    Pr_l = mu_l * cp_l * 1e3 / k_l
    cap = math.sqrt(sigma / (GRAV_REF * (rho_l - rho_v)))

    # The database keys the mixture-of-experts routing on the latent heat, not
    # on the chemical family, so high-pressure water is tagged 'refrigerant'.
    fluid_family = "water" if hfg > 1000 else ("refrigerant" if hfg > 130 else "PFC")

    # --- 2. inlet subcooling ------------------------------------------------
    # DT_sub is the primary quantity of the database and DH_in = cp_l * DT_sub
    # holds exactly in the released file; T_in is the reading, so either can be
    # entered and the other follows.
    mode = str(payload.get("subcooling_mode") or "T_in")
    T_in = _f(payload.get("T_in_C"))
    DT_sub = _f(payload.get("DT_sub_K"))
    if mode == "both":
        # Both readings are kept exactly as given. A campaign may report a
        # subcooling that is not T_sat - T_in for the tabulated saturation curve,
        # and the released records keep both numbers, so a record transcribed
        # from the database reproduces exactly. No remark is raised: it would
        # appear on most examples and explains nothing the user can act on.
        if T_in is None or DT_sub is None:
            raise DeriveError("mode 'both' needs T_in_C and DT_sub_K")
    elif mode == "DT_sub":
        if DT_sub is None:
            raise DeriveError("inlet subcooling DT_sub_K is required in this mode")
        T_in = Tsat - DT_sub
    else:
        if T_in is None:
            raise DeriveError("inlet temperature T_in_C is required in this mode")
        DT_sub = Tsat - T_in
    DH_in = cp_l * DT_sub
    DH_norm = DH_in / hfg

    # --- 3. gravity ---------------------------------------------------------
    micro = 1.0 if orient == "microgravity_ISS" else 0.0
    gravity = 0.0 if micro else GRAV_REF
    gravity_ratio = gravity / GRAV_REF

    # --- 4. geometry tokens and the channel scale ---------------------------
    geom_in = dict(payload.get("geometry") or {})
    geo: Dict[str, Optional[float]] = {k: _f(geom_in.get(k)) for k in GEOMETRY_KEYS}
    cats = {c: (str(geom_in.get(c)).strip() or None) if geom_in.get(c) not in (None, "") else None
            for c in GEOMETRY_CATS}

    heated = _f(payload.get("heated_length_m"))
    G = _f(payload.get("G_kg_m2s"))

    if geom in POOL_LIKE:
        # No channel: the scale is the capillary length, the heated length is
        # that same scale, and the flow block emits no token at all.
        char = cap
        heated = cap
        G = None
    else:
        src = CHAR_LENGTH_FROM.get(geom)
        char = None
        if src is not None and geo.get(src) is not None:
            char = geo[src] * 1e-3
        override = _f(payload.get("char_length_mm"))
        if override is not None:
            char = override * 1e-3
        if char is None or char <= 0:
            need = src or "char_length_mm"
            raise DeriveError(f"the channel scale is undefined: {need} is required for {geom}")
        if heated is not None and heated <= 0:
            heated = None
        if G is not None and G < 0:
            raise DeriveError("mass flux G_kg_m2s cannot be negative")

    # geometry tokens that mirror a quantity already collected
    mirror = HEATED_MIRROR.get(geom)
    if mirror is not None:
        geo[mirror] = heated * 1e3 if heated is not None else None

    # --- 5. flow and channel groups ----------------------------------------
    if G is None:
        Re_l = We_l = None
    else:
        Re_l = G * char / mu_l
        We_l = G * G * char / (rho_l * sigma)
    L_over_D = heated / char if heated is not None else None
    Bond = GRAV_REF * (rho_l - rho_v) * char ** 2 / sigma
    conf_Co = cap / char

    if geom == "helical_coil":
        d, Dc = geo.get("helical_Dtube_mm"), geo.get("helical_Dcoil_mm")
        if Re_l is not None and d and Dc and Dc > 0:
            geo["helical_Dean"] = Re_l * math.sqrt(d / Dc)
        else:
            geo["helical_Dean"] = None

    # --- 6. anchors ---------------------------------------------------------
    q_zuber = zuber_kW_m2(rho_l, rho_v, hfg, sigma)
    if q_zuber is None:
        raise DeriveError("the Zuber anchor is undefined for this state; check the pressure")
    # The released Bo-We column builds the Weber number of an annulus on the
    # heated equivalent diameter rather than on the hydraulic diameter that
    # carries the common layer. That convention is reproduced here so the token
    # a trained checkpoint receives is the one it was trained on.
    We_anchor = We_l
    if geom == "annulus" and G is not None and geo.get("annulus_De_mm"):
        De = geo["annulus_De_mm"] * 1e-3
        We_anchor = G * G * De / (rho_l * sigma)
    q_kand = kandlikar_kW_m2(G, We_anchor, rho_ratio, hfg)
    q_bow = (bowring_kW_m2(P, G, char, heated, DH_in)
             if (fluid == "water" and geom in BOWRING_GEOMS) else None)

    record: Dict[str, object] = {
        "fluid": fluid, "fluid_family": fluid_family,
        "geometry_type": geom, "orientation": orient,
        "P_kPa": P, "P_red": P_red, "P_rho_l": rho_l, "P_rho_v": rho_v, "P_hfg": hfg,
        "P_sigma": sigma, "P_mu_l": mu_l, "P_cp_l": cp_l, "P_k_l": k_l,
        "rho_ratio": rho_ratio, "Pr_l": Pr_l, "capillary_length_m": cap,
        "T_in_C": T_in, "DT_sub_K": DT_sub, "DH_in_kJ_kg": DH_in, "DH_norm": DH_norm,
        "gravity_m_s2": gravity, "gravity_ratio": gravity_ratio, "is_microgravity": micro,
        "G_kg_m2s": G, "Re_l": Re_l, "We_l": We_l,
        "char_length_m": char, "heated_length_m": heated, "L_over_D": L_over_D,
        "Bond": Bond, "confinement_Co": conf_Co,
        "anchor_zuber_kW_m2": q_zuber, "anchor_bowe_kW_m2": q_kand,
        "anchor_bowring_kW_m2": q_bow, "anchor_zuber_scaled_kW_m2": q_zuber,
        **geo, **cats,
    }

    # --- 7. what the interface shows next to the prediction ------------------
    n_defined = sum(1 for v in record.values() if v is not None)
    if custom and fluid not in P_CRIT_KPA:
        n_defined -= 1                      # the fluid name is not a level the encoder knows
    report = {
        "fluid_family": fluid_family,
        "T_sat_C": Tsat,
        "custom_fluid": bool(custom),
        "properties": {"T_sat_C": Tsat, "rho_l": rho_l, "rho_v": rho_v, "hfg": hfg,
                       "sigma": sigma, "mu_l": mu_l, "cp_l": cp_l, "k_l": k_l,
                       "P_crit_kPa": s["P_crit_kPa"]},
        "groups": [
            {"title": "Fluid state",
             "note": "your own property set" if custom else "table lookup at the pressure",
             "rows": [
                 ("T_sat", Tsat, "degC"), ("P_crit", s["P_crit_kPa"], "kPa"),
                 ("P_red", P_red, "-"),
                 ("rho_l", rho_l, "kg m-3"), ("rho_v", rho_v, "kg m-3"),
                 ("h_fg", hfg, "kJ kg-1"), ("sigma", sigma, "N m-1"),
                 ("mu_l", mu_l, "Pa s"), ("cp_l", cp_l, "kJ kg-1 K-1"),
                 ("k_l", k_l, "W m-1 K-1"), ("rho_l/rho_v", rho_ratio, "-"),
                 ("Pr_l", Pr_l, "-"), ("L_cap", cap, "m")]},
            {"title": "Inlet condition", "note": "DH_in = cp_l dT_sub",
             "rows": [("T_in", T_in, "degC"), ("dT_sub", DT_sub, "K"),
                      ("dH_in", DH_in, "kJ kg-1"), ("dH_in/h_fg", DH_norm, "-")]},
            {"title": "Flow and channel",
             "note": "no channel mass flux in this configuration" if G is None
                     else "Re and We on the channel scale",
             "rows": [("G", G, "kg m-2 s-1"), ("Re_l", Re_l, "-"), ("We_l", We_l, "-"),
                      ("D_char", char, "m"), ("L_heated", heated, "m"),
                      ("L/D", L_over_D, "-"), ("Bond", Bond, "-"), ("Co", conf_Co, "-")]},
            {"title": "Gravity", "note": "groups above use the reference g = 9.81 m s-2",
             "rows": [("g", gravity, "m s-2"), ("g/g_0", gravity_ratio, "-"),
                      ("microgravity", micro, "0/1")]},
        ],
        "anchors": [
            {"name": "Zuber", "value": q_zuber, "role": "residual base",
             "note": "pool-boiling limit from saturation properties"},
            {"name": "Bo-We", "value": q_kand, "role": "input token",
             "note": "engineered flow scaling, not a calibrated correlation; needs a channel mass flux"},
            {"name": "Bowring", "value": q_bow, "role": "input token",
             "note": "water in a tube-type channel, 0.2-19 MPa, G > 0"},
        ],
        "warnings": warn,
        "notes": notes,
        "n_defined": n_defined,
    }
    return record, report


def record_to_frame(record: Dict[str, object]):
    """One record (or a list of them) as the single-row frame the encoder reads."""
    import pandas as pd
    from chfpt_core import NUM_COLS, CAT_COLS, ANCHOR_COLS

    records = record if isinstance(record, list) else [record]
    cols = NUM_COLS + ANCHOR_COLS + CAT_COLS
    data = {}
    for c in cols:
        vals = [r.get(c, None) for r in records]
        if c in CAT_COLS:
            data[c] = pd.Series(vals, dtype="object")
        else:
            data[c] = pd.Series([np.nan if v is None else float(v) for v in vals],
                                dtype="float64")
    return pd.DataFrame(data)
