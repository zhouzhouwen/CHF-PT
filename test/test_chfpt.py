#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the released CHF-PT checkpoints on a held-out split.

Loads every model/chfpt/chfpt_seed*.pt, rebuilds the network from the
configuration and the frozen preprocessor stored inside each checkpoint, so
nothing is refitted and the result does not depend on this machine, and reports
per-seed and mean +- s.d. metrics: micro-MAPE, the domain-fair macro-MAPE, the
worst-domain MAPE, the ratio statistics used in the CHF literature, and the
empirical coverage of the conformal interval calibrated on the calibration
split, which no point prediction ever touches.

Usage
    python test/test_chfpt.py                    # the test split, every seed found
    python test/test_chfpt.py --split heldout    # validation + calibration + test
    python test/test_chfpt.py --ood              # add the Mahalanobis novelty score
    python test/test_chfpt.py --seeds 0 1

Writes test/results/chfpt/: per-seed metrics, the mean +- s.d. summary, the
per-geometry breakdown and one prediction file per seed with the interval
bounds.
"""


from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = "chfpt"
MODEL_LABEL = "CHF-PT"
USES_ANCHOR = True
KIND = "chfpt"     # forward signature used by the shared helpers


# ===========================================================================
# 1. Dataset schema, loading, split, leakage policy, target
# ===========================================================================
ROOT = Path(__file__).resolve().parents[1]          # release root (data/ train/ test/ model/)
DEFAULT_DATA = ROOT / "data" / "CHF_dataset_v1.1.xlsx"
DEFAULT_SHEET = "CHF_data"
DEFAULT_MODEL_DIR = ROOT / "model"

ID_COL = "row_id"
TARGET_COL = "CHF_kW_m2"
SPLIT_COL = "split"
TAG_COL = "flow_feature_imputation"
ANCHOR_BASE = "anchor_zuber_kW_m2"                  # residual base of CHF-PT

# --- common layer: 4 categorical identities -------------------------------
CAT_COMMON = ["fluid", "fluid_family", "geometry_type", "orientation"]

# --- common layer: 27 numeric quantities in four physical groups ----------
NUM_FLUID_STATE = [
    "P_kPa", "P_red", "P_rho_l", "P_rho_v", "P_hfg", "P_sigma", "P_mu_l",
    "P_cp_l", "P_k_l", "rho_ratio", "Pr_l", "capillary_length_m",
    "T_in_C", "DT_sub_K", "DH_in_kJ_kg", "DH_norm",
]                                                                   # 16
NUM_GRAVITY = ["gravity_m_s2", "gravity_ratio", "is_microgravity"]  # 3
NUM_FLOW = ["G_kg_m2s", "Re_l", "We_l"]                             # 3
NUM_CHANNEL = ["char_length_m", "heated_length_m", "L_over_D", "Bond", "confinement_Co"]  # 5
NUM_COMMON = NUM_FLUID_STATE + NUM_GRAVITY + NUM_FLOW + NUM_CHANNEL  # 27

# --- geometry-specific layer: 24 numeric + 3 categorical ------------------
NUM_GEOMETRY = [
    "tube_Di_mm", "annulus_De_mm", "annulus_Dh_mm", "plate_Dh_mm", "rect_Dh_mm",
    "rod_Drod_mm", "rod_pitch_mm", "rod_Lh_mm",
    "helical_Dtube_mm", "helical_Do_mm", "helical_Dcoil_mm", "helical_pitch_mm",
    "helical_Lh_mm", "helical_Dean",
    "fin_width_um", "fin_height_um", "fin_spacing_um", "fin_coverage",
    "fin_porosity", "fin_roughness", "fin_MBL_total_um",
    "spray_angle_deg", "spray_Tsat_C", "spray_DTsub_C",
]                                                                   # 24
CAT_GEOMETRY = ["fin_shape", "fin_array", "surface_material"]       # 3

# Token order below is part of the trained schema: the tokenizer holds one
# identity vector per column index, so these lists must not be reordered.
NUM_COLS = NUM_COMMON + NUM_GEOMETRY                                # 51
CAT_COLS = CAT_COMMON + CAT_GEOMETRY                                # 7

# --- physics anchors ------------------------------------------------------
# The released file carries the three independent anchors. The trained schema
# additionally holds a fourth token, the convective correction of Appendix A,
# which reduces to the Zuber value identically over the whole corpus; it is
# materialised here so a released checkpoint sees the exact schema it was
# trained with, and so a model trained from scratch with these scripts is
# schema-identical to the released ones.
ANCHOR_FILE_COLS = ["anchor_zuber_kW_m2", "anchor_bowe_kW_m2", "anchor_bowring_kW_m2"]
ANCHOR_DEGENERATE = "anchor_zuber_scaled_kW_m2"
# The two auxiliary columns were renamed in September 2026 for accuracy: the
# Bo-We flow-scaling token is an engineered boiling-number scaling and not the
# Kandlikar correlation, and the fourth column is the convectively scaled Zuber
# feature. Files and checkpoints written before the rename are mapped onto the
# current names when they are loaded, so both keep working unchanged.
ANCHOR_ALIASES = {
    "anchor_kandlikar_kW_m2": "anchor_bowe_kW_m2",
    "Bo\u2013We flow-scaling feature": "anchor_bowe_kW_m2",
    "anchor_flow_kW_m2": "anchor_zuber_scaled_kW_m2",
}


def canonical_anchor_names(names):
    """Map legacy anchor names, as column labels or stored schema entries, to current ones."""
    return [ANCHOR_ALIASES.get(str(n).strip(), n) for n in names]
ANCHOR_COLS = ANCHOR_FILE_COLS[:1] + ["anchor_bowe_kW_m2", "anchor_bowring_kW_m2",
                                      ANCHOR_DEGENERATE]

# --- numeric transforms frozen into the preprocessor ----------------------
LOG_COLS = set([
    "P_kPa", "P_rho_l", "P_rho_v", "P_hfg", "P_sigma", "P_mu_l", "P_cp_l",
    "P_k_l", "rho_ratio", "Pr_l", "capillary_length_m",
    "char_length_m", "heated_length_m", "L_over_D", "Bond", "confinement_Co",
    "tube_Di_mm", "annulus_De_mm", "annulus_Dh_mm", "plate_Dh_mm", "rect_Dh_mm",
    "rod_Drod_mm", "rod_pitch_mm", "rod_Lh_mm",
    "helical_Dtube_mm", "helical_Do_mm", "helical_Dcoil_mm", "helical_pitch_mm",
    "helical_Lh_mm", "helical_Dean",
    "fin_width_um", "fin_height_um", "fin_spacing_um", "fin_MBL_total_um",
] + ANCHOR_COLS)
LOG1P_COLS = set(NUM_FLOW)      # G/Re/We contain genuine zeros (stagnant pools)

# --- leakage policy -------------------------------------------------------
# Cells completed during compilation with a CHF-derived surrogate. Under the
# default policy 'mask' they are set to NaN, i.e. the token is simply not
# emitted, so the measured target can never re-enter the input.
LEAKY_CELLS = {
    "tube_energybalance_L": ["heated_length_m", "L_over_D"],
    "pool_evaporative_surrogate": ["G_kg_m2s", "Re_l", "We_l"],
    "spray_evaporative_surrogate": ["G_kg_m2s", "Re_l", "We_l"],
}


def set_seed(seed: int) -> None:
    """Seed python / numpy (and torch when it is imported by this script)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def load_dataset(path: Path = DEFAULT_DATA, sheet: str = DEFAULT_SHEET,
                 leakage_policy: str = "mask", verbose: bool = True) -> pd.DataFrame:
    """Read the released Excel file and attach the training target.

    Adds three working columns used downstream:
      ``_log_anchor`` = ln(Zuber anchor), ``_resid`` = ln(CHF) - ln(anchor)
      (the CHF-PT target), and the degenerate fourth anchor token.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = canonical_anchor_names(df.columns)

    missing = [c for c in [ID_COL, TARGET_COL, SPLIT_COL, ANCHOR_BASE] + NUM_COMMON if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    for c in CAT_COLS:                                   # keep NaN as an explicit level
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), other=np.nan)

    df[ANCHOR_DEGENERATE] = df[ANCHOR_BASE].to_numpy()   # see ANCHOR_COLS comment

    if leakage_policy not in {"mask", "keep"}:
        raise ValueError("leakage_policy must be 'mask' or 'keep'")
    n_masked = 0
    if leakage_policy == "mask" and TAG_COL in df.columns:
        for tag, cols in LEAKY_CELLS.items():
            m = df[TAG_COL] == tag
            for c in cols:
                if c in df.columns:
                    n_masked += int(m.sum())
                    df.loc[m, c] = np.nan

    if df[TARGET_COL].isna().any() or (df[TARGET_COL] <= 0).any():
        raise ValueError(f"{TARGET_COL} contains NaN or non-positive values")
    if df[ANCHOR_BASE].isna().any() or (df[ANCHOR_BASE] <= 0).any():
        raise ValueError(f"{ANCHOR_BASE} contains NaN or non-positive values")

    df["_log_anchor"] = np.log(df[ANCHOR_BASE].to_numpy())
    df["_resid"] = np.log(df[TARGET_COL].to_numpy()) - df["_log_anchor"]
    df = df.reset_index(drop=True)
    if verbose:
        parts = df[SPLIT_COL].value_counts()
        print(f"[data] {path.name}: {len(df):,} records | "
              + " ".join(f"{k}={int(parts.get(k, 0)):,}" for k in ("train", "val", "cal", "test"))
              + f" | leakage_policy={leakage_policy} ({n_masked:,} cells masked)")
    return df


def split_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split by the ``split`` column shipped with the dataset.

    The partition is group-stratified 7:1:1:1 within every fluid-geometry
    domain, with near-duplicate operating points bound to one subset. It is
    distributed with the data so that every user evaluates on exactly the
    records the released checkpoints never saw.
    """
    out = {k: df[df[SPLIT_COL] == k].reset_index(drop=True) for k in ("train", "val", "cal", "test")}
    missing = [k for k, v in out.items() if len(v) == 0]
    if missing:
        raise ValueError(f"split column has no rows for {missing}")
    return out


def target_resid(df: pd.DataFrame) -> np.ndarray:
    """Training target of the frame: anchor residual, or z-scored CHF for the
    plain baselines (whichever ``_resid`` the frame currently carries)."""
    return df["_resid"].to_numpy(dtype=np.float64)


def reconstruct_chf(mu: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Invert the training target back to CHF in kW m^-2.

    CHF-PT   : CHF = exp(mu + ln(anchor))            (frame carries _log_anchor)
    baselines: CHF = mu * std + mean                 (frame carries _chf_std)
    """
    mu = np.asarray(mu, dtype=np.float64)
    if "_chf_std" in df.columns:
        return mu * df["_chf_std"].to_numpy() + df["_chf_mean"].to_numpy()
    return np.exp(mu + df["_log_anchor"].to_numpy())


def plain_baseline_view(frames: Dict[str, pd.DataFrame], mean: float = None, std: float = None):
    """Baseline target: raw CHF, z-scored on the training split, plain squared error.

    The physics anchor is CHF-PT's contribution and is withheld from every
    baseline, both as the target base and as an input feature (see
    ``Preprocessor.strip_anchors``). Pass ``mean``/``std`` to reuse the
    statistics stored in a released checkpoint instead of recomputing them.
    """
    if mean is None or std is None:
        if "train" not in frames:
            raise ValueError("plain_baseline_view needs the training split, or the chf_mean and "
                             "chf_std stored in the checkpoint being evaluated")
        chf = frames["train"][TARGET_COL].to_numpy(dtype=np.float64)
        mean, std = float(chf.mean()), float(chf.std() or 1.0)
    out = {}
    for k, f in frames.items():
        f = f.copy()
        y = f[TARGET_COL].to_numpy(dtype=np.float64)
        f["_resid"] = (y - mean) / std
        f["_chf_mean"], f["_chf_std"] = mean, std
        out[k] = f
    return out, mean, std


# ===========================================================================
# 2. Preprocessor: standardisation, categorical vocabulary, encoding
# ===========================================================================


def balanced_stats_frame(df: pd.DataFrame, per_combo: int = 300, seed: int = 0) -> pd.DataFrame:
    """Up to ``per_combo`` rows per fluid-geometry domain.

    Statistics fitted on the raw corpus would be dominated by the 98% of
    records that are water in tubes, which pushes the features of the rare
    domains onto the clipping boundary.
    """
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in df.groupby(["fluid", "geometry_type"], sort=True):
        picks.append(g if len(g) <= per_combo
                     else g.iloc[rng.choice(len(g), per_combo, replace=False)])
    return pd.concat(picks, axis=0).reset_index(drop=True)


def _apply_transform(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if kind == "log":
        return np.log(np.clip(x, 1e-30, None))
    if kind == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    return x


@dataclass
class Preprocessor:
    """Fitted once on the training split, then frozen and stored with the model."""

    num_cols: List[str] = field(default_factory=list)
    anchor_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    transform: Dict[str, str] = field(default_factory=dict)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    cat_vocab: Dict[str, List[str]] = field(default_factory=dict)
    clip: float = 6.0
    resid_mean: float = 0.0
    resid_std: float = 1.0

    def fit(self, df_train: pd.DataFrame, per_combo: int = 300, clip: float = 6.0) -> "Preprocessor":
        self.num_cols = [c for c in NUM_COLS if c in df_train.columns]
        self.anchor_cols = [c for c in ANCHOR_COLS if c in df_train.columns]
        self.cat_cols = [c for c in CAT_COLS if c in df_train.columns]
        self.clip = clip
        for c in self.num_cols + self.anchor_cols:
            self.transform[c] = ("log" if c in LOG_COLS else
                                 "log1p" if c in LOG1P_COLS else "identity")
        stats = balanced_stats_frame(df_train, per_combo=per_combo)
        for c in self.num_cols + self.anchor_cols:
            t = _apply_transform(stats[c].to_numpy(), self.transform[c])
            t = t[np.isfinite(t)]
            self.mean[c] = float(np.mean(t)) if t.size else 0.0
            s = float(np.std(t)) if t.size else 1.0
            self.std[c] = s if s > 1e-8 else 1.0
        for c in self.cat_cols:
            self.cat_vocab[c] = sorted(pd.Series(df_train[c]).dropna().astype(str).unique().tolist())
        r = target_resid(df_train)
        self.resid_mean = float(np.mean(r))
        self.resid_std = float(np.std(r)) if np.std(r) > 1e-8 else 1.0
        return self

    def strip_anchors(self) -> "Preprocessor":
        """Baseline view: drop the anchor columns from the feature schema."""
        p = copy.deepcopy(self)
        for c in list(p.anchor_cols):
            p.transform.pop(c, None); p.mean.pop(c, None); p.std.pop(c, None)
        p.anchor_cols = []
        return p

    @property
    def n_num(self) -> int:
        return len(self.num_cols) + len(self.anchor_cols)

    @property
    def n_cat(self) -> int:
        return len(self.cat_cols)

    def cat_cardinalities(self) -> List[int]:
        return [len(self.cat_vocab[c]) + 1 for c in self.cat_cols]     # index 0 = unseen / absent

    @property
    def ann_input_dim(self) -> int:
        return 2 * self.n_num                                          # value block + presence block

    def _std_block(self, df: pd.DataFrame, cols: List[str]):
        n = len(df)
        val = np.zeros((n, len(cols)), dtype=np.float32)
        mask = np.zeros((n, len(cols)), dtype=np.float32)
        c_ = self.clip
        for j, col in enumerate(cols):
            raw = df[col].to_numpy(dtype=np.float64) if col in df.columns else np.full(n, np.nan)
            present = np.isfinite(raw)
            t = _apply_transform(np.where(present, raw, 0.0), self.transform[col])
            z = (t - self.mean[col]) / self.std[col]
            z = c_ * np.tanh(z / c_)                                   # soft clip keeps tail gradients
            val[:, j] = np.where(present, z, 0.0).astype(np.float32)
            mask[:, j] = present.astype(np.float32)
        return val, mask

    def encode(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        num_val, num_mask = self._std_block(df, self.num_cols)
        anc_val, anc_mask = self._std_block(df, self.anchor_cols)
        n = len(df)
        cat_code = np.zeros((n, len(self.cat_cols)), dtype=np.int64)
        for j, col in enumerate(self.cat_cols):
            levels = {lv: i + 1 for i, lv in enumerate(self.cat_vocab[col])}
            s = df[col].astype("object") if col in df.columns else pd.Series([np.nan] * n)
            cat_code[:, j] = np.array([levels.get(str(v), 0) if pd.notna(v) else 0 for v in s],
                                      dtype=np.int64)
        return {"num_val": num_val, "num_mask": num_mask,
                "anc_val": anc_val, "anc_mask": anc_mask, "cat_code": cat_code}

    def ann_matrix(self, df: pd.DataFrame):
        e = self.encode(df)
        dense = np.concatenate([e["num_val"], e["anc_val"], e["num_mask"], e["anc_mask"]],
                               axis=1).astype(np.float32)
        return dense, e["cat_code"]

    def tree_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw numerics (NaN preserved) + pandas categoricals, consumed natively
        by XGBoost and LightGBM; trees need no scaling or log transform."""
        out = df[self.num_cols + self.anchor_cols].copy()
        for c in self.cat_cols:
            out[c] = pd.Categorical(df[c].astype("object"), categories=self.cat_vocab[c])
        return out

    def to_dict(self) -> dict:
        return {"num_cols": self.num_cols, "anchor_cols": self.anchor_cols,
                "cat_cols": self.cat_cols, "transform": self.transform, "mean": self.mean,
                "std": self.std, "cat_vocab": self.cat_vocab, "clip": self.clip,
                "resid_mean": self.resid_mean, "resid_std": self.resid_std}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        p = cls()
        for k, v in d.items():
            setattr(p, k, v)
        # checkpoints trained before the September 2026 anchor rename store the
        # old column names in their schema; map them so the same weights read
        # the renamed dataset columns
        p.anchor_cols = canonical_anchor_names(getattr(p, "anchor_cols", []))
        for attr in ("transform", "mean", "std"):
            m = getattr(p, attr, None)
            if isinstance(m, dict):
                setattr(p, attr, {ANCHOR_ALIASES.get(k, k): v for k, v in m.items()})
        return p


# ===========================================================================
# 3. Metrics
# ===========================================================================


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Point-accuracy metrics, including the ratio statistics used in the CHF
    literature (P/M = predicted over measured)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return {k: float("nan") for k in ["n", "MAPE", "MedAPE", "P90APE", "MAE", "RMSE",
                                          "R2_lin", "R2_log", "Bias", "MeanPM", "StdPM",
                                          "RMSPE", "NRMSE", "within20", "within30"]}
    a = 100.0 * np.abs(yp - yt) / np.maximum(np.abs(yt), 1e-12)
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1e-12
    lt, lp = np.log(np.clip(yt, 1e-12, None)), np.log(np.clip(yp, 1e-12, None))
    ss_res_l = float(np.sum((lp - lt) ** 2))
    ss_tot_l = float(np.sum((lt - lt.mean()) ** 2)) or 1e-12
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pm = yp / np.maximum(np.abs(yt), 1e-12)
    return {
        "n": int(yt.size),
        "MAPE": float(np.mean(a)),
        "MedAPE": float(np.median(a)),
        "P90APE": float(np.percentile(a, 90)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": rmse,
        "R2_lin": float(1.0 - ss_res / ss_tot),
        "R2_log": float(1.0 - ss_res_l / ss_tot_l),
        "Bias": float(np.mean(err)),
        "MeanPM": float(np.mean(pm)),
        "StdPM": float(np.std(pm, ddof=1)) if yt.size > 1 else 0.0,
        "RMSPE": float(100.0 * np.sqrt(np.mean((err / np.maximum(np.abs(yt), 1e-12)) ** 2))),
        "NRMSE": float(100.0 * rmse / max(np.mean(yt), 1e-12)),
        "within20": float(np.mean(a <= 20.0)),
        "within30": float(np.mean(a <= 30.0)),
    }


def domain_mape(df: pd.DataFrame, y_true, y_pred) -> pd.Series:
    """MAPE of every fluid-geometry domain present in ``df``."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ape = 100.0 * np.abs(yp - yt) / np.maximum(np.abs(yt), 1e-12)
    d = df.reset_index(drop=True).assign(_ape=ape)
    return d.groupby(["fluid", "geometry_type"])["_ape"].mean()


def macro_mape(df: pd.DataFrame, y_true, y_pred) -> float:
    """Domain-fair error: the mean over fluid-geometry domains of each domain's
    MAPE, so a rare domain counts as much as the water-tube bulk. This is the
    primary metric and the target of every early-stopping and tuning decision."""
    per = domain_mape(df, y_true, y_pred)
    return float(per.mean()) if len(per) else float("inf")


def worst_domain_mape(df: pd.DataFrame, y_true, y_pred) -> float:
    per = domain_mape(df, y_true, y_pred)
    return float(per.max()) if len(per) else float("inf")


def breakdown(df: pd.DataFrame, y_true, y_pred, by: str) -> pd.DataFrame:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    d = df.reset_index(drop=True)
    rows = []
    for key, idx in d.groupby(by).groups.items():
        ii = d.index.get_indexer(idx)
        rows.append({by: key, **compute_metrics(yt[ii], yp[ii])})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ===========================================================================
# 4. Neural helpers: tensors and batched inference
# ===========================================================================


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(name: str = None) -> "torch.device":
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_tensors(pp: Preprocessor, frame: pd.DataFrame, device, need_dense: bool = False):
    """The whole corpus fits in memory, so tensors stay resident and mini-batches
    are drawn by shuffled indices; no DataLoader is needed."""
    frame = frame.reset_index(drop=True)
    enc = pp.encode(frame)
    num_val = np.concatenate([enc["num_val"], enc["anc_val"]], axis=1)
    num_mask = np.concatenate([enc["num_mask"], enc["anc_mask"]], axis=1)
    t = {"num_val": torch.tensor(num_val, device=device),
         "num_mask": torch.tensor(num_mask, device=device),
         "cat_code": torch.tensor(enc["cat_code"], device=device),
         "r": torch.tensor(target_resid(frame), dtype=torch.float32, device=device)}
    if need_dense:
        dense, _ = pp.ann_matrix(frame)
        t["dense"] = torch.tensor(dense, device=device)
    return t


def model_forward(model, kind: str, t, idx):
    if kind == "ann":
        return model(t["dense"][idx], t["cat_code"][idx])
    return model(t["num_val"][idx], t["num_mask"][idx], t["cat_code"][idx])


@torch.no_grad()
def predict_batched(model, kind: str, t, keys=("mu",), batch: int = 4096):
    model.eval()
    n = t["r"].shape[0]
    acc = {k: [] for k in keys}
    for s in range(0, n, batch):
        idx = torch.arange(s, min(s + batch, n), device=t["r"].device)
        out = model_forward(model, kind, t, idx)
        for k in keys:
            acc[k].append(out[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


# ===========================================================================
# 5. Model definition, identical to train/train_chfpt.py
# ===========================================================================


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch for a sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = 1 - self.p
        mask = torch.empty((x.shape[0],) + (1,) * (x.dim() - 1), device=x.device).bernoulli_(keep)
        return x / keep * mask


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        h = int(d * mult)
        self.w1, self.w2, self.wo = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        return self.wo(self.do(F.silu(self.w1(x)) * self.w2(x)))


class AttnPool(nn.Module):
    """One learnable query attends over the tokens a record actually carries."""

    def __init__(self, d, heads):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, key_padding_mask=None):
        q = self.q.expand(tokens.shape[0], -1, -1)
        z, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(z.squeeze(1))


class FeatureTokenizer(nn.Module):
    """Turn one record into a variable-length sequence of key-value tokens.

    numeric token j   = identity(j) + value * projection(j)
    categorical token = level embedding + slot embedding
    A feature the record does not define emits no token: its position is removed
    from attention through the key-padding mask, which is what lets one model
    serve geometries with different feature sets.
    """

    def __init__(self, n_num: int, cat_cards: List[int], d: int):
        super().__init__()
        self.num_id = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.num_val = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d, padding_idx=0) for c in cat_cards])
        self.cat_slot = nn.Parameter(torch.randn(len(cat_cards), d) * 0.02)

    def forward(self, num_val, num_mask, cat_code):
        num_tok = self.num_id.unsqueeze(0) + num_val.unsqueeze(-1) * self.num_val.unsqueeze(0)
        cat_toks = [emb(cat_code[:, k]) + self.cat_slot[k] for k, emb in enumerate(self.cat_emb)]
        cat_tok = (torch.stack(cat_toks, dim=1) if cat_toks
                   else num_tok.new_zeros(num_val.shape[0], 0, num_tok.shape[-1]))
        tokens = torch.cat([num_tok, cat_tok], dim=1)
        present = torch.cat([num_mask, (cat_code != 0).float()], dim=1)
        return tokens, present < 0.5


class PreNormBlock(nn.Module):
    """Pre-norm self-attention over the tokens of one record, then SwiGLU."""

    def __init__(self, d, heads, ff_mult, dropout, drop_path):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = SwiGLU(d, ff_mult, dropout)
        self.dp = DropPath(drop_path)
        self.do = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dp(self.do(a))
        return x + self.dp(self.ff(self.n2(x)))


class FluidMoE(nn.Module):
    """Fluid-aware mixture of experts on the pooled latent: five experts with
    top-2 routing, so a record is processed by the experts its fluid and
    geometry select, with a switch-style load-balancing term."""

    def __init__(self, d, n_experts=5, topk=2, hidden_mult=2.0, dropout=0.1):
        super().__init__()
        self.n_experts, self.topk = n_experts, topk
        self.gate = nn.Linear(d, n_experts)
        h = int(d * hidden_mult)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout), nn.Linear(h, d))
             for _ in range(n_experts)])
        self.norm = nn.LayerNorm(d)

    def forward(self, z):
        probs = F.softmax(self.gate(z), dim=-1)
        topv, topi = probs.topk(self.topk, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        gate = torch.zeros_like(probs).scatter_(1, topi, topv)
        out = (torch.stack([e(z) for e in self.experts], dim=1) * gate.unsqueeze(-1)).sum(1)
        aux = self.n_experts * torch.sum(probs.mean(0) * (gate > 0).float().mean(0))
        return self.norm(z + out), aux


class CHFPT(nn.Module):
    """CHF-PT: schema-aware key-value Transformer with a physics anchor.

    tokens -> six pre-norm blocks -> attention pooling -> fluid-aware MoE ->
    three heads. The model regresses the log-residual to the Zuber limit, and
    the bias of the mean head starts at the training residual mean so training
    begins at the anchor rather than at zero.
    """

    def __init__(self, n_num: int, cat_cards: List[int], cfg: dict, resid_mean: float = 0.0):
        super().__init__()
        d = int(cfg.get("d_model", 384))
        heads = int(cfg.get("n_heads", 8))
        n_layers = int(cfg.get("n_layers", 6))
        dropout = float(cfg.get("dropout", 0.1))
        drop_path = float(cfg.get("drop_path", 0.1))
        ff_mult = float(cfg.get("ff_mult", 4.0))
        self.tok = FeatureTokenizer(n_num, cat_cards, d)
        rates = torch.linspace(0, drop_path, n_layers).tolist()
        self.blocks = nn.ModuleList([PreNormBlock(d, heads, ff_mult, dropout, rates[i])
                                     for i in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.pool = AttnPool(d, heads)
        self.moe = FluidMoE(d, int(cfg.get("moe_experts", 5)), int(cfg.get("moe_topk", 2)),
                            dropout=dropout)
        self.mu_head = nn.Linear(d, 1)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, resid_mean)
        self.q_head = nn.Linear(d, 3)          # centre offset, lower gap, upper gap
        self.sigma_head = nn.Linear(d, 1)

    def forward(self, num_val, num_mask, cat_code, **_):
        tokens, kpm = self.tok(num_val, num_mask, cat_code)
        for blk in self.blocks:
            tokens = blk(tokens, key_padding_mask=kpm)
        z = self.pool(self.norm(tokens), key_padding_mask=kpm)
        z, moe_aux = self.moe(z)
        mu = self.mu_head(z).squeeze(-1)
        q = self.q_head(z)
        q50 = mu + q[:, 0]
        q05 = q50 - F.softplus(q[:, 1])        # soft-plus gaps keep quantiles non-crossing
        q95 = q50 + F.softplus(q[:, 2])
        return {"mu": mu, "q05": q05, "q50": q50, "q95": q95,
                "log_sigma": self.sigma_head(z).squeeze(-1), "z": z, "moe_aux": moe_aux}


def build_model(cfg: dict, pp: Preprocessor) -> CHFPT:
    return CHFPT(pp.n_num, pp.cat_cardinalities(), cfg.get("model", {}), pp.resid_mean)


# ===========================================================================
# 6. Checkpoint loading
# ===========================================================================


CKPT_FORMAT = "chf-pt-release-1"


def load_neural_checkpoint(path: Path, map_location="cpu"):
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path} is not a {CKPT_FORMAT} checkpoint")
    return ck["config"], Preprocessor.from_dict(ck["preprocessor"]), ck["state_dict"], ck


# ===========================================================================
# 7. Evaluation helpers
# ===========================================================================


def evaluate_split(frame: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Point metrics plus the two domain-fair metrics of the paper."""
    y_true = frame[TARGET_COL].to_numpy(dtype=np.float64)
    m = compute_metrics(y_true, y_pred)
    m["macro_MAPE"] = macro_mape(frame, y_true, y_pred)
    m["worst_domain_MAPE"] = worst_domain_mape(frame, y_true, y_pred)
    return m


def per_row_table(frame: pd.DataFrame, y_pred: np.ndarray, extra: dict = None) -> pd.DataFrame:
    y_true = frame[TARGET_COL].to_numpy(dtype=np.float64)
    out = pd.DataFrame({
        ID_COL: frame[ID_COL].to_numpy(),
        "fluid": frame["fluid"].to_numpy(),
        "geometry_type": frame["geometry_type"].to_numpy(),
        "y_true_kW_m2": y_true, "y_pred_kW_m2": y_pred,
        "ape_pct": 100.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-12),
    })
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def summarise_seeds(rows: List[dict], keys: List[str]) -> pd.DataFrame:
    """mean +- s.d. over seeds, the form used in every table of the paper."""
    d = pd.DataFrame(rows)
    rec = {}
    for k in keys:
        if k in d.columns:
            rec[f"{k}_mean"] = float(d[k].mean())
            rec[f"{k}_std"] = float(d[k].std(ddof=1)) if len(d) > 1 else 0.0
    return pd.DataFrame([rec])


SUMMARY_KEYS = ["MAPE", "macro_MAPE", "worst_domain_MAPE", "MedAPE", "P90APE", "MAE", "RMSE",
                "R2_lin", "R2_log", "MeanPM", "StdPM", "RMSPE", "NRMSE", "within20", "coverage90"]


def report(model_label: str, split: str, per_seed: List[dict], out_dir: Path,
           preds: Dict[int, pd.DataFrame], by_domain: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(out_dir / f"metrics_per_seed_{split}.csv", index=False)
    summ = summarise_seeds(per_seed, SUMMARY_KEYS)
    summ.insert(0, "n_seeds", len(per_seed))
    summ.insert(0, "split", split)
    summ.insert(0, "model", model_label)
    summ.to_csv(out_dir / f"metrics_summary_{split}.csv", index=False)
    (out_dir / f"metrics_summary_{split}.json").write_text(
        json.dumps({"model": model_label, "split": split, "n_seeds": len(per_seed),
                    **{k: v for k, v in summ.iloc[0].items() if k not in ("model", "split", "n_seeds")}},
                   indent=2, default=float))
    by_domain.to_csv(out_dir / f"metrics_by_domain_{split}.csv", index=False)
    for s, d in preds.items():
        d.to_csv(out_dir / f"predictions_{split}_seed{s}.csv", index=False)

    print(f"\n=== {model_label} | split={split} | {len(per_seed)} seeds | "
          f"n={int(ps['n'].iloc[0])} records ===")
    hdr = f"{'seed':>5} {'MAPE':>8} {'macro':>8} {'worst':>8} {'MedAPE':>8} {'RMSE':>9} {'R2_lin':>8}"
    if "coverage90" in ps.columns:
        hdr += f" {'cov90':>7}"
    print(hdr)
    for _, r in ps.iterrows():
        line = (f"{int(r['seed']):>5} {r['MAPE']:>8.3f} {r['macro_MAPE']:>8.3f} "
                f"{r['worst_domain_MAPE']:>8.3f} {r['MedAPE']:>8.3f} {r['RMSE']:>9.2f} {r['R2_lin']:>8.4f}")
        if "coverage90" in ps.columns:
            line += f" {r['coverage90']:>7.3f}"
        print(line)
    s0 = summ.iloc[0]
    line = (f"{'mean':>5} {s0['MAPE_mean']:>8.3f} {s0['macro_MAPE_mean']:>8.3f} "
            f"{s0['worst_domain_MAPE_mean']:>8.3f} {s0['MedAPE_mean']:>8.3f} "
            f"{s0['RMSE_mean']:>9.2f} {s0['R2_lin_mean']:>8.4f}")
    if "coverage90_mean" in s0:
        line += f" {s0['coverage90_mean']:>7.3f}"
    print(line)
    line = (f"{'s.d.':>5} {s0['MAPE_std']:>8.3f} {s0['macro_MAPE_std']:>8.3f} "
            f"{s0['worst_domain_MAPE_std']:>8.3f} {s0['MedAPE_std']:>8.3f} "
            f"{s0['RMSE_std']:>9.2f} {s0['R2_lin_std']:>8.4f}")
    if "coverage90_std" in s0:
        line += f" {s0['coverage90_std']:>7.3f}"
    print(line)
    print(f"-> {out_dir}")


def find_checkpoints(model_dir: Path, model_name: str, suffix: str, seeds=None) -> List[Path]:
    files = sorted(Path(model_dir).glob(f"{model_name}_seed*{suffix}"))
    if seeds:
        keep = {int(s) for s in seeds}
        files = [f for f in files if int(re.search(r"seed(\d+)", f.name).group(1)) in keep]
    if not files:
        raise FileNotFoundError(
            f"no {model_name} checkpoints in {model_dir} (expected {model_name}_seed*{suffix})")
    return files


# ===========================================================================
# 8. Conformal intervals and the out-of-distribution score
# ===========================================================================
def conformal_interval(pred_split: dict, frame: pd.DataFrame, pred_cal: dict,
                       cal_frame: pd.DataFrame, alpha: float = 0.10):
    """Conformalised quantile regression in log space.

    The two quantile heads give a nominal 90% band; the calibration split, which
    never enters any point prediction, supplies the correction that turns it
    into a finite-sample guarantee.
    """
    la_cal = cal_frame["_log_anchor"].to_numpy()
    lo_cal, hi_cal = pred_cal["q05"] + la_cal, pred_cal["q95"] + la_cal
    y_cal = np.log(cal_frame[TARGET_COL].to_numpy())
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Q = float(np.quantile(scores, q_level, method="higher"))
    la = frame["_log_anchor"].to_numpy()
    return np.exp(pred_split["q05"] + la - Q), np.exp(pred_split["q95"] + la + Q), Q


def ood_score(z_split: np.ndarray, z_train: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of the pooled latent to the training distribution."""
    mu = z_train.mean(0)
    cov = np.cov(z_train, rowvar=False) + 1e-3 * np.eye(z_train.shape[1])
    prec = np.linalg.pinv(cov)
    d = z_split - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))


# ===========================================================================
# 9. Command-line entry point
# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the released CHF-PT checkpoints on a held-out split, "
                    "with conformal intervals and an optional out-of-distribution score.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--sheet", type=str, default=DEFAULT_SHEET)
    p.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR / "chfpt")
    p.add_argument("--split", default="test", choices=["test", "val", "cal", "train", "heldout", "all"])
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--out", type=Path, default=ROOT / "test" / "results" / "chfpt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--leakage_policy", choices=["mask", "keep"], default="mask")
    p.add_argument("--alpha", type=float, default=0.10, help="1 - nominal coverage")
    p.add_argument("--ood", action="store_true",
                   help="also compute the Mahalanobis score (needs a pass over the training split)")
    p.add_argument("--ood_ref", type=int, default=8000,
                   help="training records used as the reference distribution")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(args.device)
    ckpts = find_checkpoints(args.model_dir, "chfpt", ".pt", args.seeds)
    print(f"=== evaluate CHF-PT | {len(ckpts)} checkpoints | device={device} ===")

    df = load_dataset(args.data, args.sheet, leakage_policy=args.leakage_policy)
    frames = split_frames(df)
    frames["heldout"] = pd.concat([frames["val"], frames["cal"], frames["test"]],
                                  axis=0).reset_index(drop=True)
    frames["all"] = df.reset_index(drop=True)
    frame, cal_frame = frames[args.split], frames["cal"]

    ref_frame = None
    if args.ood:
        ref_frame = frames["train"]
        if len(ref_frame) > args.ood_ref:
            rng = np.random.default_rng(0)
            ref_frame = ref_frame.iloc[rng.choice(len(ref_frame), args.ood_ref, replace=False)].reset_index(drop=True)

    per_seed, preds = [], {}
    for path in ckpts:
        cfg, pp, state, ck = load_neural_checkpoint(path, map_location=device)
        model = build_model(cfg, pp).to(device)
        model.load_state_dict(state)
        keys = ("mu", "q05", "q95", "log_sigma") + (("z",) if args.ood else ())
        p_split = predict_batched(model, "chfpt", make_tensors(pp, frame, device), keys=keys)
        p_cal = predict_batched(model, "chfpt", make_tensors(pp, cal_frame, device),
                                keys=("q05", "q95"))
        y_pred = reconstruct_chf(p_split["mu"], frame)
        lo, hi, Q = conformal_interval(p_split, frame, p_cal, cal_frame, args.alpha)
        y_true = frame[TARGET_COL].to_numpy(dtype=np.float64)
        m = evaluate_split(frame, y_pred)
        m["coverage90"] = float(np.mean((y_true >= lo) & (y_true <= hi)))
        m["rel_width_med"] = float(np.median((hi - lo) / np.maximum(y_true, 1e-12)))
        m["conformal_Q"] = Q
        extra = {"q_LCB_kW_m2": lo, "q_UCB_kW_m2": hi,
                 "sigma_log": np.exp(np.clip(p_split["log_sigma"], -6, 6))}
        if args.ood:
            z_ref = predict_batched(model, "chfpt", make_tensors(pp, ref_frame, device),
                                    keys=("z",))["z"]
            extra["s_OOD"] = ood_score(p_split["z"], z_ref)
        seed = int(ck["seed"])
        per_seed.append({"seed": seed, "checkpoint": path.name, **m})
        preds[seed] = per_row_table(frame, y_pred, extra)
        print(f"  seed {seed}: MAPE {m['MAPE']:.3f} | macro {m['macro_MAPE']:.3f} "
              f"| worst {m['worst_domain_MAPE']:.3f} | coverage {m['coverage90']:.3f}")
        del model

    y_mean = np.mean([preds[s]["y_pred_kW_m2"].to_numpy() for s in sorted(preds)], axis=0)
    by_dom = pd.DataFrame({"fluid": frame["fluid"], "geometry_type": frame["geometry_type"],
                           "y_true": frame[TARGET_COL].to_numpy(), "y_pred_seed_mean": y_mean})
    by_dom = breakdown(by_dom, by_dom["y_true"], by_dom["y_pred_seed_mean"], "geometry_type")
    report("CHF-PT", args.split, per_seed, args.out, preds, by_dom)


if __name__ == "__main__":
    main()
