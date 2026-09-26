#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train CHF-PT, the physics-anchored Transformer, on the released CHF database.

CHF-PT reads one experimental record as a variable-length sequence of key-value
tokens drawn from a common physics layer, a geometry-specific layer and three
closed-form correlation anchors, and regresses the logarithmic residual to the
Zuber hydrodynamic-instability limit rather than CHF itself. A feature a record
does not define simply emits no token, which is what lets a single model span
tubes, rod bundles, helical coils, pin-fin surfaces, sprays and microgravity
channels, and lets a geometry that was never seen in training be described.

Pipeline: read the Excel file -> take the shipped 7:1:1:1 split -> mask the
CHF-derived surrogate cells -> fit the preprocessor on the training split only
-> build the model -> train with AdamW, warm-up/cosine, EMA and early stopping
on the validation macro-MAPE -> write one self-describing checkpoint.

Usage
    python train/train_chfpt.py                 # one model, seed 0
    python train/train_chfpt.py --all_seeds     # the five seeds released with the paper
    python train/train_chfpt.py --optuna        # hyper-parameter search, then train
    python train/train_chfpt.py --smoke         # three-epoch pipeline check

Writes model/chfpt/chfpt_seed<k>.pt (weights, configuration and the frozen
preprocessor in one file) and train/logs/chfpt/chfpt_seed<k>_loss.csv.

Dependencies: numpy, pandas, openpyxl, torch (optuna only for --optuna).
Runtime: roughly 20 minutes per seed on one modern GPU.
"""


from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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


DEFAULT_CFG = {
    "model_name": "chfpt",
    "stats_per_combo": 300,
    "clip": 6.0,
    "model": {
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "dropout": 0.05273524860770856,
        "drop_path": 0.1,
        "ff_mult": 4.0,
        "moe_experts": 5,
        "moe_topk": 2
    },
    "train": {
        "epochs": 120,
        "batch_size": 512,
        "lr": 0.0019296453124486486,
        "weight_decay": 1.7288904484374606e-05,
        "warmup_steps": 500,
        "min_lr_ratio": 0.02,
        "patience": 25,
        "ema_decay": 0.999,
        "grad_clip": 1.0,
        "amp": True
    },
    "loss": {
        "w_huber": 1.0,
        "w_mape": 1.0,
        "w_quant": 0.15,
        "w_nll": 0.1,
        "w_moe": 0.01,
        "huber_delta": 1.0
    }
}


SEARCH_SPACE = {
    "study": {
        "n_trials": 30,
        "seed": 42
    },
    "fixed": {
        "trial_epochs": 40,
        "trial_patience": 12
    },
    "search": {
        "d_model": {
            "type": "categorical",
            "choices": [
                384,
                512,
                640
            ]
        },
        "n_layers": {
            "type": "categorical",
            "choices": [
                6,
                8,
                10
            ]
        },
        "dropout": {
            "type": "uniform",
            "low": 0.05,
            "high": 0.2
        },
        "lr": {
            "type": "loguniform",
            "low": 1e-05,
            "high": 0.002
        },
        "weight_decay": {
            "type": "loguniform",
            "low": 1e-06,
            "high": 0.001
        }
    }
}


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
# additionally holds a fourth token, a convectively scaled Zuber feature,
# which reduces to the Zuber value identically over the whole corpus; it is
# materialised here so a released checkpoint sees the exact schema it was
# trained with, and so a model trained from scratch with these scripts is
# schema-identical to the released ones.
ANCHOR_FILE_COLS = ["anchor_zuber_kW_m2", "anchor_bowe_kW_m2", "anchor_bowring_kW_m2"]
ANCHOR_DEGENERATE = "anchor_zuber_scaled_kW_m2"
# The released checkpoints store two anchor columns under earlier names; they are
# mapped onto the current names when they are loaded.
ANCHOR_ALIASES = {
    "anchor_kandlikar_kW_m2": "anchor_bowe_kW_m2",
    "Bo\u2013We flow-scaling feature": "anchor_bowe_kW_m2",
    "anchor_flow_kW_m2": "anchor_zuber_scaled_kW_m2",
}


def canonical_anchor_names(names):
    """Map stored anchor names, as column labels or schema entries, to the current ones."""
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
        # the released checkpoints store the earlier anchor names in their schema;
        # map them so the same weights read the current dataset columns
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
# 4. Neural machinery: tensors, EMA, schedule, loss, training loop
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


class EMA:
    """Exponential moving average of the weights, used for validation and for
    the saved checkpoint. The warm-up ``min(decay, (1+t)/(10+t))`` lets the
    shadow track the model closely at the start instead of being pinned to the
    random initialisation, which matters for best-checkpoint selection."""

    def __init__(self, model: "torch.nn.Module", decay: float = 0.999):
        self.decay, self.num_updates = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def store(self, model) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}

    @torch.no_grad()
    def copy_to(self, model) -> None:
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model) -> None:
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = {}


def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.02) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if total <= warmup:
        return 1.0
    prog = (step - warmup) / max(1, total - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(1.0, prog)))


class EarlyStopper:
    def __init__(self, patience: int = 25, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.count, self.should_stop = math.inf, 0, False

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best, self.count = value, 0
            return True
        self.count += 1
        self.should_stop = self.count >= self.patience
        return False


# --- loss ------------------------------------------------------------------
def composite_loss(out, r, w: dict):
    """CHF-PT loss in residual space.

    Huber on the residual, a smooth surrogate |exp(mu-r)-1| = |pred/true - 1|
    for the relative error the model is judged on, pinball loss on the three
    quantile heads, a Gaussian likelihood on the predicted log-sigma, and the
    switch-style load-balancing term of the mixture of experts.
    """
    comps = {}
    mu = out["mu"]
    comps["huber"] = F.huber_loss(mu, r, delta=float(w.get("huber_delta", 1.0)), reduction="mean")
    comps["mape"] = torch.mean(torch.abs(torch.exp(torch.clamp(mu - r, -10, 10)) - 1.0))
    if "q05" in out:
        def pinball(pred, q):
            e = r - pred
            return torch.mean(torch.maximum(q * e, (q - 1) * e))
        comps["quant"] = (pinball(out["q05"], 0.05) + pinball(out["q50"], 0.50)
                          + pinball(out["q95"], 0.95)) / 3.0
    if "log_sigma" in out:
        ls = torch.clamp(out["log_sigma"], -6, 6)
        comps["nll"] = torch.mean(0.5 * torch.exp(-2 * ls) * (mu - r) ** 2 + ls)
    if "moe_aux" in out:
        comps["moe"] = out["moe_aux"]
    total = (float(w.get("w_huber", 1.0)) * comps["huber"]
             + float(w.get("w_mape", 1.0)) * comps["mape"])
    for key, wk, default in (("quant", "w_quant", 0.15), ("nll", "w_nll", 0.10),
                             ("moe", "w_moe", 0.01)):
        if key in comps:
            total = total + float(w.get(wk, default)) * comps[key]
    comps["total"] = total
    return comps


def mse_loss(out, r, w: dict):
    """Baseline loss: plain squared error on the z-scored raw CHF target."""
    m = ((out["mu"] - r) ** 2).mean()
    return {"total": m, "mse": m}


# --- training loop ---------------------------------------------------------
def fit_neural(model, kind: str, cfg: dict, pp: Preprocessor, frames, device,
               log_path: Path = None, verbose: bool = True):
    """AdamW with decoupled weight decay, linear warm-up and cosine decay, EMA
    weights for validation, bf16 autocast on CUDA, gradient clipping and early
    stopping on the validation macro-MAPE. Returns the best EMA weights."""
    tr = cfg["train"]
    loss_fn = composite_loss if cfg.get("loss_mode", "composite") == "composite" else mse_loss
    need_dense = kind == "ann"
    model = model.to(device)
    t_train = make_tensors(pp, frames["train"], device, need_dense)
    t_val = make_tensors(pp, frames["val"], device, need_dense)
    val_df = frames["val"].reset_index(drop=True)
    val_true = val_df[TARGET_COL].to_numpy()

    n = t_train["r"].shape[0]
    bs = int(tr.get("batch_size", 512))
    epochs = int(tr.get("epochs", 120))
    steps_per_epoch = max(1, math.ceil(n / bs))
    total_steps = epochs * steps_per_epoch
    warmup = int(tr.get("warmup_steps", min(500, total_steps // 20 + 1)))

    decay, no_decay = [], []
    for pn, p in model.named_parameters():                 # no weight decay on
        if not p.requires_grad:                            # norms, biases and embeddings
            continue
        (no_decay if (p.ndim <= 1 or "emb" in pn or "num_id" in pn or "num_val" in pn
                      or "cat_slot" in pn or "q" == pn.split(".")[-1]) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tr.get("weight_decay", 1e-4))},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(tr.get("lr", 3e-4)), betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, warmup, total_steps, float(tr.get("min_lr_ratio", 0.02))))
    ema = EMA(model, decay=float(tr.get("ema_decay", 0.999)))
    stopper = EarlyStopper(patience=int(tr.get("patience", 25)))
    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    clip = float(tr.get("grad_clip", 1.0))
    weights = cfg.get("loss", {})

    history, best, best_state, best_epoch = [], math.inf, None, -1
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        agg = {}
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                comps = loss_fn(model_forward(model, kind, t_train, idx), t_train["r"][idx], weights)
                loss = comps["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: lower the learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step(); ema.update(model)
            for k, v in comps.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach()) * len(idx)
        agg = {k: v / n for k, v in agg.items()}

        ema.store(model); ema.copy_to(model)                       # validate on EMA weights
        mu_val = predict_batched(model, kind, t_val)["mu"]
        pred = reconstruct_chf(mu_val, val_df)
        val_mape = float(np.mean(100.0 * np.abs(pred - val_true) / np.maximum(val_true, 1e-12)))
        val_macro = macro_mape(val_df, val_true, pred)
        improved = stopper.step(val_macro)
        if improved:
            best, best_epoch = val_macro, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(model)

        row = {"epoch": epoch, "lr": sched.get_last_lr()[0], "val_mape": val_mape,
               "val_macro_mape": val_macro,
               **{f"loss_{k}": agg.get(k, float("nan")) for k in
                  ("total", "huber", "mape", "quant", "nll", "moe", "mse")}}
        history.append(row)
        if log_path is not None:
            _append_csv(log_path, row)
        if verbose:
            print(f"  ep {epoch:3d} | lr {row['lr']:.2e} | train {agg['total']:.4f} "
                  f"| val MAPE {val_mape:6.3f} | val macro-MAPE {val_macro:6.3f}"
                  + ("  *" if improved else ""))
        if stopper.should_stop:
            if verbose:
                print(f"  [early stop] no macro-MAPE improvement for {stopper.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_macro_mape": best, "best_epoch": best_epoch,
            "best_mape": min((h["val_mape"] for h in history), default=float("nan")),
            "epochs_run": len(history), "history": history, "state_dict": best_state}


def _append_csv(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ===========================================================================
# 5. Model definition
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
# 6. Checkpoint format and hyper-parameter search
# ===========================================================================


CKPT_FORMAT = "chf-pt-release-1"


def save_neural_checkpoint(path: Path, model_name: str, seed: int, state_dict, cfg: dict,
                           pp: Preprocessor, summary: dict) -> Path:
    """One self-describing file per (model, seed): weights, the exact
    configuration and the frozen preprocessor, so evaluation never refits
    anything and never depends on the training environment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CKPT_FORMAT, "model_name": model_name, "seed": int(seed),
                "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
                "config": cfg, "preprocessor": pp.to_dict(), "summary": summary,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, path)
    return path


def sample_params(trial, search: dict) -> dict:
    out = {}
    for name, sp in search.items():
        t = sp["type"]
        if t == "categorical":
            out[name] = trial.suggest_categorical(name, sp["choices"])
        elif t == "uniform":
            out[name] = trial.suggest_float(name, sp["low"], sp["high"])
        elif t == "loguniform":
            out[name] = trial.suggest_float(name, sp["low"], sp["high"], log=True)
        elif t == "int":
            out[name] = trial.suggest_int(name, sp["low"], sp["high"], step=int(sp.get("step", 1)))
        else:
            raise ValueError(f"unknown search type '{t}' for '{name}'")
    return out


def apply_params(cfg: dict, params: dict) -> dict:
    """Route sampled hyper-parameters into the model or train sub-configuration."""
    TRAIN_KEYS = {"lr", "weight_decay", "batch_size", "epochs", "warmup_steps",
                  "min_lr_ratio", "patience", "ema_decay", "grad_clip"}
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {}); cfg.setdefault("train", {})
    for k, v in params.items():
        (cfg["train"] if k in TRAIN_KEYS else cfg["model"])[k] = v
    return cfg


def run_hpo(cfg: dict, space: dict, objective_fn, n_trials: int = None, tag: str = "chfpt"):
    """Optuna search minimising the validation macro-MAPE, the same criterion
    used for early stopping. ``objective_fn(cfg) -> score``."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = space.get("study", {})
    sampler = optuna.samplers.TPESampler(seed=int(st.get("seed", 42)), multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"{tag}_task1")
    fixed = space.get("fixed", {})

    def _obj(trial):
        params = sample_params(trial, space["search"])
        tcfg = apply_params(cfg, params)
        for k, v in fixed.items():
            if k in ("trial_epochs", "trial_patience"):
                tcfg["train"]["epochs" if k == "trial_epochs" else "patience"] = int(v)
        try:
            score = float(objective_fn(tcfg))
        except FloatingPointError as e:            # a diverged trial must not kill the study
            print(f"  [trial {trial.number}] diverged: {e}")
            return float("inf")
        print(f"  [trial {trial.number}] val macro-MAPE = {score:.4f}  {params}")
        return score

    study.optimize(_obj, n_trials=int(n_trials or st.get("n_trials", 30)), show_progress_bar=False)
    print(f"[optuna] best val macro-MAPE = {study.best_value:.4f}")
    print(f"[optuna] best params = {json.dumps(study.best_params)}")
    return study.best_params


# ===========================================================================
# 7. Command-line entry point
# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CHF-PT on the released CHF database (in-distribution task).")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset .xlsx")
    p.add_argument("--sheet", type=str, default=DEFAULT_SHEET)
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR / "chfpt",
                   help="directory for the trained checkpoints")
    p.add_argument("--seed", type=int, default=0, help="single training seed")
    p.add_argument("--all_seeds", action="store_true",
                   help="train seeds 0-4, the five runs released with the paper")
    p.add_argument("--epochs", type=int, default=None, help="override the number of epochs")
    p.add_argument("--leakage_policy", choices=["mask", "keep"], default="mask",
                   help="'mask' blanks the CHF-derived surrogate cells (default, used in the paper)")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument("--config", type=Path, default=None, help="JSON overriding the built-in config")
    p.add_argument("--optuna", action="store_true", help="hyper-parameter search before training")
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny run that checks the pipeline")
    return p


def load_config(args) -> dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if args.config is not None:
        user = json.loads(Path(args.config).read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.smoke:
        cfg["train"]["epochs"] = min(int(cfg["train"].get("epochs", 3)), 3)
        cfg["train"]["patience"] = 99
    return cfg


def prepare(args, cfg):
    """Load, split, apply the model's data view, fit the preprocessor on train."""
    df = load_dataset(args.data, args.sheet, leakage_policy=args.leakage_policy)
    frames = split_frames(df)
    if args.smoke:
        rng = np.random.default_rng(0)
        for k, n in (("train", 3000), ("val", 1000)):
            if len(frames[k]) > n:
                frames[k] = frames[k].iloc[rng.choice(len(frames[k]), n, replace=False)].reset_index(drop=True)
    if USES_ANCHOR:
        cfg["loss_mode"] = "composite"
    else:
        # Baselines regress raw CHF with plain squared error and never see the
        # anchor, as a target base or as an input feature.
        frames, m, s = plain_baseline_view(frames)
        cfg["loss_mode"] = "mse"; cfg["chf_mean"], cfg["chf_std"] = m, s
    pp = Preprocessor().fit(frames["train"], per_combo=int(cfg.get("stats_per_combo", 300)),
                            clip=float(cfg.get("clip", 6.0)))
    if not USES_ANCHOR:
        pp = pp.strip_anchors()
    print(f"[schema] {pp.n_num} numeric tokens "
          f"({len(pp.num_cols)} features + {len(pp.anchor_cols)} anchors) "
          f"+ {pp.n_cat} categorical tokens | residual mean {pp.resid_mean:+.4f}")
    return df, frames, pp


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args)
    device = get_device(args.device)
    out_dir = Path(args.out)
    log_dir = ROOT / "train" / "logs" / "chfpt"
    print(f"=== train CHF-PT | device={device} ===")
    df, frames, pp = prepare(args, cfg)

    if args.optuna:
        def objective(tcfg):
            set_seed(0)
            m = build_model(tcfg, pp)
            return fit_neural(m, KIND, tcfg, pp, frames, device, verbose=False)["best_macro_mape"]
        best = run_hpo(cfg, SEARCH_SPACE, objective, args.n_trials, tag="chfpt")
        cfg = apply_params(cfg, best)

    seeds = [0, 1, 2, 3, 4] if args.all_seeds else [args.seed]
    rows = []
    for seed in seeds:
        print(f"\n--- CHF-PT | seed {seed} ---")
        cfg_s = copy.deepcopy(cfg); cfg_s["seed"] = seed
        set_seed(seed)
        model = build_model(cfg_s, pp)
        print(f"trainable parameters: {count_params(model):,}")
        t0 = time.time()
        res = fit_neural(model, KIND, cfg_s, pp, frames, device,
                         log_path=log_dir / f"chfpt_seed{seed}_loss.csv")
        summary = {"best_macro_mape": res["best_macro_mape"], "best_mape": res["best_mape"],
                   "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                   "params": count_params(model), "minutes": round((time.time() - t0) / 60, 2),
                   "n_train": int(len(frames["train"])), "n_val": int(len(frames["val"]))}
        path = save_neural_checkpoint(out_dir / f"chfpt_seed{seed}.pt", "chfpt", seed,
                                      res["state_dict"], cfg_s, pp, summary)
        rows.append({"seed": seed, **summary})
        print(f"[saved] {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        print(f"[done]  best val macro-MAPE {res['best_macro_mape']:.3f} "
              f"at epoch {res['best_epoch']} ({summary['minutes']} min)")

    if len(rows) > 1:
        t = pd.DataFrame(rows)
        print("\n=== validation summary over seeds ===")
        print(t.to_string(index=False))
        print(f"macro-MAPE {t.best_macro_mape.mean():.3f} +- {t.best_macro_mape.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
