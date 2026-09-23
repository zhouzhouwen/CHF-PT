#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CHF-PT schema, preprocessor and network — extracted verbatim from the release.

Every block below is a byte-identical copy of the corresponding block of
``CHF-PT_release/test/test_chfpt.py`` (which is itself identical to
``train/train_chfpt.py``), so a checkpoint served by this demo is rebuilt by
exactly the code that trained and evaluated it. Nothing here is refitted: the
frozen preprocessor and the configuration travel inside the .pt file.

Do not edit. To re-sync after a new release, re-run ``prepare_assets.py --sync``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. Dataset schema  (verbatim: test_chfpt.py lines 60-130)
# ===========================================================================
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
# Anchor columns renamed in September 2026 for accuracy: the Bo-We flow-scaling
# token is not the Kandlikar correlation, and the fourth column is the
# convectively scaled Zuber feature. Data files and checkpoints written before
# the rename are mapped onto the current names when they are loaded.
ANCHOR_ALIASES = {
    "anchor_bowe_kW_m2": "anchor_bowe_kW_m2",
    "Bo\u2013We flow-scaling feature": "anchor_bowe_kW_m2",
    "anchor_zuber_scaled_kW_m2": "anchor_zuber_scaled_kW_m2",
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


# ===========================================================================
# 2. Preprocessor  (verbatim: test_chfpt.py lines 272-394)
# ===========================================================================


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
        # checkpoints trained before the September 2026 anchor rename
        p.anchor_cols = canonical_anchor_names(getattr(p, "anchor_cols", []))
        for attr in ("transform", "mean", "std"):
            m = getattr(p, attr, None)
            if isinstance(m, dict):
                setattr(p, attr, {ANCHOR_ALIASES.get(k, k): v for k, v in m.items()})
        return p


# ===========================================================================
# 3. Network  (verbatim: test_chfpt.py lines 528-689)
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
# 4. Checkpoint loading  (verbatim: test_chfpt.py lines 697-705)
# ===========================================================================


CKPT_FORMAT = "chf-pt-release-1"


def load_neural_checkpoint(path: Path, map_location="cpu"):
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path} is not a {CKPT_FORMAT} checkpoint")
    return ck["config"], Preprocessor.from_dict(ck["preprocessor"]), ck["state_dict"], ck

