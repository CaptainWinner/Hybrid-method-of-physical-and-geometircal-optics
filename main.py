#!/usr/bin/env python3
"""
Interactive utility to merge scattering .dat files.

Folder layout for automated batch mode (inside the script/project folder):
- "геометрическая оптика"
- "физическая оптика вперед"
- "физическая оптика назад"
- "результаты" (created automatically)

Typical filenames:
Physical optics:
    Column_L1000_lmd0.532_T0_25.dat
    Column_L1000_lmd0.532_T170_170.625.dat

Geometrical optics:
    RC_L1000.0_lm0.532.dat

Main features:
- preview and merging are controlled by M11
- the final saved file contains ALL common columns
- physical-optics regions are never modified
- in geometrical-optics regions, all Mueller-matrix columns M.. are scaled by the
  same local factor that was applied to M11:
      scale(theta) = M11_new(theta) / M11_old(theta)
- after M11 preview, user may inspect any other numeric column with the same preview layout

Modes:
1) simple trim
2) local Hermite splice
3) separate multiplicative tapers for left and right splice regions

Top-level workflows:
1) manual interactive
2) automated batch with preset parameters
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Global configuration ----------

INPUT_DIR_CANDIDATES = {
    "go": [
        "геометрическая оптика",
        "геом оптика",
        "geometrical optics",
        "geometrical_optics",
    ],
    "physical_forward": [
        "физическая оптика вперед",
        "физ оптика вперед",
        "physical optics forward",
        "physical_optics_forward",
    ],
    "physical_backward": [
        "физическая оптика назад",
        "физ оптика назад",
        "physical optics backward",
        "physical_optics_backward",
    ],
}
RESULTS_DIR_NAME = "результаты"

AUTO_PRESET = {
    "merge_mode": "hermite",   # "hermite" or "taper"
    "save_preview": True,
    "save_diagnostic": True,

    # Hermite preset
    "hermite": {
        "degree": 2,
        "left_fit_width_deg": 2.0,
        "right_fit_width_deg": 2.0,
        "left_transition_width_deg": 2.0,
        "right_transition_width_deg": 2.0,
        "po_weight": 1.0,
        "correction_space": "log",
    },

    # Two-sided taper preset
    "taper": {
        "left_theta_offset_deg": 2.0,   # left_theta_star = left_boundary + offset
        "right_theta_offset_deg": 3.0,  # right_theta_star = right_boundary - offset
        "left_fit_width_deg": 2.0,
        "right_fit_width_deg": 2.0,
        "left_po_weight": 5.0,
        "right_po_weight": 5.0,
        "left_taper_kind": "smoothstep",
        "right_taper_kind": "smoothstep",
        "left_factor_policy": "both",
        "right_factor_policy": "both",
    },
}


# ---------- I/O helpers ----------

def guess_delimiter(file_path: str | Path) -> str:
    path = Path(file_path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample_lines = []
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            sample_lines.append(line)
            if len(sample_lines) >= 5:
                break

    sample = "".join(sample_lines)
    if not sample:
        return r"\s+"

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        delim = dialect.delimiter
        if delim == " ":
            return r"\s+"
        return delim
    except csv.Error:
        pass

    if "," in sample:
        return ","
    if ";" in sample:
        return ";"
    if "\t" in sample:
        return "\t"
    return r"\s+"


def read_dat(file_path: str | Path) -> pd.DataFrame:
    delim = guess_delimiter(file_path)
    kwargs = dict(comment="#", engine="python")
    df = pd.read_csv(file_path, sep=delim, **kwargs)
    df = df.dropna(axis=1, how="all")
    return df


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(name).strip().lower())


def sanitize_column_name(name: str) -> str:
    raw = str(name).strip().strip('"').strip("'")
    key = normalize_name(raw)

    if key in {"theta", "tetha", "tetta", "scangle"}:
        return "theta"
    if key == "name":
        return "Name"
    if key == "l":
        return "L"
    if key == "csa":
        return "CSA"
    if re.fullmatch(r"m\d\d", key):
        return key.upper()

    return raw


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [sanitize_column_name(c) for c in out.columns]

    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        print("\nWarning: duplicated column names detected after normalization:")
        print(duplicated)
        print("Keeping the first occurrence of each duplicated column.\n")
        out = out.loc[:, ~out.columns.duplicated()].copy()

    return out


def resolve_column_name(columns: Iterable[str], query: str) -> str:
    normalized = {normalize_name(c): c for c in columns}
    key = normalize_name(query)
    if key in normalized:
        return normalized[key]

    for nk, orig in normalized.items():
        if key and key in nk:
            return orig

    raise KeyError(f"Column '{query}' not found.")


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(df)

    if "theta" not in out.columns:
        raise KeyError(f"Column theta/tetta/tetha not found. Columns: {list(out.columns)}")
    if "M11" not in out.columns:
        raise KeyError(f"Column M11 not found. Columns: {list(out.columns)}")

    out = out.sort_values("theta").drop_duplicates(subset=["theta"], keep="first")
    out = out.reset_index(drop=True)
    return out


def validate_frames(left_po: pd.DataFrame, go: pd.DataFrame, right_po: pd.DataFrame) -> None:
    left_max = float(left_po["theta"].max())
    right_min = float(right_po["theta"].min())

    if left_max >= right_min:
        raise ValueError(
            "Left physical-optics range overlaps the right one without a middle region. "
            f"left_max={left_max}, right_min={right_min}"
        )

    common_cols = set(left_po.columns) & set(go.columns) & set(right_po.columns)
    if "theta" not in common_cols or "M11" not in common_cols:
        raise ValueError("All three files must contain tetta/theta/tetha and M11.")


def load_and_align_triplet(left_path: str | Path, go_path: str | Path, right_path: str | Path):
    left_po = prepare_df(read_dat(left_path))
    go = prepare_df(read_dat(go_path))
    right_po = prepare_df(read_dat(right_path))

    validate_frames(left_po, go, right_po)

    common_cols = [c for c in go.columns if c in left_po.columns and c in right_po.columns]

    left_po = left_po[common_cols].copy()
    go = go[common_cols].copy()
    right_po = right_po[common_cols].copy()

    return left_po, go, right_po, common_cols


# ---------- Math helpers ----------

def interp_series(x_src: np.ndarray, y_src: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x_src, y_src)


def safe_positive(values: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.clip(values.astype(float), eps, None)


def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    den_safe = np.clip(np.asarray(den, dtype=float), eps, None)
    return np.asarray(num, dtype=float) / den_safe


def local_polyfit_value_slope(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    degree: int = 2,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        raise ValueError("Not enough points to estimate slope.")

    u = x - x0
    deg_eff = min(degree, len(x) - 1)

    if weights is None:
        coeffs = np.polyfit(u, y, deg=deg_eff)
    else:
        coeffs = np.polyfit(u, y, deg=deg_eff, w=weights)

    poly = np.poly1d(coeffs)
    dpoly = np.polyder(poly)

    value = float(poly(0.0))
    slope = float(dpoly(0.0))
    return value, slope


def estimate_boundary_value_slope(
    df: pd.DataFrame,
    boundary: float,
    fit_width_deg: float,
    side: str,
    space: str = "log",
    degree: int = 2,
    edge_weight: float = 5.0,
) -> tuple[float, float]:
    if side == "left":
        mask = (
            (df["theta"] >= (boundary - fit_width_deg)) &
            (df["theta"] <= boundary)
        )
    elif side == "right":
        mask = (
            (df["theta"] >= boundary) &
            (df["theta"] <= (boundary + fit_width_deg))
        )
    else:
        raise ValueError("side must be 'left' or 'right'")

    win = df.loc[mask, ["theta", "M11"]].copy()
    if len(win) < 2:
        raise ValueError(
            f"Too few points for boundary slope estimation: side={side}, "
            f"boundary={boundary}, fit_width_deg={fit_width_deg}"
        )

    x = win["theta"].to_numpy(float)
    y = win["M11"].to_numpy(float)

    if space == "log":
        y = np.log(safe_positive(y))
    elif space != "linear":
        raise ValueError("space must be 'log' or 'linear'")

    dist = np.abs(x - boundary)
    proximity = 1.0 - np.clip(dist / max(fit_width_deg, 1e-12), 0.0, 1.0)
    weights = 1.0 + edge_weight * proximity

    return local_polyfit_value_slope(x, y, x0=boundary, degree=degree, weights=weights)


def cubic_hermite_segment(
    x: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    dy0: float,
    y1: float,
    dy1: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    h = x1 - x0
    if h <= 0:
        raise ValueError("x1 must be greater than x0")

    t = (x - x0) / h

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    return h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1


def estimate_boundary_value(
    df: pd.DataFrame,
    boundary: float,
    fit_width_deg: float,
    side: str,
    degree: int = 1,
    edge_weight: float = 5.0,
) -> float:
    value, _ = estimate_boundary_value_slope(
        df=df,
        boundary=boundary,
        fit_width_deg=fit_width_deg,
        side=side,
        space="linear",
        degree=degree,
        edge_weight=edge_weight,
    )
    return float(value)


def transition_weight(theta: np.ndarray, theta_start: float, theta_end: float, kind: str = "linear") -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    span = max(theta_end - theta_start, 1e-12)
    s = np.clip((theta - theta_start) / span, 0.0, 1.0)

    if kind == "linear":
        return s
    if kind == "smoothstep":
        return s * s * (3.0 - 2.0 * s)
    if kind == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * s))
    raise ValueError("kind must be 'linear', 'smoothstep' or 'cosine'")


def compute_target_factor(go_boundary_value: float, po_boundary_value: float) -> float:
    eps = 1e-300
    base = max(abs(go_boundary_value), eps)
    rel = abs(po_boundary_value - go_boundary_value) / base

    if po_boundary_value >= go_boundary_value:
        factor = 1.0 + rel
    else:
        factor = 1.0 - rel

    return max(factor, eps)


def apply_factor_policy(target_factor: float, factor_policy: str) -> float:
    factor_policy = factor_policy.lower()

    if factor_policy == "both":
        return target_factor
    if factor_policy == "only_lower":
        return min(target_factor, 1.0)
    if factor_policy == "only_raise":
        return max(target_factor, 1.0)

    raise ValueError("factor_policy must be 'both', 'only_lower' or 'only_raise'")


def build_left_taper_factor(
    theta_mid: np.ndarray,
    left_boundary: float,
    left_theta_star: float,
    target_factor: float,
    taper_kind: str,
) -> np.ndarray:
    if left_theta_star <= left_boundary:
        raise ValueError("For left taper: left_theta_star must be > left_boundary.")

    factor = np.ones_like(theta_mid, dtype=float)
    mask = (theta_mid >= left_boundary) & (theta_mid <= left_theta_star)
    if np.any(mask):
        w = transition_weight(theta_mid[mask], left_boundary, left_theta_star, kind=taper_kind)
        factor[mask] = target_factor + (1.0 - target_factor) * w
    return factor


def build_right_taper_factor(
    theta_mid: np.ndarray,
    right_theta_star: float,
    right_boundary: float,
    target_factor: float,
    taper_kind: str,
) -> np.ndarray:
    if right_theta_star >= right_boundary:
        raise ValueError("For right taper: right_theta_star must be < right_boundary.")

    factor = np.ones_like(theta_mid, dtype=float)
    mask = (theta_mid >= right_theta_star) & (theta_mid <= right_boundary)
    if np.any(mask):
        w = transition_weight(theta_mid[mask], right_theta_star, right_boundary, kind=taper_kind)
        factor[mask] = 1.0 + (target_factor - 1.0) * w
    return factor


def get_scalable_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.fullmatch(r"M\d\d", str(c))]


def get_previewable_columns(df: pd.DataFrame) -> list[str]:
    mueller_cols = sorted(
        list({c for c in df.columns if re.fullmatch(r"M\d\d", str(c))}),
        key=lambda x: (int(x[1]), int(x[2]))
    )

    other_numeric = []
    seen = set(mueller_cols) | {"theta"}

    for c in df.columns:
        if c in seen:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            other_numeric.append(c)
            seen.add(c)

    return mueller_cols + other_numeric


def apply_scale_to_go_region(go_mid: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    """
    Apply the same row-wise multiplicative scale to all Mueller-matrix GO columns.
    Other columns remain unchanged.

    Robust to duplicated column names in the original dataframe.
    """
    out = go_mid.copy()
    scale = np.asarray(scale, dtype=float)

    scalable_mask = [bool(re.fullmatch(r"M\d\d", str(c))) for c in out.columns]
    if not any(scalable_mask):
        return out

    numeric_block = out.loc[:, scalable_mask].apply(pd.to_numeric, errors="coerce")
    out.loc[:, scalable_mask] = numeric_block.mul(scale, axis=0)

    return out


# ---------- Merge logic ----------

def merge_simple(left_po: pd.DataFrame, go: pd.DataFrame, right_po: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    left_boundary = float(left_po["theta"].max())
    right_boundary = float(right_po["theta"].min())

    go_mid = go[(go["theta"] > left_boundary) & (go["theta"] < right_boundary)].copy()
    result = pd.concat([left_po.copy(), go_mid, right_po.copy()], ignore_index=True)
    result = result.sort_values("theta").reset_index(drop=True)

    debug = {
        "mode": "simple",
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "theta_mid": go_mid["theta"].to_numpy(float),
        "scale": np.ones(len(go_mid), dtype=float),
        "corr": np.zeros(len(go_mid), dtype=float),
    }
    return result, debug


def merge_weighted_hermite(
    left_po: pd.DataFrame,
    go: pd.DataFrame,
    right_po: pd.DataFrame,
    degree: int,
    left_fit_width_deg: float,
    right_fit_width_deg: float,
    left_transition_width_deg: float,
    right_transition_width_deg: float,
    po_weight: float,
    correction_space: str = "log",
) -> tuple[pd.DataFrame, dict]:
    left_boundary = float(left_po["theta"].max())
    right_boundary = float(right_po["theta"].min())

    if left_boundary >= right_boundary:
        raise ValueError("Invalid merge boundaries.")

    if left_fit_width_deg <= 0 or right_fit_width_deg <= 0:
        raise ValueError("fit widths must be > 0.")

    if left_transition_width_deg <= 0 or right_transition_width_deg <= 0:
        raise ValueError("transition widths must be > 0.")

    if left_boundary + left_transition_width_deg >= right_boundary - right_transition_width_deg:
        raise ValueError(
            "Transition zones are too wide and overlap. "
            "Reduce left_transition_width_deg and/or right_transition_width_deg."
        )

    go_mid = go[(go["theta"] > left_boundary) & (go["theta"] < right_boundary)].copy()
    theta_mid = go_mid["theta"].to_numpy(float)
    m11_mid_old = go_mid["M11"].to_numpy(float)

    if correction_space == "log":
        y_mid = np.log(safe_positive(m11_mid_old))
    elif correction_space == "linear":
        y_mid = m11_mid_old.copy()
    else:
        raise ValueError("correction_space must be 'log' or 'linear'")

    y_new = y_mid.copy()

    left_end = left_boundary + left_transition_width_deg
    left_mask = (theta_mid >= left_boundary) & (theta_mid <= left_end)

    left_debug = None
    if np.any(left_mask):
        y0, dy0 = estimate_boundary_value_slope(
            left_po,
            boundary=left_boundary,
            fit_width_deg=left_fit_width_deg,
            side="left",
            space=correction_space,
            degree=degree,
            edge_weight=po_weight,
        )

        y1, dy1 = estimate_boundary_value_slope(
            go,
            boundary=left_end,
            fit_width_deg=min(left_fit_width_deg, left_transition_width_deg),
            side="right",
            space=correction_space,
            degree=degree,
            edge_weight=1.0,
        )

        y_new[left_mask] = cubic_hermite_segment(
            theta_mid[left_mask],
            x0=left_boundary,
            x1=left_end,
            y0=y0,
            dy0=dy0,
            y1=y1,
            dy1=dy1,
        )

        left_debug = {
            "x0": left_boundary,
            "x1": left_end,
            "fit_width_deg": left_fit_width_deg,
            "transition_width_deg": left_transition_width_deg,
            "y0": y0,
            "dy0": dy0,
            "y1": y1,
            "dy1": dy1,
        }

    right_start = right_boundary - right_transition_width_deg
    right_mask = (theta_mid >= right_start) & (theta_mid <= right_boundary)

    right_debug = None
    if np.any(right_mask):
        y0, dy0 = estimate_boundary_value_slope(
            go,
            boundary=right_start,
            fit_width_deg=min(right_fit_width_deg, right_transition_width_deg),
            side="left",
            space=correction_space,
            degree=degree,
            edge_weight=1.0,
        )

        y1, dy1 = estimate_boundary_value_slope(
            right_po,
            boundary=right_boundary,
            fit_width_deg=right_fit_width_deg,
            side="right",
            space=correction_space,
            degree=degree,
            edge_weight=po_weight,
        )

        y_new[right_mask] = cubic_hermite_segment(
            theta_mid[right_mask],
            x0=right_start,
            x1=right_boundary,
            y0=y0,
            dy0=dy0,
            y1=y1,
            dy1=dy1,
        )

        right_debug = {
            "x0": right_start,
            "x1": right_boundary,
            "fit_width_deg": right_fit_width_deg,
            "transition_width_deg": right_transition_width_deg,
            "y0": y0,
            "dy0": dy0,
            "y1": y1,
            "dy1": dy1,
        }

    if correction_space == "log":
        m11_mid_new = np.exp(y_new)
    else:
        m11_mid_new = y_new

    scale = safe_ratio(m11_mid_new, m11_mid_old)
    go_mid_scaled = apply_scale_to_go_region(go_mid, scale)

    result = pd.concat([left_po.copy(), go_mid_scaled, right_po.copy()], ignore_index=True)
    result = result.sort_values("theta").reset_index(drop=True)

    debug = {
        "mode": "hermite",
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "theta_mid": theta_mid,
        "m11_old": m11_mid_old,
        "m11_new": m11_mid_new,
        "scale": scale,
        "corr": y_new - y_mid,
        "left_transition": left_debug,
        "right_transition": right_debug,
    }
    return result, debug


def merge_two_sided_taper(
    left_po: pd.DataFrame,
    go: pd.DataFrame,
    right_po: pd.DataFrame,
    left_theta_star: float,
    right_theta_star: float,
    left_fit_width_deg: float,
    right_fit_width_deg: float,
    left_po_weight: float,
    right_po_weight: float,
    left_taper_kind: str = "linear",
    right_taper_kind: str = "linear",
    left_factor_policy: str = "both",
    right_factor_policy: str = "both",
) -> tuple[pd.DataFrame, dict]:
    left_boundary = float(left_po["theta"].max())
    right_boundary = float(right_po["theta"].min())

    if left_boundary >= right_boundary:
        raise ValueError("Invalid merge boundaries.")

    if left_fit_width_deg <= 0 or right_fit_width_deg <= 0:
        raise ValueError("fit widths must be > 0.")

    if left_theta_star <= left_boundary:
        raise ValueError("left_theta_star must be greater than left boundary.")

    if right_theta_star >= right_boundary:
        raise ValueError("right_theta_star must be less than right boundary.")

    if left_theta_star >= right_theta_star:
        raise ValueError(
            "left_theta_star must be less than right_theta_star "
            "so that left and right taper regions do not overlap."
        )

    go_mid = go[(go["theta"] > left_boundary) & (go["theta"] < right_boundary)].copy()
    theta_mid = go_mid["theta"].to_numpy(float)
    old_m11 = go_mid["M11"].to_numpy(float)

    go_left_boundary_value = estimate_boundary_value(
        df=go,
        boundary=left_boundary,
        fit_width_deg=left_fit_width_deg,
        side="right",
        degree=1,
        edge_weight=1.0,
    )

    po_left_boundary_value = estimate_boundary_value(
        df=left_po,
        boundary=left_boundary,
        fit_width_deg=left_fit_width_deg,
        side="left",
        degree=1,
        edge_weight=left_po_weight,
    )

    left_target_factor = compute_target_factor(go_left_boundary_value, po_left_boundary_value)
    left_target_factor = apply_factor_policy(left_target_factor, left_factor_policy)

    go_right_boundary_value = estimate_boundary_value(
        df=go,
        boundary=right_boundary,
        fit_width_deg=right_fit_width_deg,
        side="left",
        degree=1,
        edge_weight=1.0,
    )

    po_right_boundary_value = estimate_boundary_value(
        df=right_po,
        boundary=right_boundary,
        fit_width_deg=right_fit_width_deg,
        side="right",
        degree=1,
        edge_weight=right_po_weight,
    )

    right_target_factor = compute_target_factor(go_right_boundary_value, po_right_boundary_value)
    right_target_factor = apply_factor_policy(right_target_factor, right_factor_policy)

    left_factor = build_left_taper_factor(
        theta_mid=theta_mid,
        left_boundary=left_boundary,
        left_theta_star=left_theta_star,
        target_factor=left_target_factor,
        taper_kind=left_taper_kind,
    )

    right_factor = build_right_taper_factor(
        theta_mid=theta_mid,
        right_theta_star=right_theta_star,
        right_boundary=right_boundary,
        target_factor=right_target_factor,
        taper_kind=right_taper_kind,
    )

    total_factor = left_factor * right_factor
    go_mid_scaled = apply_scale_to_go_region(go_mid, total_factor)

    result = pd.concat([left_po.copy(), go_mid_scaled, right_po.copy()], ignore_index=True)
    result = result.sort_values("theta").reset_index(drop=True)

    debug = {
        "mode": "two_sided_taper",
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "left_theta_star": left_theta_star,
        "right_theta_star": right_theta_star,
        "left_target_factor": left_target_factor,
        "right_target_factor": right_target_factor,
        "theta_mid": theta_mid,
        "left_factor": left_factor,
        "right_factor": right_factor,
        "factor": total_factor,
        "scale": total_factor,
        "y_old": old_m11,
        "y_new": old_m11 * total_factor,
        "corr": old_m11 * total_factor - old_m11,
    }
    return result, debug


# ---------- Plotting ----------

def collect_finite_values(*arrays: np.ndarray) -> np.ndarray:
    parts = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def choose_scale_mode(data: np.ndarray, prefer_log: bool = False) -> tuple[str, Optional[float]]:
    """
    Returns:
        ("linear", None)
        ("log", None)
        ("symlog", linthresh)
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return "linear", None

    pos = data[data > 0]
    neg = data[data < 0]
    zero_count = np.count_nonzero(data == 0)

    if pos.size == data.size:
        if prefer_log:
            return "log", None
        ratio = pos.max() / max(pos.min(), 1e-300)
        if ratio >= 50:
            return "log", None
        return "linear", None

    if pos.size > 0 and neg.size > 0:
        abs_nonzero = np.abs(data[data != 0])
        if abs_nonzero.size == 0:
            return "linear", None
        linthresh = float(np.percentile(abs_nonzero, 20))
        linthresh = max(linthresh, 1e-8)
        return "symlog", linthresh

    if pos.size > 0 and neg.size == 0 and zero_count > 0:
        abs_nonzero = np.abs(pos)
        linthresh = float(np.percentile(abs_nonzero, 20)) if abs_nonzero.size else 1.0
        linthresh = max(linthresh, 1e-8)
        return "symlog", linthresh

    if neg.size == data.size:
        abs_nonzero = np.abs(neg)
        ratio = abs_nonzero.max() / max(abs_nonzero.min(), 1e-300)
        if ratio >= 50:
            linthresh = float(np.percentile(abs_nonzero, 20))
            linthresh = max(linthresh, 1e-8)
            return "symlog", linthresh
        return "linear", None

    return "linear", None


def compute_y_limits_local(data: np.ndarray, scale_mode: str) -> tuple[float, float]:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return 1e-12, 1.0

    if scale_mode == "log":
        data = data[data > 0]
        if data.size == 0:
            return 1e-12, 1.0
        ymin = max(float(data.min()) / 3.0, 1e-300)
        ymax = float(data.max()) * 3.0
        return ymin, ymax

    ymin = float(np.min(data))
    ymax = float(np.max(data))

    if np.isclose(ymin, ymax):
        pad = max(abs(ymin) * 0.1, 1.0)
        return ymin - pad, ymax + pad

    span = ymax - ymin
    return ymin - 0.1 * span, ymax + 0.1 * span


def get_window_values(df: pd.DataFrame, column: str, x_min: float, x_max: float) -> np.ndarray:
    mask = (df["theta"] >= x_min) & (df["theta"] <= x_max)
    return df.loc[mask, column].to_numpy(dtype=float)


def apply_axis_scale(ax, scale_mode: str, linthresh: Optional[float]) -> None:
    if scale_mode == "log":
        ax.set_yscale("log")
    elif scale_mode == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh if linthresh is not None else 1.0)


def plot_preview(
    left_po: pd.DataFrame,
    go: pd.DataFrame,
    right_po: pd.DataFrame,
    merged: pd.DataFrame,
    column: str = "M11",
    show_raw_go: bool = True,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    zoom_deg: float = 10.0,
    force_log_for_m11: bool = True,
) -> None:
    left_boundary = float(left_po["theta"].max())
    right_boundary = float(right_po["theta"].min())

    if column not in merged.columns:
        raise KeyError(f"Column '{column}' is not present in merged data.")

    all_data = collect_finite_values(
        left_po[column].to_numpy(float),
        go[column].to_numpy(float),
        right_po[column].to_numpy(float),
        merged[column].to_numpy(float),
    )

    scale_mode, linthresh = choose_scale_mode(
        all_data,
        prefer_log=(force_log_for_m11 and column == "M11"),
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), constrained_layout=True)

    windows = [
        (0.0, 180.0),
        (max(0.0, left_boundary - zoom_deg), min(180.0, left_boundary + zoom_deg)),
        (max(0.0, right_boundary - zoom_deg), min(180.0, right_boundary + zoom_deg)),
    ]

    ax = axes[0]
    x_min, x_max = windows[0]
    if show_raw_go:
        ax.plot(go["theta"], go[column], linestyle="--", label="Geometrical optics (original)")
    ax.plot(merged["theta"], merged[column], linewidth=1.8, label="Merged result")
    ax.plot(left_po["theta"], left_po[column], linewidth=2.0, label="Physical optics (left)")
    ax.plot(right_po["theta"], right_po[column], linewidth=2.0, label="Physical optics (right)")
    local_data = collect_finite_values(
        get_window_values(left_po, column, x_min, x_max),
        get_window_values(go, column, x_min, x_max),
        get_window_values(right_po, column, x_min, x_max),
        get_window_values(merged, column, x_min, x_max),
    )
    ymin, ymax = compute_y_limits_local(local_data, scale_mode)
    apply_axis_scale(ax, scale_mode, linthresh)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("theta, deg")
    ax.set_ylabel(column)
    ax.legend()

    ax = axes[1]
    x_min, x_max = windows[1]
    if show_raw_go:
        ax.plot(go["theta"], go[column], linestyle="--", label="Geometrical optics (original)")
    ax.plot(merged["theta"], merged[column], linewidth=1.8, label="Merged result")
    ax.plot(left_po["theta"], left_po[column], linewidth=2.0, label="Physical optics (left)")
    local_data = collect_finite_values(
        get_window_values(left_po, column, x_min, x_max),
        get_window_values(go, column, x_min, x_max),
        get_window_values(merged, column, x_min, x_max),
    )
    ymin, ymax = compute_y_limits_local(local_data, scale_mode)
    apply_axis_scale(ax, scale_mode, linthresh)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("theta, deg")
    ax.set_ylabel(column)
    ax.legend()

    ax = axes[2]
    x_min, x_max = windows[2]
    if show_raw_go:
        ax.plot(go["theta"], go[column], linestyle="--", label="Geometrical optics (original)")
    ax.plot(merged["theta"], merged[column], linewidth=1.8, label="Merged result")
    ax.plot(right_po["theta"], right_po[column], linewidth=2.0, label="Physical optics (right)")
    local_data = collect_finite_values(
        get_window_values(go, column, x_min, x_max),
        get_window_values(right_po, column, x_min, x_max),
        get_window_values(merged, column, x_min, x_max),
    )
    ymin, ymax = compute_y_limits_local(local_data, scale_mode)
    apply_axis_scale(ax, scale_mode, linthresh)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("theta, deg")
    ax.set_ylabel(column)
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_correction_debug(
    debug: dict,
    correction_space: str,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    theta_mid = debug["theta_mid"]
    corr = debug["corr"]

    plt.figure(figsize=(10, 4.5))
    plt.plot(theta_mid, corr, linewidth=2.0, label="Applied correction")
    plt.axvline(debug["left_boundary"], linestyle="--", alpha=0.8, label="Left boundary")
    plt.axvline(debug["right_boundary"], linestyle="--", alpha=0.8, label="Right boundary")
    plt.grid(True, alpha=0.25)
    plt.xlabel("theta, deg")
    ylabel = "Correction to log(M11)" if correction_space == "log" else "Correction to M11"
    plt.ylabel(ylabel)
    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close()


def plot_factor_debug(
    debug: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    theta_mid = debug["theta_mid"]
    left_factor = debug["left_factor"]
    right_factor = debug["right_factor"]
    total_factor = debug["factor"]

    plt.figure(figsize=(10, 4.8))
    plt.plot(theta_mid, left_factor, linewidth=1.6, label="Left factor")
    plt.plot(theta_mid, right_factor, linewidth=1.6, label="Right factor")
    plt.plot(theta_mid, total_factor, linewidth=2.2, label="Total factor")

    plt.axhline(1.0, linestyle=":", alpha=0.7, label="f = 1")
    plt.axhline(debug["left_target_factor"], linestyle="--", alpha=0.7,
                label=f"Left target = {debug['left_target_factor']:.4g}")
    plt.axhline(debug["right_target_factor"], linestyle="--", alpha=0.7,
                label=f"Right target = {debug['right_target_factor']:.4g}")

    plt.axvline(debug["left_boundary"], linestyle="--", alpha=0.8, label="Left boundary")
    plt.axvline(debug["left_theta_star"], linestyle="--", alpha=0.8, label="Left critical point")
    plt.axvline(debug["right_theta_star"], linestyle="--", alpha=0.8, label="Right critical point")
    plt.axvline(debug["right_boundary"], linestyle="--", alpha=0.8, label="Right boundary")

    plt.grid(True, alpha=0.25)
    plt.xlabel("theta, deg")
    plt.ylabel("f(theta)")
    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close()


# ---------- Saving / preview helpers ----------

def save_dat(df: pd.DataFrame, file_path: str | Path, delimiter: str = "\t") -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, sep=delimiter, index=False, float_format="%.10e")


def make_intermediate_plot_paths(base_dir: Path, mode_name: str, run_index: int) -> tuple[Path, Path]:
    plots_dir = base_dir / "intermediate_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{run_index:03d}_{mode_name}_{timestamp}"

    preview_path = plots_dir / f"{base}_preview.png"
    correction_path = plots_dir / f"{base}_correction.png"
    return preview_path, correction_path


def make_column_plot_path(base_preview_path: Optional[str | Path], column: str) -> Optional[Path]:
    if base_preview_path is None:
        return None

    path = Path(base_preview_path)
    safe_col = re.sub(r"[^a-zA-Z0-9_\-]+", "_", column)
    return path.with_name(f"{path.stem}_{safe_col}{path.suffix}")


def show_main_and_optional_column_plots(
    left_po: pd.DataFrame,
    go: pd.DataFrame,
    right_po: pd.DataFrame,
    merged: pd.DataFrame,
    preview_path: Optional[str | Path] = None,
) -> None:
    plot_preview(
        left_po=left_po,
        go=go,
        right_po=right_po,
        merged=merged,
        column="M11",
        show_raw_go=True,
        save_path=preview_path,
        show=True,
        force_log_for_m11=True,
    )

    if preview_path is not None:
        print(f"Intermediate preview for M11 saved: {preview_path}")

    available_cols = get_previewable_columns(merged)

    while ask_yes_no("Do you want to preview another column?", default=False):
        print("Available columns:")
        print(", ".join(available_cols))

        query = input("Enter column name: ").strip()
        if not query:
            print("Empty column name.")
            continue

        try:
            col = resolve_column_name(available_cols, query)
        except KeyError as e:
            print(str(e))
            continue

        extra_path = make_column_plot_path(preview_path, col)
        plot_preview(
            left_po=left_po,
            go=go,
            right_po=right_po,
            merged=merged,
            column=col,
            show_raw_go=True,
            save_path=extra_path,
            show=True,
            force_log_for_m11=True,
        )

        if extra_path is not None:
            print(f"Intermediate preview for {col} saved: {extra_path}")


# ---------- Filename parsing / automated batch ----------

PHYSICAL_RE = re.compile(
    r"(?:^|_)L(?P<L>-?\d+(?:\.\d+)?)_(?:lm|lmd)(?P<lam>-?\d+(?:\.\d+)?)_T(?P<t1>-?\d+(?:\.\d+)?)_(?P<t2>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
GO_RE = re.compile(
    r"(?:^|_)L(?P<L>-?\d+(?:\.\d+)?)_(?:lm|lmd)(?P<lam>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def canonical_float_str(x: float, digits: int = 6) -> str:
    s = f"{float(x):.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def make_size_key(L: float, lam: float) -> tuple[float, float]:
    return round(float(L), 6), round(float(lam), 6)


def parse_physical_filename(path: Path) -> Optional[dict]:
    m = PHYSICAL_RE.search(path.stem)
    if not m:
        return None

    L = float(m.group("L"))
    lam = float(m.group("lam"))
    t1 = float(m.group("t1"))
    t2 = float(m.group("t2"))
    return {
        "path": path,
        "L": L,
        "lam": lam,
        "t1": t1,
        "t2": t2,
        "key": make_size_key(L, lam),
    }


def parse_go_filename(path: Path) -> Optional[dict]:
    m = GO_RE.search(path.stem)
    if not m:
        return None

    L = float(m.group("L"))
    lam = float(m.group("lam"))
    return {
        "path": path,
        "L": L,
        "lam": lam,
        "key": make_size_key(L, lam),
    }


def resolve_existing_subdir(base_dir: Path, candidates: list[str], create: bool = False) -> Optional[Path]:
    for name in candidates:
        p = base_dir / name
        if p.exists() and p.is_dir():
            return p

    if create:
        p = base_dir / candidates[0]
        p.mkdir(parents=True, exist_ok=True)
        return p

    return None


def scan_physical_folder(folder: Path) -> dict[tuple[float, float], list[dict]]:
    index: dict[tuple[float, float], list[dict]] = {}
    for path in sorted(folder.glob("*.dat")):
        meta = parse_physical_filename(path)
        if meta is None:
            continue
        index.setdefault(meta["key"], []).append(meta)
    return index


def scan_go_folder(folder: Optional[Path], label: str) -> dict[tuple[float, float], list[dict]]:
    index: dict[tuple[float, float], list[dict]] = {}
    if folder is None:
        return index

    for path in sorted(folder.glob("*.dat")):
        meta = parse_go_filename(path)
        if meta is None:
            continue
        meta["source_label"] = label
        index.setdefault(meta["key"], []).append(meta)
    return index


def choose_left_right_physical(files: list[dict]) -> Optional[tuple[dict, dict]]:
    if len(files) < 2:
        return None

    ordered = sorted(files, key=lambda d: (d["t1"], d["t2"]))
    left = ordered[0]
    right = ordered[-1]

    if left["path"] == right["path"]:
        return None
    return left, right


def build_automated_cases(base_dir: Path) -> list[dict]:
    go_dir = resolve_existing_subdir(base_dir, INPUT_DIR_CANDIDATES["go"])
    physical_forward_dir = resolve_existing_subdir(base_dir, INPUT_DIR_CANDIDATES["physical_forward"])
    physical_backward_dir = resolve_existing_subdir(base_dir, INPUT_DIR_CANDIDATES["physical_backward"])

    if go_dir is None:
        raise FileNotFoundError(
            f"Geometrical optics folder not found. Expected one of: {INPUT_DIR_CANDIDATES['go']}"
        )
    if physical_forward_dir is None:
        raise FileNotFoundError(
            f"Physical optics FORWARD folder not found. Expected one of: {INPUT_DIR_CANDIDATES['physical_forward']}"
        )
    if physical_backward_dir is None:
        raise FileNotFoundError(
            f"Physical optics BACKWARD folder not found. Expected one of: {INPUT_DIR_CANDIDATES['physical_backward']}"
        )

    go_index = scan_go_folder(go_dir, "geom")
    phys_forward_index = scan_physical_folder(physical_forward_dir)
    phys_backward_index = scan_physical_folder(physical_backward_dir)

    all_keys = sorted(set(go_index) | set(phys_forward_index) | set(phys_backward_index))
    cases: list[dict] = []

    for key in all_keys:
        go_candidates = go_index.get(key, [])
        if not go_candidates:
            continue

        phys_files = []
        phys_files.extend(phys_forward_index.get(key, []))
        phys_files.extend(phys_backward_index.get(key, []))

        chosen = choose_left_right_physical(phys_files)
        if chosen is None:
            continue
        left_meta, right_meta = chosen

        for go_meta in go_candidates:
            L, lam = key
            cases.append({
                "key": key,
                "L": L,
                "lam": lam,
                "left_physical": left_meta,
                "go": go_meta,
                "right_physical": right_meta,
                "source_label": "geom",
            })

    return cases


def save_batch_summary(rows: list[dict], results_dir: Path) -> Path:
    summary_path = results_dir / "batch_summary.csv"
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def build_result_basename(case: dict, merge_mode: str) -> str:
    L = canonical_float_str(case["L"])
    lam = canonical_float_str(case["lam"])
    src = case["source_label"]
    return f"merged_{merge_mode}_{src}_L{L}_lm{lam}"


def run_automated_batch(base_dir: Path) -> None:
    results_dir = resolve_existing_subdir(base_dir, [RESULTS_DIR_NAME], create=True)
    assert results_dir is not None

    cases = build_automated_cases(base_dir)
    if not cases:
        print("No matching automated cases were found.")
        print("Check folder names and filenames.")
        return

    print(f"\nAutomated mode: {len(cases)} case(s) found.")
    print(f"Preset merge mode: {AUTO_PRESET['merge_mode']}")
    print(f"Results folder: {results_dir}")

    summary_rows: list[dict] = []

    for idx, case in enumerate(cases, start=1):
        base_name = build_result_basename(case, AUTO_PRESET["merge_mode"])
        out_dat = results_dir / f"{base_name}.dat"
        out_png = results_dir / f"{base_name}.png"
        out_diag = results_dir / f"{base_name}_diagnostic.png"

        print(f"\n[{idx}/{len(cases)}] Processing L={canonical_float_str(case['L'])}, "
              f"lm={canonical_float_str(case['lam'])}, source={case['source_label']}")

        row = {
            "L": case["L"],
            "lambda": case["lam"],
            "source_label": case["source_label"],
            "left_physical": str(case["left_physical"]["path"]),
            "go_file": str(case["go"]["path"]),
            "right_physical": str(case["right_physical"]["path"]),
            "output_dat": str(out_dat),
            "output_png": str(out_png),
            "status": "started",
            "message": "",
        }

        try:
            left_po, go, right_po, common_cols = load_and_align_triplet(
                case["left_physical"]["path"],
                case["go"]["path"],
                case["right_physical"]["path"],
            )
            row["common_cols_count"] = len(common_cols)

            if AUTO_PRESET["merge_mode"] == "hermite":
                cfg = AUTO_PRESET["hermite"]
                merged, debug = merge_weighted_hermite(
                    left_po=left_po,
                    go=go,
                    right_po=right_po,
                    degree=cfg["degree"],
                    left_fit_width_deg=cfg["left_fit_width_deg"],
                    right_fit_width_deg=cfg["right_fit_width_deg"],
                    left_transition_width_deg=cfg["left_transition_width_deg"],
                    right_transition_width_deg=cfg["right_transition_width_deg"],
                    po_weight=cfg["po_weight"],
                    correction_space=cfg["correction_space"],
                )
                if AUTO_PRESET["save_diagnostic"]:
                    plot_correction_debug(
                        debug,
                        correction_space=cfg["correction_space"],
                        save_path=out_diag,
                        show=False,
                    )

            elif AUTO_PRESET["merge_mode"] == "taper":
                cfg = AUTO_PRESET["taper"]
                left_boundary = float(left_po["theta"].max())
                right_boundary = float(right_po["theta"].min())

                merged, debug = merge_two_sided_taper(
                    left_po=left_po,
                    go=go,
                    right_po=right_po,
                    left_theta_star=left_boundary + cfg["left_theta_offset_deg"],
                    right_theta_star=right_boundary - cfg["right_theta_offset_deg"],
                    left_fit_width_deg=cfg["left_fit_width_deg"],
                    right_fit_width_deg=cfg["right_fit_width_deg"],
                    left_po_weight=cfg["left_po_weight"],
                    right_po_weight=cfg["right_po_weight"],
                    left_taper_kind=cfg["left_taper_kind"],
                    right_taper_kind=cfg["right_taper_kind"],
                    left_factor_policy=cfg["left_factor_policy"],
                    right_factor_policy=cfg["right_factor_policy"],
                )
                if AUTO_PRESET["save_diagnostic"]:
                    plot_factor_debug(
                        debug,
                        save_path=out_diag,
                        show=False,
                    )

            else:
                raise ValueError(f"Unsupported AUTO_PRESET merge_mode: {AUTO_PRESET['merge_mode']}")

            if AUTO_PRESET["save_preview"]:
                plot_preview(
                    left_po=left_po,
                    go=go,
                    right_po=right_po,
                    merged=merged,
                    column="M11",
                    show_raw_go=True,
                    save_path=out_png,
                    show=False,
                    force_log_for_m11=True,
                )

            save_dat(merged, out_dat, delimiter="\t")

            row["status"] = "ok"
            row["message"] = "saved"

        except Exception as e:
            row["status"] = "error"
            row["message"] = str(e)
            print(f"  Error: {e}")

        summary_rows.append(row)

    summary_path = save_batch_summary(summary_rows, results_dir)
    print(f"\nBatch completed. Summary saved: {summary_path}")


# ---------- CLI ----------

def ask_path(prompt: str) -> str:
    while True:
        value = input(prompt).strip().strip('"')
        if Path(value).exists():
            return value
        print("File not found. Please try again.")


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return float(default)
    return float(raw.replace(",", "."))


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return int(default)
    return int(raw)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "д", "да"}


def run_manual_session(base_dir: Path) -> None:
    print("=== Manual interactive mode ===")
    print("Need three files: left physical optics, geometrical optics, right physical optics.\n")

    left_path = ask_path("Path to LEFT physical-optics file: ")
    go_path = ask_path("Path to GEOMETRICAL-optics file: ")
    right_path = ask_path("Path to RIGHT physical-optics file: ")

    left_po, go, right_po, common_cols = load_and_align_triplet(left_path, go_path, right_path)

    print("\nCommon columns:")
    print(common_cols)

    print("\nFiles loaded.")
    print(f"  Left physical optics:  {left_po['theta'].min():.4g} .. {left_po['theta'].max():.4g}°")
    print(f"  Geometrical optics:    {go['theta'].min():.4g} .. {go['theta'].max():.4g}°")
    print(f"  Right physical optics: {right_po['theta'].min():.4g} .. {right_po['theta'].max():.4g}°")
    print(f"  Number of common columns in final file: {len(common_cols)}")

    detected_delim = guess_delimiter(go_path)
    save_delim = "\t" if detected_delim == r"\s+" else detected_delim

    run_index = 1

    while True:
        print("\nChoose merge mode:")
        print("  1 - simple trim of geometrical optics")
        print("  2 - local Hermite splice for M11")
        print("  3 - separate multiplicative tapers on left and right")
        mode = input("Your choice [1/2/3]: ").strip() or "1"

        if mode == "1":
            merged, debug = merge_simple(left_po, go, right_po)
            preview_path, correction_path = make_intermediate_plot_paths(base_dir, "simple", run_index)

            show_main_and_optional_column_plots(
                left_po=left_po,
                go=go,
                right_po=right_po,
                merged=merged,
                preview_path=preview_path,
            )

        elif mode == "2":
            degree = ask_int(
                "Local polynomial degree for slope estimation (1=linear, 2=quadratic, 3=cubic)",
                2,
            )

            left_fit_width_deg = ask_float(
                "Fit window width for LEFT splice (value and slope estimation), deg",
                5.0,
            )

            right_fit_width_deg = ask_float(
                "Fit window width for RIGHT splice (value and slope estimation), deg",
                5.0,
            )

            left_transition_width_deg = ask_float(
                "Actual smooth transition width on the left, deg",
                2.0,
            )

            right_transition_width_deg = ask_float(
                "Actual smooth transition width on the right, deg",
                2.0,
            )

            po_weight = ask_float("Extra weight of physical-optics points near the boundary", 5.0)
            correction_space = input("Correction space: log or linear [log]: ").strip().lower() or "log"

            merged, debug = merge_weighted_hermite(
                left_po=left_po,
                go=go,
                right_po=right_po,
                degree=degree,
                left_fit_width_deg=left_fit_width_deg,
                right_fit_width_deg=right_fit_width_deg,
                left_transition_width_deg=left_transition_width_deg,
                right_transition_width_deg=right_transition_width_deg,
                po_weight=po_weight,
                correction_space=correction_space,
            )

            preview_path, correction_path = make_intermediate_plot_paths(base_dir, "hermite", run_index)

            show_main_and_optional_column_plots(
                left_po=left_po,
                go=go,
                right_po=right_po,
                merged=merged,
                preview_path=preview_path,
            )

            if ask_yes_no("Show diagnostic correction plot?", default=False):
                plot_correction_debug(
                    debug,
                    correction_space=correction_space,
                    save_path=correction_path,
                    show=True,
                )
                print(f"Diagnostic plot saved: {correction_path}")

        elif mode == "3":
            left_boundary = float(left_po["theta"].max())
            right_boundary = float(right_po["theta"].min())

            left_theta_star = ask_float(
                "Left critical point theta*_left where left taper ends, deg",
                left_boundary + 2.0,
            )

            right_theta_star = ask_float(
                "Right critical point theta*_right where right taper starts, deg",
                right_boundary - 3.0,
            )

            left_fit_width_deg = ask_float(
                "Fit window width for LEFT boundary mismatch estimate, deg",
                2.0,
            )

            right_fit_width_deg = ask_float(
                "Fit window width for RIGHT boundary mismatch estimate, deg",
                2.0,
            )

            left_po_weight = ask_float(
                "Extra weight of physical-optics points near LEFT boundary",
                5.0,
            )

            right_po_weight = ask_float(
                "Extra weight of physical-optics points near RIGHT boundary",
                5.0,
            )

            left_taper_kind = input(
                "Left taper type: linear / smoothstep / cosine [smoothstep]: "
            ).strip().lower() or "smoothstep"

            right_taper_kind = input(
                "Right taper type: linear / smoothstep / cosine [smoothstep]: "
            ).strip().lower() or "smoothstep"

            left_factor_policy = input(
                "LEFT factor policy: both / only_lower / only_raise [both]: "
            ).strip().lower() or "both"

            right_factor_policy = input(
                "RIGHT factor policy: both / only_lower / only_raise [both]: "
            ).strip().lower() or "both"

            if left_factor_policy not in {"both", "only_lower", "only_raise"}:
                print("Left factor policy must be both, only_lower or only_raise.")
                continue

            if right_factor_policy not in {"both", "only_lower", "only_raise"}:
                print("Right factor policy must be both, only_lower or only_raise.")
                continue

            merged, debug = merge_two_sided_taper(
                left_po=left_po,
                go=go,
                right_po=right_po,
                left_theta_star=left_theta_star,
                right_theta_star=right_theta_star,
                left_fit_width_deg=left_fit_width_deg,
                right_fit_width_deg=right_fit_width_deg,
                left_po_weight=left_po_weight,
                right_po_weight=right_po_weight,
                left_taper_kind=left_taper_kind,
                right_taper_kind=right_taper_kind,
                left_factor_policy=left_factor_policy,
                right_factor_policy=right_factor_policy,
            )

            preview_path, correction_path = make_intermediate_plot_paths(base_dir, "two_sided_taper", run_index)

            show_main_and_optional_column_plots(
                left_po=left_po,
                go=go,
                right_po=right_po,
                merged=merged,
                preview_path=preview_path,
            )

            if ask_yes_no("Show diagnostic factor plot?", default=False):
                plot_factor_debug(
                    debug,
                    save_path=correction_path,
                    show=True,
                )
                print(f"Diagnostic plot saved: {correction_path}")

        else:
            print("Unknown mode.")
            continue

        if ask_yes_no("Are you satisfied and want to save the final file?", default=True):
            results_dir = resolve_existing_subdir(base_dir, [RESULTS_DIR_NAME], create=True)
            assert results_dir is not None

            out_name = input("Final output filename [merged_result.dat]: ").strip().strip('"')
            if not out_name:
                out_name = "merged_result.dat"

            out_path = results_dir / out_name
            preview_path = out_path.with_suffix(".png")

            plot_preview(
                left_po=left_po,
                go=go,
                right_po=right_po,
                merged=merged,
                column="M11",
                show_raw_go=True,
                save_path=preview_path,
                show=False,
                force_log_for_m11=True,
            )
            save_dat(merged, out_path, delimiter=save_delim)

            print(f"\nDone. Final file saved: {out_path}")
            print(f"Final M11 preview saved: {preview_path}")
            break

        if not ask_yes_no("Continue experimenting with the same files?", default=True):
            print("Finished without saving.")
            break

        run_index += 1


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    print("=== Scattering file merger ===")
    print(f"Base directory: {base_dir}")
    print("\nChoose workflow:")
    print("  1 - manual interactive analysis")
    print("  2 - automated batch analysis")
    workflow = input("Your choice [1/2]: ").strip() or "1"

    if workflow == "1":
        run_manual_session(base_dir)
    elif workflow == "2":
        run_automated_batch(base_dir)
    else:
        print("Unknown workflow.")


if __name__ == "__main__":
    main()
