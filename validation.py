#!/usr/bin/env python3
"""
Validation of a hybrid GO-PO solution against a full Physical Optics solution.

Physical Optics (PO) is treated as the exact/reference solution.
Hybrid GO-PO is treated as the approximate solution.

Reference PO may be a headerless Excel file; merged/hybrid may remain .dat.\nOutputs are saved as tab-separated .dat files plus diagnostic .png figures.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================
REFERENCE_FILE = Path(
    r"C:\Users\thefo\OneDrive\Desktop\написание статьи\Column_div2_L100_N8_lmd0.532_mbs_po_gpu_mpi_float_fast.dat"
)

# Result produced by the hybrid GO-PO algorithm
HYBRID_FILE = Path(
    r"C:\Users\thefo\OneDrive\Desktop\написание статьи\experimenting\результаты\for_graph_100"
)

ASK_PATHS_INTERACTIVELY = True
OUTPUT_DIR = Path("validation_100")
EPS = 1e-300
# Threshold used only for the denominator log10(M11_PO).
# Points where log10(M11_PO) is essentially zero are excluded from
# the relative-log statistic to avoid division by zero.
LOG_DENOM_EPS = 1e-12
SAVE_ALL_MUELLER_PLOTS = False


# ============================================================
# BASIC HELPERS
# ============================================================

def ask_path(prompt: str) -> Path:
    while True:
        raw = input(prompt).strip().strip('"')
        path = Path(raw)
        if path.exists():
            return path
        print(f"File not found: {path}")


def save_dat(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
        float_format="%.10e",
    )


# ============================================================
# READING FILES
# ============================================================

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
        delimiter = dialect.delimiter
        return r"\s+" if delimiter == " " else delimiter
    except csv.Error:
        pass

    if "," in sample:
        return ","
    if ";" in sample:
        return ";"
    if "\t" in sample:
        return "\t"
    return r"\s+"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(name).strip().lower())


def sanitize_column_name(name: str) -> str:
    raw = str(name).strip().strip('"').strip("'")
    key = normalize_name(raw)

    if key in {
        "theta", "tetta", "tetha", "scangle",
        "scatteringangle", "scatterangle", "angle",
    }:
        return "theta"

    if re.fullmatch(r"m\d\d", key):
        return key.upper()

    return raw


def read_scattering_file(
    path: str | Path,
    template_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read a scattering file from .dat/.txt/.csv or Excel (.xlsx/.xls/.xlsm).

    Excel files are assumed to have NO header row. If template_columns is
    supplied, the Excel columns are named by position using that template.
    This is convenient when the headerless Physical Optics Excel file has
    exactly the same column order as the merged/hybrid file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    excel_suffixes = {".xlsx", ".xls", ".xlsm"}

    if path.suffix.lower() in excel_suffixes:
        # IMPORTANT: header=None preserves the first data row.
        df = pd.read_excel(path, header=None, sheet_name=0)
        df = df.dropna(axis=1, how="all")

        if template_columns is None:
            raise ValueError(
                f"Excel file {path.name} has no column names. "
                "Read the merged/hybrid file first and pass its columns "
                "as template_columns."
            )

        if len(df.columns) != len(template_columns):
            raise ValueError(
                f"Column count mismatch for headerless Excel file {path.name}: "
                f"Excel has {len(df.columns)} columns, while the merged/hybrid "
                f"template has {len(template_columns)} columns.\n"
                f"Template columns: {template_columns}"
            )

        # Assign exactly the same column names/order as in the merged file.
        df.columns = list(template_columns)

    else:
        delimiter = guess_delimiter(path)
        df = pd.read_csv(path, sep=delimiter, comment="#", engine="python")
        df = df.dropna(axis=1, how="all")
        df.columns = [sanitize_column_name(c) for c in df.columns]

    # Standardize names also after assigning the Excel template.
    df.columns = [sanitize_column_name(c) for c in df.columns]

    if df.columns.duplicated().any():
        duplicated = list(df.columns[df.columns.duplicated()])
        print(f"Warning: duplicated columns in {path.name}: {duplicated}")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if "theta" not in df.columns:
        raise KeyError(
            f"Angle column not found in {path.name}.\n"
            f"Columns: {list(df.columns)}"
        )
    if "M11" not in df.columns:
        raise KeyError(f"M11 not found in {path.name}.")

    for col in df.columns:
        if col == "theta" or re.fullmatch(r"M\d\d", str(col)):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["theta"])
    df = (
        df.sort_values("theta")
        .drop_duplicates(subset=["theta"], keep="first")
        .reset_index(drop=True)
    )
    return df


# ============================================================
# ALIGNMENT
# ============================================================

def get_common_mueller_columns(reference: pd.DataFrame, hybrid: pd.DataFrame) -> list[str]:
    columns = []
    for i in range(1, 5):
        for j in range(1, 5):
            col = f"M{i}{j}"
            if col in reference.columns and col in hybrid.columns:
                columns.append(col)
    return columns


def interpolate_hybrid_to_reference(
    reference: pd.DataFrame,
    hybrid: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    PO is the reference and is never interpolated.
    HYBRID is interpolated onto the PO angular grid.
    M11 is interpolated in log10-space when all hybrid M11 values are positive.
    Other Mueller elements are interpolated linearly.
    """
    theta_ref_all = reference["theta"].to_numpy(dtype=float)
    theta_hybrid = hybrid["theta"].to_numpy(dtype=float)

    theta_min = max(np.nanmin(theta_ref_all), np.nanmin(theta_hybrid))
    theta_max = min(np.nanmax(theta_ref_all), np.nanmax(theta_hybrid))
    mask = (theta_ref_all >= theta_min) & (theta_ref_all <= theta_max)
    theta_ref = theta_ref_all[mask]

    result = pd.DataFrame({"theta": theta_ref})

    for col in columns:
        ref_values = reference.loc[mask, col].to_numpy(dtype=float)
        hybrid_values = hybrid[col].to_numpy(dtype=float)
        result[f"{col}_PO"] = ref_values

        if col == "M11" and np.all(np.isfinite(hybrid_values)) and np.all(hybrid_values > 0):
            log_hybrid = np.log10(hybrid_values)
            interpolated_log = np.interp(theta_ref, theta_hybrid, log_hybrid)
            interpolated = 10.0 ** interpolated_log
        else:
            interpolated = np.interp(theta_ref, theta_hybrid, hybrid_values)

        result[f"{col}_HYBRID"] = interpolated

    return result


# ============================================================
# POINTWISE ERRORS
# ============================================================

def calculate_pointwise_errors(aligned: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = aligned.copy()

    for col in columns:
        ref = result[f"{col}_PO"].to_numpy(dtype=float)
        approx = result[f"{col}_HYBRID"].to_numpy(dtype=float)

        difference = approx - ref
        abs_error = np.abs(difference)

        result[f"{col}_difference"] = difference
        result[f"{col}_absolute_error"] = abs_error

        signed_relative = np.full(len(result), np.nan)
        relative = np.full(len(result), np.nan)

        valid_reference = np.isfinite(ref) & (np.abs(ref) > EPS)

        signed_relative[valid_reference] = (
            difference[valid_reference] / ref[valid_reference] * 100.0
        )
        relative[valid_reference] = (
            abs_error[valid_reference] / np.abs(ref[valid_reference]) * 100.0
        )

        result[f"{col}_signed_relative_error_percent"] = signed_relative
        result[f"{col}_relative_error_percent"] = relative

        if col == "M11":
            valid_log = (
                np.isfinite(ref) & np.isfinite(approx) & (ref > 0) & (approx > 0)
            )

            ratio = np.full(len(result), np.nan)
            log_ratio = np.full(len(result), np.nan)
            multiplicative_difference = np.full(len(result), np.nan)

            ratio[valid_log] = approx[valid_log] / ref[valid_log]
            log_ratio[valid_log] = np.log10(ratio[valid_log])
            multiplicative_difference[valid_log] = np.maximum(
                ratio[valid_log], 1.0 / ratio[valid_log]
            )

            result["M11_ratio_HYBRID_to_PO"] = ratio
            result["M11_log10_ratio"] = log_ratio
            result["M11_absolute_log10_error"] = np.abs(log_ratio)
            result["M11_multiplicative_difference"] = multiplicative_difference

            # ------------------------------------------------------------
            # Main logarithmic relative statistic requested for the paper:
            #
            # log1 = log10(M11_PO)      -- reference Physical Optics
            # log2 = log10(M11_HYBRID)  -- hybrid method
            #
            # signed value:
            #     (log1 - log2) / log1 * 100 %
            #
            # absolute value is used as the error magnitude in the
            # summary statistics.
            # ------------------------------------------------------------
            log1 = np.full(len(result), np.nan)
            log2 = np.full(len(result), np.nan)
            signed_log_relative = np.full(len(result), np.nan)
            absolute_log_relative = np.full(len(result), np.nan)

            log1[valid_log] = np.log10(ref[valid_log])
            log2[valid_log] = np.log10(approx[valid_log])

            valid_log_relative = (
                valid_log
                & np.isfinite(log1)
                & np.isfinite(log2)
                & (np.abs(log1) > LOG_DENOM_EPS)
            )

            signed_log_relative[valid_log_relative] = (
                (log1[valid_log_relative] - log2[valid_log_relative])
                / log1[valid_log_relative]
                * 100.0
            )

            absolute_log_relative[valid_log_relative] = np.abs(
                signed_log_relative[valid_log_relative]
            )

            result["M11_log10_PO"] = log1
            result["M11_log10_HYBRID"] = log2
            result["M11_signed_relative_log_difference_percent"] = signed_log_relative
            result["M11_absolute_relative_log_error_percent"] = absolute_log_relative

    return result


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics_for_element(df: pd.DataFrame, col: str) -> dict:
    ref = df[f"{col}_PO"].to_numpy(dtype=float)
    approx = df[f"{col}_HYBRID"].to_numpy(dtype=float)

    valid = np.isfinite(ref) & np.isfinite(approx)
    ref = ref[valid]
    approx = approx[valid]

    if len(ref) == 0:
        return {"element": col, "N": 0}

    diff = approx - ref
    abs_diff = np.abs(diff)

    bias = np.mean(diff)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    reference_rms = np.sqrt(np.mean(ref ** 2))
    nrmse_rms_percent = (
        rmse / reference_rms * 100.0 if reference_rms > EPS else np.nan
    )
    max_absolute_error = np.max(abs_diff)

    valid_rel = np.abs(ref) > EPS
    relative_errors = abs_diff[valid_rel] / np.abs(ref[valid_rel]) * 100.0
    signed_relative_errors = diff[valid_rel] / ref[valid_rel] * 100.0

    if relative_errors.size:
        mean_relative = np.mean(relative_errors)
        median_relative = np.median(relative_errors)
        p90_relative = np.percentile(relative_errors, 90)
        p95_relative = np.percentile(relative_errors, 95)
        p99_relative = np.percentile(relative_errors, 99)
        max_relative = np.max(relative_errors)
        mean_signed_relative = np.mean(signed_relative_errors)
        within_1 = np.mean(relative_errors <= 1) * 100.0
        within_5 = np.mean(relative_errors <= 5) * 100.0
        within_10 = np.mean(relative_errors <= 10) * 100.0
        within_20 = np.mean(relative_errors <= 20) * 100.0
    else:
        mean_relative = median_relative = p90_relative = p95_relative = np.nan
        p99_relative = max_relative = mean_signed_relative = np.nan
        within_1 = within_5 = within_10 = within_20 = np.nan

    correlation = np.corrcoef(ref, approx)[0, 1] if len(ref) > 1 else np.nan

    ss_res = np.sum((ref - approx) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan

    statistics = {
        "element": col,
        "N": len(ref),
        "bias": bias,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE_RMS_percent": nrmse_rms_percent,
        "max_absolute_error": max_absolute_error,
        "mean_relative_error_percent": mean_relative,
        "median_relative_error_percent": median_relative,
        "mean_signed_relative_error_percent": mean_signed_relative,
        "P90_relative_error_percent": p90_relative,
        "P95_relative_error_percent": p95_relative,
        "P99_relative_error_percent": p99_relative,
        "max_relative_error_percent": max_relative,
        "points_with_error_below_1_percent": within_1,
        "points_with_error_below_5_percent": within_5,
        "points_with_error_below_10_percent": within_10,
        "points_with_error_below_20_percent": within_20,
        "correlation": correlation,
        "R_squared": r_squared,
    }

    if col == "M11":
        valid_log = (ref > 0) & (approx > 0)
        ref_log = ref[valid_log]
        approx_log = approx[valid_log]

        if len(ref_log):
            log_ratio = np.log10(approx_log / ref_log)
            abs_log_ratio = np.abs(log_ratio)
            rms_log10_error = np.sqrt(np.mean(log_ratio ** 2))
            mean_abs_log10 = np.mean(abs_log_ratio)
            median_abs_log10 = np.median(abs_log_ratio)
            p95_abs_log10 = np.percentile(abs_log_ratio, 95)
            max_abs_log10 = np.max(abs_log_ratio)
            geometric_mean_ratio = 10.0 ** np.mean(log_ratio)
            factors = np.maximum(approx_log / ref_log, ref_log / approx_log)
            mean_factor = np.mean(factors)
            median_factor = np.median(factors)
            p95_factor = np.percentile(factors, 95)
            max_factor = np.max(factors)

            # Main relative statistic in logarithmic values:
            # (log1 - log2) / log1 * 100 %, where
            # log1 = log10(M11_PO), log2 = log10(M11_HYBRID).
            log1_values = np.log10(ref_log)
            log2_values = np.log10(approx_log)

            valid_log_relative = (
                np.isfinite(log1_values)
                & np.isfinite(log2_values)
                & (np.abs(log1_values) > LOG_DENOM_EPS)
            )

            signed_log_relative = (
                (log1_values[valid_log_relative] - log2_values[valid_log_relative])
                / log1_values[valid_log_relative]
                * 100.0
            )
            abs_log_relative = np.abs(signed_log_relative)

            if abs_log_relative.size:
                n_log_relative = len(abs_log_relative)
                mean_signed_log_relative = np.mean(signed_log_relative)
                mean_abs_log_relative = np.mean(abs_log_relative)
                median_abs_log_relative = np.median(abs_log_relative)
                rms_log_relative = np.sqrt(np.mean(signed_log_relative ** 2))
                p90_abs_log_relative = np.percentile(abs_log_relative, 90)
                p95_abs_log_relative = np.percentile(abs_log_relative, 95)
                p99_abs_log_relative = np.percentile(abs_log_relative, 99)
                max_abs_log_relative = np.max(abs_log_relative)

                log_rel_within_1 = np.mean(abs_log_relative <= 1.0) * 100.0
                log_rel_within_5 = np.mean(abs_log_relative <= 5.0) * 100.0
                log_rel_within_10 = np.mean(abs_log_relative <= 10.0) * 100.0
                log_rel_within_20 = np.mean(abs_log_relative <= 20.0) * 100.0
            else:
                n_log_relative = 0
                mean_signed_log_relative = np.nan
                mean_abs_log_relative = np.nan
                median_abs_log_relative = np.nan
                rms_log_relative = np.nan
                p90_abs_log_relative = np.nan
                p95_abs_log_relative = np.nan
                p99_abs_log_relative = np.nan
                max_abs_log_relative = np.nan
                log_rel_within_1 = np.nan
                log_rel_within_5 = np.nan
                log_rel_within_10 = np.nan
                log_rel_within_20 = np.nan
        else:
            rms_log10_error = mean_abs_log10 = median_abs_log10 = np.nan
            p95_abs_log10 = max_abs_log10 = geometric_mean_ratio = np.nan
            mean_factor = median_factor = p95_factor = max_factor = np.nan

            n_log_relative = 0
            mean_signed_log_relative = np.nan
            mean_abs_log_relative = np.nan
            median_abs_log_relative = np.nan
            rms_log_relative = np.nan
            p90_abs_log_relative = np.nan
            p95_abs_log_relative = np.nan
            p99_abs_log_relative = np.nan
            max_abs_log_relative = np.nan
            log_rel_within_1 = np.nan
            log_rel_within_5 = np.nan
            log_rel_within_10 = np.nan
            log_rel_within_20 = np.nan

        statistics.update({
            "RMS_log10_error": rms_log10_error,
            "mean_absolute_log10_error": mean_abs_log10,
            "median_absolute_log10_error": median_abs_log10,
            "P95_absolute_log10_error": p95_abs_log10,
            "max_absolute_log10_error": max_abs_log10,
            "geometric_mean_HYBRID_PO_ratio": geometric_mean_ratio,
            "mean_multiplicative_difference": mean_factor,
            "median_multiplicative_difference": median_factor,
            "P95_multiplicative_difference": p95_factor,
            "max_multiplicative_difference": max_factor,

            # Main statistics based on:
            # |(log10(PO) - log10(HYBRID)) / log10(PO)| * 100 %
            "N_log_relative": n_log_relative,
            "mean_signed_relative_log_difference_percent": mean_signed_log_relative,
            "mean_absolute_relative_log_error_percent": mean_abs_log_relative,
            "median_absolute_relative_log_error_percent": median_abs_log_relative,
            "RMS_relative_log_error_percent": rms_log_relative,
            "P90_absolute_relative_log_error_percent": p90_abs_log_relative,
            "P95_absolute_relative_log_error_percent": p95_abs_log_relative,
            "P99_absolute_relative_log_error_percent": p99_abs_log_relative,
            "max_absolute_relative_log_error_percent": max_abs_log_relative,
            "log_relative_points_error_below_1_percent": log_rel_within_1,
            "log_relative_points_error_below_5_percent": log_rel_within_5,
            "log_relative_points_error_below_10_percent": log_rel_within_10,
            "log_relative_points_error_below_20_percent": log_rel_within_20,
        })

    return statistics


def calculate_all_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        calculate_statistics_for_element(df, col)
        for col in columns
    ])


# ============================================================
# FIGURES
# ============================================================

def plot_m11_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["theta"], df["M11_PO"], linewidth=2, label="Physical optics (reference)")
    ax.plot(
        df["theta"], df["M11_HYBRID"],
        linestyle="--", linewidth=1.7, label="Hybrid GO-PO"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel(r"$M_{11}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_M11_PO_vs_HYBRID.png", dpi=300)
    plt.close(fig)


def plot_m11_relative_error(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["theta"], df["M11_relative_error_percent"], linewidth=1.5)
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel(r"Relative error of $M_{11}$, %")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "02_M11_relative_error.png", dpi=300)
    plt.close(fig)


def plot_m11_logarithmic_error(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["theta"], df["M11_log10_ratio"], linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel(r"$\log_{10}(M_{11}^{hybrid}/M_{11}^{PO})$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "03_M11_logarithmic_error.png", dpi=300)
    plt.close(fig)


def plot_m11_ratio(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["theta"], df["M11_ratio_HYBRID_to_PO"], linewidth=1.5)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Exact agreement")
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel(r"$M_{11}^{hybrid}/M_{11}^{PO}$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_M11_ratio.png", dpi=300)
    plt.close(fig)


def plot_m11_multiplicative_difference(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["theta"], df["M11_multiplicative_difference"], linewidth=1.5)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Exact agreement")
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel("Multiplicative difference")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_M11_multiplicative_difference.png", dpi=300)
    plt.close(fig)


def plot_m11_relative_log_difference(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Signed relative difference of logarithmic M11 values:

        (log10(M11_PO) - log10(M11_HYBRID)) / log10(M11_PO) * 100 %

    PO is the reference solution.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["theta"],
        df["M11_signed_relative_log_difference_percent"],
        linewidth=1.5,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1, label="Exact agreement")
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel(
        r"$(\log_{10}M_{11}^{PO}-\log_{10}M_{11}^{hybrid})/"
        r"\log_{10}M_{11}^{PO}\cdot100\%,\ \%$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "06_M11_relative_difference_of_logs_percent.png",
        dpi=300,
    )
    plt.close(fig)


def plot_m11_absolute_relative_log_error(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Absolute magnitude of the main relative-log statistic.
    This is the quantity used for mean/median/percentile error statistics.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["theta"],
        df["M11_absolute_relative_log_error_percent"],
        linewidth=1.5,
    )
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel("Absolute relative error of log(M11), %")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / "07_M11_absolute_relative_log_error_percent.png",
        dpi=300,
    )
    plt.close(fig)


def choose_symlog_linthresh(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.abs(values[values != 0])
    if values.size == 0:
        return 1e-8
    return max(float(np.percentile(values, 20)), 1e-12)


def plot_mueller_element(df: pd.DataFrame, column: str, output_dir: Path) -> None:
    ref = df[f"{column}_PO"].to_numpy(dtype=float)
    hyb = df[f"{column}_HYBRID"].to_numpy(dtype=float)
    linthresh = choose_symlog_linthresh(np.concatenate([ref, hyb]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["theta"], ref, linewidth=2, label="Physical optics (reference)")
    ax.plot(df["theta"], hyb, linestyle="--", linewidth=1.7, label="Hybrid GO-PO")
    ax.set_xlabel("Scattering angle, deg")
    ax.set_ylabel(column)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}_comparison.png", dpi=300)
    plt.close(fig)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_m11_summary(statistics: pd.DataFrame) -> None:
    row = statistics[statistics["element"] == "M11"]
    if row.empty:
        return

    r = row.iloc[0]

    print("\n===============================================")
    print(" M11: HYBRID vs PHYSICAL OPTICS REFERENCE")
    print("===============================================\n")
    print(f"Number of compared points: {int(r['N'])}")

    print("\nMAIN STATISTIC: relative difference of logarithmic values")
    print("Formula: (log10(PO) - log10(HYBRID)) / log10(PO) * 100 %")
    print(f"Valid logarithmic points: {int(r['N_log_relative'])}")
    print(
        f"Mean absolute relative log error: "
        f"{r['mean_absolute_relative_log_error_percent']:.6g} %"
    )
    print(
        f"Median absolute relative log error: "
        f"{r['median_absolute_relative_log_error_percent']:.6g} %"
    )
    print(
        f"RMS relative log error: "
        f"{r['RMS_relative_log_error_percent']:.6g} %"
    )
    print(
        f"90th percentile absolute relative log error: "
        f"{r['P90_absolute_relative_log_error_percent']:.6g} %"
    )
    print(
        f"95th percentile absolute relative log error: "
        f"{r['P95_absolute_relative_log_error_percent']:.6g} %"
    )
    print(
        f"99th percentile absolute relative log error: "
        f"{r['P99_absolute_relative_log_error_percent']:.6g} %"
    )
    print(
        f"Maximum absolute relative log error: "
        f"{r['max_absolute_relative_log_error_percent']:.6g} %"
    )
    print(
        f"Mean signed relative log difference: "
        f"{r['mean_signed_relative_log_difference_percent']:.6g} %"
    )

    print("\nAdditional statistics in the original linear values:")
    print(f"Mean relative error: {r['mean_relative_error_percent']:.6g} %")
    print(f"Median relative error: {r['median_relative_error_percent']:.6g} %")
    print(f"95th percentile error: {r['P95_relative_error_percent']:.6g} %")
    print(f"Maximum relative error: {r['max_relative_error_percent']:.6g} %")
    print(f"RMSE: {r['RMSE']:.6g}")
    print(f"Normalized RMSE: {r['NRMSE_RMS_percent']:.6g} %")
    print(f"Bias: {r['bias']:.6g}")
    print(f"Correlation: {r['correlation']:.8f}")
    print(f"R^2: {r['R_squared']:.8f}")
    print("\nLogarithmic comparison:")
    print(f"Mean |log10(HYBRID/PO)|: {r['mean_absolute_log10_error']:.6g}")
    print(f"RMS log10 error: {r['RMS_log10_error']:.6g}")
    print(f"95th percentile |log10 ratio|: {r['P95_absolute_log10_error']:.6g}")
    print(f"Geometric mean HYBRID/PO ratio: {r['geometric_mean_HYBRID_PO_ratio']:.6g}")
    print(f"Median multiplicative difference: {r['median_multiplicative_difference']:.6g}")
    print(f"95th percentile multiplicative difference: {r['P95_multiplicative_difference']:.6g}")
    print("\nFraction of angular points:")
    print(f"Error <= 1%:  {r['points_with_error_below_1_percent']:.2f} %")
    print(f"Error <= 5%:  {r['points_with_error_below_5_percent']:.2f} %")
    print(f"Error <= 10%: {r['points_with_error_below_10_percent']:.2f} %")
    print(f"Error <= 20%: {r['points_with_error_below_20_percent']:.2f} %")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if ASK_PATHS_INTERACTIVELY:
        print("=== Validation: Hybrid GO-PO vs full Physical Optics ===\n")
        print("Physical Optics is treated as the exact/reference solution.\n")
        reference_file = ask_path("Path to FULL PHYSICAL OPTICS reference file: ")
        hybrid_file = ask_path("Path to HYBRID GO-PO file: ")
    else:
        reference_file = REFERENCE_FILE
        hybrid_file = HYBRID_FILE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the merged/hybrid file first because it contains the column names.
    # A headerless Excel PO file then receives exactly these names by position.
    print("\nReading hybrid GO-PO result...")
    hybrid = read_scattering_file(hybrid_file)

    print("Reading Physical Optics reference...")
    reference = read_scattering_file(
        reference_file,
        template_columns=list(hybrid.columns),
    )

    mueller_columns = get_common_mueller_columns(reference, hybrid)
    if not mueller_columns:
        raise ValueError("No common Mueller matrix elements found.")

    print("\nCommon Mueller elements:")
    print(", ".join(mueller_columns))
    print(
        f"\nPO angular range: {reference['theta'].min():.6g} .. "
        f"{reference['theta'].max():.6g} deg"
    )
    print(
        f"Hybrid angular range: {hybrid['theta'].min():.6g} .. "
        f"{hybrid['theta'].max():.6g} deg"
    )

    aligned = interpolate_hybrid_to_reference(reference, hybrid, mueller_columns)
    errors = calculate_pointwise_errors(aligned, mueller_columns)

    save_dat(errors, OUTPUT_DIR / "pointwise_errors.dat")

    statistics = calculate_all_statistics(errors, mueller_columns)
    save_dat(statistics, OUTPUT_DIR / "statistics_all_Mueller_elements.dat")

    m11_statistics = statistics[statistics["element"] == "M11"].copy()
    save_dat(m11_statistics, OUTPUT_DIR / "statistics_M11.dat")

    # A compact table containing only the main supervisor-requested
    # logarithmic relative statistics.
    log_relative_columns = [
        "element",
        "N_log_relative",
        "mean_signed_relative_log_difference_percent",
        "mean_absolute_relative_log_error_percent",
        "median_absolute_relative_log_error_percent",
        "RMS_relative_log_error_percent",
        "P90_absolute_relative_log_error_percent",
        "P95_absolute_relative_log_error_percent",
        "P99_absolute_relative_log_error_percent",
        "max_absolute_relative_log_error_percent",
        "log_relative_points_error_below_1_percent",
        "log_relative_points_error_below_5_percent",
        "log_relative_points_error_below_10_percent",
        "log_relative_points_error_below_20_percent",
    ]
    save_dat(
        m11_statistics[log_relative_columns],
        OUTPUT_DIR / "statistics_M11_relative_logs.dat",
    )

    comparison_columns = ["theta"]
    for col in mueller_columns:
        comparison_columns.extend([f"{col}_PO", f"{col}_HYBRID"])
    save_dat(errors[comparison_columns], OUTPUT_DIR / "PO_vs_HYBRID_aligned.dat")

    plot_m11_comparison(errors, OUTPUT_DIR)
    plot_m11_relative_error(errors, OUTPUT_DIR)
    plot_m11_logarithmic_error(errors, OUTPUT_DIR)
    plot_m11_ratio(errors, OUTPUT_DIR)
    plot_m11_multiplicative_difference(errors, OUTPUT_DIR)
    plot_m11_relative_log_difference(errors, OUTPUT_DIR)
    plot_m11_absolute_relative_log_error(errors, OUTPUT_DIR)

    if SAVE_ALL_MUELLER_PLOTS:
        for col in mueller_columns:
            if col != "M11":
                plot_mueller_element(errors, col, OUTPUT_DIR)

    print_m11_summary(statistics)

    print("\n===============================================")
    print(f"All results saved to:\n{OUTPUT_DIR.resolve()}")
    print("\nDAT output files:")
    print("  pointwise_errors.dat")
    print("  statistics_M11.dat")
    print("  statistics_M11_relative_logs.dat")
    print("  statistics_all_Mueller_elements.dat")
    print("  PO_vs_HYBRID_aligned.dat")
    print("===============================================")


if __name__ == "__main__":
    main()
