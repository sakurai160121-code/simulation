"""
Create paper-ready plots from a per-load/per-method/per-run CSV.

Expected input columns:
- load
- method
- run_id
- avg_tat
- user6_tat
- user7_tat
- user8_tat
- user15_tat
- user16_tat
- user17_tat
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size":        17,
    "axes.labelsize":   18,
    "axes.titlesize":   17,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  15,
    "legend.title_fontsize": 15,
    "font.sans-serif":  ["Yu Gothic", "Hiragino Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


METHOD_ORDER = ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]

METHOD_STYLES = {
    "No Sharing": {"color": "#4d4d4d", "linestyle": "--", "marker": "o", "alpha": 1.0, "linewidth": 2.0},
    "FCFS": {"color": "#1f77b4", "linestyle": ":", "marker": "s", "alpha": 0.45, "linewidth": 2.0},
    "Owner Priority": {"color": "#ff7f0e", "linestyle": "-.", "marker": "^", "alpha": 1.0, "linewidth": 2.0},
    "Preemptive": {"color": "#2ca02c", "linestyle": "-", "marker": "D", "alpha": 1.0, "linewidth": 3.0},
}

TIER_USER_MAP = {
    "tier7": ("user6_tat", "user15_tat"),
    "tier8": ("user7_tat", "user16_tat"),
    "tier9": ("user8_tat", "user17_tat"),
}

TIER_LABELS = {
    "tier7": "Tier7",
    "tier8": "Tier8",
    "tier9": "Tier9",
}


def load_results(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    required_columns = [
        "load",
        "method",
        "run_id",
        "avg_tat",
        "user6_tat",
        "user7_tat",
        "user8_tat",
        "user15_tat",
        "user16_tat",
        "user17_tat",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print("Missing columns: " + ", ".join(missing_columns))
        raise ValueError("Input CSV is missing required columns")

    return df


def add_tier_tat_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tier7_tat"] = (out["user6_tat"] + out["user15_tat"]) / 2.0
    out["tier8_tat"] = (out["user7_tat"] + out["user16_tat"]) / 2.0
    out["tier9_tat"] = (out["user8_tat"] + out["user17_tat"]) / 2.0

    LOW_USERS  = [0, 1, 2, 9, 10, 11]
    MID_USERS  = [3, 4, 5, 12, 13, 14]
    HIGH_USERS = [6, 7, 8, 15, 16, 17]

    for col, users in [
        ("tier_low_tat",  LOW_USERS),
        ("tier_mid_tat",  MID_USERS),
        ("tier_high_tat", HIGH_USERS),
    ]:
        if col not in out.columns:
            user_cols = [f"user{u}_tat" for u in users if f"user{u}_tat" in out.columns]
            if user_cols:
                vals = out[user_cols].replace(0, np.nan)
                out[col] = vals.mean(axis=1)

    return out


def summarize_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(["load", "method"], as_index=False)[metric_col]
        .agg(mean="mean", min="min", max="max")
        .sort_values(["load", "method"])
        .reset_index(drop=True)
    )
    return summary


def compute_protection_ratio(df: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(index=["load", "run_id"], columns="method", values=tier_col, aggfunc="first")
    if "No Sharing" not in pivot.columns:
        raise ValueError("No Sharing column is required for protection ratio computation")

    baseline = pivot["No Sharing"]
    valid_baseline = baseline.notna() & (baseline > 0)
    invalid_baseline_count = int((~valid_baseline).sum())
    if invalid_baseline_count > 0:
        warnings.warn(
            f"Excluded {invalid_baseline_count} run(s) because the No Sharing baseline was missing or non-positive.",
            stacklevel=2,
        )

    ratio_frames: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        if method == "No Sharing" or method not in pivot.columns:
            continue

        ratio_series = pivot[method] / baseline
        valid_ratio = valid_baseline & ratio_series.notna() & np.isfinite(ratio_series)
        excluded_count = int((~valid_ratio).sum())
        if excluded_count > 0:
            warnings.warn(
                f"Excluded {excluded_count} run(s) for {method} because the ratio was missing or invalid.",
                stacklevel=2,
            )

        ratio_df = ratio_series[valid_ratio].reset_index(name="protection_ratio")
        ratio_df["method"] = method
        ratio_frames.append(ratio_df)

    if not ratio_frames:
        return pd.DataFrame(columns=["load", "run_id", "method", "protection_ratio"])

    return pd.concat(ratio_frames, ignore_index=True)


def _positive_floor(values: list[np.ndarray]) -> float:
    positives = [array[array > 0] for array in values if np.any(array > 0)]
    if not positives:
        return 1e-6
    merged = np.concatenate(positives)
    return float(np.min(merged) * 0.5)


def _style_for_method(method: str) -> dict[str, object]:
    if method not in METHOD_STYLES:
        raise KeyError(f"Unknown method: {method}")
    return METHOD_STYLES[method].copy()


def plot_band(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    methods: list[str],
    ylabel: str,
    title: str = "",
    log_scale: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    if summary_df.empty:
        return

    load_ticks = sorted(summary_df["load"].dropna().unique().tolist())
    floor = _positive_floor([
        summary_df["mean"].to_numpy(dtype=float),
        summary_df["min"].to_numpy(dtype=float),
        summary_df["max"].to_numpy(dtype=float),
    ])

    for method in methods:
        method_df = summary_df[summary_df["method"] == method].sort_values("load")
        if method_df.empty:
            continue

        x = method_df["load"].to_numpy(dtype=float)
        mean = method_df["mean"].to_numpy(dtype=float)
        min_value = method_df["min"].to_numpy(dtype=float)
        max_value = method_df["max"].to_numpy(dtype=float)

        if log_scale:
            mean = np.where(mean > 0, mean, floor)
            min_value = np.where(min_value > 0, min_value, floor)
            max_value = np.where(max_value > 0, max_value, floor)

        style = _style_for_method(method)
        ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=method,
        )
        ax.fill_between(
            x,
            min_value,
            max_value,
            color=style["color"],
            alpha=0.15,
        )

    ax.set_xlabel("System Load")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(load_ticks)
    if log_scale:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_overall_avg_tat(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "overall_avg_tat_band.png"
    if "avg_tat" not in df.columns:
        raise ValueError("Column 'avg_tat' not found in DataFrame")

    summary = summarize_metric(df, "avg_tat")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(
        ax,
        summary,
        METHOD_ORDER,
        ylabel="System Average TAT [s]",
        title="",
        log_scale=True,
    )
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_tier_group_band(
    df: pd.DataFrame,
    tier_col: str,
    ylabel: str,
    panel_label: str,
    output_path: Path,
    methods: list[str] | None = None,
) -> Path:
    if tier_col not in df.columns:
        raise ValueError(f"Column '{tier_col}' not found in DataFrame")

    if methods is None:
        methods = METHOD_ORDER

    summary = summarize_metric(df, tier_col)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(
        ax,
        summary,
        methods,
        ylabel=ylabel,
        title="",
        log_scale=True,
    )
    ax.text(0.02, 0.95, panel_label, transform=ax.transAxes, va="top", ha="left", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


SHARING_METHODS = ["FCFS", "Owner Priority", "Preemptive"]


def plot_low_tier_tat_band(df: pd.DataFrame, output_dir: str | Path) -> Path:
    return _plot_tier_group_band(
        df, "tier_low_tat",
        ylabel="Low Tier (Tier 1–3) Average TAT [s]",
        panel_label="(a)",
        output_path=Path(output_dir) / "low_tier_tat_band.png",
        methods=SHARING_METHODS,
    )


def plot_mid_tier_tat_band(df: pd.DataFrame, output_dir: str | Path) -> Path:
    return _plot_tier_group_band(
        df, "tier_mid_tat",
        ylabel="Mid Tier (Tier 4–6) Average TAT [s]",
        panel_label="(b)",
        output_path=Path(output_dir) / "mid_tier_tat_band.png",
        methods=SHARING_METHODS,
    )


def plot_high_tier_tat_band(
    df: pd.DataFrame,
    output_dir: str | Path,
    include_no_sharing: bool = False,
    output_filename: str = "high_tier_tat_band.png",
) -> Path:
    methods = METHOD_ORDER if include_no_sharing else SHARING_METHODS
    return _plot_tier_group_band(
        df, "tier_high_tat",
        ylabel="High Tier (Tier 7–9) Average TAT [s]",
        panel_label="(c)",
        output_path=Path(output_dir) / output_filename,
        methods=methods,
    )


def plot_tier_tat_combined_band(df: pd.DataFrame, tier_name: str, tier_col: str, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / f"{tier_name}_tat_combined_band.png"
    tier_label = TIER_LABELS.get(tier_name, tier_name)

    tat_summary = summarize_metric(df, tier_col)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(
        ax,
        tat_summary,
        METHOD_ORDER,
        ylabel=f"{tier_label} User TAT [s]",
        title="",
        log_scale=True,
    )
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ratio_without_fcfs(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "protection_ratio_without_fcfs.png"
    tier9_ratio_df = compute_protection_ratio(df, "tier9_tat")
    tier9_summary = summarize_metric(tier9_ratio_df, "protection_ratio")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(
        ax,
        tier9_summary,
        ["Owner Priority", "Preemptive"],
        ylabel="Tier9 Protection Ratio",
        title="",
        log_scale=False,
        ylim=(0.7, 2.0),
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready TAT and protection-ratio plots from CSV")
    parser.add_argument("--csv-path", required=True, help="Input CSV with per-load/per-method/per-run results")
    parser.add_argument("--output-dir", required=True, help="Directory to store generated PNG files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv_path)
    df = add_tier_tat_columns(df)

    outputs = [
        plot_tier_tat_combined_band(df, "tier9", "tier9_tat", output_dir),
        plot_tier_tat_combined_band(df, "tier8", "tier8_tat", output_dir),
        plot_ratio_without_fcfs(df, output_dir),
    ]

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()