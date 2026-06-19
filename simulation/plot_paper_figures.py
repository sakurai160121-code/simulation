"""
Generate paper-ready figures for EURO-Par submission.

Input: trial_results.csv produced by run_random_hetero_fixed_load_web.py
  Columns: load, run_id, method, avg_tat,
           tier_low_tat, tier_mid_tat, tier_high_tat, tier9_tat,
           user0_tat … user17_tat

Output (in --output-dir):
  fig1_overall_avg_tat.pdf     Overall average TAT (log scale)
  fig2_tier_based_tat.pdf      Low/Mid/High tier TAT 3-panel (log scale)
  fig3_tier9_tat.pdf           Tier9 TAT vs load (all 4 methods, log scale)
  fig4_protection_ratio.pdf    Tier9 Protection Ratio (broken y-axis)
  fig5_pr_zoom.pdf             PR zoomed to [0,2] excl. FCFS spike
  hetero_uniform.pdf           Heterogeneous 4-panel: uniform scenario
  hetero_low_heavy.pdf         Heterogeneous 4-panel: low-heavy scenario
  hetero_high_heavy.pdf        Heterogeneous 4-panel: high-heavy scenario
  hetero_random.pdf            Heterogeneous 4-panel: random scenario
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────
METHOD_ORDER = ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]
PR_METHODS   = ["FCFS", "Owner Priority", "Preemptive"]

METHOD_STYLES: dict[str, dict] = {
    "No Sharing":     {"color": "#4d4d4d", "linestyle": "--", "marker": "o",  "linewidth": 2.0, "alpha": 1.0},
    "FCFS":           {"color": "#1f77b4", "linestyle": ":",  "marker": "s",  "linewidth": 2.0, "alpha": 0.70},
    "Owner Priority": {"color": "#ff7f0e", "linestyle": "-.", "marker": "^",  "linewidth": 2.0, "alpha": 1.0},
    "Preemptive":     {"color": "#2ca02c", "linestyle": "-",  "marker": "D",  "linewidth": 3.0, "alpha": 1.0},
}

BAR_COLORS = {
    "No Sharing":     "#4d4d4d",
    "FCFS":           "#1f77b4",
    "Owner Priority": "#ff7f0e",
    "Preemptive":     "#2ca02c",
}

# Tier group config: (user_ids, label, background shade)
TIER_GROUPS = [
    (list(range(0, 3)),  "Low\n(T1–3)",  "#fff0f0"),
    (list(range(3, 6)),  "Mid\n(T4–6)",  "#f0f0ff"),
    (list(range(6, 9)),  "High\n(T7–9)", "#f0fff0"),
    (list(range(9, 12)), "Low\n(T1–3)",  "#fff0f0"),
    (list(range(12,15)), "Mid\n(T4–6)",  "#f0f0ff"),
    (list(range(15,18)), "High\n(T7–9)", "#f0fff0"),
]

LOAD_TICKS = [round(0.1 * i, 1) for i in range(1, 11)]
TIER9_COL  = "tier9_tat"
TIER_COLS  = {
    "Low (Tier 1–3)":  "tier_low_tat",
    "Mid (Tier 4–6)":  "tier_mid_tat",
    "High (Tier 7–9)": "tier_high_tat",
}

# ── RC params for LNCS-like font sizes ────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize":7,
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    "figure.dpi":     300,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})


# ── Helpers ───────────────────────────────────────────────────────────
def _style(method: str) -> dict:
    return METHOD_STYLES[method].copy()


def _summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate mean/min/max of value_col across run_id, per load/method."""
    return (
        df.groupby(["load", "method"], as_index=False)[value_col]
        .agg(mean="mean", lo="min", hi="max")
        .sort_values(["load", "method"])
        .reset_index(drop=True)
    )


def _draw_band_on_ax(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    ylabel: str = "",
    log: bool = False,
    ylim: tuple | None = None,
    legend: bool = True,
) -> None:
    floor = 1e-6
    if log:
        pos = summary["mean"][summary["mean"] > 0]
        if len(pos):
            floor = float(pos.min() * 0.5)

    for m in methods:
        sub = summary[summary["method"] == m].sort_values("load")
        if sub.empty:
            continue
        x    = sub["load"].to_numpy(float)
        mean = sub["mean"].to_numpy(float)
        lo   = sub["lo"].to_numpy(float)
        hi   = sub["hi"].to_numpy(float)
        if log:
            mean = np.where(mean > 0, mean, floor)
            lo   = np.where(lo   > 0, lo,   floor)
            hi   = np.where(hi   > 0, hi,   floor)

        st    = _style(m)
        alpha = st.pop("alpha")
        ax.plot(x, mean, label=m, alpha=alpha, **st)
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12)

    ax.set_xlabel("System Load")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(LOAD_TICKS)
    ax.tick_params(axis="x", rotation=30)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="upper left")


# ── Figure 3: Tier9 TAT (helper, called before fig3_protection_ratio) ─
def plot_fig3_tier9_tat(df: pd.DataFrame, out: Path) -> Path:
    """Tier9 TAT vs system load, all 4 methods, log scale."""
    if TIER9_COL not in df.columns:
        raise ValueError(f"Column '{TIER9_COL}' not found.")
    summary = _summarize(df, TIER9_COL)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    _draw_band_on_ax(ax, summary, METHOD_ORDER,
                     ylabel="Tier9 Average TAT [s]", log=True)
    plt.tight_layout()

    path = out / "fig3_tier9_tat.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 5: PR zoomed (without FCFS spike) ─────────────────────────
def plot_fig5_pr_zoom(df: pd.DataFrame, out: Path) -> Path:
    """PR zoomed to [0, 2] showing only Owner Priority and Preemptive."""
    if TIER9_COL not in df.columns:
        raise ValueError(f"Column '{TIER9_COL}' not found.")
    _, summary = _compute_pr_summary(df)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    _draw_band_on_ax(ax, summary, ["Owner Priority", "Preemptive"],
                     ylabel="Protection Ratio (PR)", log=False,
                     ylim=(0.0, 2.0), legend=False)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="PR = 1 (ideal)")
    ax.legend(loc="upper left")
    ax.set_title("Protection Ratio (FCFS excluded for scale)")
    plt.tight_layout()

    path = out / "fig5_pr_zoom.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 1: Overall Average TAT ────────────────────────────────────
def plot_fig1(df: pd.DataFrame, out: Path) -> Path:
    summary = _summarize(df, "avg_tat")

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    _draw_band_on_ax(ax, summary, METHOD_ORDER,
                     ylabel="System Average TAT [s]", log=True)
    plt.tight_layout()

    path = out / "fig1_overall_avg_tat.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 2: Tier-based TAT (3-panel) ──────────────────────────────
def plot_fig2(df: pd.DataFrame, out: Path) -> Path:
    panel_labels = ["(a)", "(b)", "(c)"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=False)

    for ax, (tier_label, col), lbl in zip(axes, TIER_COLS.items(), panel_labels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        summary = _summarize(df, col)
        _draw_band_on_ax(ax, summary, METHOD_ORDER,
                         ylabel="Average TAT [s]", log=True,
                         legend=(ax is axes[0]))
        ax.set_title(f"{lbl} {tier_label}")

    plt.tight_layout()
    path = out / "fig2_tier_based_tat.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 3: Protection Ratio (broken y-axis) ───────────────────────
def _add_break_marks(ax_top: plt.Axes, ax_bot: plt.Axes) -> None:
    """Draw diagonal break marks between the two subplots."""
    d = 0.012
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)


def _compute_pr_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return (ratio_df, summary_df) for Tier9 PR."""
    pivot = df.pivot_table(
        index=["load", "run_id"], columns="method", values=TIER9_COL, aggfunc="first"
    )
    baseline = pivot.get("No Sharing")
    if baseline is None:
        raise ValueError("No Sharing column missing in CSV.")

    rows: list[dict] = []
    for m in PR_METHODS:
        if m not in pivot.columns:
            continue
        ratio = pivot[m] / baseline
        valid = baseline.notna() & (baseline > 0) & ratio.notna() & np.isfinite(ratio)
        for (load, run_id), r in ratio[valid].items():
            rows.append({"load": load, "run_id": run_id, "method": m, "ratio": float(r)})

    ratio_df = pd.DataFrame(rows)
    return ratio_df, _summarize(ratio_df, "ratio")


def plot_fig3(df: pd.DataFrame, out: Path) -> Path:
    if TIER9_COL not in df.columns:
        raise ValueError(f"Column '{TIER9_COL}' not found in trial CSV.")

    ratio_df, summary = _compute_pr_summary(df)

    # Decide whether broken axis is needed
    fcfs_max = summary.loc[summary["method"] == "FCFS", "hi"].max()
    others_max = summary.loc[summary["method"] != "FCFS", "hi"].max()
    CLIP = max(others_max * 1.4, 2.0)   # breakpoint just above non-FCFS lines
    USE_BREAK = (fcfs_max > CLIP * 1.6)

    if USE_BREAK:
        path = _plot_fig3_broken(summary, fcfs_max, CLIP, out)
    else:
        path = _plot_fig3_simple(summary, out)
    return path


def _plot_fig3_simple(summary: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    _draw_band_on_ax(ax, summary, PR_METHODS,
                     ylabel="Protection Ratio (PR)", log=False, ylim=(0.0, None))
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="PR = 1 (ideal)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = out / "fig4_protection_ratio.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_fig3_broken(summary: pd.DataFrame, fcfs_max: float, clip: float,
                      out: Path) -> Path:
    """
    Two-panel broken y-axis figure.
    Bottom panel: 0 → clip  (shows all three methods with PR=1 baseline)
    Top panel   : clip → fcfs_max * 1.1  (shows only FCFS spike)
    """
    top_ceil = fcfs_max * 1.12

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(5.0, 4.8),
        gridspec_kw={"height_ratios": [1.0, 2.5], "hspace": 0.06},
    )

    def _draw_lines(ax: plt.Axes, methods: list[str]) -> None:
        for m in methods:
            sub = summary[summary["method"] == m].sort_values("load")
            if sub.empty:
                continue
            x    = sub["load"].to_numpy(float)
            mean = sub["mean"].to_numpy(float)
            lo   = sub["lo"].to_numpy(float)
            hi   = sub["hi"].to_numpy(float)
            st    = _style(m)
            alpha = st.pop("alpha")
            ax.plot(x, mean, label=m, alpha=alpha, **st)
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12)

    # Bottom panel: all methods, zoomed to [0, clip]
    _draw_lines(ax_bot, PR_METHODS)
    ax_bot.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="PR = 1 (ideal)")
    ax_bot.set_ylim(0.0, clip)
    ax_bot.set_ylabel("Protection Ratio (PR)")
    ax_bot.set_xlabel("System Load")
    ax_bot.set_xticks(LOAD_TICKS)
    ax_bot.tick_params(axis="x", rotation=30)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper left", fontsize=7)

    # Top panel: FCFS only, range [clip, top_ceil]
    _draw_lines(ax_top, ["FCFS"])
    ax_top.set_ylim(clip, top_ceil)
    ax_top.set_ylabel("")
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False, bottom=False)
    # y-axis ticks: a few clean values above clip
    step = _nice_step(top_ceil - clip, n_ticks=4)
    top_start = np.ceil(clip / step) * step
    ax_top.yaxis.set_major_locator(
        ticker.FixedLocator(np.arange(top_start, top_ceil, step))
    )
    ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Hide inner spines
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)

    # Diagonal break marks
    _add_break_marks(ax_top, ax_bot)

    # Shared y-label via fig.text
    fig.text(0.01, 0.55, "Protection Ratio (PR)", va="center",
             rotation="vertical", fontsize=9)
    ax_bot.set_ylabel("")  # already set via fig.text

    # Annotate peak FCFS value
    fcfs_sub = summary[summary["method"] == "FCFS"].sort_values("load")
    peak_load = fcfs_sub.loc[fcfs_sub["mean"].idxmax(), "load"]
    peak_val  = fcfs_sub["mean"].max()
    ax_top.annotate(
        f"FCFS peak\n≈{peak_val:.1f}",
        xy=(peak_load, peak_val),
        xytext=(peak_load - 0.15, peak_val * 0.88),
        fontsize=7, color=BAR_COLORS["FCFS"],
        arrowprops=dict(arrowstyle="->", color=BAR_COLORS["FCFS"], lw=0.8),
    )

    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.13, hspace=0.06)
    path = out / "fig4_protection_ratio.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _nice_step(span: float, n_ticks: int = 4) -> float:
    """Return a human-readable tick step for the given span."""
    raw = span / n_ticks
    exp = np.floor(np.log10(raw))
    frac = raw / (10 ** exp)
    for nice in [1, 2, 2.5, 5, 10]:
        if frac <= nice:
            return nice * (10 ** exp)
    return 10 ** (exp + 1)


# ── Heterogeneous PR 3-panel side-by-side ────────────────────────────
def plot_hetero_pr_3panel(imgs_dir: Path, out: Path) -> Path:
    """
    Combine Low-Heavy / High-Heavy / Random PR broken-axis PNGs into
    one 3-panel horizontal figure for compact presentation.
    """
    scenarios = [
        ("low_heavy",  "Low-Heavy\n(Low=0.7, High=0.1)"),
        ("high_heavy", "High-Heavy\n(Low=0.1, High=0.7)"),
        ("random",     "Random"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, (sc, label) in zip(axes, scenarios):
        img_path = imgs_dir / f"{sc}_pr_broken.png"
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img, aspect="auto", interpolation="lanczos")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{img_path.name}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(label, fontsize=9, pad=4)
        ax.axis("off")
    plt.tight_layout(pad=0.3, w_pad=0.5)
    path = out / "hetero_pr_3panel.png"
    fig.savefig(path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    return path


# ── Heterogeneous 4-panel composite figures ───────────────────────────
_HETERO_PANELS = [
    ("low_tier_tat.png",     "(a) Low tier TAT"),
    ("high_tier_tat.png",    "(b) High tier TAT"),
    ("mid_tier_tat.png",     "(c) Mid tier TAT"),
    ("protection_ratio.png", "(d) Tier9 Protection Ratio"),
]

def plot_hetero_combined(scenario: str, hetero_base: Path, out: Path) -> Path:
    """Composite the 4 PNG panels for one hetero scenario into a single PNG.

    Saved as PNG (not PDF) to guarantee pdflatex compatibility when the
    source panels were themselves created as raster images.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5))
    for ax, (fname, label) in zip(axes.flat, _HETERO_PANELS):
        img_path = hetero_base / scenario / fname
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img, aspect="auto", interpolation="lanczos")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{fname}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(label, fontsize=9, pad=3)
        ax.axis("off")
    plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.3)
    path = out / f"hetero_{scenario}.png"   # PNG for pdflatex compatibility
    fig.savefig(path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    return path


# ── Figure 4: Per-user TAT bar chart (UNUSED — kept for reference) ────
def plot_fig4(df: pd.DataFrame, out: Path, target_load: float = 0.5) -> Path:
    """
    Grouped horizontal bar chart: per-user average TAT for 3 sharing methods
    at a single representative system load. No Sharing shown as a dashed
    reference line per user.

    Users are sorted by tier (0-8 = Group A, 9-17 = Group B, same tiers).
    Y-axis is log scale.
    """
    user_cols = [f"user{i}_tat" for i in range(18)]
    missing = [c for c in user_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing user TAT columns: {missing}")

    # Pick the closest available load
    available_loads = sorted(df["load"].unique())
    load = min(available_loads, key=lambda x: abs(x - target_load))

    sub = df[np.isclose(df["load"], load, atol=0.05)].copy()

    # Mean over run_id per method
    methods_bar = ["FCFS", "Owner Priority", "Preemptive"]
    means: dict[str, np.ndarray] = {}
    for m in METHOD_ORDER:
        ms = sub[sub["method"] == m]
        means[m] = np.array([ms[c].mean() for c in user_cols])

    n_users = 18
    x = np.arange(n_users)
    bar_w = 0.22
    offsets = [-bar_w, 0.0, bar_w]

    fig, ax = plt.subplots(figsize=(12.0, 4.0))

    for m, dx in zip(methods_bar, offsets):
        vals = means[m]
        bars = ax.bar(x + dx, vals, width=bar_w,
                      color=BAR_COLORS[m], label=m, alpha=0.85)

    # No Sharing as reference markers (×)
    ns = means["No Sharing"]
    ax.scatter(x, ns, marker="x", color=BAR_COLORS["No Sharing"],
               zorder=5, s=40, linewidths=1.5, label="No Sharing")

    ax.set_yscale("log")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Average TAT [s]")
    ax.set_title(f"Per-user Average TAT  (system load ≈ {load:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_users)], fontsize=7)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, which="both", axis="y", alpha=0.25)

    # Tier background shading
    TIER_SHADE = [
        (range(0,  3), "#ffd6d6", "Low"),    # Tier1-3, Group A
        (range(3,  6), "#d6d6ff", "Mid"),
        (range(6,  9), "#d6ffd6", "High"),
        (range(9,  12),"#ffd6d6", "Low"),    # Tier1-3, Group B
        (range(12, 15),"#d6d6ff", "Mid"),
        (range(15, 18),"#d6ffd6", "High"),
    ]
    ymin, ymax = ax.get_ylim()
    for uid_range, shade, label in TIER_SHADE:
        lo_x = min(uid_range) - 0.5
        hi_x = max(uid_range) + 0.5
        ax.axvspan(lo_x, hi_x, color=shade, alpha=0.3, zorder=0)

    # Group separators and labels
    for sep in [2.5, 5.5, 8.5, 11.5, 14.5]:
        ax.axvline(sep, color="gray", linewidth=0.7, linestyle=":")
    ax.axvline(8.5, color="gray", linewidth=1.5, linestyle="--")  # A/B group boundary
    ax.text(4.0,  ymax * 0.6, "Group A", ha="center", fontsize=8, color="gray")
    ax.text(13.0, ymax * 0.6, "Group B", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    path = out / "fig6_user_tat_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────
def generate_all(
    trial_csv: str | Path,
    output_dir: str | Path,
    hetero_base: str | Path | None = None,
) -> list[Path]:
    df  = pd.read_csv(trial_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_fig1(df, out),
        plot_fig2(df, out),
        plot_fig3_tier9_tat(df, out),
        plot_fig3(df, out),        # fig4_protection_ratio
        plot_fig5_pr_zoom(df, out),
    ]

    if hetero_base is not None:
        hbase = Path(hetero_base)
        for scenario in ["uniform", "low_heavy", "high_heavy", "random"]:
            paths.append(plot_hetero_combined(scenario, hbase, out))

    for p in paths:
        print(f"Saved: {p}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper figures from trial_results.csv"
    )
    parser.add_argument("--csv",          required=True,
                        help="Path to trial_results.csv")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory to save PDF figures")
    parser.add_argument("--hetero-base",  default=None,
                        help="Base dir of hetero_scenarios/ (optional)")
    args = parser.parse_args()
    generate_all(args.csv, args.output_dir, hetero_base=args.hetero_base)
