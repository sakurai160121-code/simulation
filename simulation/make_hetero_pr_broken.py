"""
Generate broken-axis Tier9 Protection Ratio figures for all 4 hetero scenarios.
Runs N_TRIALS per scenario and saves imgs/<scenario>_pr_broken.png.

Features:
  - Progress saved to PROGRESS_CSV after every trial (resume on crash)
  - Each trial wrapped in try/except with MAX_RETRY retries
  - Figure generated immediately after each scenario completes
  - No file I/O for task patterns (avoids OneDrive lock issues)

Usage:
    py -3 make_hetero_pr_broken.py              # run / resume
    py -3 make_hetero_pr_broken.py --replot     # regenerate figures from saved CSV only
"""
from __future__ import annotations
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
from task_patterns import generate_task_arrivals, generate_task_types, generate_task_sizes
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption

# ── Settings ──────────────────────────────────────────────────────────
LOAD_POINTS  = [round(0.1 * i, 1) for i in range(1, 11)]
N_TRIALS     = 100
SEED         = 42
SIM_TIME     = 864000             # 10 days
MAX_RETRY    = 3
TRAINING_STD = 600000.0
INFER_MEAN   = 9580.0;   INFER_STD  = 7000.0
TRAIN_MEAN   = 412180.0

LOW_USERS  = [0, 1, 2, 9, 10, 11]
MID_USERS  = [3, 4, 5, 12, 13, 14]

FIXED_SCENARIOS: dict[str, list[float]] = {
    "uniform":    [0.3] * 18,
    "low_heavy":  [0.7 if u in LOW_USERS else (0.3 if u in MID_USERS else 0.1)
                   for u in range(18)],
    "high_heavy": [0.1 if u in LOW_USERS else (0.3 if u in MID_USERS else 0.7)
                   for u in range(18)],
    "random":     None,  # re-sampled each trial
}

METHODS = [
    ("No Sharing",     SimulatorNoSharing),
    ("FCFS",           SimulatorWithSharing),
    ("Owner Priority", SimulatorWithOwnerPriority),
    ("Preemptive",     SimulatorWithOwnerPreemption),
]

METHOD_STYLES = {
    "No Sharing":     {"color": "#4d4d4d", "linestyle": "--", "marker": "o",  "linewidth": 2.0},
    "FCFS":           {"color": "#1f77b4", "linestyle": ":",  "marker": "s",  "linewidth": 2.0},
    "Owner Priority": {"color": "#ff7f0e", "linestyle": "-.", "marker": "^",  "linewidth": 2.0},
    "Preemptive":     {"color": "#2ca02c", "linestyle": "-",  "marker": "D",  "linewidth": 3.0},
}
LOAD_TICKS  = LOAD_POINTS
TIER9_USERS = [8, 17]

OUT_DIR      = Path(__file__).parent.parent / "imgs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_CSV = Path(__file__).parent / "outputs" / "hetero_pr_progress.csv"
PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────
def _lognormal_params(mean, std):
    s2 = math.log(1 + (std / mean) ** 2)
    return math.log(mean) - 0.5 * s2, math.sqrt(s2)


def setup_load(target_load: float, user_ratios: list[float]) -> None:
    for task_type, mean, std in [
        ("inference", INFER_MEAN, INFER_STD),
        ("training",  TRAIN_MEAN, TRAINING_STD),
    ]:
        mu, sigma = _lognormal_params(mean, std)
        config.TASK_SIZE_DISTRIBUTION[task_type]["lognormal_mean"]  = mu
        config.TASK_SIZE_DISTRIBUTION[task_type]["lognormal_sigma"] = sigma
        config.EXPECTED_TASK_SIZE[task_type] = mean

    config.SIMULATION_TIME = SIM_TIME

    e_tasks   = [(1 - r) * INFER_MEAN + r * TRAIN_MEAN for r in user_ratios]
    total_e   = sum(e_tasks)
    total_cap = sum(
        config.GPU_PERFORMANCE_LEVELS[t] * len(users)
        for t, users in config.GPU_TIER_ASSIGNMENT.items()
    )
    lam = target_load * total_cap / total_e
    config.ARRIVAL_RATE  = lam
    config.ARRIVAL_RATES = {str(i): lam for i in range(config.NUM_USERS)}


def _run_one_trial(scenario_name: str, load: float, trial: int,
                   fixed_ratios) -> list[dict]:
    """Run one trial for all 4 methods. Returns list of row dicts."""
    seed = SEED + trial * 100
    np.random.seed(seed)
    config.RANDOM_SEED = seed

    ratios = (fixed_ratios if fixed_ratios is not None
              else list(np.random.uniform(0, 1, 18)))
    setup_load(load, ratios)

    scenario_cfg = {
        "training_ratio":      float(np.mean(ratios)),
        "user_training_ratios": {str(i): r for i, r in enumerate(ratios)},
    }
    arrivals = generate_task_arrivals()
    types    = generate_task_types(arrivals, scenario_cfg)
    sizes    = generate_task_sizes(arrivals, types)
    patterns = {
        "arrivals": arrivals, "types": types, "sizes": sizes,
        "config": {
            "num_users":       config.NUM_USERS,
            "arrival_rate":    config.ARRIVAL_RATE,
            "arrival_rates":   config.ARRIVAL_RATES,
            "simulation_time": config.SIMULATION_TIME,
            "random_seed":     config.RANDOM_SEED,
            "scenario_name":   f"{scenario_name}_{load}_{trial}",
            "scenario":        scenario_cfg,
        }
    }

    rows = []
    for method_name, SimClass in METHODS:
        sim   = SimClass(patterns)
        tasks = sim.run()
        tats  = [t.completion_time - t.arrival_time
                 for t in tasks
                 if t.user_id in TIER9_USERS and t.completion_time is not None]
        rows.append({
            "scenario": scenario_name,
            "load":     load,
            "trial":    trial,
            "method":   method_name,
            "tier9_tat": float(np.mean(tats)) if tats else float("nan"),
        })
    return rows


def _load_progress() -> pd.DataFrame:
    if PROGRESS_CSV.exists():
        try:
            return pd.read_csv(PROGRESS_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=["scenario", "load", "trial", "method", "tier9_tat"])


def _append_progress(rows: list[dict]) -> None:
    df_new = pd.DataFrame(rows)
    if PROGRESS_CSV.exists():
        df_new.to_csv(PROGRESS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(PROGRESS_CSV, index=False)


# ── Simulation ────────────────────────────────────────────────────────
def run_scenario(scenario_name: str, done: set[tuple]) -> pd.DataFrame:
    """
    Run all methods × all loads × N_TRIALS.
    Skips (load, trial) pairs already in `done`.
    Saves progress to CSV after each trial.
    Returns full DataFrame for this scenario (from CSV).
    """
    print(f"\n=== {scenario_name} ===")
    fixed_ratios = FIXED_SCENARIOS[scenario_name]
    total = len(LOAD_POINTS) * N_TRIALS
    finished = sum(1 for (l, t) in done)

    for load in LOAD_POINTS:
        for trial in range(N_TRIALS):
            if (load, trial) in done:
                continue

            for attempt in range(1, MAX_RETRY + 1):
                try:
                    rows = _run_one_trial(scenario_name, load, trial, fixed_ratios)
                    _append_progress(rows)
                    finished += 1
                    print(f"  [{finished}/{total}] load={load} trial={trial}", end="\r")
                    break
                except Exception as e:
                    print(f"\n  WARN attempt {attempt}/{MAX_RETRY} failed "
                          f"(load={load} trial={trial}): {e}")
                    if attempt == MAX_RETRY:
                        print(f"  SKIP load={load} trial={trial} after {MAX_RETRY} retries")

    print(f"\n  {scenario_name} done.")
    prog = _load_progress()
    return prog[prog["scenario"] == scenario_name]


# ── Plot ──────────────────────────────────────────────────────────────
def _compute_pr_summary(df: pd.DataFrame) -> pd.DataFrame:
    pivot    = df.pivot_table(index=["load", "trial"], columns="method",
                              values="tier9_tat", aggfunc="first")
    baseline = pivot.get("No Sharing")
    rows = []
    for m in ["FCFS", "Owner Priority", "Preemptive"]:
        if m not in pivot.columns:
            continue
        ratio = pivot[m] / baseline
        valid = baseline.notna() & (baseline > 0) & ratio.notna() & np.isfinite(ratio)
        for (load, trial), r in ratio[valid].items():
            rows.append({"load": load, "method": m, "ratio": float(r)})
    rdf = pd.DataFrame(rows)
    return rdf.groupby(["load", "method"], as_index=False)["ratio"].agg(
        mean="mean", lo="min", hi="max")


def _add_break_marks(ax_top, ax_bot):
    d  = 0.012
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kw); ax_top.plot((1-d, 1+d), (-d, +d), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1-d, 1+d), **kw); ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kw)


def _nice_step(span, n=4):
    raw = span / n
    exp = np.floor(np.log10(max(raw, 1e-9)))
    frac = raw / (10 ** exp)
    for s in [1, 2, 2.5, 5, 10]:
        if frac <= s:
            return s * (10 ** exp)
    return 10 ** (exp + 1)


def plot_pr_broken(summary: pd.DataFrame, scenario: str) -> Path:
    PR_METHODS = ["FCFS", "Owner Priority", "Preemptive"]

    fcfs_max   = summary.loc[summary["method"] == "FCFS", "hi"].max()
    others_max = summary.loc[summary["method"] != "FCFS", "hi"].max()
    CLIP      = max(others_max * 1.5, 2.0)
    USE_BREAK = (fcfs_max > CLIP * 1.5) and not np.isnan(fcfs_max)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 300, "pdf.fonttype": 42,
    })

    def _draw(ax, methods):
        for m in methods:
            sub = summary[summary["method"] == m].sort_values("load")
            if sub.empty:
                continue
            x, mean, lo, hi = (sub[c].to_numpy(float)
                                for c in ["load", "mean", "lo", "hi"])
            st = METHOD_STYLES[m].copy()
            ax.plot(x, mean, label=m, **st)
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12)

    if USE_BREAK:
        top_ceil = fcfs_max * 1.12
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(5.0, 4.5),
            gridspec_kw={"height_ratios": [1.0, 2.5], "hspace": 0.06})

        _draw(ax_bot, PR_METHODS)
        ax_bot.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="PR = 1 (ideal)")
        ax_bot.set_ylim(0.0, CLIP)
        ax_bot.set_xlabel("System Load"); ax_bot.set_xticks(LOAD_TICKS)
        ax_bot.tick_params(axis="x", rotation=30)
        ax_bot.grid(True, alpha=0.3); ax_bot.legend(loc="upper left", fontsize=7)

        _draw(ax_top, ["FCFS"])
        ax_top.set_ylim(CLIP, top_ceil); ax_top.grid(True, alpha=0.3)
        ax_top.tick_params(labelbottom=False, bottom=False)
        step      = _nice_step(top_ceil - CLIP, 4)
        top_start = np.ceil(CLIP / step) * step
        ax_top.yaxis.set_major_locator(
            ticker.FixedLocator(np.arange(top_start, top_ceil, step)))
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        _add_break_marks(ax_top, ax_bot)

        fcfs_sub = summary[summary["method"] == "FCFS"].sort_values("load")
        idx      = fcfs_sub["mean"].idxmax()
        pl, pv   = fcfs_sub.loc[idx, "load"], fcfs_sub.loc[idx, "mean"]
        ax_top.annotate(f"FCFS peak\n≈{pv:.1f}", xy=(pl, pv),
                        xytext=(pl - 0.15, pv * 0.88), fontsize=7,
                        color=METHOD_STYLES["FCFS"]["color"],
                        arrowprops=dict(arrowstyle="->",
                                        color=METHOD_STYLES["FCFS"]["color"], lw=0.8))
        fig.text(0.02, 0.55, "Protection Ratio (PR)",
                 va="center", rotation="vertical", fontsize=9)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.13, hspace=0.06)
    else:
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        _draw(ax, PR_METHODS)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="PR = 1 (ideal)")
        ax.set_ylim(0.0, max(others_max * 1.2, 2.0))
        ax.set_xlabel("System Load"); ax.set_ylabel("Protection Ratio (PR)")
        ax.set_xticks(LOAD_TICKS); ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
        plt.tight_layout()

    out = OUT_DIR / f"{scenario}_pr_broken.png"
    fig.savefig(out, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate figures from saved CSV without running simulations")
    args = parser.parse_args()

    if args.replot:
        prog = _load_progress()
        if prog.empty:
            print("No saved progress found. Run without --replot first.")
            sys.exit(1)
        for scenario in ["uniform", "low_heavy", "high_heavy", "random"]:
            df = prog[prog["scenario"] == scenario]
            if df.empty:
                print(f"No data for {scenario}, skipping.")
                continue
            summary = _compute_pr_summary(df)
            plot_pr_broken(summary, scenario)
        print("\nAll figures regenerated from saved CSV.")
        sys.exit(0)

    prog = _load_progress()
    for scenario in ["uniform", "low_heavy", "high_heavy", "random"]:
        sc_done = set()
        if not prog.empty and "scenario" in prog.columns:
            sc_prog = prog[prog["scenario"] == scenario]
            # a (load, trial) is done when all 4 methods are present
            counts = sc_prog.groupby(["load", "trial"])["method"].count()
            sc_done = {(l, t) for (l, t), c in counts.items() if c >= 4}
        if len(sc_done) == len(LOAD_POINTS) * N_TRIALS:
            print(f"\n=== {scenario} (already complete, replotting) ===")
            df = prog[prog["scenario"] == scenario]
        else:
            df = run_scenario(scenario, sc_done)
            prog = _load_progress()  # reload after saving

        summary = _compute_pr_summary(df)
        plot_pr_broken(summary, scenario)

    print("\nAll done. Check imgs/*_pr_broken.png")
