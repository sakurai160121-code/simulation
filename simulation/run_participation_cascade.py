"""
Participation cascade simulation at fixed load=0.8.

Each user starts with 50% probability of initial participation (random draw).
Each iteration users decide whether to stay based on: shared_TAT <= standalone_TAT.
High-tier owners leave first under FCFS (cascade), while Preemptive
keeps them because ownership is guaranteed.

Outputs (./outputs/participation_cascade/):
  cascade_high_tier.png      High-tier participation over iterations (3 methods)
  cascade_stacked_3panel.png Stacked bar per scenario (Low/Mid/High over iterations)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

import config
from task_patterns import save_patterns, load_patterns
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from simulation_iterative_wrapper import IterativeOptimizer

plt.rcParams["font.sans-serif"] = ["Yu Gothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Parameters ────────────────────────────────────────────────────────
TARGET_LOAD     = 0.8
TRAINING_RATIO  = 0.3
INFERENCE_MEAN  = 9580.0
INFERENCE_STD   = 7000.0
TRAINING_MEAN   = 412180.0
TRAINING_STD    = 600000.0
SIMULATION_TIME = 864000   # 10 days
SEED            = 42
MAX_ITERATIONS  = 10

# ── User groups ───────────────────────────────────────────────────────
LOW_USERS  = [0, 1, 2, 9, 10, 11]
MID_USERS  = [3, 4, 5, 12, 13, 14]
HIGH_USERS = [6, 7, 8, 15, 16, 17]

# ── Output ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./outputs/participation_cascade")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────
METHOD_STYLES = {
    "FCFS":           {"color": "#1f77b4", "linestyle": ":",  "marker": "s",  "linewidth": 2.5},
    "Owner Priority": {"color": "#ff7f0e", "linestyle": "-.", "marker": "^",  "linewidth": 2.5},
    "Preemptive":     {"color": "#2ca02c", "linestyle": "-",  "marker": "D",  "linewidth": 3.0},
}
GROUP_STYLES = {
    "Low":  {"color": "#4e79a7", "hatch": "",    "edgecolor": "#2a4d6e", "label": "Low (Tier 1–3)"},
    "Mid":  {"color": "#f28e2b", "hatch": "///", "edgecolor": "#a85d10", "label": "Mid (Tier 4–6)"},
    "High": {"color": "#c0392b", "hatch": "...", "edgecolor": "#7b1a12", "label": "High (Tier 7–9)"},
}


def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    sigma2 = math.log(1.0 + (std / mean) ** 2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, math.sqrt(sigma2)


def setup_load(target_load: float, training_ratio: float) -> None:
    for task_type, mean, std in [
        ("inference", INFERENCE_MEAN, INFERENCE_STD),
        ("training",  TRAINING_MEAN,  TRAINING_STD),
    ]:
        mu, sigma = _lognormal_params(mean, std)
        config.TASK_SIZE_DISTRIBUTION[task_type]["lognormal_mean"] = mu
        config.TASK_SIZE_DISTRIBUTION[task_type]["lognormal_sigma"] = sigma
        config.EXPECTED_TASK_SIZE[task_type] = mean

    config.SIMULATION_TIME = SIMULATION_TIME

    inf_size   = config.EXPECTED_TASK_SIZE["inference"]
    train_size = config.EXPECTED_TASK_SIZE["training"]
    e_task = (1.0 - training_ratio) * inf_size + training_ratio * train_size
    total_task_size = e_task * config.NUM_USERS

    total_capacity = sum(
        config.GPU_PERFORMANCE_LEVELS[t] * len(users)
        for t, users in config.GPU_TIER_ASSIGNMENT.items()
    )
    lam = target_load * total_capacity / total_task_size
    config.ARRIVAL_RATE = lam
    config.ARRIVAL_RATES = {str(i): lam for i in range(config.NUM_USERS)}


def run_cascade() -> dict[str, dict[str, list[int]]]:
    setup_load(TARGET_LOAD, TRAINING_RATIO)
    config.RANDOM_SEED = SEED
    np.random.seed(SEED)

    scenario_cfg = {
        "training_ratio": TRAINING_RATIO,
        "inference_ratio": 1.0 - TRAINING_RATIO,
        "user_training_ratios": {str(i): TRAINING_RATIO for i in range(config.NUM_USERS)},
    }
    save_patterns(scenario_name=f"cascade_load_{TARGET_LOAD}", scenario=scenario_cfg)
    patterns = load_patterns()

    initial_participation = {i: bool(np.random.random() < 0.5) for i in range(config.NUM_USERS)}
    n_init = sum(initial_participation.values())
    print(f"Initial participation: {n_init}/{config.NUM_USERS} users (50% random draw)")

    scenarios = [
        (SimulatorWithSharing,         "FCFS"),
        (SimulatorWithOwnerPriority,   "Owner Priority"),
        (SimulatorWithOwnerPreemption, "Preemptive"),
    ]

    histories: dict[str, dict[str, list[int]]] = {}

    for sim_class, name in scenarios:
        print(f"\n{'='*60}")
        print(f" Scenario: {name}")
        print(f"{'='*60}")
        optimizer = IterativeOptimizer(task_patterns=patterns)
        optimizer.run_iterative_optimization(
            sim_class, name,
            max_iterations=MAX_ITERATIONS,
            initial_participation=copy.deepcopy(initial_participation),
        )
        ph = optimizer.participation_history
        histories[name] = {
            "low":  [sum(1 for u in LOW_USERS  if ph[i][u]) for i in range(len(ph))],
            "mid":  [sum(1 for u in MID_USERS  if ph[i][u]) for i in range(len(ph))],
            "high": [sum(1 for u in HIGH_USERS if ph[i][u]) for i in range(len(ph))],
        }
        print(f"\nFinal high-tier participants: {histories[name]['high'][-1]}/6")

    return histories


def plot_results(histories: dict[str, dict[str, list[int]]]) -> None:
    scenarios  = list(histories.keys())
    n_iters    = len(next(iter(histories.values()))["high"])
    iters      = list(range(1, n_iters + 1))

    # ── Figure 1: High-tier participation for all 3 scenarios ─────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in scenarios:
        st = METHOD_STYLES[name]
        ax.plot(iters, histories[name]["high"], label=name, **st)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("High-tier Participants (/ 6)", fontsize=12)
    ax.set_xticks(iters)
    ax.set_ylim(-0.3, 6.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title(f"High-tier Participation Cascade  (load={TARGET_LOAD})", fontsize=11)
    plt.tight_layout()
    path1 = OUTPUT_DIR / "cascade_high_tier.png"
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Figure 2: Stacked bar (Low/Mid/High) — 3 panels, monochrome-safe ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    x = np.arange(n_iters)
    for ax, name in zip(axes, scenarios):
        h = histories[name]
        low_v  = np.array(h["low"],  dtype=float)
        mid_v  = np.array(h["mid"],  dtype=float)
        high_v = np.array(h["high"], dtype=float)

        for vals, bottom, key in [
            (low_v,  np.zeros(n_iters),  "Low"),
            (mid_v,  low_v,               "Mid"),
            (high_v, low_v + mid_v,       "High"),
        ]:
            st = GROUP_STYLES[key]
            ax.bar(
                x, vals, bottom=bottom,
                label=st["label"],
                color=st["color"],
                hatch=st["hatch"],
                edgecolor=st["edgecolor"],
                linewidth=0.6,
                width=0.75,
            )

        ax.set_title(name, fontsize=17, fontweight="bold", pad=8)
        ax.set_xlabel("Iteration", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(iters, fontsize=13)
        ax.set_ylim(-0.5, 19)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, alpha=0.35, axis="y", linestyle="--")
        if ax is axes[0]:
            ax.set_ylabel("Number of Participants", fontsize=15)
            ax.legend(loc="upper right", fontsize=12, framealpha=0.9,
                      handlelength=2.0, handleheight=1.2)

    plt.tight_layout(rect=[0, 0, 1, 1])
    path2 = OUTPUT_DIR / "cascade_stacked_3panel.png"
    fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")


HISTORY_JSON = OUTPUT_DIR / "cascade_histories.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip simulation and regenerate graphs from saved JSON")
    args = parser.parse_args()

    if args.replot:
        if not HISTORY_JSON.exists():
            print(f"ERROR: {HISTORY_JSON} not found. Run without --replot first.")
            sys.exit(1)
        with open(HISTORY_JSON, encoding="utf-8") as f:
            histories = json.load(f)
        print("Loaded histories from JSON. Regenerating graphs...")
    else:
        print(f"=== Participation Cascade Simulation ===")
        print(f"Load: {TARGET_LOAD},  training_ratio: {TRAINING_RATIO}")
        print(f"Start: 50% random initial participation (seed={SEED})")
        print(f"Iterations: {MAX_ITERATIONS}\n")
        histories = run_cascade()
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(histories, f, indent=2)
        print(f"Saved histories: {HISTORY_JSON}")

    plot_results(histories)
    print("\nDone. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
