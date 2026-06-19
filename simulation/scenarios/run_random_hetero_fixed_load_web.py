"""
Web UI向け: 均一到着率 + 負荷率スイープ評価

- 到着率は全ユーザーで同一（lambda_u = lambda_uniform）
- 全体負荷率を 0.1, 0.2, ..., 1.0 でスイープ
- 各負荷率で trial_count 回実行し平均化
- 学習/推論比率はユーザーごとに指定可能
- 出力:
  - trial_results.csv
  - summary_by_load.csv
  - load_setup.csv
  - overall_avg_tat_log10_by_load.png
    - tier9_tat_combined_band.png
    - protection_ratio_vs_load.png
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.core import config
from simulation.analysis.results import ResultAnalyzer
from simulation.engine.simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation.engine.simulation_with_sharing import SimulatorWithSharing
from simulation.engine.simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from simulation.engine.simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation.plotting.plot_paper_graphs_from_csv import (
    add_tier_tat_columns,
    plot_high_tier_tat_band,
    plot_low_tier_tat_band,
    plot_mid_tier_tat_band,
    plot_overall_avg_tat,
    plot_ratio_without_fcfs,
    plot_tier_tat_combined_band,
)
from simulation.engine.task_patterns import save_patterns

plt.rcParams["font.sans-serif"] = ["Yu Gothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHODS: list[tuple[str, type]] = [
    ("No Sharing", SimulatorNoSharing),
    ("FCFS", SimulatorWithSharing),
    ("Owner Priority", SimulatorWithOwnerPriority),
    ("Preemptive", SimulatorWithOwnerPreemption),
]
LOAD_POINTS: list[float] = [round(0.1 * i, 1) for i in range(1, 11)]
TIER9_USERS = [8, 17]


def linear_stats_to_lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """実数空間の平均・標準偏差を対数正規分布のμ,σへ変換する。"""
    m = float(mean)
    s = float(std)
    if m <= 0:
        raise ValueError("mean must be > 0")
    if s < 0:
        raise ValueError("std must be >= 0")
    if s == 0:
        return math.log(m), 0.0
    sigma2 = math.log(1.0 + (s * s) / (m * m))
    mu = math.log(m) - 0.5 * sigma2
    sigma = math.sqrt(sigma2)
    return mu, sigma


def apply_custom_task_distribution(task_type: str, mean: float, std: float) -> None:
    """タスク種別ごとの分布パラメータと期待サイズを更新する。"""
    mu, sigma = linear_stats_to_lognormal_params(mean, std)
    if task_type not in config.TASK_SIZE_DISTRIBUTION:
        raise ValueError(f"unknown task type: {task_type}")
    config.TASK_SIZE_DISTRIBUTION[task_type]["lognormal_mean"] = float(mu)
    config.TASK_SIZE_DISTRIBUTION[task_type]["lognormal_sigma"] = float(sigma)
    config.EXPECTED_TASK_SIZE[task_type] = float(mean)


def parse_float_list(value: str) -> list[float]:
    text = (value or "").strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_output_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("./outputs/random_hetero_fixed_load/custom_web", ts)
    os.makedirs(out, exist_ok=True)
    return out


def compute_total_gpu_capacity() -> float:
    total_capacity = 0.0
    for tier_name, users in config.GPU_TIER_ASSIGNMENT.items():
        total_capacity += float(config.GPU_PERFORMANCE_LEVELS[tier_name]) * len(users)
    return total_capacity


def resolve_user_training_ratios(
    global_training_ratio: float,
    user_training_ratios: list[float] | None,
) -> list[float]:
    """ユーザー別学習比率を確定する（未指定はグローバル値で埋める）。"""
    num_users = int(config.NUM_USERS)
    base = min(max(float(global_training_ratio), 0.0), 1.0)

    if not user_training_ratios:
        return [base for _ in range(num_users)]

    ratios = [base for _ in range(num_users)]
    for i, val in enumerate(user_training_ratios[:num_users]):
        ratios[i] = min(max(float(val), 0.0), 1.0)
    return ratios


def compute_uniform_arrival_rate_for_load(
    target_load: float,
    user_training_ratios: list[float],
) -> tuple[float, list[float], float, float]:
    """
    全ユーザー均一到着率を、全体負荷率から計算する。

    total_load = (lambda_uniform * sum_u E[S_u]) / total_gpu_capacity
    -> lambda_uniform = target_load * total_gpu_capacity / sum_u E[S_u]
    """
    inf_size = float(config.EXPECTED_TASK_SIZE["inference"])
    train_size = float(config.EXPECTED_TASK_SIZE["training"])

    user_expected_sizes = [
        (1.0 - r) * inf_size + r * train_size
        for r in user_training_ratios
    ]
    total_expected_task_size = float(sum(user_expected_sizes))
    total_capacity = compute_total_gpu_capacity()

    if total_expected_task_size <= 0 or total_capacity <= 0:
        return 0.0, user_expected_sizes, total_expected_task_size, total_capacity

    lambda_uniform = float(target_load) * total_capacity / total_expected_task_size
    return max(0.0, lambda_uniform), user_expected_sizes, total_expected_task_size, total_capacity


def apply_uniform_arrival_rates(lambda_uniform: float) -> None:
    config.ARRIVAL_RATE = float(lambda_uniform)
    config.ARRIVAL_RATES = {str(i): float(lambda_uniform) for i in range(config.NUM_USERS)}


def run_single_trial(
    target_load: float,
    run_id: int,
    base_seed: int,
    user_training_ratios: list[float],
    inference_mean: float,
    inference_std: float,
    training_mean: float,
    training_std: float,
    simulation_time: int,
    tier_rates: dict[str, float],
    acp_resident_gpu_count: int,
    acp_resident_gpu_rates: list[float] | None,
    inf_overhead: float,
    train_overhead: float,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    """1負荷率・1試行分を4方式で実行する。"""
    apply_custom_task_distribution("inference", inference_mean, inference_std)
    apply_custom_task_distribution("training", training_mean, training_std)

    config.SIMULATION_TIME = int(simulation_time)

    for tier_name, rate in tier_rates.items():
        config.GPU_PERFORMANCE_LEVELS[tier_name] = float(rate)

    config.set_acp_resident_gpu_profiles(acp_resident_gpu_count, acp_resident_gpu_rates or [])
    config.INTERRUPTION_OVERHEAD_FACTOR_INFERENCE = float(inf_overhead)
    config.INTERRUPTION_OVERHEAD_FACTOR_TRAINING = float(train_overhead)

    lambda_uniform, user_expected_sizes, total_expected_task_size, total_capacity = compute_uniform_arrival_rate_for_load(
        target_load=target_load,
        user_training_ratios=user_training_ratios,
    )
    apply_uniform_arrival_rates(lambda_uniform)

    scenario = {
        "training_ratio": float(np.mean(user_training_ratios)),
        "inference_ratio": 1.0 - float(np.mean(user_training_ratios)),
        "user_training_ratios": {str(i): float(r) for i, r in enumerate(user_training_ratios)},
    }

    # 負荷率ごと・試行ごとにシードを変える
    config.RANDOM_SEED = int(base_seed + round(target_load * 1000) * 100 + run_id)
    task_patterns_data = save_patterns(scenario_name=f"uniform_load_{target_load:.1f}", scenario=scenario)

    rows: list[dict[str, float | int | str]] = []

    for method_name, simulator_class in METHODS:
        simulator = simulator_class(task_patterns=task_patterns_data)
        tasks = simulator.run()
        analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode=method_name)

        system_stats = analyzer.get_system_statistics()
        user_stats = analyzer.get_user_statistics()
        user_tat_map = {int(s["user_id"]): float(s["avg_tat"]) for s in user_stats}

        LOW_TIER_USERS = [0, 1, 2, 9, 10, 11]
        MID_TIER_USERS = [3, 4, 5, 12, 13, 14]
        HIGH_TIER_USERS = [6, 7, 8, 15, 16, 17]

        def _tier_mean(users: list[int]) -> float:
            vals = [user_tat_map.get(u, 0.0) for u in users]
            pos = [v for v in vals if v > 0.0]
            return float(np.mean(pos)) if pos else 0.0

        row: dict[str, float | int | str] = {
            "load": float(target_load),
            "run_id": int(run_id),
            "method": method_name,
            "avg_tat": float(system_stats["avg_tat"]),
            "tier_low_tat": _tier_mean(LOW_TIER_USERS),
            "tier_mid_tat": _tier_mean(MID_TIER_USERS),
            "tier_high_tat": _tier_mean(HIGH_TIER_USERS),
            "tier9_tat": _tier_mean(TIER9_USERS),
            "completed_tasks": int(system_stats["completed_tasks"]),
            "total_tasks": int(system_stats["total_tasks"]),
            "lambda_uniform": float(lambda_uniform),
        }
        for uid in range(config.NUM_USERS):
            row[f"user{uid}_tat"] = float(user_tat_map.get(uid, 0.0))
        rows.append(row)

    setup_info = {
        "load": float(target_load),
        "lambda_uniform": float(lambda_uniform),
        "total_expected_task_size_sum": float(total_expected_task_size),
        "total_capacity": float(total_capacity),
        "mean_user_expected_task_size": float(np.mean(user_expected_sizes)),
    }
    return rows, setup_info


def aggregate_by_load(trial_results_df: pd.DataFrame) -> pd.DataFrame:
    base_metrics = ["avg_tat", "tier_low_tat", "tier_mid_tat", "tier_high_tat", "tier9_tat"]
    user_metrics = [f"user{uid}_tat" for uid in range(18)]
    metrics = [m for m in base_metrics + user_metrics if m in trial_results_df.columns]
    agg = (
        trial_results_df
        .groupby(["load", "method"], as_index=False)[metrics]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    # MultiIndex列をフラット化
    agg.columns = [
        "_".join([str(c) for c in col if str(c) and str(c) != ""]).strip("_")
        for col in agg.columns.to_flat_index()
    ]
    agg = agg.rename(columns={"load_": "load", "method_": "method"})
    return agg


def plot_overall_avg_tat_log10(summary_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, _ in METHODS:
        mdf = summary_df[summary_df["method"] == method_name].sort_values("load")
        ax.plot(
            mdf["load"],
            mdf["avg_tat_mean"],
            marker="o",
            linewidth=2.0,
            linestyle="--" if method_name == "No Sharing" else "-",
            label=method_name,
        )

    ax.set_xlabel("System Load")
    ax.set_ylabel("Average TAT [s] (log10)")
    ax.set_title("Average TAT vs System Load")
    ax.set_yscale("log", base=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(LOAD_POINTS)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_tier9_tat_combined_band(summary_df: pd.DataFrame, output_path: str) -> None:
    """Tier9 mean TAT with min-max bands in a two-panel figure."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    left_methods = ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]
    right_methods = ["No Sharing", "Owner Priority", "Preemptive"]

    color_map = {
        "No Sharing": "#4d4d4d",
        "FCFS": "#1f77b4",
        "Owner Priority": "#ff7f0e",
        "Preemptive": "#2ca02c",
    }
    marker_map = {
        "No Sharing": "o",
        "FCFS": "s",
        "Owner Priority": "^",
        "Preemptive": "D",
    }
    line_style_map = {
        "No Sharing": "--",
        "FCFS": ":",
        "Owner Priority": "-.",
        "Preemptive": "-",
    }

    def get_style(method_name: str) -> dict[str, object]:
        return {
            "color": color_map[method_name],
            "linestyle": line_style_map[method_name],
            "linewidth": 3.0 if method_name == "Preemptive" else 2.0,
            "alpha": 0.5 if method_name == "FCFS" else 1.0,
        }

    def prepare_series(method_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mdf = summary_df[summary_df["method"] == method_name].sort_values("load")
        load = mdf["load"].values.astype(float)
        mean = mdf["tier9_mean_tat_mean"].values.astype(float)
        min_value = mdf["tier9_mean_tat_min"].values.astype(float)
        max_value = mdf["tier9_mean_tat_max"].values.astype(float)
        return load, mean, min_value, max_value

    def positive_floor(arrays: list[np.ndarray]) -> float:
        positive_values = [array[array > 0] for array in arrays if np.any(array > 0)]
        if not positive_values:
            return 1e-6
        merged = np.concatenate(positive_values)
        return float(np.min(merged) * 0.5)

    def draw_panel(ax: plt.Axes, method_names: list[str], use_log_scale: bool) -> None:
        collected: list[np.ndarray] = []
        for method_name in method_names:
            _, mean, min_value, max_value = prepare_series(method_name)
            collected.extend([mean, min_value, max_value])

        floor = positive_floor(collected)

        for method_name in method_names:
            load, mean, min_value, max_value = prepare_series(method_name)
            if use_log_scale:
                mean = np.where(mean > 0, mean, floor)
                min_value = np.where(min_value > 0, min_value, floor)
                max_value = np.where(max_value > 0, max_value, floor)

            ax.plot(
                load,
                mean,
                marker=marker_map[method_name],
                label=method_name,
                **get_style(method_name),
            )
            ax.fill_between(
                load,
                min_value,
                max_value,
                alpha=0.12 if method_name == "FCFS" else 0.15,
                color=color_map[method_name],
            )

        ax.set_xlabel("System Load")
        ax.set_ylabel("High-performance User TAT [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(LOAD_POINTS)
        if use_log_scale:
            ax.set_yscale("log", base=10)

    draw_panel(axes[0], left_methods, use_log_scale=True)
    axes[0].set_title("(a) Overall view (log scale)")

    draw_panel(axes[1], right_methods, use_log_scale=False)
    axes[1].set_title("(b) Zoomed view without FCFS")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_protection_ratio_with_errorbars(summary_df: pd.DataFrame, output_path: str) -> None:
    """Protection Ratio (TAT_method / TAT_no_sharing) with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # No Sharing のデータを取得
    no_sharing_df = summary_df[summary_df["method"] == "No Sharing"].sort_values("load")
    no_sharing_mean = no_sharing_df["tier9_mean_tat_mean"].values
    no_sharing_min = no_sharing_df["tier9_mean_tat_min"].values
    no_sharing_max = no_sharing_df["tier9_mean_tat_max"].values
    
    for method_name, _ in METHODS:
        mdf = summary_df[summary_df["method"] == method_name].sort_values("load")
        
        y_mean = mdf["tier9_mean_tat_mean"].values
        y_min = mdf["tier9_mean_tat_min"].values
        y_max = mdf["tier9_mean_tat_max"].values
        
        # 比率計算（ゼロ除算対策）
        ratio_mean = np.where(no_sharing_mean > 0, y_mean / no_sharing_mean, 1.0)
        ratio_min = np.where(no_sharing_mean > 0, y_min / no_sharing_mean, 1.0)
        ratio_max = np.where(no_sharing_mean > 0, y_max / no_sharing_mean, 1.0)
        
        # エラーバー（非対称）
        ratio_err_lower = ratio_mean - ratio_min
        ratio_err_upper = ratio_max - ratio_mean
        
        ax.errorbar(
            mdf["load"].values,
            ratio_mean,
            yerr=[ratio_err_lower, ratio_err_upper],
            marker="o",
            linewidth=2.0,
            linestyle="--" if method_name == "No Sharing" else "-",
            capsize=4,
            label=method_name,
        )
    
    # y=1 の基準線
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline (y=1.0)")
    
    ax.set_xlabel("System Load", fontsize=12)
    ax.set_ylabel("Protection Ratio (TAT / TAT_NoSharing)", fontsize=12)
    ax.set_title("Protection Ratio vs System Load", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(LOAD_POINTS)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_uniform_load_sweep(
    trial_count: int = 10,
    training_ratio: float = 0.5,
    user_training_ratios: list[float] | None = None,
    inference_mean: float = 9580.0,
    inference_std: float = 7000.0,
    training_mean: float = 412180.0,
    training_std: float = 600000.0,
    simulation_time: int = 864000,
    seed: int = 42,
    tier_rates: dict[str, float] | None = None,
    acp_resident_gpu_count: int = 0,
    acp_resident_gpu_rates: list[float] | None = None,
    inf_overhead: float = 0.2,
    train_overhead: float = 0.2,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    output_dir = build_output_dir()

    if tier_rates is None:
        tier_rates = config.GPU_PERFORMANCE_LEVELS.copy()

    resolved_ratios = resolve_user_training_ratios(training_ratio, user_training_ratios)

    all_rows: list[dict[str, float | int | str]] = []
    setup_rows: list[dict[str, float]] = []

    for load in LOAD_POINTS:
        print(f"[Load {load:.1f}] running {trial_count} trials...")
        for run_id in range(int(trial_count)):
            trial_rows, setup_info = run_single_trial(
                target_load=load,
                run_id=run_id,
                base_seed=int(seed),
                user_training_ratios=resolved_ratios,
                inference_mean=float(inference_mean),
                inference_std=float(inference_std),
                training_mean=float(training_mean),
                training_std=float(training_std),
                simulation_time=int(simulation_time),
                tier_rates=tier_rates,
                acp_resident_gpu_count=int(acp_resident_gpu_count),
                acp_resident_gpu_rates=acp_resident_gpu_rates,
                inf_overhead=float(inf_overhead),
                train_overhead=float(train_overhead),
            )
            all_rows.extend(trial_rows)
            if run_id == 0:
                setup_rows.append(setup_info)

    trial_results_df = pd.DataFrame(all_rows)
    summary_df = aggregate_by_load(trial_results_df)
    load_setup_df = pd.DataFrame(setup_rows)

    trial_csv = os.path.join(output_dir, "trial_results.csv")
    summary_csv = os.path.join(output_dir, "summary_by_load.csv")
    setup_csv = os.path.join(output_dir, "load_setup.csv")

    trial_results_df.to_csv(trial_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    load_setup_df.to_csv(setup_csv, index=False)

    paper_df = add_tier_tat_columns(trial_results_df)
    overall_tat_path   = plot_overall_avg_tat(paper_df, output_dir)
    low_tier_path      = plot_low_tier_tat_band(paper_df, output_dir)
    mid_tier_path      = plot_mid_tier_tat_band(paper_df, output_dir)
    high_tier_path     = plot_high_tier_tat_band(paper_df, output_dir)
    tier9_path         = plot_tier_tat_combined_band(paper_df, "tier9", "tier9_tat", output_dir)
    tier8_path         = plot_tier_tat_combined_band(paper_df, "tier8", "tier8_tat", output_dir)
    ratio_without_fcfs_path = plot_ratio_without_fcfs(paper_df, output_dir)

    from plot_paper_figures import generate_all as _gen_paper_figs
    paper_out = os.path.join(output_dir, "paper_figures")
    _gen_paper_figs(trial_csv, paper_out)

    print(f"Saved: {trial_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {setup_csv}")
    print(f"Saved: {overall_tat_path}")
    print(f"Saved: {low_tier_path}")
    print(f"Saved: {mid_tier_path}")
    print(f"Saved: {high_tier_path}")
    print(f"Saved: {tier9_path}")
    print(f"Saved: {tier8_path}")
    print(f"Saved: {ratio_without_fcfs_path}")

    print(f"CUSTOM_OUTPUT_DIR={output_dir}")
    return output_dir, trial_results_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniform-arrival load sweep simulation")
    parser.add_argument("--trial_count", type=int, default=10)
    parser.add_argument("--training_ratio", type=float, default=0.5)
    parser.add_argument("--user_training_ratios", type=str, default="")
    parser.add_argument("--inference_mean", type=float, default=9580.0)
    parser.add_argument("--inference_std", type=float, default=7000.0)
    parser.add_argument("--training_mean", type=float, default=412180.0)
    parser.add_argument("--training_std", type=float, default=600000.0)
    parser.add_argument("--simulation_time", type=int, default=864000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier_rates", type=str, default="")
    parser.add_argument("--acp_resident_gpu_count", type=int, default=0)
    parser.add_argument("--acp_resident_gpu_rates", type=str, default="")
    parser.add_argument("--inf_overhead", type=float, default=0.2)
    parser.add_argument("--train_overhead", type=float, default=0.2)

    args = parser.parse_args()

    tier_rates_dict = config.GPU_PERFORMANCE_LEVELS.copy()
    if args.tier_rates:
        tier_names = list(config.GPU_PERFORMANCE_LEVELS.keys())
        tier_values = parse_float_list(args.tier_rates)
        for name, value in zip(tier_names, tier_values):
            tier_rates_dict[name] = float(value)

    user_training_ratios = parse_float_list(args.user_training_ratios)
    acp_rates = parse_float_list(args.acp_resident_gpu_rates) if args.acp_resident_gpu_rates else None

    out_dir, trial_df, summary_df = run_uniform_load_sweep(
        trial_count=args.trial_count,
        training_ratio=args.training_ratio,
        user_training_ratios=user_training_ratios,
        inference_mean=args.inference_mean,
        inference_std=args.inference_std,
        training_mean=args.training_mean,
        training_std=args.training_std,
        simulation_time=args.simulation_time,
        seed=args.seed,
        tier_rates=tier_rates_dict,
        acp_resident_gpu_count=args.acp_resident_gpu_count,
        acp_resident_gpu_rates=acp_rates,
        inf_overhead=args.inf_overhead,
        train_overhead=args.train_overhead,
    )

    print("\n=== Summary (head) ===")
    print(summary_df.head(20).to_string(index=False))
    print(f"\nOutput directory: {out_dir}")
