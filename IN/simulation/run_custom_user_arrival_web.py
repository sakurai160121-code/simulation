"""
Web UI向けのユーザー別負荷率シミュレーション実行スクリプト
各ユーザー(0-17)の負荷率を個別に指定し、負荷率から到着率を計算して4方式を比較する。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from results import ResultAnalyzer
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from task_patterns import load_patterns, save_patterns


plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Windows環境で日本語ログが文字化けしないようUTF-8出力に固定
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


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


def build_output_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("./outputs/multi_load/custom_user_arrival_web", ts)
    os.makedirs(out, exist_ok=True)
    return out


def parse_float_list(value: str) -> list[float]:
    """カンマ区切りの数値列を float リストへ変換する。"""
    text = (value or "").strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_user_rates(value: str) -> list[float]:
    """ユーザー0-17用の到着率を読み取る。"""
    rates = parse_float_list(value)
    if len(rates) != config.NUM_USERS:
        raise ValueError(f"user-rates must have {config.NUM_USERS} values")
    if any(rate < 0 for rate in rates):
        raise ValueError("user-rates must be >= 0")
    return rates


def parse_user_load_rates(value: str) -> list[float]:
    """ユーザー0-17用の負荷率を読み取る。"""
    load_rates = parse_float_list(value)
    if len(load_rates) != config.NUM_USERS:
        raise ValueError(f"user-load-rates must have {config.NUM_USERS} values")
    if any(rate < 0 for rate in load_rates):
        raise ValueError("user-load-rates must be >= 0")
    return load_rates


def parse_user_training_ratios(value: str) -> list[float]:
    """ユーザー0-17用の学習タスク比率を読み取る。"""
    ratios = parse_float_list(value)
    if len(ratios) != config.NUM_USERS:
        raise ValueError(f"user-training-ratios must have {config.NUM_USERS} values")
    if any((ratio < 0.0 or ratio > 1.0) for ratio in ratios):
        raise ValueError("user-training-ratios must be in [0, 1]")
    return ratios


def apply_user_arrival_rates(user_rates: list[float]) -> None:
    """ユーザー別到着率を config.ARRIVAL_RATES に反映する。"""
    config.ARRIVAL_RATES = {str(user_id): float(user_rates[user_id]) for user_id in range(config.NUM_USERS)}
    config.ARRIVAL_RATE = sum(user_rates) / float(config.NUM_USERS)


def get_user_capacity(user_id: int) -> float:
    """ユーザーuのGPU能力 C_u を返す。"""
    for tier_name, users in config.GPU_TIER_ASSIGNMENT.items():
        if user_id in users:
            return float(config.GPU_PERFORMANCE_LEVELS[tier_name])
    return 0.0


def apply_user_arrival_rates(user_rates: list[float]) -> None:
    """ユーザー別到着率を config.ARRIVAL_RATES に反映する。"""
    config.ARRIVAL_RATES = {str(user_id): float(user_rates[user_id]) for user_id in range(config.NUM_USERS)}
    config.ARRIVAL_RATE = sum(user_rates) / float(config.NUM_USERS)


def run_custom_scenario(
    training_ratio: float,
    inference_mean: float,
    inference_std: float,
    training_mean: float,
    training_std: float,
    simulation_time: int,
    seed: int,
    tier_rates: dict[str, float],
    user_load_rates: list[float],
    user_training_ratios: list[float],
    acp_resident_gpu_count: int = 0,
    acp_resident_gpu_rates: list[float] | None = None,
    inf_overhead: float = 0.2,
    train_overhead: float = 0.2,
) -> str:
    training_ratio = float(training_ratio)
    if not 0.0 <= training_ratio <= 1.0:
        raise ValueError("training_ratio must be in [0, 1]")
    inference_ratio = 1.0 - training_ratio

    scenario_name = "custom_user_arrival_scenario"
    avg_user_training_ratio = float(sum(user_training_ratios) / float(config.NUM_USERS))
    scenario = {
        "training_ratio": training_ratio,
        "inference_ratio": inference_ratio,
        "user_training_ratios": {str(i): float(user_training_ratios[i]) for i in range(config.NUM_USERS)},
        "user_inference_ratios": {str(i): float(1.0 - user_training_ratios[i]) for i in range(config.NUM_USERS)},
        "avg_user_training_ratio": avg_user_training_ratio,
        "avg_user_inference_ratio": 1.0 - avg_user_training_ratio,
    }

    config.SIMULATION_TIME = int(simulation_time)
    config.CURRENT_TASK_SCENARIO_NAME = scenario_name
    config.CURRENT_TASK_SCENARIO = scenario.copy()

    apply_custom_task_distribution("inference", inference_mean, inference_std)
    apply_custom_task_distribution("training", training_mean, training_std)

    for tier_name, rate in tier_rates.items():
        config.GPU_PERFORMANCE_LEVELS[tier_name] = float(rate)

    config.set_acp_resident_gpu_profiles(acp_resident_gpu_count, acp_resident_gpu_rates)

    config.INTERRUPTION_OVERHEAD_FACTOR_INFERENCE = float(inf_overhead)
    config.INTERRUPTION_OVERHEAD_FACTOR_TRAINING = float(train_overhead)

    user_rates = user_load_rates  # 直接到着率として使用
    apply_user_arrival_rates(user_rates)

    config.RANDOM_SEED = int(seed)
    save_patterns(scenario_name=scenario_name, scenario=scenario)
    task_patterns_data = load_patterns()

    output_dir = build_output_dir()

    methods = [
        ("No Sharing", SimulatorNoSharing, "no_sharing"),
        ("FCFS", SimulatorWithSharing, "with_sharing"),
        ("Owner Priority", SimulatorWithOwnerPriority, "with_sharing_owner_priority"),
        ("Preemptive", SimulatorWithOwnerPreemption, "with_sharing_owner_preemption"),
    ]

    overall_rows: list[dict[str, float | str]] = []
    user_rows: list[dict[str, float | int | str]] = []

    for method_name, sim_class, mode in methods:
        sim = sim_class(task_patterns=task_patterns_data)
        tasks = sim.run()
        analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode=mode)
        stats = analyzer.get_system_statistics()
        user_stats = analyzer.get_user_statistics()

        overall_rows.append(
            {
                "method": method_name,
                "avg_tat": float(stats.get("avg_tat", 0.0)),
                "avg_waiting_time": float(stats.get("avg_waiting_time", 0.0)),
                "avg_service_time": float(stats.get("avg_service_time", 0.0)),
                "avg_assignment_delay": float(stats.get("avg_assignment_delay", 0.0)),
                "completed_tasks": int(stats.get("completed_tasks", 0)),
                "total_tasks": int(stats.get("total_tasks", 0)),
            }
        )

        for row in user_stats:
            user_rows.append(
                {
                    "method": method_name,
                    "user_id": int(row.get("user_id", -1)),
                    "tier": str(row.get("tier", "")),
                    "avg_tat": float(row.get("avg_tat", 0.0)),
                    "avg_waiting_time": float(row.get("avg_waiting_time", 0.0)),
                    "completion_rate_cutoff": float(row.get("completion_rate_cutoff", 0.0)),
                    "final_completion_rate": float(row.get("final_completion_rate", 0.0)),
                    "total_tasks": int(row.get("total_tasks", 0)),
                    "completed_tasks": int(row.get("completed_tasks", 0)),
                }
            )

    results_json = {
        "scenario_name": scenario_name,
        "scenario": scenario,
        "simulation_time": int(simulation_time),
        "seed": int(seed),
        "task_size_params": {
            "inference": {"mean": float(inference_mean), "std": float(inference_std)},
            "training": {"mean": float(training_mean), "std": float(training_std)},
        },
        "gpu_performance_levels": tier_rates,
        "acp_resident_gpu": {
            "count": int(config.ACP_RESIDENT_GPU_COUNT),
            "rates": [profile["processing_rate"] for profile in config.ACP_RESIDENT_GPU_PROFILES],
        },
        "interruption_overhead_factors": {
            "inference": float(inf_overhead),
            "training": float(train_overhead),
        },
        "load_rates_by_user": {str(i): float(user_load_rates[i]) for i in range(config.NUM_USERS)},
        "training_ratios_by_user": {str(i): float(user_training_ratios[i]) for i in range(config.NUM_USERS)},
        "inference_ratios_by_user": {str(i): float(1.0 - user_training_ratios[i]) for i in range(config.NUM_USERS)},
        "arrival_rates_by_user": {str(i): float(user_rates[i]) for i in range(config.NUM_USERS)},
        "overall_results": overall_rows,
    }

    json_path = os.path.join(output_dir, "user_arrival_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    overall_csv_path = os.path.join(output_dir, "user_arrival_overall_results.csv")
    pd.DataFrame(overall_rows).to_csv(overall_csv_path, index=False, encoding="utf-8-sig")

    user_df = pd.DataFrame(user_rows)
    user_long_csv_path = os.path.join(output_dir, "user_arrival_user_tat_results_long.csv")
    user_df.to_csv(user_long_csv_path, index=False, encoding="utf-8-sig")

    user_pivot_df = user_df.pivot_table(
        index="method",
        columns="user_id",
        values="avg_tat",
        aggfunc="mean",
    ).reindex(["No Sharing", "FCFS", "Owner Priority", "Preemptive"])
    user_pivot_df = user_pivot_df.reindex(sorted(user_pivot_df.columns), axis=1)
    user_pivot_df.columns = [f"user{int(col)}" for col in user_pivot_df.columns]

    user_csv_path = os.path.join(output_dir, "user_arrival_user_tat_results.csv")
    user_pivot_df.to_csv(user_csv_path, index=True, encoding="utf-8-sig")

    # ユーザー0〜8/9〜17の平均TATを、横軸=ユーザーID・縦軸=log10で可視化
    if not user_df.empty:
        scenarios = ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]
        colors = {
            "No Sharing": "#7f7f7f",
            "FCFS": "#1f77b4",
            "Owner Priority": "#ff7f0e",
            "Preemptive": "#2ca02c",
        }
        hatches = {
            "No Sharing": "///",
            "FCFS": "\\\\",
            "Owner Priority": "xx",
            "Preemptive": "..",
        }

        def render_user_range_graph(target_users: list[int], title: str, filename: str) -> None:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(target_users), dtype=float)
            width = 0.18

            for idx, scenario_name in enumerate(scenarios):
                values = []
                for user_id in target_users:
                    row = user_df[(user_df["user_id"] == user_id) & (user_df["method"] == scenario_name)]
                    v = float(row.iloc[0]["avg_tat"]) if not row.empty else float("nan")
                    # log軸で表示できるよう、0以下は最小正数へ置換
                    if not np.isnan(v) and v <= 0.0:
                        v = 1e-9
                    values.append(v)

                offset = (idx - (len(scenarios) - 1) / 2.0) * width
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    label=scenario_name,
                    color=colors[scenario_name],
                    hatch=hatches[scenario_name],
                    edgecolor="black",
                    linewidth=0.8,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([str(user_id) for user_id in target_users])
            ax.set_xlabel("User ID")
            ax.set_ylabel("Average TAT (log10 scale)")
            ax.set_title(title)
            ax.set_yscale("log", base=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(ncol=2, fontsize=9)
            plt.tight_layout()

            graph_path = os.path.join(output_dir, filename)
            plt.savefig(graph_path, dpi=300, bbox_inches="tight")
            plt.close()

        render_user_range_graph(list(range(0, 9)), "Users 0-8: Average TAT by User ID (Bar, log10)", "user_0_to_8_tat_by_scenario.png")
        render_user_range_graph(list(range(9, 18)), "Users 9-17: Average TAT by User ID (Bar, log10)", "user_9_to_17_tat_by_scenario.png")

    return os.path.abspath(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run user-specific-load simulation for web UI")
    parser.add_argument("--training-ratio", type=float, required=True)
    parser.add_argument("--inference-mean", type=float, required=True)
    parser.add_argument("--inference-std", type=float, required=True)
    parser.add_argument("--training-mean", type=float, required=True)
    parser.add_argument("--training-std", type=float, required=True)
    parser.add_argument("--simulation-time", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tier1-rate", type=float, required=True)
    parser.add_argument("--tier2-rate", type=float, required=True)
    parser.add_argument("--tier3-rate", type=float, required=True)
    parser.add_argument("--tier4-rate", type=float, required=True)
    parser.add_argument("--tier5-rate", type=float, required=True)
    parser.add_argument("--tier6-rate", type=float, required=True)
    parser.add_argument("--tier7-rate", type=float, required=True)
    parser.add_argument("--tier8-rate", type=float, required=True)
    parser.add_argument("--tier9-rate", type=float, required=True)
    parser.add_argument("--user-load-rates", type=str, required=True)
    parser.add_argument("--user-training-ratios", type=str, required=True)
    parser.add_argument("--acp-resident-gpu-count", type=int, required=True)
    parser.add_argument("--acp-resident-gpu-rates", type=str, required=True)
    parser.add_argument("--inf-overhead", type=float, required=True)
    parser.add_argument("--train-overhead", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tier_rates = {
        "tier1": args.tier1_rate,
        "tier2": args.tier2_rate,
        "tier3": args.tier3_rate,
        "tier4": args.tier4_rate,
        "tier5": args.tier5_rate,
        "tier6": args.tier6_rate,
        "tier7": args.tier7_rate,
        "tier8": args.tier8_rate,
        "tier9": args.tier9_rate,
    }
    user_load_rates = parse_user_load_rates(args.user_load_rates)
    user_training_ratios = parse_user_training_ratios(args.user_training_ratios)
    acp_rates = parse_float_list(args.acp_resident_gpu_rates)

    out = run_custom_scenario(
        training_ratio=args.training_ratio,
        inference_mean=args.inference_mean,
        inference_std=args.inference_std,
        training_mean=args.training_mean,
        training_std=args.training_std,
        simulation_time=args.simulation_time,
        seed=args.seed,
        tier_rates=tier_rates,
        user_load_rates=user_load_rates,
        user_training_ratios=user_training_ratios,
        acp_resident_gpu_count=args.acp_resident_gpu_count,
        acp_resident_gpu_rates=acp_rates,
        inf_overhead=args.inf_overhead,
        train_overhead=args.train_overhead,
    )
    print(f"CUSTOM_OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
