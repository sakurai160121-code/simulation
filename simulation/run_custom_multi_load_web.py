"""
Web UI向けのカスタム負荷率シミュレーション実行スクリプト
推論/学習比率、タスクサイズ分布、シミュレーション時間を外部入力で受け取る。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import pandas as pd

import config
import run_multi_load_scenarios as multi


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
    out = os.path.join("./outputs/multi_load/custom_web", ts)
    os.makedirs(out, exist_ok=True)
    return out


def parse_float_list(value: str) -> list[float]:
    """カンマ区切りの数値列を float リストへ変換する。"""
    text = (value or "").strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def run_custom_scenario(
    training_ratio: float,
    inference_mean: float,
    inference_std: float,
    training_mean: float,
    training_std: float,
    simulation_time: int,
    seed: int,
    tier_rates: dict[str, float],
    acp_resident_gpu_count: int = 0,
    acp_resident_gpu_rates: list[float] | None = None,
    inf_overhead: float = 0.2,
    train_overhead: float = 0.2,
) -> str:
    # シナリオ設定
    training_ratio = float(training_ratio)
    if not 0.0 <= training_ratio <= 1.0:
        raise ValueError("training_ratio must be in [0, 1]")
    inference_ratio = 1.0 - training_ratio
    scenario_name = "custom_web_scenario"
    scenario = {
        "training_ratio": training_ratio,
        "inference_ratio": inference_ratio,
    }

    # シミュレーション時間とシナリオタイトルを上書き
    multi.SIMULATION_TIME_OVERRIDE = int(simulation_time)
    multi.SCENARIO_TITLES[scenario_name] = (
        f"Custom Scenario: {inference_ratio:.2f} Inference / {training_ratio:.2f} Training"
    )

    # タスクサイズ分布設定
    apply_custom_task_distribution("inference", inference_mean, inference_std)
    apply_custom_task_distribution("training", training_mean, training_std)

    output_dir = build_output_dir()
    target_load_rates = list(config.LOAD_RATES)
    seed_shift = int(seed) - 42

    # GPUティア性能を上書き
    for tier_name, rate in tier_rates.items():
        config.GPU_PERFORMANCE_LEVELS[tier_name] = float(rate)

    # ACP常駐GPUの台数と個別性能を上書き
    config.set_acp_resident_gpu_profiles(acp_resident_gpu_count, acp_resident_gpu_rates)

    # プリエンプト時オーバーヘッド係数を上書き
    config.INTERRUPTION_OVERHEAD_FACTOR_INFERENCE = float(inf_overhead)
    config.INTERRUPTION_OVERHEAD_FACTOR_TRAINING = float(train_overhead)

    scenario_results = {
        "No Sharing": [],
        "FCFS": [],
        "Owner Priority": [],
        "Preemptive": [],
    }
    user_avg_tat_results = {
        "No Sharing": [],
        "FCFS": [],
        "Owner Priority": [],
        "Preemptive": [],
    }
    user_avg_tat_rows: list[dict[str, float | int | str]] = []
    group_scenario_results = {
        "low": {"No Sharing": [], "FCFS": [], "Owner Priority": [], "Preemptive": []},
        "mid": {"No Sharing": [], "FCFS": [], "Owner Priority": [], "Preemptive": []},
        "high": {"No Sharing": [], "FCFS": [], "Owner Priority": [], "Preemptive": []},
    }
    actual_load_rates: list[float] = []

    for idx, load_rate in enumerate(target_load_rates):
        results, group_results, actual_load, load_user_avg_tat = multi.run_simulation_at_load(
            load_rate,
            scenario_name,
            scenario,
            seed_offset=idx + seed_shift,
            show_acp_counts=False,
        )
        actual_load_rates.append(actual_load)

        for scenario_label, avg_wait in results.items():
            scenario_results[scenario_label].append(avg_wait)
            user_avg_tat_results[scenario_label].append(
                {
                    "target_load": float(load_rate),
                    "actual_load": float(actual_load),
                    "user_avg_tat": load_user_avg_tat[scenario_label],
                }
            )
            for user_id_str, avg_tat in load_user_avg_tat[scenario_label].items():
                user_avg_tat_rows.append(
                    {
                        "target_load": float(load_rate),
                        "actual_load": float(actual_load),
                        "scenario": scenario_label,
                        "user_id": int(user_id_str),
                        "avg_tat": float(avg_tat),
                    }
                )

        for group_key in ["low", "mid", "high"]:
            for scenario_label in ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]:
                group_scenario_results[group_key][scenario_label].append(group_results[group_key][scenario_label])

    multi.plot_scenario_results(
        output_dir,
        scenario_name,
        target_load_rates,
        scenario_results,
        group_scenario_results,
    )

    results_json = {
        "scenario_name": scenario_name,
        "scenario_title": multi.SCENARIO_TITLES[scenario_name],
        "scenario": scenario,
        "simulation_time": int(simulation_time),
        "seed": int(seed),
        "task_size_params": {
            "inference": {
                "mean": float(inference_mean),
                "std": float(inference_std),
            },
            "training": {
                "mean": float(training_mean),
                "std": float(training_std),
            },
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
        "target_load_rates": target_load_rates,
        "actual_load_rates": actual_load_rates,
        "results": scenario_results,
        "group_results": group_scenario_results,
        "user_average_tat_results": user_avg_tat_results,
    }

    json_path = os.path.join(output_dir, "load_rate_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    df_results = pd.DataFrame(
        {
            "Target Load": target_load_rates,
            "Measured Load": actual_load_rates,
            "No Sharing": scenario_results["No Sharing"],
            "FCFS": scenario_results["FCFS"],
            "Owner Priority": scenario_results["Owner Priority"],
            "Preemptive": scenario_results["Preemptive"],
        }
    )
    csv_path = os.path.join(output_dir, "load_rate_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    df_user_tat = pd.DataFrame(user_avg_tat_rows)
    user_tat_csv_path = os.path.join(output_dir, "user_average_tat_results.csv")
    df_user_tat.to_csv(user_tat_csv_path, index=False, encoding="utf-8-sig")

    return os.path.abspath(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run custom multi-load simulation for web UI")
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
        acp_resident_gpu_count=args.acp_resident_gpu_count,
        acp_resident_gpu_rates=acp_rates,
        inf_overhead=args.inf_overhead,
        train_overhead=args.train_overhead,
    )
    # Streamlit側で拾うためのマーカー
    print(f"CUSTOM_OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
