"""
高性能ユーザー保護の機械学習分析用ユーティリティ。

条件サンプリング、4方式の実行、指標集計、データセット生成に使う。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from results import ResultAnalyzer
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from task_patterns import generate_task_arrivals, generate_task_sizes, generate_task_types


HIGH_TIER_NAMES = ["tier7", "tier8", "tier9"]
METHOD_ORDER = ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]
SHARING_METHODS = ["FCFS", "Owner Priority", "Preemptive"]
METHOD_TO_SIM = {
    "No Sharing": (SimulatorNoSharing, "no_sharing"),
    "FCFS": (SimulatorWithSharing, "with_sharing"),
    "Owner Priority": (SimulatorWithOwnerPriority, "with_sharing_owner_priority"),
    "Preemptive": (SimulatorWithOwnerPreemption, "with_sharing_owner_preemption"),
}


def get_high_tier_user_ids() -> list[int]:
    """tier7〜tier9に属するユーザーIDを返す。"""
    user_ids: list[int] = []
    for tier_name in HIGH_TIER_NAMES:
        user_ids.extend(config.GPU_TIER_ASSIGNMENT[tier_name])
    return sorted(user_ids)


def get_tier_names() -> list[str]:
    return [f"tier{i}" for i in range(1, 10)]


def apply_tier_arrival_rates(tier_arrival_rates: dict[str, float]) -> dict[str, float]:
    """tierごとの到着率を各ユーザーへ展開する。"""
    user_rates: dict[str, float] = {}
    for tier_name, user_ids in config.GPU_TIER_ASSIGNMENT.items():
        rate = float(tier_arrival_rates.get(tier_name, 0.0))
        for user_id in user_ids:
            user_rates[str(user_id)] = rate
    config.ARRIVAL_RATES = user_rates
    if user_rates:
        config.ARRIVAL_RATE = float(np.mean(list(user_rates.values())))
    return user_rates


def build_acp_summary(acp_resident_gpu_rates: list[float]) -> dict[str, float | int | list[float]]:
    """ACP常駐GPUの要約値を作る。"""
    rates = [float(rate) for rate in acp_resident_gpu_rates]
    count = len(rates)
    return {
        "acp_count": int(count),
        "acp_avg_perf": float(np.mean(rates)) if rates else 0.0,
        "acp_min_perf": float(np.min(rates)) if rates else 0.0,
        "acp_max_perf": float(np.max(rates)) if rates else 0.0,
        "acp_resident_gpu_rates": rates,
    }


def sample_condition(rng: np.random.Generator, acp_rate_candidates: list[float] | None = None) -> dict[str, Any]:
    """ランダムに1条件を生成する。"""
    tier_arrival_rates = {
        tier_name: float(rng.uniform(0.2, 1.5))
        for tier_name in get_tier_names()
    }
    acp_count = int(rng.integers(0, 9))
    candidates = acp_rate_candidates or [82.6, 110.0, 180.5, 233.0, 311.84]
    acp_resident_gpu_rates = [float(rng.choice(candidates)) for _ in range(acp_count)]
    training_ratio = float(rng.uniform(0.0, 1.0))
    inf_overhead = float(rng.uniform(0.0, 0.5))
    train_overhead = float(rng.uniform(0.0, 0.5))
    return {
        "tier_arrival_rates": tier_arrival_rates,
        "acp_resident_gpu_count": acp_count,
        "acp_resident_gpu_rates": acp_resident_gpu_rates,
        "training_ratio": training_ratio,
        "inf_overhead": inf_overhead,
        "train_overhead": train_overhead,
    }


def set_task_scenario(training_ratio: float) -> dict[str, float]:
    """training/inference比率をconfigへ反映する。"""
    training_ratio = float(np.clip(training_ratio, 0.0, 1.0))
    scenario = {
        "training_ratio": training_ratio,
        "inference_ratio": 1.0 - training_ratio,
    }
    config.CURRENT_TASK_SCENARIO_NAME = "ml_high_performance_protection"
    config.CURRENT_TASK_SCENARIO = scenario.copy()
    return scenario


def apply_overhead_factors(inf_overhead: float, train_overhead: float) -> None:
    config.INTERRUPTION_OVERHEAD_FACTOR_INFERENCE = float(inf_overhead)
    config.INTERRUPTION_OVERHEAD_FACTOR_TRAINING = float(train_overhead)


def build_task_patterns(scenario: dict[str, float]) -> dict[str, Any]:
    """現在のconfigからタスクパターンをメモリ上で生成する。"""
    task_arrivals = generate_task_arrivals()
    task_types = generate_task_types(task_arrivals, scenario)
    task_sizes = generate_task_sizes(task_arrivals, task_types)
    return {
        "arrivals": task_arrivals,
        "sizes": task_sizes,
        "types": task_types,
        "config": {
            "num_users": config.NUM_USERS,
            "arrival_rate": config.ARRIVAL_RATE,
            "arrival_rates": config.ARRIVAL_RATES,
            "simulation_time": config.SIMULATION_TIME,
            "random_seed": config.RANDOM_SEED,
            "scenario_name": config.CURRENT_TASK_SCENARIO_NAME,
            "scenario": scenario,
        },
    }


def compute_tier_avg_tat_map(user_stats: list[dict[str, Any]]) -> dict[str, float]:
    """各tierの平均TATを計算する。"""
    tier_values: dict[str, list[float]] = {tier_name: [] for tier_name in get_tier_names()}
    for row in user_stats:
        tier = str(row.get("tier", ""))
        if tier in tier_values:
            tier_values[tier].append(float(row.get("avg_tat", 0.0)))

    return {
        f"{tier_name}_avg_tat": float(np.mean(values)) if values else 0.0
        for tier_name, values in tier_values.items()
    }


def compute_high_tier_avg_tat(user_stats: list[dict[str, Any]]) -> float:
    """tier7〜tier9の平均TATを計算する。"""
    high_user_ids = set(get_high_tier_user_ids())
    values = [float(row.get("avg_tat", 0.0)) for row in user_stats if int(row.get("user_id", -1)) in high_user_ids]
    return float(np.mean(values)) if values else 0.0


def run_all_methods(task_patterns_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """4方式を実行し、統計情報を返す。"""
    outputs: dict[str, dict[str, Any]] = {}
    for method_name in METHOD_ORDER:
        sim_class, mode = METHOD_TO_SIM[method_name]
        sim = sim_class(task_patterns=task_patterns_data)
        tasks = sim.run()
        analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode=mode)
        system_stats = analyzer.get_system_statistics()
        user_stats = analyzer.get_user_statistics()
        tier_avg_tat = compute_tier_avg_tat_map(user_stats)
        high_tier_avg_tat = compute_high_tier_avg_tat(user_stats)
        outputs[method_name] = {
            "tasks": tasks,
            "system_stats": system_stats,
            "user_stats": user_stats,
            "tier_avg_tat": tier_avg_tat,
            "high_tier_avg_tat": high_tier_avg_tat,
        }
    return outputs


def condition_to_row(condition_id: int, condition: dict[str, Any], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """1条件分の学習用行を作る。"""
    row: dict[str, Any] = {
        "condition_id": int(condition_id),
        "random_seed": int(config.RANDOM_SEED),
        "simulation_time": int(config.SIMULATION_TIME),
        "training_ratio": float(condition["training_ratio"]),
        "inf_overhead": float(condition["inf_overhead"]),
        "train_overhead": float(condition["train_overhead"]),
        **{f"{tier_name}_rate": float(rate) for tier_name, rate in condition["tier_arrival_rates"].items()},
    }

    acp_summary = build_acp_summary(condition["acp_resident_gpu_rates"])
    row.update(
        {
            "acp_count": int(acp_summary["acp_count"]),
            "acp_avg_perf": float(acp_summary["acp_avg_perf"]),
            "acp_min_perf": float(acp_summary["acp_min_perf"]),
            "acp_max_perf": float(acp_summary["acp_max_perf"]),
        }
    )

    no_sharing_high_tier = outputs["No Sharing"]["high_tier_avg_tat"]
    no_sharing_high_tier = float(no_sharing_high_tier) if no_sharing_high_tier > 0 else np.nan

    for method_name in METHOD_ORDER:
        system_stats = outputs[method_name]["system_stats"]
        tier_avg_tat = outputs[method_name]["tier_avg_tat"]
        high_tier_avg_tat = float(outputs[method_name]["high_tier_avg_tat"])
        row.update(
            {
                f"{method_name.lower().replace(' ', '_')}_avg_tat": float(system_stats.get("avg_tat", 0.0)),
                f"{method_name.lower().replace(' ', '_')}_avg_waiting_time": float(system_stats.get("avg_waiting_time", 0.0)),
                f"{method_name.lower().replace(' ', '_')}_avg_service_time": float(system_stats.get("avg_service_time", 0.0)),
                f"{method_name.lower().replace(' ', '_')}_high_tier_avg_tat": high_tier_avg_tat,
            }
        )
        for tier_name in get_tier_names():
            row[f"{method_name.lower().replace(' ', '_')}_{tier_name}_avg_tat"] = float(tier_avg_tat[f"{tier_name}_avg_tat"])

        if np.isnan(no_sharing_high_tier) or no_sharing_high_tier <= 0:
            ratio = np.nan
        else:
            ratio = high_tier_avg_tat / no_sharing_high_tier
        row[f"{method_name.lower().replace(' ', '_')}_high_tier_tat_ratio"] = float(ratio)

    share_rows = {
        method: row[f"{method.lower().replace(' ', '_')}_high_tier_tat_ratio"]
        for method in SHARING_METHODS
    }
    finite_share_rows = {method: value for method, value in share_rows.items() if np.isfinite(value)}
    if finite_share_rows:
        best_method = min(finite_share_rows, key=finite_share_rows.get)
        best_ratio = float(finite_share_rows[best_method])
    else:
        best_method = "Unknown"
        best_ratio = np.nan

    row["best_method_for_high_tier"] = best_method
    row["best_method_for_high_tier_ratio"] = best_ratio
    return row


def create_dataset_row(condition_id: int, condition: dict[str, Any], seed: int) -> dict[str, Any]:
    """1条件を実行して学習用1行を返す。"""
    config.RANDOM_SEED = int(seed)
    apply_tier_arrival_rates(condition["tier_arrival_rates"])
    scenario = set_task_scenario(condition["training_ratio"])
    apply_overhead_factors(condition["inf_overhead"], condition["train_overhead"])
    config.set_acp_resident_gpu_profiles(condition["acp_resident_gpu_count"], condition["acp_resident_gpu_rates"])

    task_patterns_data = build_task_patterns(scenario)
    outputs = run_all_methods(task_patterns_data)
    return condition_to_row(condition_id, condition, outputs)


def build_dataset(n_samples: int, seed: int, simulation_time: int, acp_rate_candidates: list[float] | None = None) -> pd.DataFrame:
    """条件をランダム生成してデータセットを作る。"""
    config.SIMULATION_TIME = int(simulation_time)
    base_rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for condition_id in range(int(n_samples)):
        condition_seed = int(base_rng.integers(0, 2**31 - 1))
        condition_rng = np.random.default_rng(condition_seed)
        condition = sample_condition(condition_rng, acp_rate_candidates=acp_rate_candidates)
        rows.append(create_dataset_row(condition_id, condition, seed=condition_seed))
    return pd.DataFrame(rows)


def dataset_feature_columns() -> list[str]:
    """学習に使う特徴量列を返す。"""
    return [
        *(f"tier{i}_rate" for i in range(1, 10)),
        "acp_count",
        "acp_avg_perf",
        "training_ratio",
        "inf_overhead",
        "train_overhead",
    ]


def best_method_label_order() -> list[str]:
    return ["FCFS", "Owner Priority", "Preemptive"]
