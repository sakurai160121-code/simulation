"""
負荷率を到着率制御で変化させ、
タスクシナリオ（inference/training比率）ごとに4方式を実行・可視化するスクリプト
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Windows環境でUnicode出力を有効化
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import config
from task_patterns import save_patterns, load_patterns
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from results import ResultAnalyzer

# 出力ディレクトリ設定
OUTPUT_ROOT_DIR = './outputs/multi_load'
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

# このスクリプト内でのシミュレーション時間（秒）
SIMULATION_TIME_OVERRIDE = 864000

# ユーザーグループ定義
LOW_PERF_USERS = [0, 1, 2, 9, 10, 11]      # tier1-3
MID_PERF_USERS = [3, 4, 5, 12, 13, 14]     # tier4-6
HIGH_PERF_USERS = [6, 7, 8, 15, 16, 17]    # tier7-9

SCENARIO_TITLES = {
    "scenario1_all_inference": "Scenario1: 100% Inference",
    "scenario2_25_training": "Scenario2: 25% Training",
    "scenario3_50_training": "Scenario3: 50% Training",
    "scenario4_75_training": "Scenario4: 75% Training",
    "scenario5_all_training": "Scenario5: 100% Training",
}


def apply_simulation_time_override():
    """負荷率評価用にシミュレーション時間を上書き"""
    config.SIMULATION_TIME = SIMULATION_TIME_OVERRIDE


def set_current_scenario(scenario_name, scenario):
    """現在のシナリオをconfigへ反映"""
    config.CURRENT_TASK_SCENARIO_NAME = scenario_name
    config.CURRENT_TASK_SCENARIO = {
        "training_ratio": float(scenario.get("training_ratio", 0.0)),
        "inference_ratio": float(scenario.get("inference_ratio", 0.0)),
    }


def compute_total_gpu_capacity():
    """ユーザー所有GPUのみの総処理能力 C = Σμ_u"""
    total_capacity = 0.0
    for tier_name, users in config.GPU_TIER_ASSIGNMENT.items():
        tier_capacity = config.GPU_PERFORMANCE_LEVELS[tier_name]
        total_capacity += tier_capacity * len(users)
    return total_capacity


def get_user_capacity(user_id):
    """ユーザーuのGPU能力 C_u（今回は各ユーザー1GPU）"""
    for tier_name, users in config.GPU_TIER_ASSIGNMENT.items():
        if user_id in users:
            return config.GPU_PERFORMANCE_LEVELS[tier_name]
    return 0.0


def update_arrival_rates_for_load(target_load, scenario):
    """
    指定負荷率ρとシナリオ比率から
    Λ = ρ * ΣC_u / E[S]_s
    λ_u = Λ * C_u / ΣC_v
    を計算してARRIVAL_RATESへ反映
    """
    total_capacity = compute_total_gpu_capacity()
    expected_size = config.get_scenario_expected_task_size(scenario)

    if total_capacity <= 0 or expected_size <= 0:
        raise ValueError("total_capacity または expected_size が不正です")

    lambda_total = float(target_load) * total_capacity / expected_size

    user_caps = {str(u): get_user_capacity(u) for u in range(config.NUM_USERS)}
    cap_sum = sum(user_caps.values())
    if cap_sum <= 0:
        raise ValueError("ユーザー容量の合計が0です")

    # ARRIVAL_RATESをin-place更新して、既存モジュールの参照を壊さない
    if not isinstance(config.ARRIVAL_RATES, dict):
        config.ARRIVAL_RATES = {}
    config.ARRIVAL_RATES.clear()
    for user_id_str, cap in user_caps.items():
        config.ARRIVAL_RATES[user_id_str] = lambda_total * cap / cap_sum

    # 後方互換のため平均到着率も更新
    config.ARRIVAL_RATE = lambda_total / float(config.NUM_USERS)

    return lambda_total


def compute_total_work(tasks, task_patterns_data):
    """タスク総実行量（TFLOPs）を合算"""
    total_work = 0.0
    sizes = task_patterns_data.get("sizes", {})
    for task in tasks:
        if task.total_work is not None:
            total_work += task.total_work
            continue
        user_sizes = sizes.get(str(task.user_id), {})
        job_size = user_sizes.get(str(task.arrival_time))
        if job_size is not None:
            total_work += job_size
    return total_work


def compute_group_avg_tat(tasks, user_group):
    """指定ユーザーグループの平均TAT（到着から完了まで）"""
    group_tasks = [t for t in tasks if t.user_id in user_group and t.completion_time is not None]
    if not group_tasks:
        return 0.0
    total_tat = sum(t.get_turnaround_time() for t in group_tasks if t.get_turnaround_time() is not None)
    return total_tat / len(group_tasks)


def count_acp_assigned_tasks(tasks):
    """ACP常駐GPUへ割り当てられたタスク数を返す。"""
    count = 0
    for task in tasks:
        assigned_gpu = getattr(task, "assigned_gpu", None)
        if assigned_gpu is None:
            continue
        if str(assigned_gpu.gpu_id).startswith("acp_"):
            count += 1
    return count


def run_simulation_at_load(load_rate, scenario_name, scenario, seed_offset, show_acp_counts=True):
    """指定シナリオ・負荷率で4方式を実行"""
    print(f"\n{'='*80}")
    print(f"[{scenario_name}] 負荷率 {load_rate:.1f} のシミュレーション開始")
    print(f"{'='*80}")

    apply_simulation_time_override()
    set_current_scenario(scenario_name, scenario)

    lambda_total = update_arrival_rates_for_load(load_rate, scenario)
    print("到着率設定:")
    print(f"  Λ (total): {lambda_total:.6f}")
    print(f"  λ_0: {config.ARRIVAL_RATES.get('0', 0.0):.6f}")
    print(f"  λ_8: {config.ARRIVAL_RATES.get('8', 0.0):.6f}")
    print(f"  λ_17: {config.ARRIVAL_RATES.get('17', 0.0):.6f}")

    # 負荷率ごとに異なるシードでパターン生成
    config.RANDOM_SEED = 42 + seed_offset
    save_patterns(scenario_name=scenario_name, scenario=scenario)
    task_patterns_data = load_patterns()

    results = {}
    group_results = {
        "low": {},
        "mid": {},
        "high": {}
    }
    user_avg_tat_results = {}
    actual_load_rate = None

    # 共有なし
    print("\n  共有なし...")
    sim = SimulatorNoSharing(task_patterns=task_patterns_data)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode="no_sharing")
    stats = analyzer.get_system_statistics()
    user_stats = analyzer.get_user_statistics()
    user_avg_tat_results["No Sharing"] = {str(item["user_id"]): float(item["avg_tat"]) for item in user_stats}
    results["No Sharing"] = stats['avg_tat']
    group_results["low"]["No Sharing"] = compute_group_avg_tat(tasks, LOW_PERF_USERS)
    group_results["mid"]["No Sharing"] = compute_group_avg_tat(tasks, MID_PERF_USERS)
    group_results["high"]["No Sharing"] = compute_group_avg_tat(tasks, HIGH_PERF_USERS)
    print(f"    -> 平均TAT: {stats['avg_tat']:.2f}秒")
    print(f"    -> 平均待ち時間: {stats['avg_waiting_time']:.2f}秒")
    print(f"    -> 平均サービス時間: {stats['avg_service_time']:.2f}秒")
    print(f"    -> 平均割り当て遅延: {stats['avg_assignment_delay']:.2f}秒")

    if actual_load_rate is None:
        total_capacity = compute_total_gpu_capacity()
        total_work = compute_total_work(tasks, task_patterns_data)
        actual_load_rate = total_work / (total_capacity * SIMULATION_TIME_OVERRIDE) if total_capacity > 0 else 0.0

    # FCFS
    print("\n  FCFS...")
    sim = SimulatorWithSharing(task_patterns=task_patterns_data)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode="with_sharing")
    stats = analyzer.get_system_statistics()
    user_stats = analyzer.get_user_statistics()
    user_avg_tat_results["FCFS"] = {str(item["user_id"]): float(item["avg_tat"]) for item in user_stats}
    results["FCFS"] = stats['avg_tat']
    group_results["low"]["FCFS"] = compute_group_avg_tat(tasks, LOW_PERF_USERS)
    group_results["mid"]["FCFS"] = compute_group_avg_tat(tasks, MID_PERF_USERS)
    group_results["high"]["FCFS"] = compute_group_avg_tat(tasks, HIGH_PERF_USERS)
    if show_acp_counts:
        print(f"    -> ACP割当タスク数: {count_acp_assigned_tasks(tasks)}")
    print(f"    -> 平均TAT: {stats['avg_tat']:.2f}秒")
    print(f"    -> 平均待ち時間: {stats['avg_waiting_time']:.2f}秒")
    print(f"    -> 平均サービス時間: {stats['avg_service_time']:.2f}秒")
    print(f"    -> 平均割り当て遅延: {stats['avg_assignment_delay']:.2f}秒")

    # 所有者優先
    print("\n  所有者優先...")
    sim = SimulatorWithOwnerPriority(task_patterns=task_patterns_data)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode="with_sharing_owner_priority")
    stats = analyzer.get_system_statistics()
    user_stats = analyzer.get_user_statistics()
    user_avg_tat_results["Owner Priority"] = {str(item["user_id"]): float(item["avg_tat"]) for item in user_stats}
    results["Owner Priority"] = stats['avg_tat']
    group_results["low"]["Owner Priority"] = compute_group_avg_tat(tasks, LOW_PERF_USERS)
    group_results["mid"]["Owner Priority"] = compute_group_avg_tat(tasks, MID_PERF_USERS)
    group_results["high"]["Owner Priority"] = compute_group_avg_tat(tasks, HIGH_PERF_USERS)
    if show_acp_counts:
        print(f"    -> ACP割当タスク数: {count_acp_assigned_tasks(tasks)}")
    print(f"    -> 平均TAT: {stats['avg_tat']:.2f}秒")
    print(f"    -> 平均待ち時間: {stats['avg_waiting_time']:.2f}秒")
    print(f"    -> 平均サービス時間: {stats['avg_service_time']:.2f}秒")
    print(f"    -> 平均割り当て遅延: {stats['avg_assignment_delay']:.2f}秒")

    # プリエンプティブ方式
    print("\n  プリエンプティブ方式...")
    sim = SimulatorWithOwnerPreemption(task_patterns=task_patterns_data)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, config.NUM_USERS, mode="with_sharing_owner_preemption")
    stats = analyzer.get_system_statistics()
    user_stats = analyzer.get_user_statistics()
    user_avg_tat_results["Preemptive"] = {str(item["user_id"]): float(item["avg_tat"]) for item in user_stats}
    results["Preemptive"] = stats['avg_tat']
    group_results["low"]["Preemptive"] = compute_group_avg_tat(tasks, LOW_PERF_USERS)
    group_results["mid"]["Preemptive"] = compute_group_avg_tat(tasks, MID_PERF_USERS)
    group_results["high"]["Preemptive"] = compute_group_avg_tat(tasks, HIGH_PERF_USERS)
    if show_acp_counts:
        print(f"    -> ACP割当タスク数: {count_acp_assigned_tasks(tasks)}")
    print(f"    -> 平均TAT: {stats['avg_tat']:.2f}秒")
    print(f"    -> 平均待ち時間: {stats['avg_waiting_time']:.2f}秒")
    print(f"    -> 平均サービス時間: {stats['avg_service_time']:.2f}秒")
    print(f"    -> 平均割り当て遅延: {stats['avg_assignment_delay']:.2f}秒")

    return results, group_results, actual_load_rate, user_avg_tat_results


def plot_scenario_results(output_dir, scenario_name, target_load_rates, scenario_results, group_scenario_results):
    """1シナリオ分のグラフを保存"""
    groups = [
        ("Overall", None, "all"),
        ("Low-Performance GPUs", "low", "low"),
        ("Mid-Performance GPUs", "mid", "mid"),
        ("High-Performance GPUs", "high", "high"),
    ]

    scenario_colors = {
        "No Sharing": "#9467bd",
        "FCFS": "#ff7f0e",
        "Owner Priority": "#1f77b4",
        "Preemptive": "#2ca02c"
    }
    scenario_markers = {
        "No Sharing": "o",
        "FCFS": "s",
        "Owner Priority": "^",
        "Preemptive": "D"
    }

    title_prefix = SCENARIO_TITLES.get(scenario_name, scenario_name)

    for group_name, group_key, file_suffix in groups:
        fig, ax = plt.subplots(figsize=(8, 6))
        scenarios_to_plot = ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]

        for scenario_label in scenarios_to_plot:
            if group_key is None:
                data = scenario_results[scenario_label]
            else:
                data = group_scenario_results[group_key][scenario_label]

            ax.plot(
                target_load_rates,
                data,
                marker=scenario_markers[scenario_label],
                label=scenario_label,
                linewidth=2.5,
                markersize=8,
                color=scenario_colors[scenario_label],
                linestyle='-'
            )

        ax.set_title(f"{title_prefix} | {group_name}", fontsize=13, fontweight='bold')
        ax.set_xlabel('System Load', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average TAT (s)', fontsize=16, fontweight='bold')
        ax.set_xticks(target_load_rates)
        ax.set_yscale('log', base=10)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'load_rate_{file_suffix}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """メイン処理"""
    start_time = datetime.now()
    print("\n" + "="*80)
    print("シナリオ別・負荷率別シミュレーション実行")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    target_load_rates = list(config.LOAD_RATES)

    try:
        for scenario_name, scenario in config.TASK_SCENARIOS.items():
            print("\n" + "="*80)
            print(f"シナリオ開始: {scenario_name} ({SCENARIO_TITLES.get(scenario_name, '')})")
            print("="*80)

            scenario_output_dir = os.path.join(OUTPUT_ROOT_DIR, scenario_name)
            os.makedirs(scenario_output_dir, exist_ok=True)

            scenario_results = {
                "No Sharing": [],
                "FCFS": [],
                "Owner Priority": [],
                "Preemptive": [],
            }
            group_scenario_results = {
                "low": {"No Sharing": [], "FCFS": [], "Owner Priority": [], "Preemptive": []},
                "mid": {"No Sharing": [], "FCFS": [], "Owner Priority": [], "Preemptive": []},
                "high": {"No Sharing": [], "FCFS": [], "Owner Priority": [], "Preemptive": []},
            }
            actual_load_rates = []

            for idx, load_rate in enumerate(target_load_rates):
                results, group_results, actual_load, user_avg_tat = run_simulation_at_load(
                    load_rate,
                    scenario_name,
                    scenario,
                    seed_offset=idx,
                )
                actual_load_rates.append(actual_load)

                for scenario_label, avg_wait in results.items():
                    scenario_results[scenario_label].append(avg_wait)

                for group_key in ["low", "mid", "high"]:
                    for scenario_label in ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]:
                        group_scenario_results[group_key][scenario_label].append(group_results[group_key][scenario_label])

            plot_scenario_results(
                scenario_output_dir,
                scenario_name,
                target_load_rates,
                scenario_results,
                group_scenario_results,
            )

            # JSON保存
            results_json = {
                "scenario_name": scenario_name,
                "scenario_title": SCENARIO_TITLES.get(scenario_name, scenario_name),
                "scenario": scenario,
                "target_load_rates": target_load_rates,
                "actual_load_rates": actual_load_rates,
                "results": scenario_results,
                "group_results": group_scenario_results,
            }
            json_path = os.path.join(scenario_output_dir, 'load_rate_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)

            # CSV保存
            import pandas as pd
            df_results = pd.DataFrame({
                "Target Load": target_load_rates,
                "Measured Load": actual_load_rates,
                "No Sharing": scenario_results["No Sharing"],
                "FCFS": scenario_results["FCFS"],
                "Owner Priority": scenario_results["Owner Priority"],
                "Preemptive": scenario_results["Preemptive"],
            })
            csv_path = os.path.join(scenario_output_dir, 'load_rate_results.csv')
            df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')

            print(f"✓ シナリオ結果を保存: {scenario_output_dir}")

        end_time = datetime.now()
        elapsed = end_time - start_time
        print("\n" + "="*80)
        print("完了")
        print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {elapsed}")
        print("="*80)
        print(f"\n結果は {OUTPUT_ROOT_DIR} 配下に保存されました\n")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
