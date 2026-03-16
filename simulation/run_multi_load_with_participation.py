"""
負荷率を変化させて反復最適化シミュレーションを実行し、
負荷率と参加者数の関係をグラフ化するスクリプト
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import copy

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import task_patterns
from task_patterns import save_patterns, load_patterns
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from simulation_iterative_wrapper import IterativeOptimizer
from results import ResultAnalyzer
import config
from config import NUM_USERS, GPU_PERFORMANCE_LEVELS, GPU_TIER_ASSIGNMENT, TASK_SIZE_MEANS, EPOCHS

# 出力ディレクトリ設定
OUTPUT_DIR = './outputs/multi_load'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# このスクリプト内でのシミュレーション時間（秒）
SIMULATION_TIME_OVERRIDE = 86400

# ユーザーグループ定義
LOW_PERF_USERS = [0, 1, 2, 9, 10, 11]      # tier1-3
MID_PERF_USERS = [3, 4, 5, 12, 13, 14]     # tier4-6
HIGH_PERF_USERS = [6, 7, 8, 15, 16, 17]    # tier7-9

FIXED_ARRIVAL_RATE = 0.005

def apply_simulation_time_override():
    """負荷率グラフ用にシミュレーション時間を上書き"""
    config.SIMULATION_TIME = SIMULATION_TIME_OVERRIDE
    task_patterns.SIMULATION_TIME = SIMULATION_TIME_OVERRIDE

def update_arrival_rate_fixed():
    """到着率を固定（負荷率はバッチサイズで調整）"""
    config.ARRIVAL_RATE = FIXED_ARRIVAL_RATE
    config.ARRIVAL_RATES = {str(i): FIXED_ARRIVAL_RATE for i in range(18)}

def compute_batch_size_for_load(target_load):
    """負荷率0.1のバッチサイズを基準にスケールして算出"""
    total_capacity = compute_total_gpu_capacity()
    mean_task_size_sum = sum(TASK_SIZE_MEANS.values())
    mean_epochs = sum(EPOCHS.values()) / len(EPOCHS)
    base_batch = 0.1 * total_capacity / (FIXED_ARRIVAL_RATE * mean_task_size_sum * mean_epochs)
    scale = target_load / 0.1
    batch_size = base_batch * scale
    return max(1, int(round(batch_size)))

def update_batch_sizes_for_load(target_load):
    """負荷率に合わせて全ユーザーのバッチサイズを更新"""
    batch_size = compute_batch_size_for_load(target_load)
    for i in range(18):
        config.BATCH_SIZES[i] = batch_size
    return batch_size

def compute_total_gpu_capacity():
    """全GPUの総処理能力（TFLOPS）を合算"""
    total_capacity = 0.0
    for tier_name, users in GPU_TIER_ASSIGNMENT.items():
        tier_capacity = GPU_PERFORMANCE_LEVELS[tier_name]
        total_capacity += tier_capacity * len(users)
    return total_capacity

def compute_total_work_from_patterns(patterns):
    """タスクパターンから総実行量（TFLOPS）を合算"""
    total_work = 0.0
    sizes = patterns.get("sizes", {})
    for user_id, user_sizes in sizes.items():
        for _, job_size in user_sizes.items():
            total_work += job_size
    return total_work

def run_scenario_with_participation(scenario_class, participation_status, suppress_output=True):
    """参加者を限定して1シナリオを実行"""
    apply_simulation_time_override()
    
    patterns = load_patterns()
    
    # 参加ユーザーのGPUのみを共有プールに
    participating_users = [uid for uid, v in participation_status.items() if v]
    
    if suppress_output:
        from io import StringIO
        import contextlib
        
        f = StringIO()
        with contextlib.redirect_stdout(f):
            sim = scenario_class(task_patterns=patterns, participating_users=participating_users)
            all_tasks_shared = sim.run()
    else:
        sim = scenario_class(task_patterns=patterns, participating_users=participating_users)
        all_tasks_shared = sim.run()
    
    # 非参加ユーザーのシナリオ（共有なしは全ユーザーで実行してからフィルタ）
    non_participating_users = [uid for uid, v in participation_status.items() if not v]
    if suppress_output:
        f = StringIO()
        with contextlib.redirect_stdout(f):
            sim_no_sharing = SimulatorNoSharing(task_patterns=patterns)
            all_tasks_no_sharing_all = sim_no_sharing.run()
    else:
        sim_no_sharing = SimulatorNoSharing(task_patterns=patterns)
        all_tasks_no_sharing_all = sim_no_sharing.run()

    all_tasks_no_sharing = [t for t in all_tasks_no_sharing_all if t.user_id in non_participating_users]
    
    all_tasks = all_tasks_shared + all_tasks_no_sharing
    
    return analyze_results(all_tasks, participation_status)

def analyze_results(all_tasks, participation_status):
    """結果分析"""
    results = {}
    group_results = {"low": {}, "mid": {}, "high": {}}
    
    for user_id in range(NUM_USERS):
        user_tasks = [t for t in all_tasks if t.user_id == user_id and t.completion_time is not None]
        if user_tasks:
            avg_wait = sum(t.get_waiting_time() for t in user_tasks) / len(user_tasks)
            results[user_id] = avg_wait
        else:
            results[user_id] = 0.0
    
    # グループ別計算
    for group_name, user_group in [("low", LOW_PERF_USERS), ("mid", MID_PERF_USERS), ("high", HIGH_PERF_USERS)]:
        group_tasks = [t for t in all_tasks if t.user_id in user_group and t.completion_time is not None]
        if group_tasks:
            avg_wait = sum(t.get_waiting_time() for t in group_tasks) / len(group_tasks)
            group_results[group_name]["共有"] = avg_wait
        else:
            group_results[group_name]["共有"] = 0.0
    
    return results, group_results

def decide_participation(scenario_class, current_participation_status, target_user_id, current_stats=None):
    """参加/不参加の意思決定"""
    if current_participation_status.get(target_user_id, False):
        # 参加中：脱退判定
        test_participation = copy.deepcopy(current_participation_status)
        test_participation[target_user_id] = False
        
        _, group_results_without = run_scenario_with_participation(scenario_class, test_participation, suppress_output=True)
        
        shared_wait = current_stats[target_user_id] if current_stats and target_user_id in current_stats else 0
        standalone_wait = group_results_without.get("default", {}).get("共有", 0)
        
        threshold = standalone_wait * 1.05
        if shared_wait > threshold:
            return False
        return True
    else:
        # 不参加中：参加判定
        test_participation = copy.deepcopy(current_participation_status)
        test_participation[target_user_id] = True
        
        _, group_results_with = run_scenario_with_participation(scenario_class, test_participation, suppress_output=True)
        
        shared_wait = group_results_with.get("default", {}).get("共有", 0)
        standalone_wait = current_stats[target_user_id] if current_stats and target_user_id in current_stats else 0
        
        threshold = standalone_wait * 1.05
        if shared_wait <= threshold:
            return True
        return False

def run_iterative_optimization_at_load(scenario_class, scenario_name, load_rate, seed_offset, max_iterations=10):
    """指定された負荷率で反復最適化を実行"""
    
    apply_simulation_time_override()
    update_arrival_rate_fixed()
    batch_size = update_batch_sizes_for_load(load_rate)
    
    # タスクパターン生成
    config.RANDOM_SEED = 42 + seed_offset
    np.random.seed(42 + seed_offset)
    save_patterns()

    # 実測負荷率を算出して表示
    patterns = load_patterns()
    total_capacity = compute_total_gpu_capacity()
    total_work = compute_total_work_from_patterns(patterns)
    actual_load_rate = total_work / (total_capacity * SIMULATION_TIME_OVERRIDE)
    print(f"  バッチサイズ: {batch_size}")
    print(f"  実測負荷率: {actual_load_rate:.6f}")

    # 反復最適化ラッパーを実行
    optimizer = IterativeOptimizer()
    final_participation, _, _, _, _, _ = optimizer.run_iterative_optimization(
        scenario_class,
        scenario_name,
        max_iterations=max_iterations,
        initial_participation=None
    )

    # 最終参加者数をグループごとに計算
    low_count = sum(1 for uid in LOW_PERF_USERS if final_participation[uid])
    mid_count = sum(1 for uid in MID_PERF_USERS if final_participation[uid])
    high_count = sum(1 for uid in HIGH_PERF_USERS if final_participation[uid])
    total_count = sum(final_participation.values())
    
    return low_count, mid_count, high_count, total_count, optimizer.participation_history

def main():
    """メイン処理"""
    start_time = datetime.now()
    print("\n" + "="*80)
    print("負荷率と参加者数の関係分析")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    target_load_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 3シナリオで実行
    scenarios = [
        (SimulatorWithSharing, 'FCFS（共有・先着順）'),
        (SimulatorWithOwnerPriority, '所有者優先'),
        (SimulatorWithOwnerPreemption, 'プリエンプティブ方式')
    ]
    
    results_by_scenario = {
        scenario_name: {
            "load_rates": [],
            "low": [],
            "mid": [],
            "high": [],
            "total": []
        }
        for _, scenario_name in scenarios
    }
    
    for idx, load_rate in enumerate(target_load_rates):
        print(f"\n{'='*80}")
        print(f"負荷率 {load_rate:.1f} を実行中...")
        print(f"{'='*80}")
        
        for scenario_class, scenario_name in scenarios:
            print(f"  シナリオ: {scenario_name}")
            low, mid, high, total, history = run_iterative_optimization_at_load(
                scenario_class, scenario_name, load_rate, seed_offset=idx
            )
            results_by_scenario[scenario_name]["load_rates"].append(load_rate)
            results_by_scenario[scenario_name]["low"].append(low)
            results_by_scenario[scenario_name]["mid"].append(mid)
            results_by_scenario[scenario_name]["high"].append(high)
            results_by_scenario[scenario_name]["total"].append(total)
            print(f"    → 最終参加者数: 全体={total}/18, 低性能={low}/6, 中性能={mid}/6, 高性能={high}/6")
    
    # グラフ生成
    print("\n" + "="*80)
    print("グラフ生成中...")
    print("="*80)
    
    group_colors = {
        "low": "#1f77b4",
        "mid": "#ff7f0e",
        "high": "#2ca02c"
    }
    scenario_markers = {
        "FCFS（共有・先着順）": "s",
        "所有者優先": "^",
        "プリエンプティブ方式": "D"
    }
    
    # 各性能グループごとにシナリオ比較グラフを生成
    group_labels = {
        "low": "低性能（0,1,2,9,10,11）",
        "mid": "中性能（3,4,5,12,13,14）",
        "high": "高性能（6,7,8,15,16,17）",
    }
    for group_key in ["low", "mid", "high"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for scenario_name, results in results_by_scenario.items():
            ax.plot(
                results["load_rates"],
                results[group_key],
                marker=scenario_markers.get(scenario_name, 'o'),
                label=scenario_name,
                linewidth=2.5,
                markersize=8
            )

        ax.set_xlabel('システム負荷率', fontsize=18, fontweight='bold')
        ax.set_ylabel(f'{group_labels[group_key]}の参加者数（人）', fontsize=18, fontweight='bold')
        ax.set_xticks(target_load_rates)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.5, 6.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'participation_by_load_{group_key}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ グラフを保存: participation_by_load_{group_key}.png")
        plt.close()
    
    # 全体参加者数グラフ
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario_name, results in results_by_scenario.items():
        ax.plot(results["load_rates"], results["total"], marker=scenario_markers.get(scenario_name, 'o'), label=scenario_name,
               linewidth=2.5, markersize=8)
    
    ax.set_xlabel('システム負荷率', fontsize=18, fontweight='bold')
    ax.set_ylabel('総参加者数（人）', fontsize=18, fontweight='bold')
    ax.set_xticks(target_load_rates)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.5, 18.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'participation_by_load_all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ グラフを保存: participation_by_load_all.png")
    plt.close()
    
    # 結果をJSON保存（numpy型をPython型へ変換）
    json_safe_results = {}
    for scenario_name, results in results_by_scenario.items():
        json_safe_results[scenario_name] = {
            "load_rates": [float(x) for x in results["load_rates"]],
            "low": [int(x) for x in results["low"]],
            "mid": [int(x) for x in results["mid"]],
            "high": [int(x) for x in results["high"]],
            "total": [int(x) for x in results["total"]]
        }

    json_path = os.path.join(OUTPUT_DIR, 'participation_by_load_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 結果をJSON保存: participation_by_load_results.json")
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "="*80)
    print("完了")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"経過時間: {elapsed}")
    print("="*80)
    print(f"\n結果は {OUTPUT_DIR} に保存されました\n")

if __name__ == "__main__":
    main()
