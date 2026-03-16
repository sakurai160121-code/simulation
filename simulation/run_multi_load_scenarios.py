"""
負荷率を変化させて複数シナリオを実行し、結果をグラフ化するスクリプト
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Windows環境でUnicode出力を有効化
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import task_patterns
from task_patterns import save_patterns, load_patterns
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
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

def apply_simulation_time_override():
    """負荷率グラフ用にシミュレーション時間を上書き"""
    config.SIMULATION_TIME = SIMULATION_TIME_OVERRIDE
    task_patterns.SIMULATION_TIME = SIMULATION_TIME_OVERRIDE

FIXED_ARRIVAL_RATE = 0.005

def update_arrival_rate_fixed():
    """到着率を固定（負荷率はバッチサイズで調整）"""
    config.ARRIVAL_RATE = FIXED_ARRIVAL_RATE
    config.ARRIVAL_RATES = {str(i): FIXED_ARRIVAL_RATE for i in range(18)}

def compute_batch_size_for_load(target_load):
    """負荷率0.1のバッチサイズを基準にスケールして算出"""
    total_capacity = compute_total_gpu_capacity()
    mean_task_size_sum = sum(TASK_SIZE_MEANS.values())
    mean_epochs = sum(EPOCHS.values()) / len(EPOCHS)
    # 期待負荷率 = (λ * Σ(task_mean) * batch * epochs) / Σ(mu)
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

def compute_total_work(tasks, task_patterns):
    """タスク総実行量（TFLOPS）を合算"""
    total_work = 0.0
    sizes = task_patterns.get("sizes", {})
    for task in tasks:
        if task.total_work is not None:
            total_work += task.total_work
            continue
        user_sizes = sizes.get(str(task.user_id), {})
        job_size = user_sizes.get(str(task.arrival_time))
        if job_size is not None:
            total_work += job_size
    return total_work

def compute_group_avg_waiting_time(tasks, user_group):
    """指定ユーザーグループの平均待ち時間を計算"""
    group_tasks = [t for t in tasks if t.user_id in user_group and t.completion_time is not None]
    if not group_tasks:
        return 0.0
    total_wait = sum(t.get_waiting_time() for t in group_tasks)
    return total_wait / len(group_tasks)

def run_simulation_at_load(load_rate, seed_offset):
    """指定された負荷率で4つのシナリオを実行"""
    
    print(f"\n{'='*80}")
    print(f"【負荷率 {load_rate:.1f}】のシミュレーション開始")
    print(f"{'='*80}")
    
    # シミュレーション時間を上書き
    apply_simulation_time_override()

    # 到着率は固定
    update_arrival_rate_fixed()
    
    # 到着率確認
    print(f"設定到着率:")
    print(f"  ARRIVAL_RATE: {config.ARRIVAL_RATE:.3f} (固定)")
    batch_size = update_batch_sizes_for_load(load_rate)
    print(f"  バッチサイズ: {batch_size}（全ユーザー共通）")
    print(f"  エポック数: 全ユーザー10に統一")
    
    # 負荷率ごとに異なるランダムシードでタスクパターンを生成
    print("タスクパターン生成中...")
    config.RANDOM_SEED = 42 + seed_offset  # 負荷率ごとに異なるシードを使用
    save_patterns()
    task_patterns = load_patterns()
    
    # デバッグ：保存されたパターンの到着率を確認
    print(f"  （確認）タスクパターンのARRIVAL_RATE: {task_patterns['config']['arrival_rate']:.3f}")
    print(f"  （確認）タスクパターンのRANDOM_SEED: {task_patterns['config']['random_seed']}")
    
    results = {}
    group_results = {
        "low": {},
        "mid": {},
        "high": {}
    }
    actual_load_rate = None
    
    # シナリオ1: 共有なし
    print("\n  共有なし...")
    sim = SimulatorNoSharing(task_patterns=task_patterns)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, NUM_USERS, mode="no_sharing")
    stats = analyzer.get_system_statistics()
    results["共有なし"] = stats['avg_waiting_time']
    group_results["low"]["共有なし"] = compute_group_avg_waiting_time(tasks, LOW_PERF_USERS)
    group_results["mid"]["共有なし"] = compute_group_avg_waiting_time(tasks, MID_PERF_USERS)
    group_results["high"]["共有なし"] = compute_group_avg_waiting_time(tasks, HIGH_PERF_USERS)
    print(f"    → 平均TAT: {stats['avg_waiting_time']:.2f}秒")
    if actual_load_rate is None:
        total_capacity = compute_total_gpu_capacity()
        total_work = compute_total_work(tasks, task_patterns)
        actual_load_rate = total_work / (total_capacity * SIMULATION_TIME_OVERRIDE)
    
    # シナリオ2: FCFS（先着順）
    print("\n  FCFS（先着順）...")
    sim = SimulatorWithSharing(task_patterns=task_patterns)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, NUM_USERS, mode="with_sharing")
    stats = analyzer.get_system_statistics()
    results["FCFS"] = stats['avg_waiting_time']
    group_results["low"]["FCFS"] = compute_group_avg_waiting_time(tasks, LOW_PERF_USERS)
    group_results["mid"]["FCFS"] = compute_group_avg_waiting_time(tasks, MID_PERF_USERS)
    group_results["high"]["FCFS"] = compute_group_avg_waiting_time(tasks, HIGH_PERF_USERS)
    print(f"    → 平均TAT: {stats['avg_waiting_time']:.2f}秒")
    
    # シナリオ3: 所有者優先
    print("\n  所有者優先...")
    sim = SimulatorWithOwnerPriority(task_patterns=task_patterns)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, NUM_USERS, mode="with_sharing_owner_priority")
    stats = analyzer.get_system_statistics()
    results["所有者優先"] = stats['avg_waiting_time']
    group_results["low"]["所有者優先"] = compute_group_avg_waiting_time(tasks, LOW_PERF_USERS)
    group_results["mid"]["所有者優先"] = compute_group_avg_waiting_time(tasks, MID_PERF_USERS)
    group_results["high"]["所有者優先"] = compute_group_avg_waiting_time(tasks, HIGH_PERF_USERS)
    print(f"    → 平均TAT: {stats['avg_waiting_time']:.2f}秒")
    
    # シナリオ4: プリエンプティブ方式
    print("\n  プリエンプティブ方式...")
    sim = SimulatorWithOwnerPreemption(task_patterns=task_patterns)
    tasks = sim.run()
    analyzer = ResultAnalyzer(tasks, NUM_USERS, mode="with_sharing_owner_preemption")
    stats = analyzer.get_system_statistics()
    results["プリエンプティブ方式"] = stats['avg_waiting_time']
    group_results["low"]["プリエンプティブ方式"] = compute_group_avg_waiting_time(tasks, LOW_PERF_USERS)
    group_results["mid"]["プリエンプティブ方式"] = compute_group_avg_waiting_time(tasks, MID_PERF_USERS)
    group_results["high"]["プリエンプティブ方式"] = compute_group_avg_waiting_time(tasks, HIGH_PERF_USERS)
    print(f"    → 平均TAT: {stats['avg_waiting_time']:.2f}秒")
    
    return results, group_results, actual_load_rate

def main():
    """メイン処理"""
    start_time = datetime.now()
    print("\n" + "="*80)
    print("負荷率別シミュレーション実行")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # テスト対象の負荷率リスト
    target_load_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    actual_load_rates = []
    # 結果を格納する辞書
    scenario_results = {
        "共有なし": [],
        "FCFS": [],
        "所有者優先": [],
        "プリエンプティブ方式": []
    }
    group_scenario_results = {
        "low": {"共有なし": [], "FCFS": [], "所有者優先": [], "プリエンプティブ方式": []},
        "mid": {"共有なし": [], "FCFS": [], "所有者優先": [], "プリエンプティブ方式": []},
        "high": {"共有なし": [], "FCFS": [], "所有者優先": [], "プリエンプティブ方式": []}
    }
    
    try:
        # 各負荷率でシミュレーション実行
        for idx, load_rate in enumerate(target_load_rates):
            results, group_results, actual_load = run_simulation_at_load(load_rate, seed_offset=idx)
            actual_load_rates.append(actual_load)
            
            # 結果を集計
            for scenario, avg_wait in results.items():
                scenario_results[scenario].append(avg_wait)
            
            # グループ別結果を集計
            for group in ["low", "mid", "high"]:
                for scenario in ["共有なし", "FCFS", "所有者優先", "プリエンプティブ方式"]:
                    group_scenario_results[group][scenario].append(group_results[group][scenario])
        
        # グラフ生成
        print("\n" + "="*80)
        print("グラフ生成中...")
        print("="*80)
        
        groups = [
            ("全体平均", None, "all"),
            ("低性能GPU", "low", "low"),
            ("中性能GPU", "mid", "mid"),
            ("高性能GPU", "high", "high")
        ]
        
        scenarios = ["FCFS", "所有者優先", "プリエンプティブ方式"]
        
        # シナリオごとの色設定
        scenario_colors = {
            "共有なし": "#9467bd",
            "FCFS": "#ff7f0e",
            "所有者優先": "#1f77b4",
            "プリエンプティブ方式": "#2ca02c"
        }
        scenario_markers = {
            "共有なし": "o",
            "FCFS": "s",
            "所有者優先": "^",
            "プリエンプティブ方式": "D"
        }
        
        # 各グループごとに個別のグラフを生成して保存
        for group_name, group_key, file_suffix in groups:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            scenarios_to_plot = ["共有なし", "FCFS", "所有者優先", "プリエンプティブ方式"]
            
            for scenario in scenarios_to_plot:
                if group_key is None:
                    # 全体平均
                    data = scenario_results[scenario]
                else:
                    # 性能グループ別
                    data = group_scenario_results[group_key][scenario]
                
                ax.plot(
                    target_load_rates,
                    data,
                    marker=scenario_markers[scenario],
                    label=scenario,
                    linewidth=2.5,
                    markersize=8,
                    color=scenario_colors[scenario],
                    linestyle='-'
                )
            
            ax.set_xlabel('システム負荷率', fontsize=18, fontweight='bold')
            ax.set_ylabel('平均TAT（秒）', fontsize=18, fontweight='bold')
            ax.set_xticks(target_load_rates)
            ax.set_yscale('log', base=10)
            ax.tick_params(labelsize=18)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f'load_rate_{file_suffix}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ グラフを保存: load_rate_{file_suffix}.png")
            plt.close()
        
        # 結果をJSON形式で保存
        results_json = {
            "target_load_rates": target_load_rates,
            "actual_load_rates": actual_load_rates,
            "results": scenario_results
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'load_rate_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"✓ 結果をJSON保存: load_rate_results.json")
        
        # CSV形式でも保存
        import pandas as pd
        df_results = pd.DataFrame({
            "負荷率（設定）": target_load_rates,
            "負荷率（実測）": actual_load_rates,
            "FCFS": scenario_results["FCFS"],
            "所有者優先": scenario_results["所有者優先"],
            "プリエンプティブ方式": scenario_results["プリエンプティブ方式"]
        })
        
        csv_path = os.path.join(OUTPUT_DIR, 'load_rate_results.csv')
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 結果をCSV保存: load_rate_results.csv")
        
        # 結果サマリー表示
        print("\n" + "="*80)
        print("【シミュレーション結果サマリー】")
        print("="*80)
        print(df_results.to_string(index=False))
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print("\n" + "="*80)
        print("完了")
        print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {elapsed}")
        print("="*80)
        print(f"\n結果は {OUTPUT_DIR} に保存されました\n")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
