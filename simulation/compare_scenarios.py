"""
3つのシナリオ（非共有、共有、所有者優先）の比較分析
ユーザー別の待ち時間と完了率を可視化
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

def run_simulation(sim_module_name):
    """指定されたシミュレーションを実行して結果を返す"""
    from task_patterns import load_patterns
    patterns = load_patterns()
    
    if sim_module_name == "simulation_no_sharing":
        from simulation_no_sharing import Simulator
        sim = Simulator(task_patterns=patterns)
        sim.run()
        return sim
        
    elif sim_module_name == "simulation_with_sharing":
        from simulation_with_sharing import SimulatorWithSharing
        sim = SimulatorWithSharing(task_patterns=patterns)
        sim.run()
        return sim
        
    elif sim_module_name == "simulation_with_sharing_owner_priority":
        from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
        sim = SimulatorWithOwnerPriority(task_patterns=patterns)
        sim.run()
        return sim
        
    elif sim_module_name == "simulation_with_sharing_owner_preemption":
        from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
        sim = SimulatorWithOwnerPreemption(task_patterns=patterns)
        sim.run()
        return sim

def extract_user_stats(sim):
    """シミュレーション結果からユーザー別統計を抽出"""
    from config import NUM_USERS
    
    user_stats = {}
    for user_id in range(NUM_USERS):
        # 完了タスクはcompletion_timeが設定されているもの
        completed_tasks = [t for t in sim.all_tasks if t.user_id == user_id and t.completion_time is not None]
        total_tasks = sum(1 for t in sim.all_tasks if t.user_id == user_id)
        
        if len(completed_tasks) > 0:
            # 待ち時間 = 開始時刻 - 到着時刻
            wait_times = []
            for t in completed_tasks:
                if t.start_time is not None:
                    wait_time = t.start_time - t.arrival_time
                    wait_times.append(wait_time)
            
            avg_wait = np.mean(wait_times) if wait_times else 0
            completion_rate = len(completed_tasks) / total_tasks * 100 if total_tasks > 0 else 0
        else:
            avg_wait = 0
            completion_rate = 0
        
        user_stats[user_id] = {
            'avg_wait_time': avg_wait,
            'completion_rate': completion_rate,
            'completed': len(completed_tasks),
            'total': total_tasks
        }
    
    return user_stats

def plot_comparison(stats_no_sharing, stats_sharing, stats_owner_priority, stats_preemption):
    """4つのシナリオを比較するグラフを作成"""
    from config import NUM_USERS
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    users = list(range(NUM_USERS))
    x = np.arange(len(users))
    width = 0.20
    
    # 待ち時間の比較
    wait_no_sharing = [stats_no_sharing[u]['avg_wait_time'] for u in users]
    wait_sharing = [stats_sharing[u]['avg_wait_time'] for u in users]
    wait_owner_priority = [stats_owner_priority[u]['avg_wait_time'] for u in users]
    wait_preemption = [stats_preemption[u]['avg_wait_time'] for u in users]
    
    ax1.bar(x - 1.5*width, wait_no_sharing, width, label='非共有', color='#2E86AB')
    ax1.bar(x - 0.5*width, wait_sharing, width, label='共有', color='#A23B72')
    ax1.bar(x + 0.5*width, wait_owner_priority, width, label='所有者優先', color='#F18F01')
    ax1.bar(x + 1.5*width, wait_preemption, width, label='途中切断', color='#C73E1D')
    
    ax1.set_xlabel('ユーザーID', fontsize=11)
    ax1.set_ylabel('平均待ち時間', fontsize=11)
    ax1.set_title('ユーザー別平均待ち時間の比較', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{u}' for u in users], fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 完了率の比較
    comp_no_sharing = [stats_no_sharing[u]['completion_rate'] for u in users]
    comp_sharing = [stats_sharing[u]['completion_rate'] for u in users]
    comp_owner_priority = [stats_owner_priority[u]['completion_rate'] for u in users]
    comp_preemption = [stats_preemption[u]['completion_rate'] for u in users]
    
    ax2.bar(x - 1.5*width, comp_no_sharing, width, label='非共有', color='#2E86AB')
    ax2.bar(x - 0.5*width, comp_sharing, width, label='共有', color='#A23B72')
    ax2.bar(x + 0.5*width, comp_owner_priority, width, label='所有者優先', color='#F18F01')
    ax2.bar(x + 1.5*width, comp_preemption, width, label='途中切断', color='#C73E1D')
    
    ax2.set_xlabel('ユーザーID', fontsize=11)
    ax2.set_ylabel('完了率 (%)', fontsize=11)
    ax2.set_title('ユーザー別タスク完了率の比較', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{u}' for u in users], fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./scenario_comparison.png', dpi=150, bbox_inches='tight')
    print("\nグラフを保存しました: ./scenario_comparison.png")
    
    # 統計サマリーを表示
    print("\n" + "="*100)
    print("【シナリオ比較サマリー】")
    print("="*100)
    
    for scenario_name, stats in [("非共有", stats_no_sharing), 
                                  ("共有", stats_sharing), 
                                  ("所有者優先", stats_owner_priority),
                                  ("途中切断", stats_preemption)]:
        avg_wait = np.mean([stats[u]['avg_wait_time'] for u in users])
        avg_comp = np.mean([stats[u]['completion_rate'] for u in users])
        total_completed = sum(stats[u]['completed'] for u in users)
        total_tasks = sum(stats[u]['total'] for u in users)
        
        print(f"\n【{scenario_name}】")
        print(f"  平均待ち時間: {avg_wait:.2f}")
        print(f"  平均完了率: {avg_comp:.2f}%")
        print(f"  完了タスク数: {total_completed} / {total_tasks}")

if __name__ == "__main__":
    print("="*100)
    print("【シナリオ比較分析】")
    print("="*100)
    
    print("\n[1/3] 非共有シミュレーション実行中...")
    sim_no_sharing = run_simulation("simulation_no_sharing")
    stats_no_sharing = extract_user_stats(sim_no_sharing)
    print("完了")
    
    print("\n[2/3] 共有シミュレーション実行中...")
    sim_sharing = run_simulation("simulation_with_sharing")
    stats_sharing = extract_user_stats(sim_sharing)
    print("完了")
    
    print("\n[3/4] 所有者優先シミュレーション実行中...")
    sim_owner_priority = run_simulation("simulation_with_sharing_owner_priority")
    stats_owner_priority = extract_user_stats(sim_owner_priority)
    print("完了")
    
    print("\n[4/4] 途中切断モデルシミュレーション実行中...")
    sim_preemption = run_simulation("simulation_with_sharing_owner_preemption")
    stats_preemption = extract_user_stats(sim_preemption)
    print("完了")
    
    print("\n比較グラフ作成中...")
    plot_comparison(stats_no_sharing, stats_sharing, stats_owner_priority, stats_preemption)
    print("\n分析完了！")
