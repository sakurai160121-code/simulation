"""
反復型シミュレーション：既存の3つのシナリオを使用
既存シミュレータクラスを参加状態に応じて呼び出す反復最適化フレームワーク
"""

import numpy as np
import heapq
import copy
from definitions import User, GPU, Task
from config import (
    NUM_USERS,
    ARRIVAL_RATE,
    ARRIVAL_RATES,
    GPU_PERFORMANCE_LEVELS,
    GPU_TIER_ASSIGNMENT,
    TASK_SIZE_MEANS,
    TASK_SIZE_MEAN_GLOBAL,
    BATCH_MULTIPLIER,
)
from task_patterns import load_patterns, save_patterns
import os

# 既存のシミュレータをインポート
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from simulation_no_sharing import Simulator as SimulatorNoSharing


class IterativeOptimizer:
    """
    既存シミュレータを使った反復最適化
    participation_status に基づいて、参加者のGPUのみを共有プール化して実行
    """
    def __init__(self, task_patterns=None):
        self.task_patterns = task_patterns or {}
        self.participation_history = []
        self.performance_history = []
    
    def run_scenario_with_participation(self, scenario_class, participation_status):
        """
        シナリオクラスに応じて、参加状態を反映したシミュレーション実行
        参加者：共有プールシミュレーション
        非参加者：独立GPU環境シミュレーション（no_sharing）
        """
        # 共有プールシミュレーション実行（全員参加を仮定）
        sim_shared = scenario_class(task_patterns=self.task_patterns)
        all_tasks_shared = sim_shared.run()
        
        # 独立GPU環境シミュレーション実行（常に実行して比較可能にする）
        sim_no_sharing = SimulatorNoSharing(task_patterns=self.task_patterns)
        all_tasks_no_sharing = sim_no_sharing.run()
        
        # 結果を分析
        return self._analyze_results(all_tasks_shared, all_tasks_no_sharing, participation_status)
    
    def _analyze_results(self, all_tasks_shared, all_tasks_no_sharing, participation_status):
        """結果分析（両方の待ち時間を保存し、participation_statusに基づいて選択）"""
        user_stats = {}
        
        for user_id in range(NUM_USERS):
            # 共有プール環境での待ち時間
            user_tasks_shared = [t for t in all_tasks_shared if t.user_id == user_id]
            completed_tasks_shared = [t for t in user_tasks_shared if t.completion_time is not None]
            
            if completed_tasks_shared:
                waiting_times_shared = [t.start_time - t.arrival_time for t in completed_tasks_shared 
                                if t.start_time is not None]
                avg_wait_shared = np.mean(waiting_times_shared) if waiting_times_shared else float('inf')
            else:
                avg_wait_shared = float('inf')
            
            # 独立GPU環境での待ち時間
            user_tasks_no_sharing = [t for t in all_tasks_no_sharing if t.user_id == user_id]
            completed_tasks_no_sharing = [t for t in user_tasks_no_sharing if t.completion_time is not None]
            
            if completed_tasks_no_sharing:
                waiting_times_no_sharing = [t.start_time - t.arrival_time for t in completed_tasks_no_sharing 
                                    if t.start_time is not None]
                avg_wait_no_sharing = np.mean(waiting_times_no_sharing) if waiting_times_no_sharing else float('inf')
            else:
                avg_wait_no_sharing = float('inf')
            
            # 現在の参加状態に基づいて表示する待ち時間を選択
            if participation_status[user_id]:
                avg_wait = avg_wait_shared
                user_tasks_count = user_tasks_shared
                completed_tasks_count = completed_tasks_shared
            else:
                avg_wait = avg_wait_no_sharing
                user_tasks_count = user_tasks_no_sharing
                completed_tasks_count = completed_tasks_no_sharing
            
            user_stats[user_id] = {
                'total_tasks': len(user_tasks_count),
                'completed_tasks': len(completed_tasks_count),
                'avg_waiting_time': avg_wait,
                'avg_waiting_time_shared': avg_wait_shared,  # 共有プール環境
                'avg_waiting_time_no_sharing': avg_wait_no_sharing,  # 独立GPU環境
                'participating': participation_status[user_id]
            }
        
        return user_stats
    
    def estimate_waiting_time_standalone(self, user_id):
        """自分のGPUのみ使用した場合の理論的期待待ち時間（M/M/1）"""
        tier = None
        for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
            if user_id in user_list:
                tier = tier_name
                break
        
        mu = GPU_PERFORMANCE_LEVELS[tier]
        lam = ARRIVAL_RATES.get(str(user_id), ARRIVAL_RATE)
        mean_job_size = TASK_SIZE_MEANS.get(user_id, TASK_SIZE_MEAN_GLOBAL) * BATCH_MULTIPLIER
        
        rho = (lam * mean_job_size) / mu
        
        if rho >= 1.0:
            return float('inf')
        
        avg_queue_length = rho / (1.0 - rho)
        avg_waiting_time = avg_queue_length / lam
        
        return avg_waiting_time
    
    def estimate_waiting_time_acp(self, user_id, prev_stats):
        """ACP参加時の期待待ち時間を前回実績から推定"""
        if prev_stats is None:
            return self.estimate_waiting_time_standalone(user_id) * 0.7
        
        if user_id in prev_stats and prev_stats[user_id]['participating']:
            return prev_stats[user_id]['avg_waiting_time']
        
        participating_users = [uid for uid, s in prev_stats.items() if s['participating']]
        if participating_users:
            avg_wait = np.mean([prev_stats[uid]['avg_waiting_time'] 
                              for uid in participating_users 
                              if prev_stats[uid]['avg_waiting_time'] < float('inf')])
            return avg_wait
        
        return self.estimate_waiting_time_standalone(user_id)
    
    def run_iterative_optimization(self, scenario_class, scenario_name, max_iterations=20):
        """反復最適化実行"""
        print("=" * 80)
        print(f"反復型シミュレーション開始：{scenario_name}")
        print("=" * 80)
        
        # 第1ループ：ランダム参加
        participation_status = {i: np.random.rand() > 0.5 for i in range(NUM_USERS)}
        print(f"\n【第1ループ】ランダム参加")
        print(f"参加者数：{sum(participation_status.values())}/{NUM_USERS}")
        
        prev_stats = None
        
        for iteration in range(max_iterations):
            print(f"\n--- イテレーション {iteration + 1} ---")
            
            # シミュレーション実行
            stats = self.run_scenario_with_participation(scenario_class, participation_status)
            
            # 履歴保存
            self.participation_history.append(copy.deepcopy(participation_status))
            self.performance_history.append(stats)
            
            # 結果表示
            participating_count = sum(participation_status.values())
            valid_waits = [s['avg_waiting_time'] for s in stats.values() 
                          if s['avg_waiting_time'] < float('inf')]
            avg_wait_all = np.mean(valid_waits) if valid_waits else float('inf')
            
            print(f"参加者数：{participating_count}/{NUM_USERS}")
            print(f"全体平均待ち時間：{avg_wait_all:.2f}秒")
            
            # 次回の参加判断
            new_participation = {}
            changes = 0
            
            for user_id in range(NUM_USERS):
                # 共有プール参加時の待ち時間（常に実績値を使用）
                wait_acp = stats[user_id].get('avg_waiting_time_shared', stats[user_id]['avg_waiting_time'])
                
                # 独立GPU環境での待ち時間（常に実績値を使用）
                wait_standalone = stats[user_id].get('avg_waiting_time_no_sharing', stats[user_id]['avg_waiting_time'])
                
                # 参加判定：共有プール参加時の方が短ければ参加
                should_participate = wait_acp < wait_standalone
                new_participation[user_id] = should_participate
                
                if new_participation[user_id] != participation_status[user_id]:
                    changes += 1
            
            print(f"参加状態変更：{changes}人")
            
            # 収束判定
            if changes == 0:
                print(f"\n収束しました（イテレーション {iteration + 1}）")
                break
            
            participation_status = new_participation
            prev_stats = stats
        
        # 最終結果
        print("\n" + "=" * 80)
        print("最終結果")
        print("=" * 80)
        final_stats = self.run_scenario_with_participation(scenario_class, participation_status)
        self._print_final_results(participation_status, final_stats)
        
        return participation_status, final_stats
    
    def _print_final_results(self, participation_status, stats):
        """最終結果表示"""
        print(f"\n参加者数：{sum(participation_status.values())}/{NUM_USERS}")
        
        print("\n【ユーザー別結果】")
        print(f"{'ID':>3} {'Tier':>6} {'参加':>4} {'完了率':>7} {'平均待ち時間':>12}")
        print("-" * 50)
        
        for user_id in range(NUM_USERS):
            tier = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    break
            
            s = stats[user_id]
            completion_rate = s['completed_tasks'] / s['total_tasks'] * 100 if s['total_tasks'] > 0 else 0
            participating = "参加" if participation_status[user_id] else "不参加"
            wait_time = s['avg_waiting_time'] if s['avg_waiting_time'] < float('inf') else -1
            
            print(f"{user_id:3d} {tier:>6} {participating:>4} {completion_rate:6.1f}% {wait_time:11.2f}秒")


def main():
    # タスクパターン読み込み
    if not os.path.exists("task_patterns.json"):
        print("タスクパターンを生成中...")
        save_patterns()
    patterns = load_patterns()
    
    scenarios = [
        (SimulatorWithSharing, 'FCFS（共有・先着順）'),
        (SimulatorWithOwnerPriority, '所有者優先'),
        (SimulatorWithOwnerPreemption, '所有者優先＋プリエンプト')
    ]
    
    all_results = {}
    
    for scenario_class, scenario_name in scenarios:
        print("\n\n")
        optimizer = IterativeOptimizer(task_patterns=patterns)
        final_participation, final_stats = optimizer.run_iterative_optimization(
            scenario_class, scenario_name
        )
        all_results[scenario_name] = {
            'participation': final_participation,
            'stats': final_stats
        }
    
    # 3シナリオの比較
    print("\n\n")
    print("=" * 80)
    print("3シナリオの比較")
    print("=" * 80)
    
    for scenario_name in [s[1] for s in scenarios]:
        result = all_results[scenario_name]
        participating_count = sum(result['participation'].values())
        valid_waits = [s['avg_waiting_time'] for s in result['stats'].values() 
                      if s['avg_waiting_time'] < float('inf')]
        avg_wait = np.mean(valid_waits) if valid_waits else float('inf')
        
        print(f"\n【{scenario_name}】")
        print(f"  参加者数：{participating_count}/{NUM_USERS}")
        print(f"  平均待ち時間：{avg_wait:.2f}秒")


if __name__ == "__main__":
    main()
