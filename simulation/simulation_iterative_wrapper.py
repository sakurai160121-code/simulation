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
        参加者：共有プールシミュレーション（参加ユーザーのみ）
        非参加者：独立GPU環境シミュレーション（no_sharing）
        """
        # 参加ユーザーのタスクパターンのみを抽出
        participating_users = [uid for uid, v in participation_status.items() if v]
        filtered_patterns = self._filter_task_patterns_by_users(participating_users)
        
        # 共有プールシミュレーション実行（参加ユーザーのみ）
        sim_shared = scenario_class(task_patterns=filtered_patterns, participating_users=participating_users)
        all_tasks_shared = sim_shared.run()
        
        # 独立GPU環境シミュレーション実行（常に実行して比較可能にする）
        sim_no_sharing = SimulatorNoSharing(task_patterns=self.task_patterns)
        all_tasks_no_sharing = sim_no_sharing.run()
        
        # 結果を分析
        return self._analyze_results(all_tasks_shared, all_tasks_no_sharing, participation_status, participating_users)
    
    def _filter_task_patterns_by_users(self, participating_users):
        """参加ユーザーのタスクパターンのみを抽出"""
        if not self.task_patterns:
            return {}
        
        filtered = {}
        for key, value in self.task_patterns.items():
            if key == "arrivals":
                # arrivals は {str(user_id): [...]} の形式
                filtered[key] = {str(uid): value.get(str(uid), []) 
                                for uid in participating_users}
            elif key == "sizes":
                # sizes は {str(user_id): [...]} の形式
                filtered[key] = {str(uid): value.get(str(uid), []) 
                                for uid in participating_users}
            else:
                # その他のキーはそのまま
                filtered[key] = value
        
        return filtered
    
    def _analyze_results(self, all_tasks_shared, all_tasks_no_sharing, participation_status, participating_users=None):
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
                'participating': participation_status[user_id],
                'last_completion_time_shared': max([t.completion_time for t in completed_tasks_shared], default=None) if completed_tasks_shared else None,
                'last_completion_time_no_sharing': max([t.completion_time for t in completed_tasks_no_sharing], default=None) if completed_tasks_no_sharing else None,
                'last_completion_time': max([t.completion_time for t in completed_tasks_count], default=None) if completed_tasks_count else None
            }
        
        return user_stats
    
    def decide_participation(self, scenario_class, current_participation_status, target_user_id, standalone_wait_times, current_stats=None):
        """
        個別テストシミュレーション方式による参加判定
        
        Args:
            scenario_class: シナリオクラス（SimulatorWithSharing等）
            current_participation_status: 現在の参加状態辞書
            target_user_id: 判定対象ユーザーID
            standalone_wait_times: 全ユーザーの単独GPU待ち時間辞書
            current_stats: 現在イテレーションの結果（参加中ユーザーの実測待ち時間を利用）
        
        Returns:
            bool: 参加すべきならTrue、不参加ならFalse
        """
        # すでに参加しているユーザーは、今の共有待ち時間と単独待ち時間を比較して継続可否を判断
        if current_participation_status.get(target_user_id, False):
            # current_stats があれば共有待ち時間をそこから取得
            shared_wait = None
            if current_stats is not None:
                shared_wait = current_stats.get(target_user_id, {}).get('avg_waiting_time_shared', None)
            # 取れない場合は単独GPUと比較できないので継続（保守的）
            if shared_wait is None:
                return True
            standalone_wait = standalone_wait_times.get(target_user_id, float('inf'))
            return shared_wait <= standalone_wait

        # 現在の参加者リスト + ターゲットユーザー
        test_participants = [uid for uid, v in current_participation_status.items() if v]
        if target_user_id not in test_participants:
            test_participants.append(target_user_id)
        
        # テスト用の参加状態を作成
        test_participation = {uid: (uid in test_participants) for uid in range(NUM_USERS)}
        
        # 参加者のタスクパターンを抽出
        filtered_patterns = self._filter_task_patterns_by_users(test_participants)
        
        # 共有プールシミュレーション実行
        sim_shared = scenario_class(task_patterns=filtered_patterns, participating_users=test_participants)
        all_tasks_shared = sim_shared.run()
        
        # ターゲットユーザーの平均待ち時間を取得
        target_tasks = [t for t in all_tasks_shared if t.user_id == target_user_id]
        completed_tasks = [t for t in target_tasks if t.completion_time is not None]
        
        if completed_tasks:
            waiting_times = [t.start_time - t.arrival_time for t in completed_tasks 
                           if t.start_time is not None]
            shared_wait = np.mean(waiting_times) if waiting_times else float('inf')
        else:
            shared_wait = float('inf')
        
        # 単独GPU環境の待ち時間と比較
        standalone_wait = standalone_wait_times.get(target_user_id, float('inf'))
        
        # 参加判定：共有プール参加時の待ち時間が単独と同等以下なら参加
        should_participate = shared_wait <= standalone_wait
        
        return should_participate
    
    def run_iterative_optimization(self, scenario_class, scenario_name, max_iterations=10):
        """反復最適化実行（最大10イテレーション、毎回ユーザー別結果を出力）"""
        print("=" * 80)
        print(f"反復型シミュレーション開始：{scenario_name}")
        print("=" * 80)
        
        # 第1ループ：ランダム参加
        participation_status = {i: np.random.rand() > 0.5 for i in range(NUM_USERS)}
        print(f"\n【第1ループ】ランダム参加")
        iteration = 0
        print(f"参加者数：{sum(participation_status.values())}/{NUM_USERS}")
        
        prev_stats = None
        
        while True:
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
            
            # ユーザー別結果を表形式で出力
            print("\n【ユーザー別結果】")
            print(" ID   Tier   参加          平均待ち時間        全タスク完了時刻")
            print("-" * 65)
            user_results = self._get_user_results_summary(participation_status, stats)
            for result in user_results:
                user_id = result['user_id']
                tier = result['tier']
                participating = result['participating']
                wait_time = result['avg_waiting_time']
                last_completion = result['last_completion_time']
                
                wait_str = f"{wait_time:.2f}秒" if wait_time >= 0 else "   -秒"
                last_str = f"{last_completion:.2f}秒" if last_completion < float('inf') else "   -秒"
                
                print(f"{user_id:3d}  {tier:6s}  {participating:6s}  {wait_str:>15s}   {last_str:>15s}")

            
            # 次回の参加判断
            new_participation = {}
            changes = 0
            
            # 単独GPU環境の待ち時間を取得（全ユーザー分）
            standalone_wait_times = {}
            for user_id in range(NUM_USERS):
                standalone_wait_times[user_id] = stats[user_id].get('avg_waiting_time_no_sharing', stats[user_id]['avg_waiting_time'])
            
            # 各ユーザーについて個別テストシミュレーションで判定
            for user_id in range(NUM_USERS):
                should_participate = self.decide_participation(
                    scenario_class, 
                    participation_status, 
                    user_id, 
                    standalone_wait_times,
                    current_stats=stats
                )
                new_participation[user_id] = should_participate
                
                if new_participation[user_id] != participation_status[user_id]:
                    changes += 1
            
            print(f"\n参加状態変更：{changes}人")
            
            # 収束判定または最大反復数に達したかチェック
            if changes == 0:
                print(f"収束しました（イテレーション {iteration + 1}）")
                break
            
            if iteration + 1 >= max_iterations:
                print(f"最大イテレーション数 {max_iterations} に到達しました")
                break
            
            participation_status = new_participation
            prev_stats = stats
            iteration += 1
        
        # 最終結果を保存（表示は後でまとめて行う）
        final_stats = self.run_scenario_with_participation(scenario_class, participation_status)
        
        return participation_status, final_stats
    
    def _get_user_results_summary(self, participation_status, stats):
        """ユーザー別結果をリスト形式で返す"""
        user_results = []
        for user_id in range(NUM_USERS):
            tier = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    break
            
            s = stats[user_id]
            participating = "参加" if participation_status[user_id] else "不参加"
            wait_time = s['avg_waiting_time'] if s['avg_waiting_time'] < float('inf') else -1
            last_completion = s.get('last_completion_time', float('inf'))
            
            user_results.append({
                'user_id': user_id,
                'tier': tier,
                'participating': participating,
                'avg_waiting_time': wait_time,
                'last_completion_time': last_completion
            })
        return user_results


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
    
    # 3シナリオのユーザー別結果をまとめて表示
    print("\n\n")
    print("=" * 120)
    print("3シナリオのユーザー別結果")
    print("=" * 120)
    
    for scenario_name in [s[1] for s in scenarios]:
        result = all_results[scenario_name]
        participating_count = sum(result['participation'].values())
        print(f"\n【{scenario_name}】 参加者数：{participating_count}/{NUM_USERS}")
        print(f"{'ID':>3} {'Tier':>6} {'参加':>4} {'平均待ち時間':>15} {'全タスク完了時刻':>15}")
        print("-" * 65)
        
        for user_id in range(NUM_USERS):
            tier = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    break
            
            s = result['stats'][user_id]
            participating = "参加" if result['participation'][user_id] else "不参加"
            wait_time = s['avg_waiting_time'] if s['avg_waiting_time'] < float('inf') else -1
            last_completion = s.get('last_completion_time', float('inf'))
            
            if last_completion < float('inf'):
                last_completion_str = f"{last_completion:.2f}秒"
            else:
                last_completion_str = "N/A"
            
            print(f"{user_id:3d} {tier:>6} {participating:>4} {wait_time:14.2f}秒 {last_completion_str:>15}")


if __name__ == "__main__":
    main()
