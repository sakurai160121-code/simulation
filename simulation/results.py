"""
結果分析・出力
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from definitions import Task
from config import GPU_TIER_ASSIGNMENT, GPU_PERFORMANCE_LEVELS

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultAnalyzer:
    """
    シミュレーション結果の分析クラス
    """
    def __init__(self, tasks: List[Task], num_users: int, mode: str = "no_sharing"):
        self.tasks = tasks
        self.num_users = num_users
        self.mode = mode  # "no_sharing" または "with_sharing"
        self.completed_tasks = [t for t in tasks if t.completion_time is not None]
        
    def get_system_statistics(self):
        """システム全体の統計情報を取得"""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        completion_rate = completed_tasks / total_tasks * 100 if total_tasks > 0 else 0
        
        # 待ち時間の計算
        waiting_times = [t.get_waiting_time() for t in self.completed_tasks]
        total_waiting_time = sum(waiting_times) if waiting_times else 0
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": completion_rate,
            "total_waiting_time": total_waiting_time,
            "avg_waiting_time": avg_waiting_time,
        }
    
    def get_user_statistics(self):
        """ユーザー別の統計情報を取得"""
        user_stats = []
        
        for user_id in range(self.num_users):
            # ユーザーの性能ティアを取得
            tier = None
            processing_rate = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    processing_rate = GPU_PERFORMANCE_LEVELS[tier_name]
                    break
            
            user_tasks = [t for t in self.tasks if t.user_id == user_id]
            user_completed_tasks = [t for t in user_tasks if t.completion_time is not None]
            
            total_user_tasks = len(user_tasks)
            completed_user_tasks = len(user_completed_tasks)
            completion_rate = completed_user_tasks / total_user_tasks * 100 if total_user_tasks > 0 else 0
            
            # 待ち時間の計算
            waiting_times = [t.get_waiting_time() for t in user_completed_tasks]
            total_waiting_time = sum(waiting_times) if waiting_times else 0
            avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
            
            user_stats.append({
                "user_id": user_id,
                "tier": tier,
                "processing_rate": processing_rate,
                "total_tasks": total_user_tasks,
                "completed_tasks": completed_user_tasks,
                "completion_rate": completion_rate,
                "total_waiting_time": total_waiting_time,
                "avg_waiting_time": avg_waiting_time,
            })
        
        return user_stats
    
    def print_system_results(self):
        """システム全体の結果を表示"""
        stats = self.get_system_statistics()
        
        mode_title = "【共有なしシミュレーション】" if self.mode == "no_sharing" else "【共有ありシミュレーション（ACPプール）】"
        
        print("=" * 80)
        print(mode_title)
        print("=" * 80)
        print("【システム全体の統計情報】")
        print("=" * 80)
        print(f"発生したタスク総数：{stats['total_tasks']}")
        print(f"完了したタスク数：{stats['completed_tasks']}")
        print(f"タスク完了率：{stats['completion_rate']:.2f}%")
        print(f"待ち時間の総数：{stats['total_waiting_time']:.4f}")
        print(f"平均待ち時間：{stats['avg_waiting_time']:.4f}")
        print()
    
    def print_user_results(self):
        """ユーザー別の結果を表示"""
        user_stats = self.get_user_statistics()
        
        print("=" * 140)
        print("【ユーザー別の統計情報】")
        print("=" * 140)
        
        # DataFrameで見やすく表示
        df = pd.DataFrame(user_stats)
        df_display = df.copy()
        df_display.columns = ["ユーザーID", "性能ティア", "処理レート", "発生タスク数", "完了タスク数", "完了率(%)", "待ち時間合計", "平均待ち時間"]
        
        print(df_display.to_string(index=False))
        print()
        
        # ティア別サマリー
        print("=" * 140)
        print("【性能ティア別サマリー】")
        print("=" * 140)
        for tier_name in ["tier1", "tier2", "tier3", "tier4"]:
            tier_data = df[df['tier'] == tier_name]
            if len(tier_data) > 0:
                print(f"\n{tier_name.upper()} (処理レート: {tier_data['processing_rate'].iloc[0]})")
                print(f"  平均完了率：{tier_data['completion_rate'].mean():.2f}%")
                print(f"  平均待ち時間：{tier_data['avg_waiting_time'].mean():.4f}")
        
        # グローバルサマリー
        print("\n" + "=" * 140)
        print("【ユーザー別グローバルサマリー】")
        print("=" * 140)
        print(f"平均完了率：{df['completion_rate'].mean():.2f}%")
        print(f"最高完了率：{df['completion_rate'].max():.2f}% (User {df.loc[df['completion_rate'].idxmax(), 'user_id']:.0f})")
        print(f"最低完了率：{df['completion_rate'].min():.2f}% (User {df.loc[df['completion_rate'].idxmin(), 'user_id']:.0f})")
        print(f"平均待ち時間：{df['avg_waiting_time'].mean():.4f}")
        print(f"最小待ち時間：{df['avg_waiting_time'].min():.4f} (User {df.loc[df['avg_waiting_time'].idxmin(), 'user_id']:.0f})")
        print(f"最大待ち時間：{df['avg_waiting_time'].max():.4f} (User {df.loc[df['avg_waiting_time'].idxmax(), 'user_id']:.0f})")
        print()
    
    def plot_results(self, save_dir="./"):
        """結果のグラフを作成"""
        user_stats = self.get_user_statistics()
        df = pd.DataFrame(user_stats)
        
        mode_title = "共有なし" if self.mode == "no_sharing" else "共有あり（ACPプール）"
        
        # 3つのサブプロットを作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"シミュレーション結果（{mode_title}）", fontsize=16, fontweight='bold')
        
        # グラフ1: ユーザー別完了率
        ax1 = axes[0]
        ax1.bar(df['user_id'].astype(int), df['completion_rate'], color='steelblue', alpha=0.7)
        ax1.axhline(y=df['completion_rate'].mean(), color='red', linestyle='--', label=f'平均: {df["completion_rate"].mean():.2f}%')
        ax1.set_xlabel('ユーザーID')
        ax1.set_ylabel('完了率(%)')
        ax1.set_title('ユーザー別タスク完了率')
        ax1.set_xticks(range(0, 20, 2))
        ax1.set_xticklabels(range(0, 20, 2))
        ax1.set_ylim([0, 101])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # グラフ2: ユーザー別平均待ち時間
        ax2 = axes[1]
        colors = ['red' if x > df['avg_waiting_time'].mean() else 'green' for x in df['avg_waiting_time']]
        ax2.bar(df['user_id'].astype(int), df['avg_waiting_time'], color=colors, alpha=0.7)
        ax2.axhline(y=df['avg_waiting_time'].mean(), color='blue', linestyle='--', label=f'平均: {df["avg_waiting_time"].mean():.2f}')
        ax2.set_xlabel('ユーザーID')
        ax2.set_ylabel('平均待ち時間')
        ax2.set_title('ユーザー別平均待ち時間')
        ax2.set_xticks(range(0, 20, 2))
        ax2.set_xticklabels(range(0, 20, 2))
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # グラフ3: GPU別タスク処理割合
        ax3 = axes[2]
        gpu_task_counts = {}
        total_completed = len(self.completed_tasks)
        
        for task in self.completed_tasks:
            gpu_id = task.assigned_gpu.gpu_id if task.assigned_gpu else None
            if gpu_id is not None:
                gpu_task_counts[gpu_id] = gpu_task_counts.get(gpu_id, 0) + 1
        
        # GPU IDでソート
        sorted_gpu_ids = sorted(gpu_task_counts.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 0)
        task_counts = [gpu_task_counts[gpu_id] for gpu_id in sorted_gpu_ids]
        task_percentages = [(count / total_completed * 100) if total_completed > 0 else 0 for count in task_counts]
        
        # GPU IDを整数表示用に変換
        gpu_labels = [int(gid) if isinstance(gid, (int, str)) and str(gid).isdigit() else str(gid) for gid in sorted_gpu_ids]
        
        ax3.bar(range(len(gpu_labels)), task_percentages, color='orange', alpha=0.7)
        ax3.axhline(y=100/len(gpu_labels), color='red', linestyle='--', label=f'均等値: {100/len(gpu_labels):.2f}%')
        ax3.set_xlabel('GPU ID')
        ax3.set_ylabel('処理タスク割合(%)')
        ax3.set_title('GPU別タスク処理割合')
        ax3.set_xticks(range(0, len(gpu_labels), 2))
        ax3.set_xticklabels([gpu_labels[i] for i in range(0, len(gpu_labels), 2)])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # ファイル名をモード別に変更
        filename = f'simulation_results_{self.mode}.png'
        plt.savefig(f'{save_dir}{filename}', dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_dir}{filename}")
        plt.close()
    
    def plot_task_timeline(self, save_dir="./"):
        """タスク発生と完了のタイムラインを作成（削除）"""
        # 削除
        pass


def analyze_and_print_results(tasks: List[Task], num_users: int, mode: str = "no_sharing"):
    """結果分析と出力を実行"""
    analyzer = ResultAnalyzer(tasks, num_users, mode=mode)
    analyzer.print_system_results()
    analyzer.print_user_results()
    analyzer.plot_results()
    
    return analyzer
