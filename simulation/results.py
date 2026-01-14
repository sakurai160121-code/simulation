"""
結果分析・出力
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from definitions import Task
from config import GPU_TIER_ASSIGNMENT, GPU_PERFORMANCE_LEVELS, SIMULATION_TIME

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
        completed_by_cutoff = len([t for t in self.completed_tasks if t.completion_time is not None and t.completion_time <= SIMULATION_TIME])
        completion_rate_cutoff = completed_by_cutoff / total_tasks * 100 if total_tasks > 0 else 0
        final_completion_rate = completed_tasks / total_tasks * 100 if total_tasks > 0 else 0
        
        # 待ち時間の計算
        waiting_times = [t.get_waiting_time() for t in self.completed_tasks]
        total_waiting_time = sum(waiting_times) if waiting_times else 0
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0

        # 全タスク完了時刻（Makespan）
        makespan = max([t.completion_time for t in self.completed_tasks], default=0)
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate_cutoff": completion_rate_cutoff,
            "final_completion_rate": final_completion_rate,
            "total_waiting_time": total_waiting_time,
            "avg_waiting_time": avg_waiting_time,
            "makespan": makespan,
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
            completed_by_cutoff = len([t for t in user_completed_tasks if t.completion_time is not None and t.completion_time <= SIMULATION_TIME])
            completion_rate_cutoff = completed_by_cutoff / total_user_tasks * 100 if total_user_tasks > 0 else 0
            final_completion_rate = completed_user_tasks / total_user_tasks * 100 if total_user_tasks > 0 else 0
            
            # 待ち時間の計算
            waiting_times = [t.get_waiting_time() for t in user_completed_tasks]
            total_waiting_time = sum(waiting_times) if waiting_times else 0
            avg_waiting_time = np.mean(waiting_times) if waiting_times else 0

            # 実行した総仕事量（TFLOPs）
            total_work = sum([t.total_work for t in user_completed_tasks if t.total_work is not None])
            
            # 自分のGPU vs 他人のGPUで実行した仕事量
            own_gpu_work = sum([t.total_work for t in user_completed_tasks if t.total_work is not None and t.assigned_gpu and t.assigned_gpu.gpu_id == user_id])
            other_gpu_work = sum([t.total_work for t in user_completed_tasks if t.total_work is not None and t.assigned_gpu and t.assigned_gpu.gpu_id != user_id])

            # ユーザーの最後の完了時刻（全タスクが終わる時刻）
            last_completion_time = max([t.completion_time for t in user_completed_tasks], default=None)
            
            user_stats.append({
                "user_id": user_id,
                "tier": tier,
                "processing_rate": processing_rate,
                "total_tasks": total_user_tasks,
                "completed_tasks": completed_user_tasks,
                "completion_rate_cutoff": completion_rate_cutoff,
                "final_completion_rate": final_completion_rate,
                "total_waiting_time": total_waiting_time,
                "avg_waiting_time": avg_waiting_time,
                "total_work": total_work,
                "own_gpu_work": own_gpu_work,
                "other_gpu_work": other_gpu_work,
                "last_completion_time": last_completion_time,
            })
        
        return user_stats
    
    def print_system_results(self):
        """システム全体の結果を表示"""
        stats = self.get_system_statistics()
        
        if self.mode == "no_sharing":
            mode_title = "【共有なしシミュレーション】"
        elif self.mode == "with_sharing":
            mode_title = "【共有あり・FCFS（先着順）】"
        elif self.mode == "with_sharing_owner_priority":
            mode_title = "【共有あり・所有者優先】"
        elif self.mode == "with_sharing_owner_preemption":
            mode_title = "【共有あり・所有者優先・プリエンプト】"
        else:
            mode_title = "【シミュレーション結果】"
        
        print("=" * 80)
        print(mode_title)
        print("=" * 80)
        print("【システム全体の統計情報】")
        print("=" * 80)
        print(f"発生したタスク総数：{stats['total_tasks']}")
        print(f"完了したタスク数：{stats['completed_tasks']}")
        print(f"タスク完了率（3600秒時点）：{stats['completion_rate_cutoff']:.2f}%")
        print(f"タスク完了率（最終）：{stats['final_completion_rate']:.2f}%")
        print(f"待ち時間の総数：{stats['total_waiting_time']:.4f} 秒")
        print(f"平均待ち時間：{stats['avg_waiting_time']:.4f} 秒")
        print(f"全タスク完了時刻 (Makespan)：{stats['makespan']:.4f} 秒")
        print()
    
    def get_gpu_selection_stats(self):
        """ユーザーごとのGPU選択パターンを計算"""
        gpu_selection_stats = []
        
        for user_id in range(self.num_users):
            user_tasks = [t for t in self.tasks if t.user_id == user_id and t.assigned_gpu is not None]
            
            owner_count = 0
            other_count = 0
            
            for task in user_tasks:
                if task.assigned_gpu.gpu_id == user_id:
                    owner_count += 1
                else:
                    other_count += 1
            
            total = owner_count + other_count
            owner_rate = owner_count / total * 100 if total > 0 else 0
            
            gpu_selection_stats.append({
                "user_id": user_id,
                "owner_count": owner_count,
                "other_count": other_count,
                "owner_rate": owner_rate,
                "other_rate": 100 - owner_rate,
            })
        
        return gpu_selection_stats
    
    def print_user_results(self):
        """ユーザー別の結果を表示"""
        user_stats = self.get_user_statistics()
        gpu_stats = self.get_gpu_selection_stats()
        
        print("=" * 140)
        print("【ユーザー別の統計情報】")
        print("=" * 140)
        
        # DataFrameで見やすく表示
        df = pd.DataFrame(user_stats)
        df_display = df[['user_id', 'tier', 'processing_rate', 'total_tasks', 'completed_tasks', 'completion_rate_cutoff', 'final_completion_rate', 'total_waiting_time', 'avg_waiting_time', 'total_work', 'last_completion_time']].copy()
        df_display.columns = ["ユーザーID", "性能ティア", "処理レート", "発生タスク数", "完了タスク数", "完了率(3600s)(%)", "最終完了率(%)", "待ち時間合計", "平均待ち時間", "総仕事量", "全タスク完了時刻"]
        df_display['平均待ち時間'] = df_display['平均待ち時間'].map(lambda x: f"{x:.4f} 秒")
        df_display['待ち時間合計'] = df_display['待ち時間合計'].map(lambda x: f"{x:.4f} 秒")
        df_display['総仕事量'] = df_display['総仕事量'].map(lambda x: f"{x:.4f} TFLOPs")
        df_display['全タスク完了時刻'] = df_display['全タスク完了時刻'].map(lambda x: f"{x:.4f} 秒" if pd.notnull(x) else "N/A")
        print(df_display.to_string(index=False))
        print()
        # GPU選択パターンを表示
        print("=" * 140)
        print("【GPU選択パターン（共有モード時のみ有意）】")
        print("=" * 140)
        gpu_df = pd.DataFrame(gpu_stats)
        gpu_display = gpu_df.copy()
        gpu_display.columns = ["ユーザーID", "自分のGPU", "他人のGPU", "自分のGPU(%)", "他人のGPU(%)"]
        gpu_display["自分のGPU"] = gpu_display["自分のGPU"].astype(int)
        gpu_display["他人のGPU"] = gpu_display["他人のGPU"].astype(int)
        gpu_display["自分のGPU(%)"] = gpu_display["自分のGPU(%)"].apply(lambda x: f"{x:.1f}%")
        gpu_display["他人のGPU(%)"] = gpu_display["他人のGPU(%)"].apply(lambda x: f"{x:.1f}%")
        
        print(gpu_display.to_string(index=False))
        print()
        
        # グローバルサマリー
        print("\n" + "=" * 140)
        print("【ユーザー別グローバルサマリー】")
        print("=" * 140)
        print(f"平均完了率(3600s)：{df['completion_rate_cutoff'].mean():.2f}%")
        print(f"最高完了率(3600s)：{df['completion_rate_cutoff'].max():.2f}% (User {df.loc[df['completion_rate_cutoff'].idxmax(), 'user_id']:.0f})")
        print(f"最低完了率(3600s)：{df['completion_rate_cutoff'].min():.2f}% (User {df.loc[df['completion_rate_cutoff'].idxmin(), 'user_id']:.0f})")
        print(f"平均待ち時間：{df['avg_waiting_time'].mean():.4f} 秒")
        print(f"最小待ち時間：{df['avg_waiting_time'].min():.4f} 秒 (User {df.loc[df['avg_waiting_time'].idxmin(), 'user_id']:.0f})")
        print(f"最大待ち時間：{df['avg_waiting_time'].max():.4f} 秒 (User {df.loc[df['avg_waiting_time'].idxmax(), 'user_id']:.0f})")
        print()
    
    def plot_results(self, save_dir="./"):
        """結果のグラフを作成"""
        user_stats = self.get_user_statistics()
        df = pd.DataFrame(user_stats)
        if 'total_work' not in df:
            df['total_work'] = 0.0
        df['total_work'] = df['total_work'].fillna(0.0)
        if 'own_gpu_work' not in df:
            df['own_gpu_work'] = 0.0
        df['own_gpu_work'] = df['own_gpu_work'].fillna(0.0)
        if 'other_gpu_work' not in df:
            df['other_gpu_work'] = 0.0
        df['other_gpu_work'] = df['other_gpu_work'].fillna(0.0)
        if 'last_completion_time' not in df:
            df['last_completion_time'] = np.nan
        df['last_completion_time'] = df['last_completion_time'].fillna(np.nan)
        
        if self.mode == "no_sharing":
            mode_title = "共有なし"
        elif self.mode == "with_sharing":
            mode_title = "共有あり・FCFS（先着順）"
        elif self.mode == "with_sharing_owner_priority":
            mode_title = "共有あり・所有者優先"
        elif self.mode == "with_sharing_owner_preemption":
            mode_title = "共有あり・所有者優先・プリエンプト"
        else:
            mode_title = "シミュレーション結果"
        
        # 2x2のサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"シミュレーション結果（{mode_title}）", fontsize=16, fontweight='bold')
        
        # グラフ1: ユーザー別完了率
        ax1 = axes[0][0]
        ax1.bar(df['user_id'].astype(int), df['completion_rate_cutoff'], color='steelblue', alpha=0.7)
        ax1.axhline(y=df['completion_rate_cutoff'].mean(), color='red', linestyle='--', label=f'平均: {df["completion_rate_cutoff"].mean():.2f}%')
        ax1.set_xlabel('ユーザーID')
        ax1.set_ylabel('完了率(%)')
        ax1.set_title('ユーザー別タスク完了率（3600秒時点）')
        ax1.set_xticks(range(0, self.num_users, 2))
        ax1.set_xticklabels(range(0, self.num_users, 2))
        ax1.set_ylim([0, 101])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # グラフ2: ユーザー別平均待ち時間（表形式）
        ax2 = axes[0][1]
        ax2.axis('tight')
        ax2.axis('off')
        
        # 待ち時間の最大値を取得
        mean_wait = df['avg_waiting_time'].mean()
        
        # 表データを準備（2列に分割）
        table_data = []
        # 前半9ユーザー
        col1_data = [['ユーザーID', '平均待ち時間(秒)']]
        for i in range(9):
            uid = df.iloc[i]['user_id']
            wait = df.iloc[i]['avg_waiting_time']
            col1_data.append([f"{int(uid)}", f"{wait:.1f}"])
        
        # 後半9ユーザー
        col2_data = [['ユーザーID', '平均待ち時間(秒)']]
        for i in range(9, self.num_users):
            uid = df.iloc[i]['user_id']
            wait = df.iloc[i]['avg_waiting_time']
            col2_data.append([f"{int(uid)}", f"{wait:.1f}"])
        
        # 平均行を追加
        col1_data.append(['', ''])
        col2_data.append(['平均', f"{mean_wait:.1f}"])
        
        # 2つの列を結合
        for i in range(len(col1_data)):
            if i < len(col2_data):
                table_data.append(col1_data[i] + ['  '] + col2_data[i])
            else:
                table_data.append(col1_data[i] + ['  ', '', ''])
        
        # 表を描画
        table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # ヘッダー行のスタイル
        for i in [0, 3]:  # 2つのヘッダー列
            for j in range(len(table_data[0])):
                cell = table[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
        
        # 平均行のスタイル
        last_row = len(table_data) - 1
        for j in [3, 4]:
            cell = table[(last_row, j)]
            cell.set_facecolor('#E7E6E6')
            cell.set_text_props(weight='bold')
        
        ax2.set_title('ユーザー別平均待ち時間', fontsize=12, fontweight='bold', pad=20)
        
        # グラフ3: ユーザー別総仕事量 (TFLOPs) - 自分/他人のGPU別に色分け
        ax3 = axes[1][0]
        x_pos = df['user_id'].astype(int)
        ax3.bar(x_pos, df['own_gpu_work'], color='steelblue', alpha=0.8, label='自分のGPU')
        ax3.bar(x_pos, df['other_gpu_work'], bottom=df['own_gpu_work'], color='orange', alpha=0.8, label='他人のGPU')
        ax3.set_xlabel('ユーザーID')
        ax3.set_ylabel('総仕事量 (TFLOPs)')
        ax3.set_title('ユーザー別総仕事量（GPU別）')
        ax3.set_xticks(range(0, self.num_users, 2))
        ax3.set_xticklabels(range(0, self.num_users, 2))
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # グラフ4: 各ユーザーの全タスク完了時刻（表形式）
        ax4 = axes[1][1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 表データを準備（2列に分割）
        table_data = []
        # 前半9ユーザー
        col1_data = [['ユーザーID', '完了時刻(秒)']]
        for i in range(9):
            uid = df.iloc[i]['user_id']
            time = df.iloc[i]['last_completion_time']
            col1_data.append([f"{int(uid)}", f"{time:.1f}"])
        
        # 後半9ユーザー
        col2_data = [['ユーザーID', '完了時刻(秒)']]
        for i in range(9, self.num_users):
            uid = df.iloc[i]['user_id']
            time = df.iloc[i]['last_completion_time']
            col2_data.append([f"{int(uid)}", f"{time:.1f}"])
        
        # makespan行を追加
        makespan = df['last_completion_time'].max()
        col1_data.append(['', ''])
        col2_data.append(['最大値', f"{makespan:.1f}"])
        
        # 2つの列を結合
        for i in range(len(col1_data)):
            if i < len(col2_data):
                table_data.append(col1_data[i] + ['  '] + col2_data[i])
            else:
                table_data.append(col1_data[i] + ['  ', '', ''])
        
        # 表を描画
        table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # ヘッダー行のスタイル
        for i in [0, 3]:  # 2つのヘッダー列
            for j in range(len(table_data[0])):
                cell = table[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
        
        # 最大値行のスタイル
        last_row = len(table_data) - 1
        for j in [3, 4]:
            cell = table[(last_row, j)]
            cell.set_facecolor('#E7E6E6')
            cell.set_text_props(weight='bold')
        
        ax4.set_title('ユーザー別全タスク完了時刻', fontsize=12, fontweight='bold', pad=20)
        
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
