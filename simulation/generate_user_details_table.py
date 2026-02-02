"""
全ユーザー（0～17）の平均待ち時間詳細表を生成するスクリプト
"""

import sys
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Windows環境でUnicode出力を有効化
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 日本語フォント設定
rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from task_patterns import save_patterns, load_patterns
from simulation_no_sharing import Simulator as SimulatorNoSharing
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from results import ResultAnalyzer
from config import NUM_USERS

def run_all_simulations():
    """4つの基本シナリオを実行し、ユーザー別統計を取得"""
    
    print("\n" + "="*80)
    print("タスクパターン生成")
    print("="*80)
    
    # タスクパターン生成
    save_patterns()
    task_patterns = load_patterns()
    
    print("\n" + "="*80)
    print("4つのシミュレーションを実行中...")
    print("="*80)
    
    all_user_stats = {}
    
    # シナリオ1: 非共有
    print("\n【非共有】")
    sim = SimulatorNoSharing(task_patterns=task_patterns)
    tasks_no_sharing = sim.run()
    analyzer = ResultAnalyzer(tasks_no_sharing, NUM_USERS, mode="no_sharing")
    all_user_stats["非共有"] = analyzer.get_user_statistics()
    
    # シナリオ2: FCFS（先着順）
    print("\n【FCFS（先着順）】")
    sim = SimulatorWithSharing(task_patterns=task_patterns)
    tasks_with_sharing = sim.run()
    analyzer = ResultAnalyzer(tasks_with_sharing, NUM_USERS, mode="with_sharing")
    all_user_stats["FCFS（先着順）"] = analyzer.get_user_statistics()
    
    # シナリオ3: 所有者優先
    print("\n【所有者優先】")
    sim = SimulatorWithOwnerPriority(task_patterns=task_patterns)
    tasks_owner_priority = sim.run()
    analyzer = ResultAnalyzer(tasks_owner_priority, NUM_USERS, mode="with_sharing_owner_priority")
    all_user_stats["所有者優先"] = analyzer.get_user_statistics()
    
    # シナリオ4: プリエンプション
    print("\n【プリエンプション】")
    sim = SimulatorWithOwnerPreemption(task_patterns=task_patterns)
    tasks_owner_preemption = sim.run()
    analyzer = ResultAnalyzer(tasks_owner_preemption, NUM_USERS, mode="with_sharing_owner_preemption")
    all_user_stats["プリエンプション"] = analyzer.get_user_statistics()
    
    return all_user_stats

def create_user_table(all_user_stats):
    """ユーザー詳細表を作成・表示"""
    
    print("\n" + "="*80)
    print("ユーザー0～8の平均TAT詳細")
    print("="*80)
    
    # テーブルデータを構築
    table_data = []
    
    for user_id in range(9):  # ユーザー0～8
        row = {"ユーザーID": f"ユーザー{user_id}"}
        
        for scenario_name in ["非共有", "FCFS（先着順）", "所有者優先", "プリエンプション"]:
            user_stats_list = all_user_stats[scenario_name]
            user_stat = user_stats_list[user_id]
            avg_wait = user_stat['avg_waiting_time']
            row[scenario_name] = f"{avg_wait:.2f}秒"
        
        table_data.append(row)
    
    # DataFrameに変換して表示
    df = pd.DataFrame(table_data)
    print("\n" + df.to_string(index=False))
    
    # CSVファイルに保存
    csv_filename = "user_details_0_to_8.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSVファイルを保存: {csv_filename}")
    
    # 次に全ユーザー（0～17）の表を作成
    print("\n" + "="*80)
    print("ユーザー0～17の平均TAT詳細")
    print("="*80)
    
    table_data_all = []
    
    for user_id in range(NUM_USERS):
        row = {"ユーザーID": f"ユーザー{user_id}"}
        
        for scenario_name in ["非共有", "FCFS（先着順）", "所有者優先", "プリエンプション"]:
            user_stats_list = all_user_stats[scenario_name]
            user_stat = user_stats_list[user_id]
            avg_wait = user_stat['avg_waiting_time']
            row[scenario_name] = f"{avg_wait:.2f}秒"
        
        table_data_all.append(row)
    
    df_all = pd.DataFrame(table_data_all)
    print("\n" + df_all.to_string(index=False))
    
    # CSVファイルに保存
    csv_filename_all = "user_details_0_to_17.csv"
    df_all.to_csv(csv_filename_all, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSVファイルを保存: {csv_filename_all}")
    
    # PNG画像として保存
    save_table_as_image(df_all, "user_details_0_to_17.png")

def save_table_as_image(df, filename):
    """DataFrameをPNG画像として保存"""
    
    print(f"\n【PNG画像生成】")
    
    # テーブルデータを準備
    table_data = []
    table_data.append(df.columns.tolist())
    for _, row in df.iterrows():
        table_data.append(row.tolist())
    
    # 図を作成
    fig, ax = plt.subplots(figsize=(16, 14), dpi=100)
    ax.axis('tight')
    ax.axis('off')
    
    # テーブルを描画
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.12, 0.12, 0.12]
    )
    
    # テーブルのスタイル設定
    table.auto_set_font_size(False)
    table.set_fontsize(28)
    table.scale(2, 4.4)  # セルの高さを大きめに
    
    # ヘッダー行のスタイル（濃い青、白字）
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', size=28)
        cell.set_linewidth(0.8)
    
    # 3ユーザーずつの色分け（低性能、中性能、高性能）
    # 色：低性能（赤系）、中性能（黄系）、高性能（緑系）
    colors = [
        "#B1C4F8",  # ユーザー0-2：低性能（薄い赤）
        "#F5BF8D",  # ユーザー3-5：中性能（薄い黄）
        "#FFB3B3",  # ユーザー6-8：高性能（薄い緑）
        "#B1C4F8",  # ユーザー9-11：低性能（薄い赤）
        '#F5BF8D',  # ユーザー12-14：中性能（薄い黄）
        '#FFB3B3',  # ユーザー15-17：高性能（薄い緑）
    ]
    
    # 行の背景色を3ユーザーごとに設定
    for i in range(1, len(table_data)):
        user_idx = i - 1  # 0～17
        color_idx = (user_idx // 3) % 6  # 0-2は色0, 3-5は色1, 6-8は色2, 9-11は色0...
        
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[color_idx])
            cell.set_linewidth(0.8)
    
    # タイトルを追加
    fig.suptitle('ユーザー0～17の平均TAT詳細', fontsize=28, fontweight='bold', y=0.98)
    
    # PNG保存
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(f"✓ PNG画像を保存: {filename}")
    plt.close()

def main():
    """メイン処理"""
    start_time = datetime.now()
    print("\n" + "="*80)
    print(f"全ユーザー詳細表生成開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # すべてのシミュレーションを実行
        all_user_stats = run_all_simulations()
        
        # ユーザー詳細表を作成・表示
        create_user_table(all_user_stats)
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print("\n" + "="*80)
        print("完了")
        print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {elapsed}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
