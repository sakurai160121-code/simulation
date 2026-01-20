"""
全シミュレーション一括実行スクリプト
1. タスクパターン生成
2. 4つの基本シナリオ実行
3. 反復最適化ラッパー実行
"""

import os
import sys
from datetime import datetime

# 出力ディレクトリの設定
OUTPUT_DIR_BASE = './outputs'
OUTPUT_DIR_BASIC = './outputs/basic_scenarios'
OUTPUT_DIR_USER_COMP = './outputs/user_comparisons'
OUTPUT_DIR_TABLES = './outputs/tables'

# 出力ディレクトリを作成
for dir_path in [OUTPUT_DIR_BASE, OUTPUT_DIR_BASIC, OUTPUT_DIR_USER_COMP, OUTPUT_DIR_TABLES]:
    os.makedirs(dir_path, exist_ok=True)

# Windows環境でUnicode出力を有効化
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_task_pattern_generation():
    """タスクパターン生成"""
    print("\n" + "="*80)
    print("【ステップ1】タスクパターン生成")
    print("="*80)
    
    # 既存のタスクパターンを削除
    if os.path.exists("task_patterns.json"):
        print("既存のタスクパターンを削除...")
        os.remove("task_patterns.json")
    
    from task_patterns import save_patterns
    print("新しいタスクパターンを生成中...")
    save_patterns()
    print("✓ タスクパターン生成完了")

def run_base_scenarios():
    """4つの基本シナリオを実行"""
    print("\n" + "="*80)
    print("【ステップ2】基本シナリオ実行")
    print("="*80)
    
    scenarios = [
        ("simulation_no_sharing", "共有なし", "no_sharing"),
        ("simulation_with_sharing", "FCFS（先着順）", "with_sharing"),
        ("simulation_with_sharing_owner_priority", "所有者優先", "with_sharing_owner_priority"),
        ("simulation_with_sharing_owner_preemption", "所有者優先＋プリエンプション", "with_sharing_owner_preemption")
    ]
    
    results = {}
    all_tasks = {}
    
    for module_name, scenario_name, mode in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        if module_name == "simulation_no_sharing":
            from simulation_no_sharing import Simulator
            from task_patterns import load_patterns
            patterns = load_patterns()
            sim = Simulator(task_patterns=patterns)
            tasks = sim.run()
            
        elif module_name == "simulation_with_sharing":
            from simulation_with_sharing import SimulatorWithSharing
            from task_patterns import load_patterns
            patterns = load_patterns()
            sim = SimulatorWithSharing(task_patterns=patterns)
            tasks = sim.run()
            
        elif module_name == "simulation_with_sharing_owner_priority":
            from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
            from task_patterns import load_patterns
            patterns = load_patterns()
            sim = SimulatorWithOwnerPriority(task_patterns=patterns)
            tasks = sim.run()
            
        elif module_name == "simulation_with_sharing_owner_preemption":
            from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
            from task_patterns import load_patterns
            patterns = load_patterns()
            sim = SimulatorWithOwnerPreemption(task_patterns=patterns)
            tasks = sim.run()
        
        # タスクデータを保存
        all_tasks[mode] = tasks
        
        # 平均待ち時間を計算（到着から完了までの時間）
        completed_tasks = [t for t in tasks if t.completion_time is not None]
        if completed_tasks:
            response_times = [t.completion_time - t.arrival_time for t in completed_tasks]
            avg_wait = sum(response_times) / len(response_times) if response_times else 0
            results[scenario_name] = avg_wait
            print(f"✓ 完了 - 平均待ち時間: {avg_wait:.2f}秒")
        else:
            results[scenario_name] = float('inf')
            print(f"✓ 完了 - タスク完了なし")
    
    # 結果サマリー
    print("\n" + "-"*80)
    print("【基本シナリオ結果サマリー】")
    print("-"*80)
    for scenario_name, avg_wait in results.items():
        if avg_wait < float('inf'):
            print(f"{scenario_name:30s}: {avg_wait:>12.2f}秒")
        else:
            print(f"{scenario_name:30s}: {'N/A':>12s}")
    
    return results, all_tasks

def run_iterative_wrapper():
    """反復最適化ラッパー実行"""
    print("\n" + "="*80)
    print("【ステップ3】反復最適化ラッパー実行")
    print("="*80)
    
    from simulation_iterative_wrapper import main as wrapper_main
    wrapper_main()
    
    print("\n✓ 反復最適化ラッパー完了")

def generate_graphs(all_tasks):
    """グラフ生成"""
    print("\n" + "="*80)
    print("【グラフ生成】")
    print("="*80)
    
    from results import ResultAnalyzer
    from config import NUM_USERS
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # シナリオ別のタイトル
    scenario_titles = {
        "no_sharing": "共有なし",
        "with_sharing": "FCFS（先着順）",
        "with_sharing_owner_priority": "所有者優先",
        "with_sharing_owner_preemption": "所有者優先＋プリエンプション"
    }
    
    # Step1: シナリオ別グラフを生成
    for mode, tasks in all_tasks.items():
        print(f"\n{scenario_titles[mode]}のグラフを生成中...")
        analyzer = ResultAnalyzer(tasks, NUM_USERS, mode=mode)
        analyzer.plot_results(save_dir="./")
    
    print("\n✓ シナリオ別グラフ生成完了")
    
    # Step2: ユーザー別の比較グラフを生成
    print("\n" + "-"*80)
    print("ユーザー別比較グラフを生成中...")
    print("-"*80)
    
    # 各シナリオのユーザー統計を取得
    user_stats_by_scenario = {}
    for mode, tasks in all_tasks.items():
        analyzer = ResultAnalyzer(tasks, NUM_USERS, mode=mode)
        user_stats_by_scenario[mode] = analyzer.get_user_statistics()
    
    # ユーザーごとに比較グラフを作成
    for user_id in range(NUM_USERS):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        fig.suptitle(f"ユーザー{user_id}の各シナリオ比較", fontsize=14, fontweight='bold')
        
        # グラフ用データを準備
        scenario_names = []
        avg_wait_times = []
        completion_times = []
        other_gpu_rates = []
        my_gpu_used_by_others_counts = []
        
        for mode in ["no_sharing", "with_sharing", "with_sharing_owner_priority", "with_sharing_owner_preemption"]:
            stats_list = user_stats_by_scenario[mode]
            user_stat = stats_list[user_id]
            
            scenario_names.append(scenario_titles[mode])
            avg_wait_times.append(user_stat['avg_waiting_time'])
            completion_times.append(user_stat['last_completion_time'] if user_stat['last_completion_time'] else 0)
            
            # 他人のGPU使用割合を計算
            total_work = user_stat['total_work']
            other_work = user_stat['other_gpu_work']
            other_rate = (other_work / total_work * 100) if total_work > 0 else 0
            other_gpu_rates.append(other_rate)
            
            # 他人が自分のGPUで処理したタスク数をカウント（assigned_gpuはGPUオブジェクト）
            tasks = all_tasks[mode]
            other_on_my_gpu = sum(
                1
                for t in tasks
                if t.assigned_gpu is not None
                and getattr(t.assigned_gpu, 'gpu_id', None) == user_id
                and t.user_id != user_id
                and t.completion_time is not None
            )
            my_gpu_used_by_others_counts.append(other_on_my_gpu)
        
        # グラフ1: 平均待ち時間（左上）
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars1 = ax1.bar(range(len(scenario_names)), avg_wait_times, color=colors, alpha=0.7)
        ax1.set_xlabel('シナリオ', fontsize=10)
        ax1.set_ylabel('平均待ち時間（秒）', fontsize=10)
        ax1.set_title('平均待ち時間の比較', fontsize=11, fontweight='bold')
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for i, (bar, val) in enumerate(zip(bars1, avg_wait_times)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_wait_times)*0.02,
                    f'{val:.1f}秒', ha='center', va='bottom', fontsize=8)
        
        # グラフ2: 全タスク完了時刻（右上）
        bars2 = ax2.bar(range(len(scenario_names)), completion_times, color=colors, alpha=0.7)
        ax2.set_xlabel('シナリオ', fontsize=10)
        ax2.set_ylabel('全タスク完了時刻（秒）', fontsize=10)
        ax2.set_title('全タスク完了時刻の比較', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for i, (bar, val) in enumerate(zip(bars2, completion_times)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(completion_times)*0.02,
                    f'{val:.1f}秒', ha='center', va='bottom', fontsize=8)
        
        # 表: 他人GPU使用割合・割り込み・プリエンプション統計（下部全体）
        ax3.axis('tight')
        ax3.axis('off')
        
        # 表データを準備（3列：他GPU%、他人が自分GPUで処理した回数）
        table_data = [['シナリオ', '他GPU(%)', '他人が自分GPUで処理']] 
        for i, name in enumerate(scenario_names):
            # 利用率が0でも0.0として表示
            other_gpu_str = f"{other_gpu_rates[i]:.1f}"
            table_data.append([name, other_gpu_str, my_gpu_used_by_others_counts[i]])
        
        # 表を描画
        table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # ヘッダー行のスタイル
        for j in range(3):
            cell = table[(0, j)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white', fontsize=9)
        
        # データ行の色分け
        for i in range(1, len(table_data)):
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor(colors[i-1] if i <= len(colors) else '#CCCCCC')
                cell.set_alpha(0.3)
        
        ax3.set_title('GPU使用統計', fontsize=10, fontweight='bold', pad=20)
        
        filename = f'user_comparison_{user_id:02d}.png'
        output_path = os.path.join(OUTPUT_DIR_USER_COMP, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ ユーザー{user_id}の比較グラフを生成: {filename}")
    
    print("\n✓ 全ユーザー別比較グラフ生成完了（18ファイル）")
    
    # Step3: 特定ユーザー（3～8）のシナリオ別平均待ち時間折れ線グラフ
    print("\n" + "-"*80)
    print("特定ユーザーのシナリオ別平均待ち時間グラフを生成中...")
    print("-"*80)
    
    target_users_table = [0, 1, 2]  # 表形式出力用（0～2のみに変更）
    scenario_labels = ["共有なし", "FCFS\n(先着順)", "所有者\n優先", "所有者優先+\nプリエンプト"]
    scenario_modes = ["no_sharing", "with_sharing", "with_sharing_owner_priority", "with_sharing_owner_preemption"]
    
    # ユーザー0～2の表形式出力
    print("\n  ユーザー0～2の詳細結果を表で生成中...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['ユーザーID'] + scenario_labels]
    for user_id in target_users_table:
        row = [f'ユーザー{user_id}']
        for mode in scenario_modes:
            user_stat = user_stats_by_scenario[mode][user_id]
            avg_wait = user_stat['avg_waiting_time']
            row.append(f'{avg_wait:.2f}秒')
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2] + [0.2]*len(scenario_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2.5)
    
    # ヘッダー行のスタイリング
    for i in range(len(scenario_labels) + 1):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 交互に背景色を付ける
    for i in range(1, len(table_data)):
        for j in range(len(scenario_labels) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('ユーザー0～2の平均待ち時間詳細', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR_TABLES, 'scenario_table_users_0_to_2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ 詳細表を生成: scenario_table_users_0_to_2.png")
    
    # Step4: ユーザー3～8の折れ線グラフ（GPU性能表の3段階色に合わせる）
    print("\n" + "-"*80)
    print("ユーザー3～8のシナリオ別平均待ち時間折れ線グラフを生成中...")
    print("-"*80)
    
    # より濃い色で折れ線グラフを描画
    tier_colors = {
        # グループ1（濃い青） - Tier1-3
        "tier1": "#0d47a1",
        "tier2": "#1565c0", 
        "tier3": "#1976d2",
        # グループ2（明るい黄色） - Tier4-6
        "tier4": "#ffed4e",
        "tier5": "#ffe082",
        "tier6": "#ffd54f",
        # グループ3（濃い赤） - Tier7-9
        "tier7": "#c1272d",
        "tier8": "#d62728",
        "tier9": "#e63946"
    }
    
    # ユーザーとティアのマッピング
    user_to_tier = {
        0: "tier1", 1: "tier2", 2: "tier3",
        3: "tier4", 4: "tier5", 5: "tier6",
        6: "tier7", 7: "tier8", 8: "tier9",
        9: "tier1", 10: "tier2", 11: "tier3",
        12: "tier4", 13: "tier5", 14: "tier6",
        15: "tier7", 16: "tier8", 17: "tier9"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    target_users_line = [3, 4, 5, 6, 7, 8]  # 折れ線グラフ用
    scenario_names = ["共有なし", "FCFS", "所有者優先", "所有者優先+\nプリエンプト"]
    scenario_modes = ["no_sharing", "with_sharing", "with_sharing_owner_priority", "with_sharing_owner_preemption"]
    
    # 各ユーザーのデータを折れ線でプロット
    for user_id in target_users_line:
        tier = user_to_tier[user_id]
        line_color = tier_colors[tier]
        
        avg_wait_values = []
        for mode in scenario_modes:
            user_stat = user_stats_by_scenario[mode][user_id]
            avg_wait_values.append(user_stat['avg_waiting_time'])
        
        ax.plot(range(len(scenario_names)), avg_wait_values, marker='o', label=f'ユーザー{user_id}', 
               color=line_color, linewidth=2.5, markersize=8)
    
    ax.set_xlabel('シナリオ', fontsize=20, fontweight='bold')
    ax.set_ylabel('平均待ち時間（秒）', fontsize=20, fontweight='bold')
    ax.set_title('ユーザー3～8のシナリオ別平均待ち時間推移', fontsize=20, fontweight='bold')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper right', fontsize=14, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR_TABLES, 'users_3_to_8_line_graph.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ 折れ線グラフを生成: users_3_to_8_line_graph.png")

def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*80)
    print("基本シナリオ実行＆グラフ生成")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # ステップ1: タスクパターン生成
        run_task_pattern_generation()
        
        # ステップ2: 基本シナリオ実行
        base_results, all_tasks = run_base_scenarios()
        
        # グラフ生成
        generate_graphs(all_tasks)
        
        # 完了メッセージ
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print("\n" + "="*80)
        print("基本シナリオ実行完了")
        print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {elapsed}")
        print("="*80)
        
        print("\n" + "="*80)
        print("反復最適化ラッパーを実行するには：")
        print("  python simulation_iterative_wrapper.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
