"""
複数のconfigシナリオを実行するスクリプト
各シナリオごとに設定を変更し、独立した出力ディレクトリに結果を保存
"""

import os
import sys
import shutil
from datetime import datetime

# Windows環境でUnicode出力を有効化
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# シナリオ定義
SCENARIOS = {
    "scenario1": {
        "name": "ベースライン（到着率0.005）",
        "description": "現在の設定：到着率0.005、エポック10",
        "config_changes": []
    },
    "scenario2": {
        "name": "高エポック（エポック200）",
        "description": "エポック数を10→200に変更",
        "config_changes": [
            ('EPOCHS = {i: 10 for i in range(18)}', 'EPOCHS = {i: 200 for i in range(18)}')
        ]
    },
    "scenario3": {
        "name": "性能別到着率",
        "description": "GPU性能に応じて到着率を変化（高性能ほど高頻度）",
        "config_changes": [
            (
                'ARRIVAL_RATES = {str(i): ARRIVAL_RATE for i in range(18)}  # 全ユーザー同じ到着率',
                '''ARRIVAL_RATES = {
    # 低性能GPU（tier1-3）: 低頻度
    "0": 0.003, "9": 0.003,   # tier1
    "1": 0.003, "10": 0.003,  # tier2
    "2": 0.004, "11": 0.004,  # tier3
    # 中性能GPU（tier4-6）: 中頻度
    "3": 0.005, "12": 0.005,  # tier4
    "4": 0.006, "13": 0.006,  # tier5
    "5": 0.006, "14": 0.006,  # tier6
    # 高性能GPU（tier7-9）: 高頻度
    "6": 0.008, "15": 0.008,  # tier7
    "7": 0.009, "16": 0.009,  # tier8
    "8": 0.010, "17": 0.010,  # tier9
}'''
            )
        ]
    }
}


def backup_config():
    """config.pyのバックアップを作成"""
    if os.path.exists('config.py'):
        shutil.copy2('config.py', 'config_original.py')
        print("✓ config.pyをバックアップしました")


def restore_config():
    """config.pyをバックアップから復元"""
    if os.path.exists('config_original.py'):
        shutil.copy2('config_original.py', 'config.py')
        print("✓ config.pyを復元しました")


def apply_config_changes(changes):
    """config.pyに変更を適用"""
    if not changes:
        print("  設定変更なし（デフォルト設定を使用）")
        return
    
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old_text, new_text in changes:
        if old_text in content:
            content = content.replace(old_text, new_text)
            print(f"  ✓ 設定を変更しました")
        else:
            print(f"  ⚠ 変更対象が見つかりませんでした")
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(content)


def modify_output_paths_in_script(scenario_id):
    """run_all_simulations.pyの出力パスを一時的に変更"""
    script_path = 'run_all_simulations.py'
    backup_path = 'run_all_simulations_backup.py'
    
    # バックアップ
    shutil.copy2(script_path, backup_path)
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 出力パスを変更
    replacements = [
        ("OUTPUT_DIR_BASE = './outputs'", f"OUTPUT_DIR_BASE = './outputs/{scenario_id}'"),
        ("OUTPUT_DIR_BASIC = './outputs/basic_scenarios'", f"OUTPUT_DIR_BASIC = './outputs/{scenario_id}/basic_scenarios'"),
        ("OUTPUT_DIR_USER_COMP = './outputs/user_comparisons'", f"OUTPUT_DIR_USER_COMP = './outputs/{scenario_id}/user_comparisons'"),
        ("OUTPUT_DIR_TABLES = './outputs/tables'", f"OUTPUT_DIR_TABLES = './outputs/{scenario_id}/tables'"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)


def modify_iterative_wrapper_paths(scenario_id):
    """simulation_iterative_wrapper.pyの出力パスを一時的に変更"""
    script_path = 'simulation_iterative_wrapper.py'
    backup_path = 'simulation_iterative_wrapper_backup.py'
    
    # バックアップ
    shutil.copy2(script_path, backup_path)
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 出力パスを変更
    replacements = [
        ("OUTPUT_DIR_ITERATIVE = './outputs/iterative_results'", 
         f"OUTPUT_DIR_ITERATIVE = './outputs/{scenario_id}/iterative_results'"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)


def restore_script():
    """run_all_simulations.pyとsimulation_iterative_wrapper.pyを復元"""
    backup_path = 'run_all_simulations_backup.py'
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, 'run_all_simulations.py')
        os.remove(backup_path)
    
    backup_path = 'simulation_iterative_wrapper_backup.py'
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, 'simulation_iterative_wrapper.py')
        os.remove(backup_path)


def run_scenario(scenario_id, scenario_config):
    """単一シナリオを実行"""
    print("\n" + "="*80)
    print(f"【{scenario_id}】{scenario_config['name']}")
    print("="*80)
    print(f"説明: {scenario_config['description']}")
    
    # config.pyに変更を適用
    print("\n設定を適用中...")
    apply_config_changes(scenario_config['config_changes'])
    
    # run_all_simulations.pyの出力パスを変更
    modify_output_paths_in_script(scenario_id)
    
    # run_all_simulations.pyを実行
    print("\n基本シナリオ実行中...")
    import subprocess
    result = subprocess.run(
        [sys.executable, 'run_all_simulations.py'],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n⚠ 基本シナリオでエラーが発生しました")
    
    # simulation_iterative_wrapper.pyの出力パスを変更
    modify_iterative_wrapper_paths(scenario_id)
    
    # simulation_iterative_wrapper.pyを実行
    print("\n反復最適化ラッパー実行中...")
    result2 = subprocess.run(
        [sys.executable, 'simulation_iterative_wrapper.py'],
        capture_output=False,
        text=True
    )
    
    if result2.returncode != 0:
        print(f"\n⚠ 反復最適化ラッパーでエラーが発生しました")
    
    # スクリプトを復元
    restore_script()
    
    if result.returncode == 0 and result2.returncode == 0:
        print(f"\n✓ {scenario_id} 完了")
    else:
        print(f"\n⚠ {scenario_id} で一部エラーが発生しました")


def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*80)
    print("複数シナリオ実行開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # config.pyをバックアップ
    backup_config()
    
    try:
        # 各シナリオを実行
        for scenario_id, scenario_config in SCENARIOS.items():
            run_scenario(scenario_id, scenario_config)
            
            # config.pyを復元（次のシナリオのため）
            restore_config()
    
    except Exception as e:
        print(f"\n⚠ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 最終的にconfig.pyとスクリプトを復元
        restore_config()
        restore_script()
        
        # バックアップファイルを削除
        if os.path.exists('config_original.py'):
            os.remove('config_original.py')
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "="*80)
    print("全シナリオ実行完了")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"経過時間: {elapsed}")
    print("="*80)
    
    # 結果サマリーを表示
    print("\n結果は以下のディレクトリに保存されています:")
    for scenario_id in SCENARIOS.keys():
        print(f"  - ./outputs/{scenario_id}/")


if __name__ == "__main__":
    main()
