"""
GPU使用状況分析スクリプト（共有ありシミュレーション版）
所有者優先度がない場合のGPU使用パターンを分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    NUM_USERS,
    SIMULATION_TIME,
    RANDOM_SEED,
    GPU_TIER_ASSIGNMENT,
    GPU_PERFORMANCE_LEVELS,
)
from task_patterns import load_patterns, save_patterns
from simulation_with_sharing import SimulatorWithSharing
import os

np.random.seed(RANDOM_SEED)

# タスクパターンを生成（存在しない場合）または読み込み
if not os.path.exists("task_patterns.json"):
    print("タスクパターンを生成中...")
    save_patterns()

patterns = load_patterns()

# シミュレーション実行
print("=" * 120)
print("【GPU使用状況分析（共有あり・所有者優先度なし）】")
print("=" * 120)

sim = SimulatorWithSharing(task_patterns=patterns)
tasks = sim.run()

print(f"\nシミュレーション完了：発生タスク {len(tasks)} 件\n")

# =====================================================
# 分析1: GPU別処理タスク数
# =====================================================
print("=" * 120)
print("【GPU別処理タスク統計】")
print("=" * 120)

gpu_stats = []
for gpu_id in range(NUM_USERS):
    # GPU所有者情報
    owner_id = sim.gpu_owner[gpu_id]
    tier = None
    for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
        if gpu_id in user_list:
            tier = tier_name
            break
    processing_rate = GPU_PERFORMANCE_LEVELS[tier]
    
    # このGPUで処理されたタスク
    gpu_tasks = [t for t in tasks if t.assigned_gpu is not None and t.assigned_gpu.gpu_id == gpu_id]
    completed_tasks = [t for t in gpu_tasks if t.completion_time is not None]
    
    # 処理ユーザー別集計
    user_counts = {}
    for task in completed_tasks:
        user_id = task.user_id
        user_counts[user_id] = user_counts.get(user_id, 0) + 1
    
    # 待ち時間統計
    waiting_times = [t.get_waiting_time() for t in completed_tasks if t.get_waiting_time() is not None]
    avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
    
    gpu_stats.append({
        "gpu_id": gpu_id,
        "owner_id": owner_id,
        "tier": tier,
        "processing_rate": processing_rate,
        "assigned_tasks": len(gpu_tasks),
        "completed_tasks": len(completed_tasks),
        "unique_users": len(user_counts),
        "avg_waiting_time": avg_waiting_time,
        "user_counts": user_counts,
    })

# DataFrameで表示
df_gpu = pd.DataFrame(gpu_stats)
df_display = df_gpu[["gpu_id", "owner_id", "tier", "processing_rate", "assigned_tasks", "completed_tasks", "unique_users", "avg_waiting_time"]].copy()
df_display.columns = ["GPUID", "所有者", "ティア", "処理レート", "割り当て", "完了", "利用ユーザー数", "平均待ち時間"]

print(df_display.to_string(index=False))
print()

# =====================================================================
# 分析2: GPU別ユーザー割り当てマトリックス
# =====================================================================
print("=" * 120)
print("【GPU別ユーザー割り当てマトリックス（完了タスク数）】")
print("=" * 120)

# マトリックス作成
matrix_data = []
for gpu_id in range(NUM_USERS):
    row = {"GPU": f"GPU{gpu_id}"}
    gpu_obj = [g for g in sim.shared_gpus if g.gpu_id == gpu_id][0]
    owner_id = sim.gpu_owner[gpu_id]
    
    for user_id in range(NUM_USERS):
        gpu_tasks = [t for t in tasks if t.assigned_gpu is not None and t.assigned_gpu.gpu_id == gpu_id]
        user_tasks = [t for t in gpu_tasks if t.user_id == user_id and t.completion_time is not None]
        count = len(user_tasks)
        
        # 所有者のタスクはマークを付ける
        if user_id == owner_id and count > 0:
            row[f"User{user_id}"] = f"{count}*"
        else:
            row[f"User{user_id}"] = str(count) if count > 0 else "-"
    
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data)
print(df_matrix.to_string(index=False))
print("\n（*は所有者のタスク）")
print()

# =====================================================================
# 分析3: ユーザーから見たGPU使用パターン
# =====================================================================
print("=" * 120)
print("【ユーザー別GPU使用パターン（完了タスク数）】")
print("=" * 120)

user_gpu_pattern = []
for user_id in range(NUM_USERS):
    row = {"User": f"User{user_id}"}
    user_tasks = [t for t in tasks if t.user_id == user_id and t.completion_time is not None]
    
    for gpu_id in range(NUM_USERS):
        gpu_tasks = [t for t in user_tasks if t.assigned_gpu.gpu_id == gpu_id]
        count = len(gpu_tasks)
        if count > 0:
            row[f"GPU{gpu_id}"] = str(count)
        else:
            row[f"GPU{gpu_id}"] = "-"
    
    user_gpu_pattern.append(row)

df_user_gpu = pd.DataFrame(user_gpu_pattern)
print(df_user_gpu.to_string(index=False))
print()

# =====================================================================
# 分析4: GPU利用率（どのGPUがどれだけ使われたか）
# =====================================================================
print("=" * 120)
print("【GPU利用率統計】")
print("=" * 120)

utilization_stats = []
total_completed = len([t for t in tasks if t.completion_time is not None])

for gpu_id in range(NUM_USERS):
    owner_id = sim.gpu_owner[gpu_id]
    gpu_tasks = [t for t in tasks if t.assigned_gpu is not None and t.assigned_gpu.gpu_id == gpu_id]
    completed_tasks = [t for t in gpu_tasks if t.completion_time is not None]
    
    count = len(completed_tasks)
    rate = count / total_completed * 100 if total_completed > 0 else 0
    
    # 所有者と他人のタスク数
    owner_tasks = [t for t in completed_tasks if t.user_id == owner_id]
    other_tasks = [t for t in completed_tasks if t.user_id != owner_id]
    
    utilization_stats.append({
        "gpu_id": gpu_id,
        "owner_id": owner_id,
        "completed": count,
        "rate": rate,
        "owner_tasks": len(owner_tasks),
        "other_tasks": len(other_tasks),
    })

df_util = pd.DataFrame(utilization_stats)
df_util["rate_str"] = df_util["rate"].apply(lambda x: f"{x:.1f}%")
df_util_display = df_util[["gpu_id", "owner_id", "completed", "rate_str", "owner_tasks", "other_tasks"]].copy()
df_util_display.columns = ["GPUID", "所有者", "完了タスク数", "全体比率", "所有者タスク", "他人タスク"]

print(df_util_display.to_string(index=False))
print(f"\n平均: {df_util['rate'].mean():.1f}% （均等配分なら {100/NUM_USERS:.1f}%）")
print()

# =====================================================================
# グラフ1: GPU別完了タスク数
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("GPU使用状況分析（共有あり・所有者優先度なし）", fontsize=16, fontweight='bold')

# 左: GPU別完了タスク数（棒）＋ 利用率（折れ線・右軸）
ax1 = axes[0]
colors = ['steelblue' if sim.gpu_owner[i] == i else 'lightblue' for i in range(NUM_USERS)]
ax1.bar(df_util['gpu_id'], df_util['completed'], color=colors, alpha=0.7, label='完了タスク数')
ax1.axhline(y=df_util['completed'].mean(), color='red', linestyle='--', label=f'平均(タスク): {df_util["completed"].mean():.0f}')
ax1.set_xlabel('GPU ID')
ax1.set_ylabel('完了タスク数')
ax1.set_title('GPU別完了タスク数')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(loc='upper left')

# 右: ユーザー別GPU選択パターン（自分/他人 スタックバー）
ax2 = axes[1]
user_gpu_dist = []
for user_id in range(NUM_USERS):
    user_tasks = [t for t in tasks if t.user_id == user_id and t.completion_time is not None]
    own_gpu_count = len([t for t in user_tasks if t.assigned_gpu.gpu_id == user_id])
    other_gpu_count = len(user_tasks) - own_gpu_count
    user_gpu_dist.append({'own': own_gpu_count, 'other': other_gpu_count})

own_counts = [d['own'] for d in user_gpu_dist]
other_counts = [d['other'] for d in user_gpu_dist]

ax2.bar(range(NUM_USERS), own_counts, label='自分のGPU', alpha=0.7)
ax2.bar(range(NUM_USERS), other_counts, bottom=own_counts, label='他人のGPU', alpha=0.7)
ax2.set_xlabel('ユーザーID')
ax2.set_ylabel('完了タスク数')
ax2.set_title('ユーザー別GPU選択パターン（自分/他人）')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./gpu_usage_analysis_with_sharing.png', dpi=300, bbox_inches='tight')
print("グラフを保存しました: ./gpu_usage_analysis_with_sharing.png")
plt.close()

print("\n" + "=" * 120)
print("【サマリー】")
print("=" * 120)
print(f"総完了タスク: {total_completed}")
print(f"GPU平均利用率: {df_util['rate'].mean():.1f}%")
print(f"最も使われたGPU: GPU {df_util.loc[df_util['completed'].idxmax(), 'gpu_id']:.0f} ({df_util['completed'].max():.0f}タスク)")
print(f"最も使われないGPU: GPU {df_util.loc[df_util['completed'].idxmin(), 'gpu_id']:.0f} ({df_util['completed'].min():.0f}タスク)")
print(f"\n【重要な違い】")
print(f"所有者タスク平均: {df_util['owner_tasks'].mean():.0f}")
print(f"他人タスク平均: {df_util['other_tasks'].mean():.0f}")
