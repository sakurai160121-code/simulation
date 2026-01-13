"""
所有者優先モデルのタスク処理フロー詳細確認
"""

import numpy as np
from config import NUM_USERS, RANDOM_SEED
from task_patterns import load_patterns, save_patterns
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
import os

np.random.seed(RANDOM_SEED)

if not os.path.exists("task_patterns.json"):
    save_patterns()

patterns = load_patterns()

print("=" * 140)
print("【所有者優先モデルの詳細動作確認】")
print("=" * 140)

sim = SimulatorWithOwnerPriority(task_patterns=patterns)
sim.run()

# 完了タスクの詳細確認
completed_tasks = [t for t in sim.all_tasks if t.completion_time is not None]
print(f"\n総完了タスク: {len(completed_tasks)}")

# ユーザー別の完了タスク内訳
print("\n【ユーザー別完了タスク内訳】")
for user_id in range(NUM_USERS):
    user_completed = [t for t in completed_tasks if t.user_id == user_id]
    if user_completed:
        avg_wait = np.mean([t.start_time - t.arrival_time for t in user_completed])
        avg_service = np.mean([t.completion_time - t.start_time for t in user_completed])
        print(f"User {user_id:2d}: {len(user_completed):3d}タスク | 平均待ち時間: {avg_wait:7.2f} | 平均サービス時間: {avg_service:7.2f}")

# 所有者タスク vs ゲストタスク
print("\n【所有者 vs ゲストの比較（タスクサイズ=100の場合）】")
owner_completed = 0
guest_completed = 0
owner_waits = []
guest_waits = []

for task in completed_tasks:
    # このタスクが割り当たったGPUの所有者を確認
    gpu_id = task.assigned_gpu.gpu_id
    owner_id = sim.gpu_owner[gpu_id]
    
    wait = task.start_time - task.arrival_time
    
    if task.user_id == owner_id:
        # 所有者タスク
        owner_completed += 1
        owner_waits.append(wait)
    else:
        # ゲストタスク
        guest_completed += 1
        guest_waits.append(wait)

print(f"所有者タスク完了: {owner_completed} / 平均待ち時間: {np.mean(owner_waits) if owner_waits else 0:.2f}")
print(f"ゲストタスク完了: {guest_completed} / 平均待ち時間: {np.mean(guest_waits) if guest_waits else 0:.2f}")

# サンプル確認（最初の10タスク）
print("\n【最初の完了した10タスクの詳細】")
print("-" * 140)
print(f"{'ID':>4s} | {'User':>4s} | {'GPU':>4s} | {'所有者':>4s} | {'到着':>8s} | {'開始':>8s} | {'完了':>8s} | {'待ち':>8s} | {'サ時':>8s}")
print("-" * 140)

for i, task in enumerate(completed_tasks[:10]):
    gpu_id = task.assigned_gpu.gpu_id
    owner_id = sim.gpu_owner[gpu_id]
    is_owner = "○" if task.user_id == owner_id else "×"
    wait = task.start_time - task.arrival_time
    service = task.completion_time - task.start_time
    
    print(f"{i:4d} | {task.user_id:4d} | {gpu_id:4d} | {is_owner:>4s} | {task.arrival_time:8.2f} | {task.start_time:8.2f} | {task.completion_time:8.2f} | {wait:8.2f} | {service:8.2f}")
