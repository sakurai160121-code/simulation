"""
所有者優先度の動作検証スクリプト
特定のGPUについてタスク割り当てと実行順序を追跡
"""

import numpy as np
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from config import NUM_USERS, SIMULATION_TIME, RANDOM_SEED, GPU_TIER_ASSIGNMENT, GPU_PERFORMANCE_LEVELS, ARRIVAL_RATES
from task_patterns import load_patterns, save_patterns
import os

np.random.seed(RANDOM_SEED)

# タスクパターンを生成（存在しない場合）または読み込み
if not os.path.exists("task_patterns.json"):
    print("タスクパターンを生成中...")
    save_patterns()

patterns = load_patterns()

# シミュレーション実行（ログ付き）
class DiagnosticSimulator(SimulatorWithOwnerPriority):
    """診断用拡張シミュレータ"""
    
    def __init__(self, task_patterns=None, tracked_gpu_id=None):
        super().__init__(task_patterns)
        self.tracked_gpu_id = tracked_gpu_id or 5  # User 5のGPU（tier2）を追跡
        self.gpu_events = []  # GPUイベントのログ
    
    def process_task_arrival(self, user_id):
        """タスク到着イベント処理（ログ付き）"""
        user = self.users[user_id]
        task = user.create_task(self.current_time)
        self.all_tasks.append(task)
        
        # 最適なGPUを選択
        best_gpu = self.select_best_gpu(user_id)
        task.assigned_gpu = best_gpu
        
        # GPU所有者IDを渡して、所有者のタスクを優先化
        owner_id = self.gpu_owner[best_gpu.gpu_id]
        best_gpu.add_task(task, owner_id=owner_id)
        
        # 追跡対象GPUのイベントをログ
        if best_gpu.gpu_id == self.tracked_gpu_id:
            queue_info = [f"User{t.user_id}" for t in best_gpu.task_queue]
            is_owner = "✓所有者タスク" if user_id == owner_id else "他人のタスク"
            self.gpu_events.append({
                "time": self.current_time,
                "event": f"タスク到着",
                "user_id": user_id,
                "owner_relation": is_owner,
                "queue_length": len(best_gpu.task_queue),
                "queue": queue_info,
            })
        
        # GPUが空いていたら即座に処理開始
        if best_gpu.current_task is None:
            self.start_task_on_gpu(best_gpu, task)
        
        # 次のタスク発生をスケジュール
        arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
        next_arrival_index = len([t for t in user.tasks if t is not None])
        
        if next_arrival_index < len(arrivals):
            next_arrival = arrivals[next_arrival_index]
            if next_arrival <= SIMULATION_TIME:
                self.schedule_event(next_arrival, "task_arrival", user_id)
    
    def start_task_on_gpu(self, gpu, task):
        """GPUでタスクを開始（ログ付き）"""
        super().start_task_on_gpu(gpu, task)
        
        if gpu.gpu_id == self.tracked_gpu_id:
            owner_id = self.gpu_owner[gpu.gpu_id]
            is_owner = "✓所有者タスク" if task.user_id == owner_id else "他人のタスク"
            self.gpu_events.append({
                "time": self.current_time,
                "event": f"処理開始",
                "user_id": task.user_id,
                "owner_relation": is_owner,
                "service_time": gpu.finish_time - self.current_time,
                "finish_time": gpu.finish_time,
            })
    
    def process_gpu_finish(self, gpu_id):
        """GPU処理完了イベント処理（ログ付き）"""
        # GPU IDで対応するGPUを探す
        gpu = None
        for g in self.shared_gpus:
            if g.gpu_id == gpu_id:
                gpu = g
                break
        
        if gpu is None:
            return
        
        # 現在のタスクを完了
        task = gpu.current_task
        if gpu_id == self.tracked_gpu_id:
            self.gpu_events.append({
                "time": self.current_time,
                "event": f"処理完了",
                "user_id": task.user_id,
                "queue_length": len(gpu.task_queue),
            })
        
        task.completion_time = self.current_time
        gpu.current_task = None
        
        # キューに次のタスクがあれば処理開始
        if len(gpu.task_queue) > 0:
            next_task = gpu.task_queue.pop(0)
            self.start_task_on_gpu(gpu, next_task)

# 診断実行
print("=" * 100)
print("【所有者優先度機能の動作診断】")
print("=" * 100)

sim = DiagnosticSimulator(task_patterns=patterns, tracked_gpu_id=5)
tasks = sim.run()

# 追跡対象GPU（User 5）の情報を表示
tracked_gpu_id = 5
owner_id = sim.gpu_owner[tracked_gpu_id]
owner_lambda = ARRIVAL_RATES.get(str(owner_id), 1.0)
owner_gpu_info = None
for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
    if owner_id in user_list:
        owner_gpu_info = (tier_name, GPU_PERFORMANCE_LEVELS[tier_name])
        break

print(f"\n【追跡対象GPU: User {tracked_gpu_id}のGPU】")
print(f"  所有者: User {owner_id}")
print(f"  ティア: {owner_gpu_info[0]}, 処理レート: {owner_gpu_info[1]}")
print(f"  所有者の到着率(λ): {owner_lambda}")
print()

# イベントを時系列で表示（最初の20件と最後の10件）
print(f"\n【GPU {tracked_gpu_id}のイベントログ（合計{len(sim.gpu_events)}件）】")
print("=" * 100)
print(f"{'時刻':<10} {'イベント':<15} {'ユーザーID':<12} {'関係':<20} {'詳細':<45}")
print("=" * 100)

# 最初の15件
for i, event in enumerate(sim.gpu_events[:15]):
    time_str = f"{event['time']:.2f}"
    user_str = f"User {int(event['user_id'])}"
    owner_rel = event.get('owner_relation', '')
    
    if event['event'] == 'タスク到着':
        detail = f"キュー長: {event['queue_length']}, キュー: {event['queue']}"
    elif event['event'] == '処理開始':
        detail = f"処理時間: {event['service_time']:.2f}, 完了予定: {event['finish_time']:.2f}"
    else:  # 処理完了
        detail = f"キュー長: {event['queue_length']}"
    
    print(f"{time_str:<10} {event['event']:<15} {user_str:<12} {owner_rel:<20} {detail:<45}")

print("\n...")
print(f"\n【最後の10件のイベント】")
print("=" * 100)
for i, event in enumerate(sim.gpu_events[-10:]):
    time_str = f"{event['time']:.2f}"
    user_str = f"User {int(event['user_id'])}"
    owner_rel = event.get('owner_relation', '')
    
    if event['event'] == 'タスク到着':
        detail = f"キュー長: {event['queue_length']}, キュー: {event['queue']}"
    elif event['event'] == '処理開始':
        detail = f"処理時間: {event['service_time']:.2f}, 完了予定: {event['finish_time']:.2f}"
    else:  # 処理完了
        detail = f"キュー長: {event['queue_length']}"
    
    print(f"{time_str:<10} {event['event']:<15} {user_str:<12} {owner_rel:<20} {detail:<45}")

# 割り込みの発生を検出
print("\n" + "=" * 100)
print("【割り込み検出】")
print("=" * 100)

interruption_count = 0
for i in range(1, len(sim.gpu_events)):
    curr = sim.gpu_events[i]
    prev = sim.gpu_events[i-1]
    
    # 処理開始イベントで、前のイベントが別ユーザーの処理開始で、かつ所有者タスクの場合
    if curr['event'] == '処理開始' and prev['event'] == '処理開始':
        if curr['user_id'] == owner_id and prev['user_id'] != owner_id:
            interruption_count += 1
            print(f"時刻 {prev['time']:.2f}～{curr['time']:.2f}: "
                  f"User {int(prev['user_id'])}のタスク処理中に、"
                  f"所有者User {owner_id}のタスク到着により割り込み発生！")

if interruption_count == 0:
    print("割り込みが検出されませんでした（所有者タスクは即座に実行されます）")
else:
    print(f"\n合計 {interruption_count} 回の割り込みが発生しました")

# 最後に、所有者vs他人のGPU選択率を統計
print("\n" + "=" * 100)
print("【ユーザーのGPU選択パターン】")
print("=" * 100)

owner_gpu_selections = {}
other_gpu_selections = {}

for user_id in range(NUM_USERS):
    user_tasks = [t for t in tasks if t.user_id == user_id and t.assigned_gpu is not None]
    
    owner_count = 0
    other_count = 0
    
    for task in user_tasks:
        if task.assigned_gpu.gpu_id == user_id:
            owner_count += 1
        else:
            other_count += 1
    
    total = owner_count + other_count
    if total > 0:
        owner_rate = owner_count / total * 100
        print(f"User {user_id}: 自分のGPU選択 {owner_rate:.1f}% ({owner_count}/{total}), "
              f"他人のGPU選択 {100-owner_rate:.1f}% ({other_count}/{total})")
