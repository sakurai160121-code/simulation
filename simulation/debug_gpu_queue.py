import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from definitions import Task
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns

# テスト設定：性能50000のACP GPU 1個
config.set_acp_resident_gpu_profiles(count=1, processing_rates=[50000])

# ユーザーGPU性能を低く設定
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

task_patterns_data = load_patterns()

print("="*80)
print("テスト: 最初のミニシミュレーション（最初の3タスクまで）")
print("="*80)

# 各シミュレータを定義
def run_mini_simulation(simulator_name, simulator_class):
    print(f"\n【{simulator_name}】")
    sim = simulator_class(task_patterns=task_patterns_data)
    sim.initialize()
    
    # 最初の 3 イベント（task_arrival）を処理
    task_count = 0
    event_count = 0
    
    while task_count < 3 and event_count < 100:
        if not sim.event_queue:
            break
        
        event = sim.event_queue[0]
        
        if event[1] == "task_arrival":
            # イベント処理
            sim.process_next_event()
            event_count += 1
            
            # GPU キューの状態を確認
            acp_gpu = [g for g in sim.shared_gpus if "acp" in str(g.gpu_id).lower()][0]
            print(f"  Event {event_count}: acp_gpu.task_queue size = {len(acp_gpu.task_queue)}")
            
            # キューの最初の3タスクを表示
            for i, task in enumerate(acp_gpu.task_queue[:3]):
                print(f"    Queue[{i}]: user={task.user_id}, task_index={task.task_index}, size={task.job_size:.0f}")
            
            task_count += 1
        else:
            sim.process_next_event()
            event_count += 1

# FCFS でシミュレート
run_mini_simulation("FCFS", SimulatorWithSharing)

# Owner Priority でシミュレート
run_mini_simulation("Owner Priority", SimulatorWithOwnerPriority)

# Preemptive でシミュレート
run_mini_simulation("Preemptive", SimulatorWithOwnerPreemption)

print("\n" + "="*80)
print("GPU キューの追加順序が異なるか確認")
print("="*80)
print("Owner Priority と Preemptive では GPU.add_task() で owner_id を指定。")
print("GPU.add_task() が優先度付けで異なる順序に入れる可能性あり。")
