import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns
import heapq

# テスト設定：性能50000のACP GPU 2個
config.set_acp_resident_gpu_profiles(count=2, processing_rates=[50000, 50000])

# ユーザーGPU性能を低く設定
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

task_patterns_data = load_patterns()

print("="*80)
print("テスト: 最初の10タスクの割り当て先と予測完了時刻")
print("="*80)

# FCFS
print("\n【FCFS】")
sim_fcfs = SimulatorWithSharing(task_patterns=task_patterns_data)
sim_fcfs.initialize()

fcfs_assignment = []
for _ in range(10):
    if not sim_fcfs.event_queue:
        break
    time, event_type, data = heapq.heappop(sim_fcfs.event_queue)
    sim_fcfs.current_time = time
    
    if event_type == "task_arrival":
        user_id = data
        task_index = sim_fcfs.users[user_id].task_count
        task_type, job_size = sim_fcfs.resolve_task_profile(user_id, task_index)
        
        best_gpu = sim_fcfs.select_best_gpu(task_type)
        if best_gpu:
            t_complete = sim_fcfs.predict_completion_time(best_gpu, task_type)
            is_acp = "acp" in str(best_gpu.gpu_id)
            fcfs_assignment.append((user_id, task_index, best_gpu.gpu_id, t_complete, is_acp))
            print(f"  Task {len(fcfs_assignment):2d}: User {user_id:2d}, GPU {best_gpu.gpu_id:5s}, completion={t_complete:8.2f}秒, ACP={is_acp}")

# Owner Priority
print("\n【Owner Priority】")
sim_op = SimulatorWithOwnerPriority(task_patterns=task_patterns_data)
sim_op.initialize()

op_assignment = []
for _ in range(10):
    if not sim_op.event_queue:
        break
    time, event_type, data = heapq.heappop(sim_op.event_queue)
    sim_op.current_time = time
    
    if event_type == "task_arrival":
        user_id = data
        task_index = sim_op.users[user_id].task_count
        task_type, job_size = sim_op.resolve_task_profile(user_id, task_index)
        
        best_gpu = sim_op.select_best_gpu(user_id, task_type)
        if best_gpu:
            owner_id = sim_op.gpu_owner.get(best_gpu.gpu_id)
            if owner_id == user_id:
                t_complete = sim_op.predict_completion_time_own_gpu(best_gpu, user_id, task_type)
            else:
                t_complete = sim_op.predict_completion_time_other_gpu(best_gpu, user_id, task_type)
            
            is_acp = "acp" in str(best_gpu.gpu_id)
            op_assignment.append((user_id, task_index, best_gpu.gpu_id, t_complete, is_acp))
            print(f"  Task {len(op_assignment):2d}: User {user_id:2d}, GPU {best_gpu.gpu_id:5s}, completion={t_complete:8.2f}秒, ACP={is_acp}, owner={owner_id}")

# Preemptive
print("\n【Preemptive】")
sim_preempt = SimulatorWithOwnerPreemption(task_patterns=task_patterns_data)
sim_preempt.initialize()

preempt_assignment = []
for _ in range(10):
    if not sim_preempt.event_queue:
        break
    time, event_type, data = heapq.heappop(sim_preempt.event_queue)
    sim_preempt.current_time = time
    
    if event_type == "task_arrival":
        user_id = data
        task_index = sim_preempt.users[user_id].task_count
        task_type, job_size = sim_preempt.resolve_task_profile(user_id, task_index)
        
        best_gpu = sim_preempt.select_best_gpu_for_new(user_id, job_size, task_type)
        if best_gpu:
            owner_id = sim_preempt.gpu_owner.get(best_gpu.gpu_id)
            if owner_id == user_id:
                t_complete = sim_preempt.predict_completion_time_own_gpu(best_gpu, user_id, new_task_type=task_type)
            else:
                t_complete = sim_preempt.predict_completion_time_other_gpu(best_gpu, user_id, new_task_type=task_type, include_penalty=False)
            
            is_acp = "acp" in str(best_gpu.gpu_id)
            preempt_assignment.append((user_id, task_index, best_gpu.gpu_id, t_complete, is_acp))
            print(f"  Task {len(preempt_assignment):2d}: User {user_id:2d}, GPU {best_gpu.gpu_id:5s}, completion={t_complete:8.2f}秒, ACP={is_acp}, owner={owner_id}")

print("\n" + "="*80)
print("比較: 同じタスク位置での割り当て先と完了時刻")
print("="*80)

for i in range(min(len(fcfs_assignment), len(op_assignment), len(preempt_assignment))):
    user_f, _, gpu_f, t_f, acp_f = fcfs_assignment[i]
    user_o, _, gpu_o, t_o, acp_o = op_assignment[i]
    user_p, _, gpu_p, t_p, acp_p = preempt_assignment[i]
    
    print(f"\nTask {i+1}:")
    print(f"  FCFS:      User {user_f} → GPU {gpu_f:5s}, completion={t_f:8.2f}秒")
    print(f"  OP:        User {user_o} → GPU {gpu_o:5s}, completion={t_o:8.2f}秒")
    print(f"  Preempt:   User {user_p} → GPU {gpu_p:5s}, completion={t_p:8.2f}秒")
    
    # 割り当て先が異なるか確認
    if gpu_f != gpu_o or gpu_f != gpu_p:
        print(f"  ⚠️  割り当て先が異なる!")
    if abs(t_f - t_o) > 0.01 or abs(t_f - t_p) > 0.01:
        print(f"  ⚠️  完了時刻が異なる!")

print("\n" + "="*80)
print("結論:")
print("="*80)
print("タスク割り当ての流れを確認。")
print("ACP GPU に全タスクが割り当てられるなら、完了時刻は同じはずが、異なるか確認。")
