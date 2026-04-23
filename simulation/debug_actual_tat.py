import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns
from run_multi_load_scenarios import update_arrival_rates_for_load

# テスト設定：性能50000のACP GPU 1個
config.set_acp_resident_gpu_profiles(count=1, processing_rates=[50000])

# ユーザーGPU性能を低く設定
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

task_patterns_data = load_patterns()

print("="*80)
print("シミュレーション実行: load_rate=0.05 で各方式の TAT を比較")
print("="*80)

scenario = {"inference_ratio": 0.5, "training_ratio": 0.5}

# FCFS
print("\n【FCFS】")
update_arrival_rates_for_load(0.05, scenario)
sim_fcfs = SimulatorWithSharing(task_patterns=task_patterns_data)
tasks_fcfs = sim_fcfs.run()

# Owner Priority
print("\n【Owner Priority】")
update_arrival_rates_for_load(0.05, scenario)
sim_op = SimulatorWithOwnerPriority(task_patterns=task_patterns_data)
tasks_op = sim_op.run()

# Preemptive
print("\n【Preemptive】")
update_arrival_rates_for_load(0.05, scenario)
sim_preempt = SimulatorWithOwnerPreemption(task_patterns=task_patterns_data)
tasks_preempt = sim_preempt.run()

# TAT の計算と比較
acp_tasks_fcfs = [t for t in tasks_fcfs if "acp" in str(t.gpu_id).lower()]
acp_tasks_op = [t for t in tasks_op if "acp" in str(t.gpu_id).lower()]
acp_tasks_preempt = [t for t in tasks_preempt if "acp" in str(t.gpu_id).lower()]

print("\n" + "="*80)
print("結果: ACP GPU タスク統計")
print("="*80)
print(f"FCFS:      {len(acp_tasks_fcfs)} tasks on ACP")
print(f"OP:        {len(acp_tasks_op)} tasks on ACP")
print(f"Preempt:   {len(acp_tasks_preempt)} tasks on ACP")

if acp_tasks_fcfs and acp_tasks_op and acp_tasks_preempt:
    avg_tat_fcfs = sum(t.get_turnaround_time() for t in acp_tasks_fcfs) / len(acp_tasks_fcfs)
    avg_tat_op = sum(t.get_turnaround_time() for t in acp_tasks_op) / len(acp_tasks_op)
    avg_tat_preempt = sum(t.get_turnaround_time() for t in acp_tasks_preempt) / len(acp_tasks_preempt)
    
    print(f"\n平均 TAT:")
    print(f"FCFS:      {avg_tat_fcfs:.2f}秒")
    print(f"OP:        {avg_tat_op:.2f}秒 (差分: {avg_tat_op - avg_tat_fcfs:+.2f}秒)")
    print(f"Preempt:   {avg_tat_preempt:.2f}秒 (差分: {avg_tat_preempt - avg_tat_fcfs:+.2f}秒)")
    
    print(f"\nタスク割り当てセット同一性:")
    users_fcfs = {t.user_id for t in acp_tasks_fcfs}
    users_op = {t.user_id for t in acp_tasks_op}
    users_preempt = {t.user_id for t in acp_tasks_preempt}
    
    if users_fcfs == users_op == users_preempt:
        print(f"✓ ユーザーセット一致: {len(users_fcfs)} ユーザー")
    else:
        print(f"✗ ユーザーセット不一致!")
