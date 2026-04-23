import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns

# テスト設定：性能50000のACP GPU 2個
config.set_acp_resident_gpu_profiles(count=2, processing_rates=[50000, 50000])

# ユーザーGPU性能を低く設定（比較用）
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

task_patterns_data = load_patterns()

print("="*80)
print("GPU設定確認:")
print("="*80)
print(f"ACP GPU Count: {config.ACP_RESIDENT_GPU_COUNT}")
print(f"ACP GPU Profiles: {config.ACP_RESIDENT_GPU_PROFILES}")
acp_specs = config.get_acp_resident_gpu_specs()
print(f"ACP specs: {acp_specs}")

# 各方式でシミュレータ初期化
print("\n" + "="*80)
print("方式別のGPUリスト:")
print("="*80)

# FCFS
sim_fcfs = SimulatorWithSharing(task_patterns=task_patterns_data)
sim_fcfs.initialize()
print("\n【FCFS】 shared_gpus (最初の5個):")
for i, gpu in enumerate(sim_fcfs.shared_gpus[:5]):
    print(f"  GPU {i}: id={gpu.gpu_id}, rate={gpu.processing_rate}")

# Owner Priority
sim_op = SimulatorWithOwnerPriority(task_patterns=task_patterns_data)
sim_op.initialize()
print("\n【Owner Priority】 shared_gpus (最初の5個):")
for i, gpu in enumerate(sim_op.shared_gpus[:5]):
    owner = sim_op.gpu_owner.get(gpu.gpu_id)
    print(f"  GPU {i}: id={gpu.gpu_id}, rate={gpu.processing_rate}, owner={owner}")

# Preemptive
sim_preempt = SimulatorWithOwnerPreemption(task_patterns=task_patterns_data)
sim_preempt.initialize()
print("\n【Preemptive】 shared_gpus (最初の5個):")
for i, gpu in enumerate(sim_preempt.shared_gpus[:5]):
    owner = sim_preempt.gpu_owner.get(gpu.gpu_id)
    print(f"  GPU {i}: id={gpu.gpu_id}, rate={gpu.processing_rate}, owner={owner}")

# テスト：最初のいくつかのタスク割当で predict_completion_time を比較
print("\n" + "="*80)
print("予測完了時刻の計算差分（ACP GPU に対して）:")
print("="*80)

# ACP GPU を取得
acp_gpu_fcfs = [g for g in sim_fcfs.shared_gpus if "acp" in str(g.gpu_id).lower()][0]
acp_gpu_op = [g for g in sim_op.shared_gpus if "acp" in str(g.gpu_id).lower()][0]
acp_gpu_preempt = [g for g in sim_preempt.shared_gpus if "acp" in str(g.gpu_id).lower()][0]

print(f"\nACP GPU (FCFS): {acp_gpu_fcfs.gpu_id}, rate={acp_gpu_fcfs.processing_rate}")
print(f"ACP GPU (OP):   {acp_gpu_op.gpu_id}, rate={acp_gpu_op.processing_rate}")
print(f"ACP GPU (PRE):  {acp_gpu_preempt.gpu_id}, rate={acp_gpu_preempt.processing_rate}")

# User GPU も取得（比較用）
user_gpu_fcfs = [g for g in sim_fcfs.shared_gpus if "acp" not in str(g.gpu_id).lower()][0]
user_gpu_op = [g for g in sim_op.shared_gpus if "acp" not in str(g.gpu_id).lower()][0]

print(f"\nUser GPU (FCFS): {user_gpu_fcfs.gpu_id}, rate={user_gpu_fcfs.processing_rate}")
print(f"User GPU (OP):   {user_gpu_op.gpu_id}, rate={user_gpu_op.processing_rate}")

# predict_completion_time を比較（新規タスク, キューが空のとき）
print("\n" + "="*80)
print("新規タスク到着時の予測完了時刻（キュー空）:")
print("="*80)

print("\n【FCFS】")
for task_type in ["inference"]:
    t_acp = sim_fcfs.predict_completion_time(acp_gpu_fcfs, task_type)
    t_user = sim_fcfs.predict_completion_time(user_gpu_fcfs, task_type)
    print(f"  Task type={task_type}:")
    print(f"    ACP GPU:  {t_acp:.2f}秒")
    print(f"    User GPU: {t_user:.2f}秒")

print("\n【Owner Priority】(User 8):")
for task_type in ["inference"]:
    # ACP GPU に対する計算
    t_acp = sim_op.predict_completion_time_other_gpu(acp_gpu_op, user_id=8, new_task_type=task_type)
    # User GPU (自分のGPU) に対する計算
    t_user = sim_op.predict_completion_time_own_gpu(user_gpu_op, user_id=8, new_task_type=task_type)
    # Owner utilization 確認
    rho_own = sim_op.get_owner_utilization(acp_gpu_op)
    mu_eff_acp = sim_op.get_effective_processing_rate(acp_gpu_op, user_id=8)
    print(f"  Task type={task_type}:")
    print(f"    ACP GPU:  {t_acp:.2f}秒")
    print(f"      - owner_id={sim_op.gpu_owner.get(acp_gpu_op.gpu_id)} → effective_rate={mu_eff_acp:.1f}")
    print(f"    User GPU: {t_user:.2f}秒")

print("\n【Preemptive】(User 8):")
for task_type in ["inference"]:
    # ACP GPU に対する計算
    t_acp = sim_preempt.predict_completion_time_other_gpu(acp_gpu_preempt, user_id=8, new_task_type=task_type, include_penalty=False)
    # User GPU (自分のGPU) に対する計算
    t_user = sim_preempt.predict_completion_time_own_gpu(acp_gpu_preempt, user_id=8, new_task_type=task_type)
    # Owner utilization 確認
    rho_own = sim_preempt.get_owner_utilization(acp_gpu_preempt)
    mu_eff_acp = sim_preempt.get_effective_processing_rate(acp_gpu_preempt, user_id=8)
    print(f"  Task type={task_type}:")
    print(f"    ACP GPU:  {t_acp:.2f}秒")
    print(f"      - owner_id={sim_preempt.gpu_owner.get(acp_gpu_preempt.gpu_id)} → effective_rate={mu_eff_acp:.1f}")
    print(f"    User GPU own: {t_user:.2f}秒")

# User 0 の場合も確認（User 0 は ACP GPU の owner ではない）
print("\n【Owner Priority】(User 0 - ACP非所有):")
for task_type in ["inference"]:
    t_acp = sim_op.predict_completion_time_other_gpu(acp_gpu_op, user_id=0, new_task_type=task_type)
    owner_id = sim_op.gpu_owner.get(acp_gpu_op.gpu_id)
    mu_eff = sim_op.get_effective_processing_rate(acp_gpu_op, user_id=0)
    print(f"  ACP GPU: {t_acp:.2f}秒, owner={owner_id}, effective_rate={mu_eff:.1f}")

print("\n【Preemptive】(User 0 - ACP非所有):")
for task_type in ["inference"]:
    t_acp = sim_preempt.predict_completion_time_other_gpu(acp_gpu_preempt, user_id=0, new_task_type=task_type, include_penalty=False)
    owner_id = sim_preempt.gpu_owner.get(acp_gpu_preempt.gpu_id)
    mu_eff = sim_preempt.get_effective_processing_rate(acp_gpu_preempt, user_id=0)
    print(f"  ACP GPU: {t_acp:.2f}秒, owner={owner_id}, effective_rate={mu_eff:.1f}")

print("\n" + "="*80)
print("結論:")
print("="*80)
print("ACP GPU (owner=None) での effective_rate 計算が異なるか確認。")
print("特に Owner Priority と Preemptive での get_owner_utilization() の値を比較。")
