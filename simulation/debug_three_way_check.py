import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from collections import defaultdict
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns
from run_multi_load_scenarios import update_arrival_rates_for_load, compute_group_avg_tat

# 共有GPU/ACPの条件をユーザー要望に合わせて設定
config.set_acp_resident_gpu_profiles(count=1, processing_rates=[50000])
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

scenario = {"inference_ratio": 0.5, "training_ratio": 0.5}
load_rate = 0.05
update_arrival_rates_for_load(load_rate, scenario)
patterns = load_patterns()

print("=" * 100)
print("1) 発生したタスク時間が3方式で一致しているか")
print("=" * 100)

sims = {
    "FCFS": SimulatorWithSharing(task_patterns=patterns),
    "Owner Priority": SimulatorWithOwnerPriority(task_patterns=patterns),
    "Preemptive": SimulatorWithOwnerPreemption(task_patterns=patterns),
}
results = {}
for name, sim in sims.items():
    tasks = sim.run()
    results[name] = tasks
    print(f"\n[{name}] total tasks = {len(tasks)}")
    for task in tasks[:5]:
        print(
            f"  task_id={task.task_id}, user={task.user_id}, arrival={task.arrival_time:.6f}, "
            f"assigned={task.assigned_time:.6f}, completion={task.completion_time:.6f}, gpu={getattr(task.assigned_gpu, 'gpu_id', None)}"
        )

# arrival_time の一致確認
print("\n到着時刻の比較（task_id 順）")
base = results["FCFS"]
for other_name in ["Owner Priority", "Preemptive"]:
    other = results[other_name]
    same = all(abs(a.arrival_time - b.arrival_time) < 1e-9 for a, b in zip(base, other))
    print(f"  FCFS vs {other_name}: {'一致' if same else '不一致'}")

print("\n" + "=" * 100)
print("2) ACP上のGPUに割り当てられたタスクの assigned_time と completion_time が等しいか")
print("=" * 100)

for name, tasks in results.items():
    acp_tasks = [t for t in tasks if getattr(t.assigned_gpu, 'gpu_id', '').startswith('acp_')]
    print(f"\n[{name}] ACP割当タスク数 = {len(acp_tasks)}")
    if not acp_tasks:
        continue
    same_assigned_completion = all(abs(t.assigned_time - t.completion_time) < 1e-9 for t in acp_tasks if t.assigned_time is not None and t.completion_time is not None)
    print(f"  assigned_time == completion_time: {'YES' if same_assigned_completion else 'NO'}")
    for task in acp_tasks[:5]:
        print(
            f"  task_id={task.task_id}, user={task.user_id}, assigned={task.assigned_time:.6f}, "
            f"completion={task.completion_time:.6f}, TAT={task.get_turnaround_time():.6f}, waiting={task.get_waiting_time():.6f}, service={task.get_service_time():.6f}"
        )

print("\n" + "=" * 100)
print("3) グラフに使われる値の確認")
print("=" * 100)

for name, tasks in results.items():
    analyzer_name = name
    avg_tat = sum(t.get_turnaround_time() for t in tasks if t.get_turnaround_time() is not None) / len([t for t in tasks if t.get_turnaround_time() is not None])
    acp_count = sum(1 for t in tasks if getattr(t.assigned_gpu, 'gpu_id', '').startswith('acp_'))
    print(f"[{analyzer_name}] avg_tat={avg_tat:.6f}, ACP count={acp_count}")

print("\nグラフで表示されるのは stats['avg_tat'] の系列です。")
print("負荷率ごとに No Sharing / FCFS / Owner Priority / Preemptive の平均TATを plot_scenario_results が描画します。")

# 追加でグループ平均TATを出す
for name, tasks in results.items():
    print(f"\n[{name}] group avg TAT")
    for group_name, group in [("low", multi.LOW_PERF_USERS), ("mid", multi.MID_PERF_USERS), ("high", multi.HIGH_PERF_USERS)]:
        print(f"  {group_name}: {compute_group_avg_tat(tasks, group):.6f}")
