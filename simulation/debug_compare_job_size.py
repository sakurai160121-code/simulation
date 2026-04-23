import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from simulation_with_sharing import SimulatorWithSharing
from simulation_with_sharing_owner_priority import SimulatorWithOwnerPriority
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns
from run_multi_load_scenarios import update_arrival_rates_for_load

config.set_acp_resident_gpu_profiles(count=1, processing_rates=[50000])
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

scenario = {"inference_ratio": 0.5, "training_ratio": 0.5}
update_arrival_rates_for_load(0.05, scenario)
patterns = load_patterns()

sims = {
    "FCFS": SimulatorWithSharing(task_patterns=patterns),
    "Owner Priority": SimulatorWithOwnerPriority(task_patterns=patterns),
    "Preemptive": SimulatorWithOwnerPreemption(task_patterns=patterns),
}

results = {}
for name, sim in sims.items():
    tasks = sim.run()
    results[name] = tasks

print("=" * 100)
print("task_id / job_size / total_work / completion_time comparison")
print("=" * 100)

base = results["FCFS"]
for idx in range(5):
    t0 = base[idx]
    t1 = results["Owner Priority"][idx]
    t2 = results["Preemptive"][idx]
    print(f"\nidx={idx}")
    for name, t in [("FCFS", t0), ("Owner Priority", t1), ("Preemptive", t2)]:
        print(
            f"  {name}: id={t.task_id}, arrival={t.arrival_time:.6f}, job_size={t.job_size:.6f}, total_work={t.total_work:.6f}, "
            f"assigned={t.assigned_time:.6f}, completion={t.completion_time:.6f}, service={t.get_service_time():.6f}, gpu={getattr(t.assigned_gpu, 'gpu_id', None)}"
        )

print("\n差分（FCFS - Preemptive）")
for idx in range(5):
    t0 = base[idx]
    t2 = results["Preemptive"][idx]
    print(f"  {t0.task_id}: completion diff={t0.completion_time - t2.completion_time:.6f}, job_size diff={t0.job_size - t2.job_size:.6f}")
