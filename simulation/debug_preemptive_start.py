import sys
sys.path.insert(0, 'c:/Users/baske/OneDrive/ドキュメント/研究室/卒論/simulation')

import config
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption
from task_patterns import load_patterns
from run_multi_load_scenarios import update_arrival_rates_for_load

config.set_acp_resident_gpu_profiles(count=1, processing_rates=[50000])
for tier in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier7", "tier8", "tier9"]:
    config.GPU_PERFORMANCE_LEVELS[tier] = 100

scenario = {"inference_ratio": 0.5, "training_ratio": 0.5}
update_arrival_rates_for_load(0.05, scenario)
patterns = load_patterns()

sim = SimulatorWithOwnerPreemption(task_patterns=patterns)
sim.initialize()

original_start = sim.start_task_on_gpu

def traced_start(gpu, task):
    if task.task_id in {"user16_task0", "user4_task0"}:
        print(f"TRACE start: task={task.task_id}, job_size={task.job_size}, remaining_work={task.remaining_work}, rate={gpu.processing_rate}, current_time={sim.current_time}")
    original_start(gpu, task)
    if task.task_id in {"user16_task0", "user4_task0"}:
        print(f"TRACE finish scheduled: task={task.task_id}, gpu.finish_time={gpu.finish_time}, delta={gpu.finish_time - sim.current_time}")

sim.start_task_on_gpu = traced_start

count = 0
while sim.event_queue and count < 5:
    time, event_type, data = sim.event_queue[0]
    sim.current_time = time
    time, event_type, data = sim.event_queue.pop(0)
    if event_type == "task_arrival":
        sim.process_task_arrival(data)
    elif event_type == "gpu_finish":
        sim.process_gpu_finish(data)
    count += 1
