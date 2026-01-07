"""
GPU処理能力とタスク量の理論値検証
"""
from config import (
    NUM_USERS,
    SIMULATION_TIME,
    GPU_PERFORMANCE_LEVELS,
    GPU_TIER_ASSIGNMENT,
    TASK_SIZE_MEANS,
    TASK_SIZE_MEAN,
)
from task_patterns import load_patterns
from simulation_with_sharing_owner_preemption import SimulatorWithOwnerPreemption

def calculate_theoretical_capacity():
    """理論的なGPU処理能力を計算"""
    print("="*80)
    print("【理論GPU処理能力】")
    print("="*80)
    
    total_capacity = 0.0
    for tier, users in GPU_TIER_ASSIGNMENT.items():
        rate = GPU_PERFORMANCE_LEVELS[tier]
        num_gpus = len(users)
        tier_capacity = num_gpus * rate * SIMULATION_TIME
        total_capacity += tier_capacity
        print(f"{tier}: {num_gpus}台 × レート{rate} × 時間{SIMULATION_TIME} = {tier_capacity:.1f}")
    
    print(f"\n総GPU処理能力: {total_capacity:.1f}")
    return total_capacity

def calculate_task_workload():
    """タスクの総作業量を計算"""
    print("\n" + "="*80)
    print("【タスク総作業量】")
    print("="*80)
    
    patterns = load_patterns()
    arrivals = patterns.get("arrivals", {})
    job_sizes = patterns.get("job_sizes", {})
    
    total_work = 0.0
    total_tasks = 0
    
    for user_id in range(NUM_USERS):
        user_arrivals = arrivals.get(str(user_id), [])
        user_job_sizes = job_sizes.get(str(user_id), {})
        user_work = 0.0
        
        for arrival in user_arrivals:
            if arrival <= SIMULATION_TIME:
                job_size = user_job_sizes.get(str(arrival), TASK_SIZE_MEANS.get(str(user_id), TASK_SIZE_MEAN))
                user_work += job_size
                total_tasks += 1
        
        total_work += user_work
    
    avg_task_size = total_work / total_tasks if total_tasks > 0 else 0
    print(f"総タスク数: {total_tasks}")
    print(f"タスクサイズ平均: {avg_task_size:.2f}")
    print(f"総作業量: {total_work:.1f}")
    
    return total_work, total_tasks, avg_task_size

def analyze_actual_utilization():
    """実際のシミュレーション結果からGPU利用率を分析"""
    print("\n" + "="*80)
    print("【実シミュレーション結果】")
    print("="*80)
    
    patterns = load_patterns()
    sim = SimulatorWithOwnerPreemption(task_patterns=patterns)
    tasks = sim.run()
    
    # 完了タスクの実作業量を計算
    completed = [t for t in tasks if t.completion_time is not None]
    failed = [t for t in tasks if t.failed]
    
    print(f"\n完了タスク: {len(completed)} / {len(tasks)} ({len(completed)/len(tasks)*100:.2f}%)")
    print(f"失敗タスク: {len(failed)}")
    
    # 完了タスクの実際の作業量を計算
    completed_work = 0.0
    for t in completed:
        # 到着時に設定されたジョブサイズを取得
        patterns = load_patterns()
        job_sizes = patterns.get("job_sizes", {}).get(str(t.user_id), {})
        job_size = job_sizes.get(str(t.arrival_time), TASK_SIZE_MEANS.get(str(t.user_id), TASK_SIZE_MEAN))
        completed_work += job_size
    
    print(f"完了タスクの総作業量: {completed_work:.1f}")
    
    # GPU別の処理量を計算（レート換算した実稼働時間）
    gpu_work_done = {}
    for gpu_id in range(NUM_USERS):
        gpu_tasks = [t for t in completed if t.assigned_gpu.gpu_id == gpu_id]
        total_work = 0.0
        for t in gpu_tasks:
            job_sizes = patterns.get("job_sizes", {}).get(str(t.user_id), {})
            job_size = job_sizes.get(str(t.arrival_time), TASK_SIZE_MEANS.get(str(t.user_id), TASK_SIZE_MEAN))
            total_work += job_size
        gpu_work_done[gpu_id] = total_work
    
    print("\n【GPU別処理作業量】")
    total_work_done = 0.0
    for tier, users in GPU_TIER_ASSIGNMENT.items():
        rate = GPU_PERFORMANCE_LEVELS[tier]
        tier_work = sum(gpu_work_done.get(uid, 0) for uid in users)
        tier_capacity = len(users) * rate * SIMULATION_TIME
        # 作業量をGPU時間に換算（作業量 / レート）
        tier_gpu_time = tier_work / rate
        utilization = tier_gpu_time / (len(users) * SIMULATION_TIME) * 100
        total_work_done += tier_work
        print(f"{tier}: 作業量={tier_work:.1f}, GPU時間={tier_gpu_time:.1f} / {len(users)*SIMULATION_TIME} ({utilization:.1f}%)")
    
    print(f"\n総作業量完了: {total_work_done:.1f} / 理論容量{capacity} ({total_work_done/capacity*100:.1f}%)")

if __name__ == "__main__":
    capacity = calculate_theoretical_capacity()
    workload, num_tasks, avg_size = calculate_task_workload()
    
    print("\n" + "="*80)
    print("【理論値比較】")
    print("="*80)
    print(f"総GPU処理能力: {capacity:.1f}")
    print(f"タスク総作業量: {workload:.1f}")
    print(f"理論最大完了率: {min(100, capacity/workload*100):.2f}%")
    print(f"作業量 / 処理能力 = {workload/capacity:.2f}")
    
    if workload > capacity:
        print(f"\n⚠️ 警告: 総作業量({workload:.1f})が処理能力({capacity:.1f})を超えています！")
        print(f"   超過分: {workload - capacity:.1f}")
    
    print("\n")
    analyze_actual_utilization()
