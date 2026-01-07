"""
タスク発生パターンの生成と保存
各シミュレーションで同じパターンを使用するために事前に生成
"""

import numpy as np
import json
from config import NUM_USERS, ARRIVAL_RATE, SIMULATION_TIME, RANDOM_SEED


def generate_task_arrivals():
    """
    各ユーザーのタスク発生時刻を生成
    Returns: dict {user_id: [arrival_times]}
    """
    np.random.seed(RANDOM_SEED)
    
    task_arrivals = {}
    
    for user_id in range(NUM_USERS):
        arrivals = []
        current_time = 0.0
        
        while True:
            # ポアソン過程でタスク到着間隔を生成
            inter_arrival = np.random.exponential(1.0 / ARRIVAL_RATE)
            current_time += inter_arrival
            
            if current_time > SIMULATION_TIME:
                break
            
            arrivals.append(float(current_time))
        
        task_arrivals[str(user_id)] = arrivals
    
    return task_arrivals


def generate_service_times(task_arrivals):
    """
    各タスクの処理時間を指数分布で生成
    Returns: dict {user_id: {arrival_time: service_time}}
    """
    from config import GPU_TIER_ASSIGNMENT, GPU_PERFORMANCE_LEVELS
    
    np.random.seed(RANDOM_SEED + 1)  # 異なるシードで処理時間を生成
    
    service_times = {}
    
    for user_id_str, arrivals in task_arrivals.items():
        user_id = int(user_id_str)
        
        # ユーザーの性能ティアを決定
        tier = None
        for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
            if user_id in user_list:
                tier = tier_name
                break
        
        mean_service_time = GPU_PERFORMANCE_LEVELS[tier]
        
        service_times[user_id_str] = {}
        for arrival_time in arrivals:
            # 指数分布で処理時間を生成
            service_time = float(np.random.exponential(mean_service_time))
            service_times[user_id_str][str(arrival_time)] = service_time
    
    return service_times


def save_patterns(filename="task_patterns.json"):
    """タスク発生パターンをファイルに保存"""
    task_arrivals = generate_task_arrivals()
    service_times = generate_service_times(task_arrivals)
    
    patterns = {
        "arrivals": task_arrivals,
        "service_times": service_times,
        "config": {
            "num_users": NUM_USERS,
            "arrival_rate": ARRIVAL_RATE,
            "simulation_time": SIMULATION_TIME,
            "random_seed": RANDOM_SEED,
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"タスクパターンを保存しました: {filename}")
    return patterns


def load_patterns(filename="task_patterns.json"):
    """タスク発生パターンをファイルから読み込み"""
    with open(filename, 'r') as f:
        patterns = json.load(f)
    
    return patterns


if __name__ == "__main__":
    save_patterns()
