"""
タスク発生パターンの生成と保存
各シミュレーションで同じパターンを使用するために事前に生成
"""

import numpy as np
import json
from config import NUM_USERS, ARRIVAL_RATE, SIMULATION_TIME, TASK_SIZE_MEANS, BATCH_MULTIPLIER, RANDOM_SEED


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
            # ポアソン過程でタスク到着間隔を生成（指数分布）
            inter_arrival = np.random.exponential(1.0 / ARRIVAL_RATE)
            current_time += inter_arrival
            
            if current_time > SIMULATION_TIME:
                break
            
            arrivals.append(float(current_time))
        
        task_arrivals[str(user_id)] = arrivals
    
    return task_arrivals


def generate_task_sizes(task_arrivals):
    """
    各タスクのサイズ（仕事量）を指数分布で生成
    Returns: dict {user_id: {arrival_time: task_size}}
    """
    np.random.seed(RANDOM_SEED + 1)  # 異なるシードでタスクサイズを生成
    
    task_sizes = {}
    
    for user_id_str, arrivals in task_arrivals.items():
        user_id = int(user_id_str)
        
        # ユーザーのタスクサイズ平均を取得
        # 1タスク=100枚バッチ分にスケール
        mean_size = TASK_SIZE_MEANS[user_id] * BATCH_MULTIPLIER
        
        task_sizes[user_id_str] = {}
        for arrival_time in arrivals:
            # 指数分布でタスクサイズを生成
            size = float(np.random.exponential(mean_size))
            task_sizes[user_id_str][str(arrival_time)] = size
    
    return task_sizes


def save_patterns(filename="task_patterns.json"):
    """タスク発生パターンをファイルに保存"""
    task_arrivals = generate_task_arrivals()
    task_sizes = generate_task_sizes(task_arrivals)
    
    patterns = {
        "arrivals": task_arrivals,
        "sizes": task_sizes,
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
