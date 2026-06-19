"""
タスク発生パターンの生成と保存
各シミュレーションで同じパターンを使用するために事前に生成
"""


import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
import numpy as np
import json
from simulation.core import config


def generate_task_arrivals():
    """
    各ユーザーのタスク発生時刻を生成
    Returns: dict {user_id: [arrival_times]}
    """
    np.random.seed(config.RANDOM_SEED)
    
    task_arrivals = {}
    
    for user_id in range(config.NUM_USERS):
        arrivals = []
        current_time = 0.0
        arrival_rate = config.ARRIVAL_RATES.get(str(user_id), config.ARRIVAL_RATE)

        # 到着率が不正な場合はタスクを生成しない（エラー耐性）
        if arrival_rate is None or arrival_rate <= 0:
            task_arrivals[str(user_id)] = arrivals
            continue
        
        while True:
            # ポアソン過程でタスク到着間隔を生成（指数分布）
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            
            if current_time > config.SIMULATION_TIME:
                break
            
            arrivals.append(float(current_time))
        
        task_arrivals[str(user_id)] = arrivals
    
    return task_arrivals


def generate_task_types(task_arrivals, scenario):
    """
    シナリオ比率に従って各タスクの種別を生成
    Returns: dict {user_id: {arrival_time: "inference"|"training"}}
    """
    np.random.seed(config.RANDOM_SEED + 2)

    training_ratio = float(scenario.get("training_ratio", 0.0))
    training_ratio = min(max(training_ratio, 0.0), 1.0)
    user_training_ratios = scenario.get("user_training_ratios", {})

    def get_user_training_ratio(user_id_str):
        if isinstance(user_training_ratios, dict):
            ratio = user_training_ratios.get(user_id_str, training_ratio)
        elif isinstance(user_training_ratios, list):
            try:
                ratio = user_training_ratios[int(user_id_str)]
            except (ValueError, IndexError):
                ratio = training_ratio
        else:
            ratio = training_ratio
        return min(max(float(ratio), 0.0), 1.0)

    task_types = {}
    for user_id_str, arrivals in task_arrivals.items():
        user_ratio = get_user_training_ratio(user_id_str)
        task_types[user_id_str] = {}
        for arrival_time in arrivals:
            task_type = "training" if np.random.random() < user_ratio else "inference"
            task_types[user_id_str][str(arrival_time)] = task_type

    return task_types


def _sample_task_size(task_type):
    """タスク種別に応じたログ正規分布からタスクサイズをサンプル。"""
    dist = config.TASK_SIZE_DISTRIBUTION.get(task_type, config.TASK_SIZE_DISTRIBUTION["inference"])
    sampled = np.random.lognormal(mean=dist["lognormal_mean"], sigma=dist["lognormal_sigma"])
    clipped = np.clip(sampled, dist["clip_min"], dist["clip_max"])
    return float(clipped)


def generate_task_sizes(task_arrivals, task_types):
    """
    各タスクのサイズ（仕事量）をタスク種別別ログ正規分布で生成
    Returns: dict {user_id: {arrival_time: task_size}}
    """
    np.random.seed(config.RANDOM_SEED + 1)  # 異なるシードでタスクサイズを生成
    
    task_sizes = {}
    
    for user_id_str, arrivals in task_arrivals.items():
        task_sizes[user_id_str] = {}
        for arrival_time in arrivals:
            task_type = task_types.get(user_id_str, {}).get(str(arrival_time), "inference")
            size = _sample_task_size(task_type)
            task_sizes[user_id_str][str(arrival_time)] = size
    
    return task_sizes


def save_patterns(filename="task_patterns.json", scenario_name=None, scenario=None):
    """タスク発生パターンをファイルに保存"""
    scenario_name = scenario_name or config.CURRENT_TASK_SCENARIO_NAME
    scenario = scenario or config.CURRENT_TASK_SCENARIO

    task_arrivals = generate_task_arrivals()
    task_types = generate_task_types(task_arrivals, scenario)
    task_sizes = generate_task_sizes(task_arrivals, task_types)
    
    patterns = {
        "arrivals": task_arrivals,
        "sizes": task_sizes,
        "types": task_types,
        "config": {
            "num_users": config.NUM_USERS,
            "arrival_rate": config.ARRIVAL_RATE,
            "arrival_rates": config.ARRIVAL_RATES,
            "simulation_time": config.SIMULATION_TIME,
            "random_seed": config.RANDOM_SEED,
            "scenario_name": scenario_name,
            "scenario": scenario,
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

    # 旧形式JSONとの互換性を維持
    if "types" not in patterns:
        patterns["types"] = {}
        arrivals = patterns.get("arrivals", {})
        for user_id_str, user_arrivals in arrivals.items():
            patterns["types"][user_id_str] = {
                str(arrival_time): "inference" for arrival_time in user_arrivals
            }

    patterns.setdefault("config", {})
    patterns["config"].setdefault("scenario_name", config.CURRENT_TASK_SCENARIO_NAME)
    patterns["config"].setdefault("scenario", config.CURRENT_TASK_SCENARIO)
    
    return patterns


if __name__ == "__main__":
    save_patterns()
