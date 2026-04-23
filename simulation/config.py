"""
シミュレーション設定パラメータ
18ユーザー・9ティア構成
"""

from typing import Dict
import numpy as np

# ユーザー数
NUM_USERS = 18

# タスク到着率（ポアソン過程）
# 全ユーザー共通
ARRIVAL_RATE = 0.005  # λ=0.005 (200秒に1回タスク発生の平均)
ARRIVAL_RATES = {str(i): ARRIVAL_RATE for i in range(18)}  # 全ユーザー同じ到着率

# 評価対象の負荷率
LOAD_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# GPU性能（TFLOPS）
# 9段階ティア、各ティアに2ユーザーずつ
GPU_PERFORMANCE_LEVELS = {
    "tier1": 2.98,        # GTX 1650
    "tier2": 8.87,        # GTX 1080
    "tier3": 20.41,       # RTX 3050
    "tier4": 64.83,       # RTX 3060Ti
    "tier5": 82.60,       # RTX 2080
    "tier6": 110.00,      # RTX 2080Ti
    "tier7": 180.50,      # Titan RTX
    "tier8": 233,         # RTX 4070
    "tier9": 311.84,      # A100
}

# ACP常駐GPU（共有プールに最初から存在する常設GPU）
# 将来はこのプロファイル配列を増やすことで複数性能に拡張できる。
ACP_RESIDENT_GPU_COUNT = 0
ACP_RESIDENT_GPU_RATE = 180.5
ACP_RESIDENT_GPU_PROFILES = [
    {
        "count": ACP_RESIDENT_GPU_COUNT,
        "processing_rate": ACP_RESIDENT_GPU_RATE,
    },
]


def set_acp_resident_gpu_profiles(count: int, processing_rates: list[float] | None = None) -> None:
    """ACP常駐GPUの台数と個別性能を設定する。"""
    global ACP_RESIDENT_GPU_COUNT, ACP_RESIDENT_GPU_RATE, ACP_RESIDENT_GPU_PROFILES

    normalized_count = max(0, int(count))
    rates = [float(rate) for rate in (processing_rates or [])]
    if normalized_count == 0:
        ACP_RESIDENT_GPU_COUNT = 0
        ACP_RESIDENT_GPU_RATE = float(rates[-1]) if rates else ACP_RESIDENT_GPU_RATE
        ACP_RESIDENT_GPU_PROFILES = []
        return

    if not rates:
        rates = [float(ACP_RESIDENT_GPU_RATE)] * normalized_count
    elif len(rates) < normalized_count:
        rates = rates + [rates[-1]] * (normalized_count - len(rates))
    else:
        rates = rates[:normalized_count]

    ACP_RESIDENT_GPU_COUNT = normalized_count
    ACP_RESIDENT_GPU_RATE = float(rates[0])
    ACP_RESIDENT_GPU_PROFILES = [
        {"count": 1, "processing_rate": rate}
        for rate in rates
    ]


def get_acp_resident_gpu_specs():
    """ACP常駐GPUの個別スペック一覧を返す。"""
    specs = []
    running_index = 0
    for profile in ACP_RESIDENT_GPU_PROFILES:
        count = int(profile.get("count", 0))
        processing_rate = float(profile.get("processing_rate", ACP_RESIDENT_GPU_RATE))
        for _ in range(count):
            specs.append(
                {
                    "gpu_id": f"acp_{running_index}",
                    "processing_rate": processing_rate,
                    "owner": None,
                }
            )
            running_index += 1
    return specs

# タスクサイズ（仕事量）の平均値 [TFLOPS]
# 全ユーザー: YOLOv7-w6
TASK_SIZE_MEANS = {
    0: 0.360,    # User 0 - Tier1
    1: 0.360,    # User 1 - Tier2
    2: 0.360,    # User 2 - Tier3
    3: 0.360,    # User 3 - Tier4
    4: 0.360,    # User 4 - Tier5
    5: 0.360,    # User 5 - Tier6
    6: 0.360,    # User 6 - Tier7
    7: 0.360,    # User 7 - Tier8
    8: 0.360,    # User 8 - Tier9
    9: 0.360,    # User 9 - Tier1
    10: 0.360,   # User 10 - Tier2
    11: 0.360,   # User 11 - Tier3
    12: 0.360,   # User 12 - Tier4
    13: 0.360,   # User 13 - Tier5
    14: 0.360,   # User 14 - Tier6
    15: 0.360,   # User 15 - Tier7
    16: 0.360,   # User 16 - Tier8
    17: 0.360,   # User 17 - Tier9
}

# タスクシナリオ（training/inference 比率）
TASK_SCENARIOS = {
    "scenario1_all_inference": {"training_ratio": 0.0, "inference_ratio": 1.0},
    "scenario2_25_training": {"training_ratio": 0.25, "inference_ratio": 0.75},
    "scenario3_50_training": {"training_ratio": 0.50, "inference_ratio": 0.50},
    "scenario4_75_training": {"training_ratio": 0.75, "inference_ratio": 0.25},
    "scenario5_all_training": {"training_ratio": 1.0, "inference_ratio": 0.0},
}

# 現在実行中シナリオ（run_multi_load_scenarios.py で更新）
CURRENT_TASK_SCENARIO_NAME = "scenario1_all_inference"
CURRENT_TASK_SCENARIO = TASK_SCENARIOS[CURRENT_TASK_SCENARIO_NAME].copy()

# タスク種別ごとのサイズ分布パラメータ
TASK_SIZE_DISTRIBUTION = {
    "inference": {
        "lognormal_mean": float(np.log(8000.0)),
        "lognormal_sigma": 0.6,
        "clip_min": 1000.0,
        "clip_max": 50000.0,
    },
    "training": {
        "lognormal_mean": float(np.log(250000.0)),
        "lognormal_sigma": 1.0,
        "clip_min": 30000.0,
        "clip_max": 3000000.0,
    },
}

# 予測・概算に使う期待タスクサイズ（TFLOPs）
EXPECTED_TASK_SIZE = {
    "inference": 9580.0,
    "training": 412180.0,
}


def get_scenario_expected_task_size(scenario: Dict[str, float] = None) -> float:
    """シナリオ比率に基づく混合平均タスクサイズ E[S] を返す。"""
    s = scenario if scenario is not None else CURRENT_TASK_SCENARIO
    p_inf = float(s.get("inference_ratio", 0.0))
    p_train = float(s.get("training_ratio", 0.0))
    return p_inf * EXPECTED_TASK_SIZE["inference"] + p_train * EXPECTED_TASK_SIZE["training"]


def get_current_task_ratios() -> Dict[str, float]:
    """現在シナリオの inference/training 比率を返す。"""
    p_inf = float(CURRENT_TASK_SCENARIO.get("inference_ratio", 0.0))
    p_train = float(CURRENT_TASK_SCENARIO.get("training_ratio", 0.0))
    return {
        "inference_ratio": p_inf,
        "training_ratio": p_train,
    }


def get_expected_task_size_by_ratios(inference_ratio: float, training_ratio: float) -> float:
    """与えられた比率から混合期待タスクサイズを返す。"""
    p_inf = float(inference_ratio)
    p_train = float(training_ratio)
    return p_inf * EXPECTED_TASK_SIZE["inference"] + p_train * EXPECTED_TASK_SIZE["training"]


def get_user_expected_task_size(user_id: int) -> float:
    """
    ユーザー固有の期待タスクサイズを返す。
    現状はユーザー固有比率が無いため、現在シナリオ比率でフォールバックする。
    """
    _ = user_id
    ratios = get_current_task_ratios()
    return get_expected_task_size_by_ratios(
        ratios["inference_ratio"],
        ratios["training_ratio"],
    )

# バッチサイズ（画像枚数）
# ユーザー0～8: 3000バッチ
# ユーザー9～17: 6000バッチ
BATCH_SIZES = {
    0: 3000, 1: 3000, 2: 3000, 3: 3000, 4: 3000, 5: 3000, 6: 3000, 7: 3000, 8: 3000,
    9: 6000, 10: 6000, 11: 6000, 12: 6000, 13: 6000, 14: 6000, 15: 6000, 16: 6000, 17: 6000
}

# エポック数（全ユーザー共通）
EPOCHS = {i: 10 for i in range(18)}

# バッチ処理係数（後方互換性のため残す、実際はBATCH_SIZESとEPOCHSを使用）
BATCH_MULTIPLIER = 1000.0

# キュー長概算で使う全体平均
TASK_SIZE_MEAN_GLOBAL = sum(TASK_SIZE_MEANS.values()) / len(TASK_SIZE_MEANS)

# ユーザーをGPUティアに割り当て
# ユーザー0～8: 1000バッチ
# ユーザー9～17: 2000バッチ
GPU_TIER_ASSIGNMENT = {
    "tier1": [0, 9],      # User 0（1000バッチ）, User 9（2000バッチ）
    "tier2": [1, 10],     # User 1（1000バッチ）, User 10（2000バッチ）
    "tier3": [2, 11],     # User 2（1000バッチ）, User 11（2000バッチ）
    "tier4": [3, 12],     # User 3（1000バッチ）, User 12（2000バッチ）
    "tier5": [4, 13],     # User 4（1000バッチ）, User 13（2000バッチ）
    "tier6": [5, 14],     # User 5（1000バッチ）, User 14（2000バッチ）
    "tier7": [6, 15],     # User 6（1000バッチ）, User 15（2000バッチ）
    "tier8": [7, 16],     # User 7（1000バッチ）, User 16（2000バッチ）
    "tier9": [8, 17],     # User 8（1000バッチ）, User 17（2000バッチ）
}
# シミュレーション終了時刻
SIMULATION_TIME = 8640000  # 1時間（3600秒）

# ランダムシード（再現性のため）
RANDOM_SEED = 42

# プリエンプト時の再開・マイグレーションオーバーヘッド係数（タスク種別ごと）
# 大きいほど他人GPU選択が不利
INTERRUPTION_OVERHEAD_FACTOR_INFERENCE = 0.2
INTERRUPTION_OVERHEAD_FACTOR_TRAINING = 0.2

