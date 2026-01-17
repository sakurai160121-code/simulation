"""
シミュレーション設定パラメータ
18ユーザー・9ティア構成
"""

# ユーザー数
NUM_USERS = 18

# タスク到着率（ポアソン過程）
# 全ユーザー共通
ARRIVAL_RATE = 0.2  # λ=0.2 (5秒に1回タスク発生の平均)
ARRIVAL_RATES = {str(i): ARRIVAL_RATE for i in range(18)}  # 全ユーザー同じ到着率

# GPU性能（TFLOPS）
# 9段階ティア、各ティアに2ユーザーずつ
GPU_PERFORMANCE_LEVELS = {
    "tier1": 2.98,        # GTX 1650
    "tier2": 8.87,        # GTX 1080
    "tier3": 29.15,       # RTX 4070
    "tier4": 36.41,       # RTX 3050
    "tier5": 64.83,       # RTX 3060Ti
    "tier6": 82.60,       # RTX 2080
    "tier7": 110.00,      # RTX 2080Ti
    "tier8": 130.50,      # Titan RTX
    "tier9": 311.84,      # A100
}

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

# バッチサイズ（画像枚数）
# ユーザー0～8: 1000バッチ
# ユーザー9～17: 2000バッチ
BATCH_SIZES = {
    0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000,
    9: 2000, 10: 2000, 11: 2000, 12: 2000, 13: 2000, 14: 2000, 15: 2000, 16: 2000, 17: 2000
}

# エポック数（全ユーザー共通）
EPOCHS = {i: 1 for i in range(18)}

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
SIMULATION_TIME = 3600.0  # 1時間（3600秒）

# ランダムシード（再現性のため）
RANDOM_SEED = 42

# プリエンプト時の再開・マイグレーションオーバーヘッド係数（大きいほど他人GPU選択が不利）
INTERRUPTION_OVERHEAD_FACTOR = 0.2

