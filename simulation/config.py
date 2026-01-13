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
# ユーザー0～8: 軽量（YOLOv7-tiny）
# ユーザー9～17: 重量（YOLOv7-E6E）
TASK_SIZE_MEANS = {
    # 軽量タスク（ユーザー0～8）
    0: 0.360,    # User 0 - Tier1
    1: 0.360,    # User 1 - Tier2
    2: 0.360,    # User 2 - Tier3
    3: 0.360,    # User 3 - Tier4
    4: 0.360,    # User 4 - Tier5
    5: 0.360,    # User 5 - Tier6
    6: 0.360,    # User 6 - Tier7
    7: 0.360,    # User 7 - Tier8
    8: 0.360,    # User 8 - Tier9
    # 重量タスク（ユーザー9～17）
    9: 0.8432,    # User 9 - Tier1
    10: 0.8432,   # User 10 - Tier2
    11: 0.8432,   # User 11 - Tier3
    12: 0.8432,   # User 12 - Tier4
    13: 0.8432,   # User 13 - Tier5
    14: 0.8432,   # User 14 - Tier6
    15: 0.8432,   # User 15 - Tier7
    16: 0.8432,   # User 16 - Tier8
    17: 0.8432,   # User 17 - Tier9
}

# バッチ処理係数（1タスク=画像1000枚ぶん）
BATCH_MULTIPLIER = 1000.0

# キュー長概算で使う全体平均
TASK_SIZE_MEAN_GLOBAL = sum(TASK_SIZE_MEANS.values()) / len(TASK_SIZE_MEANS)

# ユーザーをGPUティアに割り当て
# 軽量タスク: ユーザー0～8（Tier1～Tier9）
# 重量タスク: ユーザー9～17（Tier1～Tier9）
GPU_TIER_ASSIGNMENT = {
    "tier1": [0, 9],      # User 0（軽量）, User 9（重量）
    "tier2": [1, 10],     # User 1（軽量）, User 10（重量）
    "tier3": [2, 11],     # User 2（軽量）, User 11（重量）
    "tier4": [3, 12],     # User 3（軽量）, User 12（重量）
    "tier5": [4, 13],     # User 4（軽量）, User 13（重量）
    "tier6": [5, 14],     # User 5（軽量）, User 14（重量）
    "tier7": [6, 15],     # User 6（軽量）, User 15（重量）
    "tier8": [7, 16],     # User 7（軽量）, User 16（重量）
    "tier9": [8, 17],     # User 8（軽量）, User 17（重量）
}

# シミュレーション終了時刻
SIMULATION_TIME = 3600.0  # 1時間（3600秒）

# ランダムシード（再現性のため）
RANDOM_SEED = 42

# プリエンプト時の再開・マイグレーションオーバーヘッド係数（大きいほど他人GPU選択が不利）
INTERRUPTION_OVERHEAD_FACTOR = 2.0

# タスク締切の係数（基準サービス時間×係数）
# 例: DEADLINE_FACTOR=3.0 なら、基準サービス時間の3倍までに完了できないタスクは失敗扱い
DEADLINE_FACTOR = 10.0
