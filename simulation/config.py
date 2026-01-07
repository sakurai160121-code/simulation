"""
シミュレーション設定パラメータ
"""

# ユーザー数
NUM_USERS = 20

# タスク到着率（ポアソン過程）
ARRIVAL_RATE = 2.0  # λ = 2.0（デフォルト）
# ユーザーごとの到着率（偶数ID:1.0, 奇数ID:2.0）
ARRIVAL_RATES = {
    str(i): (1.0 if i % 2 == 0 else 2.0)
    for i in range(NUM_USERS)
}

# GPUの処理性能（レート値が大きいほど速い）
# 最高性能と最低性能で約4倍差に設定
GPU_PERFORMANCE_LEVELS = {
    "tier1": 1.0,    # 最低性能（処理レート1.0）
    "tier2": 2.0,    # 中程度-低
    "tier3": 3.0,    # 中程度-高
    "tier4": 4.0,    # 最高性能（処理レート4.0）
}

# タスクサイズ（仕事量）の平均。実際のサービス時間は「タスクサイズ / GPU処理レート」で決定。
TASK_SIZE_MEAN = 2.0
# ユーザーごとのタスクサイズ平均（指定がなければデフォルト値を使用）
TASK_SIZE_MEANS = {str(i): TASK_SIZE_MEAN for i in range(NUM_USERS)}
# キュー長概算などで使う全体平均（単純平均）
TASK_SIZE_MEAN_GLOBAL = sum(TASK_SIZE_MEANS.values()) / len(TASK_SIZE_MEANS)

# ユーザーを性能ティアに割り当て
# 20人を4段階に均等割り当て（各段階5人）
GPU_TIER_ASSIGNMENT = {
    "tier1": list(range(0, 5)),      # ユーザー0-4
    "tier2": list(range(5, 10)),     # ユーザー5-9
    "tier3": list(range(10, 15)),    # ユーザー10-14
    "tier4": list(range(15, 20)),    # ユーザー15-19
}

# シミュレーション終了時刻
SIMULATION_TIME = 1000.0

# ランダムシード（再現性のため）
RANDOM_SEED = 42
