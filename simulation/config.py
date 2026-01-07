"""
シミュレーション設定パラメータ
"""

# ユーザー数
NUM_USERS = 20

# タスク到着率（ポアソン過程）
ARRIVAL_RATE = 0.1  # デフォルト（余り0のユーザー）
# user_id % 5 に応じて到着率を振り分け: 0.10, 0.20, 0.30, 0.40, 0.50
ARRIVAL_RATES = {
    str(i): (0.10 if i % 5 == 0 else 0.20 if i % 5 == 1 else 0.30 if i % 5 == 2 else 0.40 if i % 5 == 3 else 0.50)
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
TASK_SIZE_MEAN = 10.0
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

# プリエンプト時の再開・マイグレーションオーバーヘッド係数（大きいほど他人GPU選択が不利）
INTERRUPTION_OVERHEAD_FACTOR = 2.0

# タスク締切の係数（基準サービス時間×係数）
# 例: DEADLINE_FACTOR=3.0 なら、基準サービス時間の3倍までに完了できないタスクは失敗扱い
DEADLINE_FACTOR = 10.0
