"""
所有者優先度シミュレーションの計算式と割り込み動作を検証するスクリプト
"""

import numpy as np
from config import (
    ARRIVAL_RATES, TASK_SIZE_MEANS, TASK_SIZE_MEAN, GPU_PERFORMANCE_LEVELS, GPU_TIER_ASSIGNMENT
)

print("=" * 80)
print("【所有者優先度シミュレーションの計算式検証】")
print("=" * 80)

# ユーザーのティア割り当て
user_to_tier = {}
for tier, users in GPU_TIER_ASSIGNMENT.items():
    for user_id in users:
        user_to_tier[user_id] = tier

print("\n【ユーザーごとの到着率とタスクサイズ平均】\n")
for user_id in range(20):
    lambda_user = ARRIVAL_RATES.get(str(user_id), 2.0)
    s_bar = TASK_SIZE_MEANS.get(str(user_id), TASK_SIZE_MEAN)
    tier = user_to_tier[user_id]
    mu = GPU_PERFORMANCE_LEVELS[tier]
    
    rho_own = lambda_user * s_bar / mu
    
    print(f"User {user_id:2d} (ティア={tier}): λ={lambda_user}, s̄={s_bar}, μ={mu}, ρ_own={rho_own:.4f}")

print("\n" + "=" * 80)
print("【計算式の説明】")
print("=" * 80)

print("""
【1】自分のGPUを選ぶ場合：
  予想完了時刻 = max(実行中タスク残り時間, 0) + Σ(自分のキュー内タスク処理時間)
  
  理由：
  - 実行中のタスクが完了するまで待つ（割り込まない）
  - その後、自分のキュー内タスクを処理
  - キュー内他人のタスクは、自分の新規到着タスクに「割り込まれ」て後ろに下がる

【2】他人のGPUを選ぶ場合：
  実効処理レート: μ_eff = μ / (1 + ρ_own)
  
  ここで ρ_own = λ_own × s̄_own / μ （GPUの所有者の稼働率）
  
  予想完了時刻 ≈ 現在時刻 + (平均タスクサイズ / μ_eff) × キュー長
  
  理由：
  - 所有者がいつでも割り込んでくる可能性があるため
  - 実効性能が「薄まっている」と考える
  - 稼働率が高いほど（ρ_ownが大きいほど）、割り込みが多く、実効性能が低下
""")

print("\n" + "=" * 80)
print("【具体例】")
print("=" * 80)

# tier1の User 0（λ=1.0）が User 5（tier2, λ=2.0）のGPUを使うケース
user_wanting = 0  # User 0 がタスク要求
user_owner = 5    # User 5 のGPU

lambda_owner = ARRIVAL_RATES.get(str(user_owner), 2.0)
s_bar_owner = TASK_SIZE_MEANS.get(str(user_owner), TASK_SIZE_MEAN)
tier_owner = user_to_tier[user_owner]
mu_owner = GPU_PERFORMANCE_LEVELS[tier_owner]

rho_own = lambda_owner * s_bar_owner / mu_owner
mu_eff = mu_owner / (1.0 + rho_own)

print(f"\nUser {user_wanting} (tier1, λ={ARRIVAL_RATES[str(user_wanting)]}) が")
print(f"User {user_owner} (tier2, λ={lambda_owner}) のGPUを使う場合：\n")

print(f"  User {user_owner} の稼働率 ρ_own = λ × s̄ / μ")
print(f"                          = {lambda_owner} × {s_bar_owner} / {mu_owner}")
print(f"                          = {rho_own:.4f}\n")

print(f"  実効処理レート μ_eff = μ / (1 + ρ_own)")
print(f"                      = {mu_owner} / (1 + {rho_own:.4f})")
print(f"                      = {mu_eff:.4f}\n")

print(f"  ⇒ 通常性能 {mu_owner} が {mu_eff:.4f} に低下（{(1-mu_eff/mu_owner)*100:.1f}% 減少）")

print("\n" + "=" * 80)
print("【割り込みの実装】")
print("=" * 80)

print("""
現在の実装：

1. タスク到着時に select_best_gpu() で最適なGPUを選択
   - 自分のGPU：割り込み可能（キュー内の自分のタスク優先度が高い）
   - 他人のGPU：実効性能が低いため選ばれにくい

2. GPU処理完了時に process_gpu_finish() で次のタスク開始
   - キュー先頭のタスク（自分のものなら先に実行）から処理開始

3. 割り込みの効果：
   - 自分のGPU：キュー内他人のタスクより自分のタスクが先に実行される
   - これが「割り込み」の実装
   - 実際の実行順序は、到着順ではなく所有者優先順
""")
