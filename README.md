# 卒論シミュレーション（GPU共有スケジューリング評価）

本リポジトリは、18ユーザー・9ティアGPU環境を対象に、GPU共有ポリシーの違いがタスク応答時間（TAT）・完了率・参加行動に与える影響を評価するためのシミュレーションコードです。

## 目的

- 共有方式ごとの性能差を定量比較する
- 負荷率を段階的に変化させたときの傾向を確認する
- 反復判定により、ユーザーの共有参加/不参加がどのように収束するかを確認する

## 比較対象の方式

- 共有なし（各ユーザーが自GPUのみ利用）
- FCFS（先着順共有）
- 所有者優先
- 所有者優先 + プリエンプティブ方式

## リポジトリ構成（主要部分）

- `simulation/` : 実験コード本体
  - `config.py` : シミュレーション設定（ユーザー数・到着率・GPU性能・バッチ/エポックなど）
  - `task_patterns.py` : タスク到着時刻・タスクサイズ生成
  - `definitions.py` : `User` / `GPU` / `Task` 定義
  - `simulation_*.py` : 各方式のシミュレータ実装
  - `simulation_iterative_wrapper.py` : 参加判定を含む反復最適化
  - `results.py` : 統計集計・可視化
  - `run_all_simulations.py` : 基本比較 + 反復最適化の一括実行
  - `run_multi_load_scenarios.py` : 負荷率別の4方式比較
  - `run_multi_load_with_participation.py` : 負荷率別の参加者数分析
- `simulation/outputs/` : 実験生成物の保存先（画像・CSV・JSON）
- `template.potx` : 発表資料テンプレート

## 実行環境

- Python 3.10 以上
- Windows + PowerShell で動作確認
- 使用ライブラリ
  - `numpy`
  - `pandas`
  - `matplotlib`

## セットアップ

1. 仮想環境を作成
2. 仮想環境を有効化
3. パッケージをインストール

`requirements.txt` がない場合:

```bash
pip install numpy pandas matplotlib
```

## 実行手順

以下は `simulation/` ディレクトリで実行します。

### A. 基本比較 + 反復最適化

```bash
python run_all_simulations.py
```

実行内容:

1. タスクパターン生成
2. 4方式の単発比較
3. 反復最適化（参加状態更新）
4. 比較グラフ・表の出力

### B. 負荷率別の方式比較（0.1〜1.0）

```bash
python run_multi_load_scenarios.py
```

実行内容:

- 到着率は固定し、バッチサイズ調整で負荷率を制御
- 各負荷率で4方式を実行
- 全体/グループ別の平均待ち時間を保存

### C. 負荷率と参加者数の関係分析

```bash
python run_multi_load_with_participation.py
```

実行内容:

- 反復最適化を負荷率ごとに実行
- 低性能・中性能・高性能グループ別の参加人数推移を保存

## 出力先

- `simulation/outputs/basic_scenarios/` : 基本比較の図
- `simulation/outputs/iterative_results/` : 反復最適化の図
- `simulation/outputs/multi_load/` : 負荷率分析の図・CSV・JSON
- `simulation/outputs/tables/` : テーブル画像
- `simulation/outputs/user_comparisons/` : ユーザー別比較図

## 設定変更時の確認ポイント

`simulation/config.py` の変更が主要です。

- `NUM_USERS` : ユーザー数
- `ARRIVAL_RATE`, `ARRIVAL_RATES` : 到着率
- `GPU_PERFORMANCE_LEVELS`, `GPU_TIER_ASSIGNMENT` : GPU性能と割当
- `TASK_SIZE_MEANS`, `BATCH_SIZES`, `EPOCHS` : タスク規模
- `SIMULATION_TIME` : 観測時間
- `RANDOM_SEED` : 再現性

設定変更時は、`task_patterns.json` を再生成してから比較することを推奨します。

## 備考

- 既存の補助説明は `simulation/README_multi_scenarios.md` を参照
- 生成物（`simulation/outputs/`、キャッシュ、ログなど）は `.gitignore` で除外設定済み

## ライセンス

研究用途のコードです。公開・再配布の扱いは研究室ルールに従ってください。
