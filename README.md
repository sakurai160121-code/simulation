# 卒論シミュレーション（GPU共有スケジューリング評価）

18ユーザー・9ティアGPU環境を対象に、**GPU共有ポリシーの違い**が待ち時間や完了率に与える影響を比較するシミュレーションです。  
卒論で使う再現実験・可視化をまとめて実行できます。

---

## できること

- 4つの実行方式を比較
  - 共有なし
  - FCFS（先着順）
  - 所有者優先
  - 所有者優先 + プリエンプティブ方式
- 負荷率（0.1〜1.0）を変えた性能比較
- 反復最適化による「共有参加/不参加」シミュレーション
- グラフ・CSV・JSONで結果保存

---

## リポジトリ構成（主要部分）

- `docs/` : 発表資料・図（コード以外の資料）
  - `presentations/` : PDF / PPTX
  - `diagrams/` : draw.io 図
- `simulation/` : 実験コード本体
  - `config.py` : ユーザー数・GPU性能・到着率などの設定
  - `task_patterns.py` : タスク到着/サイズパターン生成
  - `run_all_simulations.py` : 基本比較 + 反復最適化の一括実行
  - `run_multi_load_scenarios.py` : 負荷率別に4方式を比較
  - `run_multi_load_with_participation.py` : 負荷率別に参加者数推移を分析
  - `results.py` : 統計集計・グラフ作成
- `simulation/outputs/` : 実験結果の保存先
- `outputs/` : 一部スクリプト実行時の出力先（ルート側）

> 補足: 既存の補助説明は `simulation/README_multi_scenarios.md` にあります。
> 
> GitHub提出では、生成物（`simulation/outputs/` など）は `.gitignore` で除外しています。

---

## 実行環境

- Python 3.10+ 推奨
- OS: Windows で確認（PowerShell）
- 主なライブラリ
  - `numpy`
  - `pandas`
  - `matplotlib`

---

## セットアップ

1. 仮想環境を作成・有効化
2. 必要パッケージをインストール

`requirements.txt` が未整備の場合は次をインストールしてください。

```bash
pip install numpy pandas matplotlib
```

---

## 使い方（よく使う3本）

作業ディレクトリを `simulation/` にして実行する想定です。

### 1) 基本比較 + 反復最適化（まずはこれ）

```bash
python run_all_simulations.py
```

- タスクパターン生成
- 4方式比較
- 反復最適化
- グラフ出力

### 2) 負荷率別の方式比較（0.1〜1.0）

```bash
python run_multi_load_scenarios.py
```

- 各負荷率で4方式を比較
- 待ち時間推移を可視化

### 3) 負荷率と参加者数の関係分析

```bash
python run_multi_load_with_participation.py
```

- 共有参加率の収束傾向を分析
- グループ別（低/中/高性能）結果を出力

---

## 主な出力

- `simulation/outputs/basic_scenarios/` : 基本比較結果
- `simulation/outputs/iterative_results/` : 反復最適化結果
- `simulation/outputs/multi_load/` : 負荷率別結果（CSV/JSON/画像）
- `simulation/outputs/tables/` : 集計テーブル
- `simulation/outputs/user_comparisons/` : ユーザー別比較

---

## 設定変更ポイント

実験条件を変える場合は `simulation/config.py` を編集します。

- `NUM_USERS` : ユーザー数
- `ARRIVAL_RATE` / `ARRIVAL_RATES` : タスク到着率
- `GPU_PERFORMANCE_LEVELS` : GPU性能（TFLOPS）
- `BATCH_SIZES`, `EPOCHS` : タスク規模
- `SIMULATION_TIME` : シミュレーション時間
- `RANDOM_SEED` : 乱数シード（再現性）

---

## 提出用メモ（GitHub）

- この `README.md` をトップに置く
- 実行に不要な生成物（大きな画像・ログ）を整理
- `simulation/outputs/` は必要な結果のみ残す
- 実験再現のため、`config.py` の主要値は README と整合させる

---

## ライセンス

研究用途のコードです。必要に応じて研究室ルールに合わせて追記してください。
