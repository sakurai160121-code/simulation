# Air Computing Pool (ACP) — GPUシェアリング・シミュレーション

学内規模のGPU共有プールを想定し、スケジューリング方式の優劣を評価するシミュレータです。
**共有なし（No Sharing） / FCFS / 所有者優先（Owner Priority） / プリエンプティブ（Preemptive）** の4方式を、負荷率・ワークロード構成・参加インセンティブの観点から比較します。

---

## 概要

| 項目 | 内容 |
|---|---|
| ユーザー数 | 18人（Low/Mid/High各6人） |
| GPUティア | 9段階（GTX 1650 → A100） |
| スケジューリング方式 | 共有なし / FCFS / 所有者優先 / プリエンプティブ |
| 主要指標 | ユーザー別・ティア別の平均TAT（Turn-Around Time） |
| 参加モデル | 反復的な合理的エージェントの参加判断（共有時TAT ≦ 単独時TATなら参加） |

---

## リポジトリ構成

```
simulation/
├── core/                               # 基盤モジュール
│   ├── config.py                       # グローバル設定（ユーザー数・GPUティア・到着率など）
│   └── definitions.py                  # Task / GPU / User のデータクラス
│
├── engine/                             # シミュレーション本体
│   ├── task_patterns.py                # タスク到着・サイズ生成（ポアソン過程＋対数正規分布）
│   ├── simulation_no_sharing.py        # ベースライン：各ユーザーが自分のGPUのみ使用
│   ├── simulation_with_sharing.py      # FCFS方式の共有
│   ├── simulation_with_sharing_owner_priority.py    # 所有者優先方式の共有
│   ├── simulation_with_sharing_owner_preemption.py  # プリエンプティブ方式の共有
│   └── simulation_iterative_wrapper.py # 参加可否を反復的に最適化するラッパー
│
├── scenarios/                          # 実行スクリプト群
│   ├── run_custom_user_arrival_web.py        # ユーザー別到着率の比較（Streamlit UIから呼び出し）
│   ├── run_random_hetero_fixed_load_web.py   # ヘテロワークロードの負荷率スイープ（Streamlit UIから呼び出し）
│   ├── run_participation_cascade.py          # 固定負荷率0.8での参加カスケード
│   ├── run_hetero_scenarios.py               # 4種のヘテロワークロードシナリオ
│   ├── run_multi_load_scenarios.py           # 複数負荷率での一括実行
│   ├── run_multi_load_with_participation.py  # 参加モデルを含む複数負荷率での一括実行
│   ├── run_custom_multi_load_web.py          # Web UI向けの複数負荷率実行
│   ├── run_all_simulations.py                # 全シミュレーションの一括実行
│   ├── run_ml_dataset_generation.py          # 機械学習用データセット生成
│   ├── generate_user_details_table.py        # ユーザー詳細テーブルの生成
│   └── make_hetero_pr_broken.py              # Protection Ratioの内訳図生成
│
├── plotting/                           # グラフ生成
│   ├── plot_paper_graphs_from_csv.py   # 試行結果CSVからの帯グラフ（平均±最小最大）生成
│   └── plot_paper_figures.py           # 論文用PDF図の生成
│
├── analysis/                           # 統計・分析
│   ├── results.py                      # 統計処理・可視化の補助関数
│   ├── high_performance_protection_ml.py     # 高性能GPU保護の機械学習分析
│   └── analyze_high_performance_protection_ml.py  # 上記分析結果の集計
│
└── outputs/                            # 実行結果の出力先（Git管理外）

streamlit_app.py                        # Web UI（シナリオ実行＋結果ビューア）
imgs/                                   # 論文・発表用に保存済みの図
```

---

## スケジューリング方式

| 方式 | 説明 |
|---|---|
| **共有なし（No Sharing）** | 各ユーザーは自分のGPUでのみタスクを実行 |
| **FCFS** | 共有プール内で到着順に処理 |
| **所有者優先（Owner Priority）** | 所有者のタスクが常にゲストのタスクより先に実行される |
| **プリエンプティブ（Preemptive）** | 所有者のタスク到着時、実行中のゲストタスクを即座に中断・置き換える |

**Protection Ratio** = 共有時TAT ÷ 共有なし時TAT
1.0以下であれば、所有者は共有によって不利益を受けていないことを示す。

---

## クイックスタート

このプロジェクトに `requirements.txt` は用意されていません。以下の主要パッケージを手動でインストールしてください（リポジトリの `.venv` には次のバージョンが導入済みです）。

```bash
pip install streamlit numpy pandas matplotlib scikit-learn
```

```bash
streamlit run streamlit_app.py
```

Web UIでは以下の実行メニューを提供しています（`SCRIPT_CATALOG` in [streamlit_app.py](streamlit_app.py) 参照）。

| メニュー | スクリプト | 内容 |
|---|---|---|
| ランダムheterogeneous固定負荷率評価 | `scenarios/run_random_hetero_fixed_load_web.py` | 到着率を全ユーザー均一に保ち、全体負荷率0.1〜1.0をスイープしてTier8/Tier9のTATとProtection Ratioを評価 |
| ユーザー別到着率で比較 | `scenarios/run_custom_user_arrival_web.py` | ユーザー0〜17ごとに負荷率を個別指定し、4方式の全体指標・ユーザー別TATを比較 |

---

## スクリプトを直接実行する

すべてのスクリプトは**リポジトリのルートディレクトリ**から実行します（`simulation/` の中からではありません）。

### ヘテロワークロードの負荷率スイープ（帯グラフ）

```bash
python simulation/scenarios/run_random_hetero_fixed_load_web.py
```

出力先 `simulation/outputs/random_hetero_fixed_load/`:
- `overall_avg_tat_band.png` — システム全体の平均TAT
- `low/mid/high_tier_tat_band.png` — ティア別グループのTAT
- `protection_ratio_without_fcfs.png` — Tier9のProtection Ratio

### 参加カスケード

```bash
python simulation/scenarios/run_participation_cascade.py
```

出力先 `simulation/outputs/participation_cascade/`:
- `cascade_high_tier.png` — High tierの各反復における参加数（3方式比較）
- `cascade_stacked_3panel.png` — シナリオ別のLow/Mid/High積み上げグラフ

再プロットのみ行う場合：

```bash
python simulation/scenarios/run_participation_cascade.py --replot
```

### ヘテロワークロードシナリオ

```bash
python simulation/scenarios/run_hetero_scenarios.py
```

4シナリオ × 各100試行を実行します。

| シナリオ | Training比率 |
|---|---|
| `uniform` | 全ユーザー: 0.3 |
| `low_heavy` | Low=0.7, Mid=0.3, High=0.1 |
| `high_heavy` | Low=0.1, Mid=0.3, High=0.7 |
| `random` | 試行ごとに均一分布から再サンプリング |

出力先 `simulation/outputs/hetero_scenarios/{scenario}/`:
- `low/mid/high_tier_tat.png`, `protection_ratio.png`

---

## 出力ディレクトリ

```
simulation/outputs/
├── random_hetero_fixed_load/     # 負荷率スイープの結果（帯グラフ＋CSV）
├── participation_cascade/        # カスケードシミュレーションのグラフ＋JSON
└── hetero_scenarios/
    ├── uniform/
    ├── low_heavy/
    ├── high_heavy/
    └── random/
```

> **注：** `simulation/outputs/` 以下の出力ファイルはGit管理対象外です（`.gitignore` 参照）。

---

## 設定（`simulation/core/config.py`）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `NUM_USERS` | 18 | ユーザー総数 |
| `SIMULATION_TIME` | 8,640,000 | 観測時間（秒） |
| `ARRIVAL_RATE` | 0.005 | タスク到着率λ（ポアソン過程、全ユーザー共通） |
| `GPU_PERFORMANCE_LEVELS` | 9ティア | GPUの相対性能（TFLOPS、2.98〜311.84） |
| `GPU_TIER_ASSIGNMENT` | dict | ティア → ユーザーIDのマッピング |
| `RANDOM_SEED` | 42 | 再現性確保のためのグローバル乱数シード |
| `ACP_RESIDENT_GPU_COUNT` | 0 | 共有プールに常設されるACP常駐GPUの台数 |

タスクサイズは**対数正規分布**に従う（`TASK_SIZE_DISTRIBUTION`）：
- Inference: 期待値 9,580 TFLOPs
- Training: 期待値 412,180 TFLOPs

---

## 結果の再現方法

1. `config.py` の `RANDOM_SEED = 42`（デフォルト）を確認する。
2. 目的のスクリプトを実行する。
3. 固定シードであれば出力は決定論的に再現される。`random` ヘテロシナリオのみ独自に `rng = np.random.default_rng(42)` を使用する。
