from __future__ import annotations

import json
import locale
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / "simulation"
OUTPUT_DIR = SIM_DIR / "outputs"

SCRIPT_CATALOG = {
    "ランダムheterogeneous固定負荷率評価": {
        "script": "run_random_hetero_fixed_load_web.py",
        "summary": "到着率を全ユーザー均一に保ちつつ、全体負荷率を0.1〜1.0でスイープし、Tier8/Tier9のTATとTier9のProtection Ratioを論文用グラフとして評価します。",
        "outputs": [
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/trial_results.csv",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/summary_by_load.csv",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/load_setup.csv",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/overall_avg_tat_band.png",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/low_tier_tat_band.png",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/mid_tier_tat_band.png",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/high_tier_tat_band.png",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/tier9_tat_combined_band.png",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/tier8_tat_combined_band.png",
            "simulation/outputs/random_hetero_fixed_load/custom_web/<timestamp>/protection_ratio_without_fcfs.png",
        ],
    },
    "ユーザー別到着率で比較": {
        "script": "run_custom_user_arrival_web.py",
        "summary": "ユーザー0〜17ごとに負荷率を個別指定し、負荷率から到着率を計算して4方式の全体指標とユーザー別TATを比較します。",
        "outputs": [
            "simulation/outputs/multi_load/custom_user_arrival_web/<timestamp>/user_arrival_results.json",
            "simulation/outputs/multi_load/custom_user_arrival_web/<timestamp>/user_arrival_overall_results.csv",
            "simulation/outputs/multi_load/custom_user_arrival_web/<timestamp>/user_arrival_user_tat_results.csv",
        ],
    },
}


GRAPH_HINTS = [
    (
        "user_comparison_",
        "ユーザー別比較図です。上段左=平均待ち時間、上段右=全タスク完了時刻、下段表=他GPU利用率などを示します。",
    ),
    (
        "scenario_table_users_0_to_2",
        "ユーザー0〜2の方式別平均待ち時間を表形式で比較した図です。",
    ),
    (
        "users_3_to_8_line_graph",
        "ユーザー3〜8の方式別平均待ち時間の折れ線比較です。線が低いほど待ち時間が短いです。",
    ),
    (
        "simulation_results_no_sharing",
        "共有なし方式の結果図です。各ユーザーの統計分布を確認します。",
    ),
    (
        "simulation_results_with_sharing",
        "FCFS共有方式の結果図です。共有なしと比べて待ち時間改善/悪化を確認します。",
    ),
    (
        "simulation_results_with_sharing_owner_priority",
        "所有者優先方式の結果図です。所有者タスクを優先した場合の影響を確認します。",
    ),
    (
        "simulation_results_with_sharing_owner_preemption",
        "プリエンプティブ方式の結果図です。割り込み許可時の性能変化を確認します。",
    ),
    (
        "load_rate_",
        "負荷率を横軸にした比較図です。負荷増加に対する各方式の耐性を見ます。",
    ),
    (
        "participation_by_load_",
        "負荷率ごとの参加者数（または参加率）を示す図です。グループ差の傾向を確認します。",
    ),
    (
        "user_0_to_8_tat_by_scenario",
        "ユーザー0〜8について、横軸=ユーザーID、縦軸=平均TATをシナリオ別の棒グラフで比較した図です。",
    ),
]


def run_script(script_name: str, args: list[str] | None = None) -> tuple[int, str]:
    def decode_output(data: bytes) -> str:
        if not data:
            return ""
        for enc in ["utf-8", locale.getpreferredencoding(False), "cp932"]:
            if not enc:
                continue
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    completed = subprocess.run(
        cmd,
        cwd=str(SIM_DIR),
        capture_output=True,
        text=False,
    )
    output = decode_output(completed.stdout) + "\n" + decode_output(completed.stderr)
    return completed.returncode, output.strip()


def graph_description(path: Path) -> str:
    name = path.name
    for key, desc in GRAPH_HINTS:
        if key in name:
            return desc
    return "このファイル専用の説明は未登録です。ファイル名と出力先から実験目的を確認してください。"


def parse_custom_output_dir(logs: str) -> Path | None:
    marker = "CUSTOM_OUTPUT_DIR="
    text = logs or ""
    index = text.find(marker)
    if index != -1:
        raw = text[index + len(marker):].splitlines()[0].strip()
        if raw:
            p = Path(raw)
            if p.exists():
                return p
    return None


def script_exists(script_name: str) -> bool:
    return (SIM_DIR / script_name).exists()


def collect_output_files() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    exts = {".png", ".csv", ".json"}
    files = [p for p in OUTPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def to_rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


st.set_page_config(page_title="GPU共有シミュレーション", layout="wide")
st.title("GPU共有シミュレーション")
st.caption("実行メニューを選び、結果ファイルを直接ブラウザで確認できます。")


if not SIM_DIR.exists():
    st.error("simulation ディレクトリが見つかりません。")
    st.stop()

run_tab, view_tab, help_tab = st.tabs(["シミュレーション実行", "結果", "グラフの見方"])

with run_tab:
    st.subheader("実行")
    mode = st.radio("実行するメニュー", list(SCRIPT_CATALOG.keys()))
    meta: dict[str, Any] = SCRIPT_CATALOG[mode]
    script = str(meta["script"])
    st.write(f"実行スクリプト: simulation/{script}")
    st.write(f"内容: {meta['summary']}")

    with st.expander("このメニューで生成される主な出力"):
        for out in meta["outputs"]:
            st.write(f"- {out}")

    exists = script_exists(script)
    if not exists:
        st.error(f"スクリプトが見つかりません: simulation/{script}")

    custom_args: list[str] = []
    if mode == "ランダムheterogeneous固定負荷率評価":
        st.markdown("### 実行設定")
        st.caption("全体負荷率は 0.1〜1.0（0.1刻み）を自動スイープします。")
        trial_count = st.number_input("各負荷率での試行回数", min_value=1, value=10, step=1)

        training_ratio = st.slider(
            "学習比率（全ユーザー共通の初期値）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="uniform_training_ratio",
        )

        for user_id in range(18):
            ratio_key = f"uniform_user_train_ratio_{user_id}"
            if ratio_key not in st.session_state:
                st.session_state[ratio_key] = float(training_ratio)

        with st.expander("ユーザー別 学習比率 (user0〜user17)", expanded=False):
            st.caption("必要時のみ個別調整してください。未調整時は上の共通値が使われます。")
            ratio_cols = st.columns([1, 2, 1, 1])
            with ratio_cols[0]:
                bulk_train_ratio = st.number_input(
                    "一括設定 学習比率",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(training_ratio),
                    step=0.01,
                    key="uniform_bulk_train_ratio",
                )
            with ratio_cols[1]:
                selected_ratio_users = st.multiselect(
                    "反映対象ユーザー",
                    options=list(range(18)),
                    default=[],
                    key="uniform_selected_train_users",
                )
            with ratio_cols[2]:
                apply_ratio_all = st.button("全ユーザー一括反映", key="uniform_apply_ratio_all")
            with ratio_cols[3]:
                apply_ratio_selected = st.button("選択ユーザー反映", key="uniform_apply_ratio_selected")

            if apply_ratio_all:
                for user_id in range(18):
                    st.session_state[f"uniform_user_train_ratio_{user_id}"] = float(bulk_train_ratio)
                st.success("全ユーザーの学習比率を反映しました。")

            if apply_ratio_selected:
                for user_id in selected_ratio_users:
                    st.session_state[f"uniform_user_train_ratio_{user_id}"] = float(bulk_train_ratio)
                st.success("選択ユーザーの学習比率を反映しました。")

            for row_idx in range(6):
                cols = st.columns(3)
                for col_idx in range(3):
                    user_id = row_idx * 3 + col_idx
                    if user_id >= 18:
                        continue
                    with cols[col_idx]:
                        st.number_input(
                            f"user{user_id} 学習比率",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            key=f"uniform_user_train_ratio_{user_id}",
                        )

        st.markdown("### タスク分布設定")
        c1, c2 = st.columns(2)
        with c1:
            inf_mean = st.number_input("推論タスク平均サイズ", min_value=1.0, value=9580.0, step=100.0, key="hetero_inf_mean")
            inf_std = st.number_input("推論タスク標準偏差", min_value=0.0, value=7000.0, step=100.0, key="hetero_inf_std")
            simulation_time = st.number_input("総シミュレーション時間（秒）", min_value=1, value=864000, step=3600, key="hetero_sim_time")
        with c2:
            train_mean = st.number_input("学習タスク平均サイズ", min_value=1.0, value=412180.0, step=1000.0, key="hetero_train_mean")
            train_std = st.number_input("学習タスク標準偏差", min_value=0.0, value=600000.0, step=1000.0, key="hetero_train_std")
            random_seed = st.number_input("乱数シード", min_value=0, value=42, step=1, key="hetero_seed")

        st.markdown("#### GPU処理能力 (TFLOPS)")
        g1, g2, g3 = st.columns(3)
        with g1:
            tier1_rate = st.number_input("tier1", min_value=0.1, value=2.98, step=0.1, key="hetero_tier1")
            tier2_rate = st.number_input("tier2", min_value=0.1, value=8.87, step=0.1, key="hetero_tier2")
            tier3_rate = st.number_input("tier3", min_value=0.1, value=20.41, step=0.1, key="hetero_tier3")
        with g2:
            tier4_rate = st.number_input("tier4", min_value=0.1, value=64.83, step=0.1, key="hetero_tier4")
            tier5_rate = st.number_input("tier5", min_value=0.1, value=82.60, step=0.1, key="hetero_tier5")
            tier6_rate = st.number_input("tier6", min_value=0.1, value=110.00, step=0.1, key="hetero_tier6")
        with g3:
            tier7_rate = st.number_input("tier7", min_value=0.1, value=180.50, step=0.1, key="hetero_tier7")
            tier8_rate = st.number_input("tier8", min_value=0.1, value=233.00, step=0.1, key="hetero_tier8")
            tier9_rate = st.number_input("tier9", min_value=0.1, value=311.84, step=0.1, key="hetero_tier9")

        st.markdown("#### ACP常駐GPU")
        acp_resident_gpu_count = st.number_input("ACP常駐GPU台数", min_value=0, value=0, step=1, key="hetero_acp_count")
        acp_resident_gpu_rates: list[float] = []
        if int(acp_resident_gpu_count) > 0:
            st.caption("台数分だけ性能を個別指定できます。")
            acp_cols = st.columns(2 if int(acp_resident_gpu_count) <= 2 else 3)
            for idx in range(int(acp_resident_gpu_count)):
                with acp_cols[idx % len(acp_cols)]:
                    acp_resident_gpu_rates.append(
                        st.number_input(
                            f"ACP GPU {idx + 1} 性能 (TFLOPS)",
                            min_value=0.1,
                            value=180.50,
                            step=0.1,
                            key=f"hetero_acp_resident_gpu_rate_{idx}",
                        )
                    )

        st.markdown("#### プリエンプト時オーバーヘッドコスト")
        o1, o2 = st.columns(2)
        with o1:
            inf_overhead = st.number_input("推論タスクオーバーヘッド係数", min_value=0.0, value=0.2, step=0.05, key="hetero_inf_overhead")
        with o2:
            train_overhead = st.number_input("学習タスクオーバーヘッド係数", min_value=0.0, value=0.2, step=0.05, key="hetero_train_overhead")

        user_training_ratios = [
            float(st.session_state[f"uniform_user_train_ratio_{uid}"])
            for uid in range(18)
        ]

        custom_args = [
            "--trial_count", str(int(trial_count)),
            "--training_ratio", str(float(training_ratio)),
            "--user_training_ratios", ",".join(str(v) for v in user_training_ratios),
            "--inference_mean", str(inf_mean),
            "--inference_std", str(inf_std),
            "--training_mean", str(train_mean),
            "--training_std", str(train_std),
            "--simulation_time", str(int(simulation_time)),
            "--seed", str(int(random_seed)),
            "--tier_rates", ",".join(str(rate) for rate in [tier1_rate, tier2_rate, tier3_rate, tier4_rate, tier5_rate, tier6_rate, tier7_rate, tier8_rate, tier9_rate]),
            "--acp_resident_gpu_count", str(int(acp_resident_gpu_count)),
            "--acp_resident_gpu_rates", ",".join(str(rate) for rate in acp_resident_gpu_rates),
            "--inf_overhead", str(inf_overhead),
            "--train_overhead", str(train_overhead),
        ]
    elif mode == "ユーザー別到着率で比較":
        st.markdown("### カスタム入力")

        c1, c2 = st.columns(2)
        with c1:
            inf_mean = st.number_input("推論タスク平均サイズ", min_value=1.0, value=9580.0, step=100.0, key="user_arrival_inf_mean")
            inf_std = st.number_input("推論タスク標準偏差", min_value=0.0, value=7000.0, step=100.0, key="user_arrival_inf_std")
            simulation_time = st.number_input("総シミュレーション時間（秒）", min_value=1, value=86400, step=3600, key="user_arrival_sim_time")
        with c2:
            train_mean = st.number_input("学習タスク平均サイズ", min_value=1.0, value=412180.0, step=1000.0, key="user_arrival_train_mean")
            train_std = st.number_input("学習タスク標準偏差", min_value=0.0, value=600000.0, step=1000.0, key="user_arrival_train_std")
            random_seed = st.number_input("乱数シード", min_value=0, value=42, step=1, key="user_arrival_seed")

        st.markdown("#### GPU処理能力 (TFLOPS)")
        g1, g2, g3 = st.columns(3)
        with g1:
            tier1_rate = st.number_input("tier1", min_value=0.1, value=2.98, step=0.1, key="user_arrival_tier1")
            tier2_rate = st.number_input("tier2", min_value=0.1, value=8.87, step=0.1, key="user_arrival_tier2")
            tier3_rate = st.number_input("tier3", min_value=0.1, value=20.41, step=0.1, key="user_arrival_tier3")
        with g2:
            tier4_rate = st.number_input("tier4", min_value=0.1, value=64.83, step=0.1, key="user_arrival_tier4")
            tier5_rate = st.number_input("tier5", min_value=0.1, value=82.60, step=0.1, key="user_arrival_tier5")
            tier6_rate = st.number_input("tier6", min_value=0.1, value=110.00, step=0.1, key="user_arrival_tier6")
        with g3:
            tier7_rate = st.number_input("tier7", min_value=0.1, value=180.50, step=0.1, key="user_arrival_tier7")
            tier8_rate = st.number_input("tier8", min_value=0.1, value=233.00, step=0.1, key="user_arrival_tier8")
            tier9_rate = st.number_input("tier9", min_value=0.1, value=311.84, step=0.1, key="user_arrival_tier9")

        for user_id in range(18):
            load_key = f"user_arrival_user_rate_{user_id}"
            train_key = f"user_arrival_user_train_ratio_{user_id}"
            if load_key not in st.session_state:
                st.session_state[load_key] = 0.5
            if train_key not in st.session_state:
                st.session_state[train_key] = 0.5

        with st.expander("ユーザー別到着率 (user0〜user17)", expanded=False):
            st.caption("必要な時だけ開いて編集できます。")
            load_apply_cols = st.columns([1, 2, 1, 1])
            with load_apply_cols[0]:
                bulk_load_rate = st.number_input(
                    "一括設定到着率",
                    min_value=0.0,
                    value=0.005,
                    step=0.00001,
                    format="%.5f",
                    key="user_arrival_bulk_load_rate",
                )
            with load_apply_cols[1]:
                selected_load_users = st.multiselect(
                    "反映対象ユーザー",
                    options=list(range(18)),
                    default=[],
                    key="user_arrival_selected_load_users",
                )
            with load_apply_cols[2]:
                apply_load_all = st.button("全ユーザー一括反映", key="user_arrival_apply_load_all")
            with load_apply_cols[3]:
                apply_load_selected = st.button("選択ユーザー反映", key="user_arrival_apply_load_selected")

            if apply_load_all:
                for user_id in range(18):
                    st.session_state[f"user_arrival_user_rate_{user_id}"] = float(bulk_load_rate)
                st.success("全ユーザーの到着率を反映しました。")

            if apply_load_selected:
                for user_id in selected_load_users:
                    st.session_state[f"user_arrival_user_rate_{user_id}"] = float(bulk_load_rate)
                st.success("選択ユーザーの到着率を反映しました。")

            for row_idx in range(6):
                cols = st.columns(3)
                for col_idx in range(3):
                    user_id = row_idx * 3 + col_idx
                    if user_id >= 18:
                        continue
                    with cols[col_idx]:
                        st.number_input(
                            f"user{user_id} 到着率",
                            min_value=0.0,
                            step=0.00001,
                            format="%.5f",
                            key=f"user_arrival_user_rate_{user_id}",
                        )

        with st.expander("ユーザー別 学習/推論 比率 (user0〜user17)", expanded=False):
            st.caption("必要な時だけ開いて編集できます。推論比率は 1 - 学習比率 で自動計算されます。")
            train_apply_cols = st.columns([1, 2, 1, 1])
            with train_apply_cols[0]:
                bulk_train_ratio = st.number_input(
                    "一括設定学習比率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    format="%.2f",
                    key="user_arrival_bulk_train_ratio",
                )
            with train_apply_cols[1]:
                selected_train_users = st.multiselect(
                    "反映対象ユーザー",
                    options=list(range(18)),
                    default=[],
                    key="user_arrival_selected_train_users",
                )
            with train_apply_cols[2]:
                apply_train_all = st.button("全ユーザー一括反映", key="user_arrival_apply_train_all")
            with train_apply_cols[3]:
                apply_train_selected = st.button("選択ユーザー反映", key="user_arrival_apply_train_selected")

            if apply_train_all:
                for user_id in range(18):
                    st.session_state[f"user_arrival_user_train_ratio_{user_id}"] = float(bulk_train_ratio)
                st.success("全ユーザーの学習比率を反映しました。")

            if apply_train_selected:
                for user_id in selected_train_users:
                    st.session_state[f"user_arrival_user_train_ratio_{user_id}"] = float(bulk_train_ratio)
                st.success("選択ユーザーの学習比率を反映しました。")

            for row_idx in range(6):
                cols = st.columns(3)
                for col_idx in range(3):
                    user_id = row_idx * 3 + col_idx
                    if user_id >= 18:
                        continue
                    with cols[col_idx]:
                        st.number_input(
                            f"user{user_id} 学習比率",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.05,
                            format="%.2f",
                            key=f"user_arrival_user_train_ratio_{user_id}",
                        )

        user_load_rates: list[float] = [float(st.session_state[f"user_arrival_user_rate_{user_id}"]) for user_id in range(18)]
        user_training_ratios: list[float] = [float(st.session_state[f"user_arrival_user_train_ratio_{user_id}"]) for user_id in range(18)]
        training_ratio = sum(user_training_ratios) / len(user_training_ratios) if user_training_ratios else 0.5

        st.markdown("#### ACP常駐GPU")
        acp_resident_gpu_count = st.number_input("ACP常駐GPU台数", min_value=0, value=0, step=1, key="user_arrival_acp_count")
        acp_resident_gpu_rates: list[float] = []
        if int(acp_resident_gpu_count) > 0:
            st.caption("台数分だけ性能を個別指定できます。")
            acp_cols = st.columns(2 if int(acp_resident_gpu_count) <= 2 else 3)
            for idx in range(int(acp_resident_gpu_count)):
                with acp_cols[idx % len(acp_cols)]:
                    acp_resident_gpu_rates.append(
                        st.number_input(
                            f"ACP GPU {idx + 1} 性能 (TFLOPS)",
                            min_value=0.1,
                            value=180.50,
                            step=0.1,
                            key=f"user_arrival_acp_rate_{idx}",
                        )
                    )

        st.markdown("#### プリエンプト時オーバーヘッドコスト")
        o1, o2 = st.columns(2)
        with o1:
            inf_overhead = st.number_input("推論タスクオーバーヘッド係数", min_value=0.0, value=0.2, step=0.05, key="user_arrival_inf_ov")
        with o2:
            train_overhead = st.number_input("学習タスクオーバーヘッド係数", min_value=0.0, value=0.2, step=0.05, key="user_arrival_train_ov")

        custom_args = [
            "--training-ratio", str(training_ratio),
            "--inference-mean", str(inf_mean),
            "--inference-std", str(inf_std),
            "--training-mean", str(train_mean),
            "--training-std", str(train_std),
            "--simulation-time", str(int(simulation_time)),
            "--seed", str(int(random_seed)),
            "--tier1-rate", str(tier1_rate),
            "--tier2-rate", str(tier2_rate),
            "--tier3-rate", str(tier3_rate),
            "--tier4-rate", str(tier4_rate),
            "--tier5-rate", str(tier5_rate),
            "--tier6-rate", str(tier6_rate),
            "--tier7-rate", str(tier7_rate),
            "--tier8-rate", str(tier8_rate),
            "--tier9-rate", str(tier9_rate),
            "--user-load-rates", ",".join(str(rate) for rate in user_load_rates),
            "--user-training-ratios", ",".join(str(ratio) for ratio in user_training_ratios),
            "--acp-resident-gpu-count", str(int(acp_resident_gpu_count)),
            "--acp-resident-gpu-rates", ",".join(str(rate) for rate in acp_resident_gpu_rates),
            "--inf-overhead", str(inf_overhead),
            "--train-overhead", str(train_overhead),
        ]

    if st.button("実行する", type="primary", disabled=not exists):
        with st.spinner("シミュレーションを実行中です..."):
            code, logs = run_script(script, custom_args)

        if code == 0:
            st.success("実行が完了しました。")
        else:
            st.error(f"実行中にエラーが発生しました（終了コード: {code}）。")

        st.text_area("実行ログ", logs or "(出力なし)", height=320)

        if code == 0 and mode in ["ランダムheterogeneous固定負荷率評価", "ユーザー別到着率で比較"]:
            output_dir = parse_custom_output_dir(logs)
            if output_dir is None:
                st.warning("出力フォルダを特定できませんでした。結果タブから確認してください。")
            else:
                st.caption(f"出力先: {output_dir}")

                if mode == "ランダムheterogeneous固定負荷率評価":
                    paper_plot_script = "plot_paper_graphs_from_csv.py"
                    trial_csv = output_dir / "trial_results.csv"
                    if trial_csv.exists():
                        post_code, post_logs = run_script(
                            paper_plot_script,
                            ["--csv-path", str(trial_csv), "--output-dir", str(output_dir)],
                        )
                        if post_code != 0:
                            st.warning(f"論文用グラフの自動生成に失敗しました（終了コード: {post_code}）。")
                            st.text_area("後処理ログ", post_logs or "(出力なし)", height=180)

                    summary_csv = output_dir / "summary_by_load.csv"
                    if summary_csv.exists():
                        try:
                            summary_df = pd.read_csv(summary_csv)
                            st.subheader("Summary Results by Load")
                            st.dataframe(summary_df, use_container_width=True)
                        except Exception as e:  # noqa: BLE001
                            st.warning(f"サマリー結果の表示に失敗しました: {e}")

                    load_setup_csv = output_dir / "load_setup.csv"
                    if load_setup_csv.exists():
                        with st.expander("負荷率ごとの到着率計算結果"):
                            st.dataframe(pd.read_csv(load_setup_csv), use_container_width=True)

                    paper_graphs = [
                        ("tier9_tat_combined_band.png", "Tier9 TAT"),
                        ("tier8_tat_combined_band.png", "Tier8 TAT"),
                        ("protection_ratio_without_fcfs.png", "Tier9 Protection Ratio"),
                    ]
                    existing_paper_graphs = [(name, title) for name, title in paper_graphs if (output_dir / name).exists()]
                    if existing_paper_graphs:
                        st.subheader("論文用グラフ")
                        cols = st.columns(2)
                        for idx, (name, title) in enumerate(existing_paper_graphs):
                            with cols[idx % 2]:
                                st.image(str(output_dir / name), caption=title, use_container_width=True)

                    if trial_csv.exists():
                        with st.expander("試行ごとの詳細結果"):
                            try:
                                trial_df = pd.read_csv(trial_csv)
                                st.dataframe(trial_df, use_container_width=True)
                            except Exception as e:  # noqa: BLE001
                                st.warning(f"試行結果の表示に失敗しました: {e}")
                else:
                    # 既存モード: ユーザー別到着率で比較
                    # 実行設定と結果数値を先に表示して、変化有無を確認しやすくする
                    result_json_name = "user_arrival_results.json"
                    result_json_path = output_dir / result_json_name
                    if result_json_path.exists():
                        try:
                            result_data = json.loads(result_json_path.read_text(encoding="utf-8"))
                            acp_info = result_data.get("acp_resident_gpu", {})
                            st.markdown("### 実行設定の確認")
                            st.write(f"ACP常駐GPU台数: {acp_info.get('count', 0)}")
                            st.write(f"ACP常駐GPU性能: {acp_info.get('rates', [])}")

                            overall_csv = output_dir / "user_arrival_overall_results.csv"
                            user_csv = output_dir / "user_arrival_user_tat_results.csv"

                            if overall_csv.exists():
                                st.markdown("### 方式別サマリー")
                                st.dataframe(pd.read_csv(overall_csv), use_container_width=True)

                            if user_csv.exists():
                                user_df = pd.read_csv(user_csv, index_col=0)
                                if not user_df.empty:
                                    st.markdown("### ユーザー別TAT表")
                                    st.dataframe(user_df, use_container_width=True)

                                    long_user_csv = output_dir / "user_arrival_user_tat_results_long.csv"
                                    if long_user_csv.exists():
                                        with st.expander("ユーザー別詳細データ"):
                                            st.dataframe(pd.read_csv(long_user_csv), use_container_width=True)
                        except Exception as e:  # noqa: BLE001
                            st.warning(f"結果JSONの表示に失敗しました: {e}")

                    graph_paths = [
                        output_dir / "user_0_to_8_tat_by_scenario.png",
                        output_dir / "user_9_to_17_tat_by_scenario.png",
                    ]
                    existing_graphs = [path for path in graph_paths if path.exists()]
                    if existing_graphs:
                        st.markdown("### ユーザー別TATグラフ")
                        cols = st.columns(len(existing_graphs))
                        for col, graph_path in zip(cols, existing_graphs):
                            with col:
                                st.image(str(graph_path), caption=graph_path.name, use_container_width=True)
with view_tab:
    st.subheader("結果ファイル")
    files = collect_output_files()

    if not files:
        st.info("表示できる結果ファイルがありません。先にシミュレーションを実行してください。")
    else:
        st.write(f"検出ファイル数: {len(files)}")

        selected = st.selectbox(
            "ファイルを選択",
            files,
            format_func=lambda p: to_rel(p),
        )

        rel = to_rel(selected)
        st.caption(rel)
        st.info(graph_description(selected))

        suffix = selected.suffix.lower()
        if suffix == ".png":
            st.image(str(selected), width="stretch")
        elif suffix == ".csv":
            try:
                df = pd.read_csv(selected)
                st.dataframe(df, width="stretch")
            except Exception as e:  # noqa: BLE001
                st.error(f"CSVの読み込みに失敗しました: {e}")
        elif suffix == ".json":
            try:
                data = json.loads(selected.read_text(encoding="utf-8"))
                st.json(data)
            except Exception as e:  # noqa: BLE001
                st.error(f"JSONの読み込みに失敗しました: {e}")

        st.markdown("---")
        st.markdown("### 最新5件")
        for p in files[:5]:
            st.write(f"- {to_rel(p)}")

with help_tab:
        st.subheader("グラフの見方（run_all_simulations.py 出力）")
        st.markdown(
                """
- **user_comparison_XX.png**
    - 左上: ユーザーXXの方式別平均待ち時間（低いほど良い）
    - 右上: ユーザーXXの方式別全タスク完了時刻（低いほど早く終わる）
    - 下段: 他GPU利用率・他人が自GPUで処理した件数

- **scenario_table_users_0_to_2.png**
    - ユーザー0〜2の方式別平均待ち時間の一覧表

- **users_3_to_8_line_graph.png**
    - ユーザー3〜8について、方式ごとの平均待ち時間を線で比較

- **load_rate_*.png**
    - 横軸が負荷率。方式ごとに負荷増加で待ち時間がどう変わるかを確認

- **participation_by_load_*.png**
    - 負荷率に対する参加者数（または参加率）の推移
                """
        )
