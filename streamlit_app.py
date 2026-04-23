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
    "カスタム設定で負荷率比較": {
        "script": "run_custom_multi_load_web.py",
        "summary": "推論/学習比率、タスク平均サイズ・標準偏差、総シミュレーション時間を指定して4グラフを生成します。",
        "outputs": [
            "simulation/outputs/multi_load/custom_web/<timestamp>/load_rate_all.png",
            "simulation/outputs/multi_load/custom_web/<timestamp>/load_rate_low.png",
            "simulation/outputs/multi_load/custom_web/<timestamp>/load_rate_mid.png",
            "simulation/outputs/multi_load/custom_web/<timestamp>/load_rate_high.png",
        ],
    },
    "ユーザー別到着率で比較": {
        "script": "run_custom_user_arrival_web.py",
        "summary": "ユーザー0〜17ごとに到着率を個別指定して、4方式の全体指標とユーザー別TATを比較します。",
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
    for line in (logs or "").splitlines():
        if line.startswith(marker):
            raw = line.replace(marker, "", 1).strip()
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
    if mode == "カスタム設定で負荷率比較":
        st.markdown("### カスタム入力")
        training_ratio = st.slider("学習タスク比率", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        inference_ratio = 1.0 - training_ratio
        st.caption(f"推論タスク比率: {inference_ratio:.2f}, 学習タスク比率: {training_ratio:.2f}")

        c1, c2 = st.columns(2)
        with c1:
            inf_mean = st.number_input("推論タスク平均サイズ", min_value=1.0, value=9580.0, step=100.0)
            inf_std = st.number_input("推論タスク標準偏差", min_value=0.0, value=7000.0, step=100.0)
            simulation_time = st.number_input("総シミュレーション時間（秒）", min_value=1, value=86400, step=3600)
        with c2:
            train_mean = st.number_input("学習タスク平均サイズ", min_value=1.0, value=412180.0, step=1000.0)
            train_std = st.number_input("学習タスク標準偏差", min_value=0.0, value=600000.0, step=1000.0)
            random_seed = st.number_input("乱数シード", min_value=0, value=42, step=1)

        st.markdown("#### GPU処理能力 (TFLOPS)")
        g1, g2, g3 = st.columns(3)
        with g1:
            tier1_rate = st.number_input("tier1", min_value=0.1, value=2.98, step=0.1)
            tier2_rate = st.number_input("tier2", min_value=0.1, value=8.87, step=0.1)
            tier3_rate = st.number_input("tier3", min_value=0.1, value=20.41, step=0.1)
        with g2:
            tier4_rate = st.number_input("tier4", min_value=0.1, value=64.83, step=0.1)
            tier5_rate = st.number_input("tier5", min_value=0.1, value=82.60, step=0.1)
            tier6_rate = st.number_input("tier6", min_value=0.1, value=110.00, step=0.1)
        with g3:
            tier7_rate = st.number_input("tier7", min_value=0.1, value=180.50, step=0.1)
            tier8_rate = st.number_input("tier8", min_value=0.1, value=233.00, step=0.1)
            tier9_rate = st.number_input("tier9", min_value=0.1, value=311.84, step=0.1)

        st.markdown("#### ACP常駐GPU")
        acp_resident_gpu_count = st.number_input("ACP常駐GPU台数", min_value=0, value=0, step=1)
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
                            key=f"acp_resident_gpu_rate_{idx}",
                        )
                    )

        st.markdown("#### プリエンプト時オーバーヘッドコスト")
        o1, o2 = st.columns(2)
        with o1:
            inf_overhead = st.number_input("推論タスクオーバーヘッド係数", min_value=0.0, value=0.2, step=0.05)
        with o2:
            train_overhead = st.number_input("学習タスクオーバーヘッド係数", min_value=0.0, value=0.2, step=0.05)

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
            "--acp-resident-gpu-count", str(int(acp_resident_gpu_count)),
            "--acp-resident-gpu-rates", ",".join(str(rate) for rate in acp_resident_gpu_rates),
            "--inf-overhead", str(inf_overhead),
            "--train-overhead", str(train_overhead),
        ]
    elif mode == "ユーザー別到着率で比較":
        st.markdown("### カスタム入力")
        training_ratio = st.slider("学習タスク比率", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="user_arrival_training_ratio")
        inference_ratio = 1.0 - training_ratio
        st.caption(f"推論タスク比率: {inference_ratio:.2f}, 学習タスク比率: {training_ratio:.2f}")

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

        st.markdown("#### ユーザー別到着率 (user0〜user17)")
        user_rates: list[float] = []
        for row_idx in range(6):
            cols = st.columns(3)
            for col_idx in range(3):
                user_id = row_idx * 3 + col_idx
                if user_id >= 18:
                    continue
                with cols[col_idx]:
                    user_rates.append(
                        st.number_input(
                            f"user{user_id} 到着率",
                            min_value=0.0,
                            value=0.005,
                            step=0.001,
                            format="%.3f",
                            key=f"user_arrival_user_rate_{user_id}",
                        )
                    )

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
            "--user-rates", ",".join(str(rate) for rate in user_rates),
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

        if code == 0 and mode in ["カスタム設定で負荷率比較", "ユーザー別到着率で比較"]:
            output_dir = parse_custom_output_dir(logs)
            if output_dir is None:
                st.warning("出力フォルダを特定できませんでした。結果タブから確認してください。")
            else:
                st.caption(f"出力先: {output_dir}")

                # 実行設定と結果数値を先に表示して、変化有無を確認しやすくする
                result_json_name = "load_rate_results.json" if mode == "カスタム設定で負荷率比較" else "user_arrival_results.json"
                result_json_path = output_dir / result_json_name
                if result_json_path.exists():
                    try:
                        result_data = json.loads(result_json_path.read_text(encoding="utf-8"))
                        acp_info = result_data.get("acp_resident_gpu", {})
                        st.markdown("### 実行設定の確認")
                        st.write(f"ACP常駐GPU台数: {acp_info.get('count', 0)}")
                        st.write(f"ACP常駐GPU性能: {acp_info.get('rates', [])}")

                        if mode == "カスタム設定で負荷率比較":
                            target_load = result_data.get("target_load_rates", [])
                            result_rows = {
                                "Target Load": target_load,
                                "No Sharing": result_data.get("results", {}).get("No Sharing", []),
                                "FCFS": result_data.get("results", {}).get("FCFS", []),
                                "Owner Priority": result_data.get("results", {}).get("Owner Priority", []),
                                "Preemptive": result_data.get("results", {}).get("Preemptive", []),
                            }
                            st.markdown("### 方式別TAT（数値）")
                            st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

                            user_tat_csv = output_dir / "user_average_tat_results.csv"
                            if user_tat_csv.exists():
                                try:
                                    user_tat_df = pd.read_csv(user_tat_csv)
                                    if not user_tat_df.empty:
                                        st.markdown("### ユーザー別TAT表")
                                        for scenario_label in ["No Sharing", "FCFS", "Owner Priority", "Preemptive"]:
                                            scenario_df = user_tat_df[user_tat_df["scenario"] == scenario_label]
                                            if scenario_df.empty:
                                                continue
                                            pivot_df = scenario_df.pivot_table(
                                                index="user_id",
                                                columns="target_load",
                                                values="avg_tat",
                                                aggfunc="mean",
                                            ).sort_index()
                                            pivot_df.columns = [f"Load {float(col):.1f}" for col in pivot_df.columns]
                                            st.markdown(f"#### {scenario_label}")
                                            st.dataframe(pivot_df, use_container_width=True)
                                except Exception as e:  # noqa: BLE001
                                    st.warning(f"ユーザー別TAT表の表示に失敗しました: {e}")
                        else:
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

                if mode == "カスタム設定で負荷率比較":
                    st.markdown("### 生成された4グラフ")
                    col_left, col_right = st.columns(2)
                    first_row = ["load_rate_all.png", "load_rate_low.png"]
                    second_row = ["load_rate_mid.png", "load_rate_high.png"]

                    for idx, name in enumerate(first_row):
                        img_path = output_dir / name
                        if img_path.exists():
                            with (col_left if idx == 0 else col_right):
                                st.image(str(img_path), caption=name, use_container_width=True)

                    col_left2, col_right2 = st.columns(2)
                    for idx, name in enumerate(second_row):
                        img_path = output_dir / name
                        if img_path.exists():
                            with (col_left2 if idx == 0 else col_right2):
                                st.image(str(img_path), caption=name, use_container_width=True)
                else:
                    graph_path = output_dir / "user_0_to_8_tat_by_scenario.png"
                    if graph_path.exists():
                        st.markdown("### ユーザー0〜8のTATグラフ")
                        st.image(str(graph_path), caption="user_0_to_8_tat_by_scenario.png", use_container_width=True)
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
