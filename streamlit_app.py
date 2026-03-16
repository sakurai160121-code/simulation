from __future__ import annotations

import json
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
    "基本比較 + 反復最適化": {
        "script": "run_all_simulations.py",
        "summary": "4方式を比較し、ユーザー別比較図・表を生成します。",
        "outputs": [
            "simulation/outputs/basic_scenarios/*.png",
            "simulation/outputs/user_comparisons/*.png",
            "simulation/outputs/tables/*.png",
        ],
    },
    "負荷率別の方式比較": {
        "script": "run_multi_load_scenarios.py",
        "summary": "負荷率0.1〜1.0で4方式を比較し、待ち時間推移を出力します。",
        "outputs": [
            "simulation/outputs/multi_load/load_rate_*.png",
            "simulation/outputs/multi_load/load_rate_results.csv",
            "simulation/outputs/multi_load/load_rate_results.json",
        ],
    },
    "負荷率と参加者数分析": {
        "script": "run_multi_load_with_participation.py",
        "summary": "負荷率ごとの参加者数の推移（低/中/高性能）を分析します。",
        "outputs": [
            "simulation/outputs/multi_load/participation_by_load_*.png",
            "simulation/outputs/multi_load/participation_by_load_results.json",
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
]


def run_script(script_name: str) -> tuple[int, str]:
    cmd = [sys.executable, script_name]
    completed = subprocess.run(
        cmd,
        cwd=str(SIM_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return completed.returncode, output.strip()


def graph_description(path: Path) -> str:
    name = path.name
    for key, desc in GRAPH_HINTS:
        if key in name:
            return desc
    return "このファイル専用の説明は未登録です。ファイル名と出力先から実験目的を確認してください。"


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

    if st.button("実行する", type="primary", disabled=not exists):
        with st.spinner("シミュレーションを実行中です..."):
            code, logs = run_script(script)

        if code == 0:
            st.success("実行が完了しました。")
        else:
            st.error(f"実行中にエラーが発生しました（終了コード: {code}）。")

        st.text_area("実行ログ", logs or "(出力なし)", height=320)

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
