"""
update_imgs.py  ―  シミュレーション全実行 → imgs/ 一括更新スクリプト

使い方:
    py -3 update_imgs.py            # 全ステップ実行
    py -3 update_imgs.py --skip-sim # シミュレーションをスキップして imgs/ だけ更新
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── パス設定 ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent                           # 卒論ルート
SIM        = ROOT / "simulation"                             # シミュレーションフォルダ
OUT        = SIM / "outputs"
IMGS       = ROOT / "imgs"

# シミュレーション出力先
PAPER_FIGS = OUT / "paper_figures"
CASCADE    = OUT / "participation_cascade"
HETERO     = OUT / "hetero_scenarios"

# 最新の trial_results.csv (random_hetero_fixed_load の最新フォルダを自動検出)
def _latest_trial_csv() -> Path:
    base = OUT / "random_hetero_fixed_load" / "custom_web"
    dirs = sorted(base.glob("*/trial_results.csv"), key=lambda p: p.parent.name)
    if not dirs:
        raise FileNotFoundError(f"trial_results.csv not found under {base}")
    return dirs[-1]

# ── Step 1a: メインシミュレーション実行 ───────────────────────────────
def run_main_simulation() -> None:
    print("[main-sim] Running uniform load sweep (100 trials, 10 days, ratio=0.3)…")
    subprocess.run(
        [sys.executable,
         str(SIM / "run_random_hetero_fixed_load_web.py"),
         "--trial_count",    "100",
         "--training_ratio", "0.3",
         "--simulation_time","864000"],
        check=True,
        cwd=str(SIM),
    )

# ── Step 1b: グラフ生成 (plot_paper_figures.py) ────────────────────────
def run_plot_figures() -> None:
    csv = _latest_trial_csv()
    print(f"[plot] Using trial CSV: {csv}")
    subprocess.run(
        [sys.executable,
         str(SIM / "plot_paper_figures.py"),
         "--csv",        str(csv),
         "--output-dir", str(PAPER_FIGS),
         "--hetero-base", str(HETERO)],
        check=True,
        cwd=str(SIM),
    )

# ── Step 2: cascade シミュレーション ──────────────────────────────────
def run_cascade() -> None:
    print("[cascade] Running participation cascade simulation…")
    subprocess.run(
        [sys.executable, str(SIM / "run_participation_cascade.py")],
        check=True,
        cwd=str(SIM),
    )

# ── Step 3: hetero PR broken 図生成 ───────────────────────────────────
def run_hetero_pr_broken() -> None:
    print("[hetero_pr] Running make_hetero_pr_broken.py…")
    subprocess.run(
        [sys.executable, str(SIM / "make_hetero_pr_broken.py")],
        check=True,
        cwd=str(SIM),
    )

# ── Step 4: imgs/ へコピー ────────────────────────────────────────────
HETERO_SCENARIOS = ["uniform", "low_heavy", "high_heavy", "random"]

def copy_to_imgs() -> None:
    IMGS.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = []

    def cp(src: Path, dst: Path) -> None:
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(dst.relative_to(ROOT))
        else:
            skipped.append(src.relative_to(ROOT))

    # paper_figures → imgs/ (フラット)
    for f in ["fig1_overall_avg_tat.pdf", "fig2_tier_based_tat.pdf",
              "fig3_tier9_tat.pdf",       "fig4_protection_ratio.pdf",
              "hetero_pr_3panel.png"]:
        cp(PAPER_FIGS / f, IMGS / f)

    # hetero 合成PNG → imgs/
    for scenario in HETERO_SCENARIOS:
        f = f"hetero_{scenario}.png"
        cp(PAPER_FIGS / f, IMGS / f)

    # cascade → imgs/ (フラット、元ファイル名のまま)
    for f in ["cascade_stacked_3panel.png"]:
        cp(CASCADE / f, IMGS / f)

    # make_hetero_pr_broken.py が imgs/ に直接書き出す *_pr_broken.png はコピー不要
    # (すでに正しい場所にある)

    print(f"\n[copy] Copied {len(copied)} file(s) to imgs/")
    for p in copied:
        print(f"  OK  {p}")
    if skipped:
        print(f"\n[copy] Skipped {len(skipped)} missing source(s):")
        for p in skipped:
            print(f"  --  {p}")

# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation runs, only refresh imgs/")
    args = parser.parse_args()

    if not args.skip_sim:
        run_main_simulation()
        run_plot_figures()
        run_cascade()
        run_hetero_pr_broken()

    copy_to_imgs()
    print("\nDone. imgs/ is up to date.")

if __name__ == "__main__":
    main()
