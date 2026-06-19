"""
4つの異種ワークロードシナリオを実行しグラフを生成する。

  uniform    - 全ユーザー training_ratio = 0.3（固定）
  low_heavy  - Low 0.7 / Mid 0.3 / High 0.1（固定）
  high_heavy - Low 0.1 / Mid 0.3 / High 0.7（固定）
  random     - 毎 trial ごとに 18 ユーザー全員の比率をランダム再サンプル

出力： outputs/hetero_scenarios/{scenario}/
  low_tier_tat.png      Mid_tier_tat.png      high_tier_tat.png（No Sharing 含む）
  protection_ratio.png
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path

import config
from run_random_hetero_fixed_load_web import (
    run_uniform_load_sweep,
    run_single_trial,
    LOAD_POINTS,
)
from plot_paper_graphs_from_csv import (
    add_tier_tat_columns,
    _plot_tier_group_band,
    summarize_metric,
    compute_protection_ratio,
    plot_band,
    METHOD_ORDER,
    SHARING_METHODS,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRIAL_COUNT = 100
SEED        = 42
OUTPUT_BASE = Path("./outputs/hetero_scenarios")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

LOW_USERS  = [0, 1, 2, 9, 10, 11]
MID_USERS  = [3, 4, 5, 12, 13, 14]

FIXED_SCENARIOS: dict[str, list[float]] = {
    "uniform":    [0.3] * 18,
    "low_heavy":  [0.7 if u in LOW_USERS else (0.3 if u in MID_USERS else 0.1) for u in range(18)],
    "high_heavy": [0.1 if u in LOW_USERS else (0.3 if u in MID_USERS else 0.7) for u in range(18)],
}

SCENARIO_LABELS = {
    "uniform":    "Uniform (all 0.3)",
    "low_heavy":  "Low-Heavy (Low=0.7, High=0.1)",
    "high_heavy": "High-Heavy (Low=0.1, High=0.7)",
    "random":     "Random (per-trial resampled)",
}


# ── graph helpers ──────────────────────────────────────────────────────
def plot_protection_ratio(df: pd.DataFrame, output_path: Path) -> Path:
    ratio_df = compute_protection_ratio(df, "tier9_tat")
    summary  = summarize_metric(ratio_df, "protection_ratio")
    fig, ax  = plt.subplots(figsize=(10, 6))
    plot_band(ax, summary, ["FCFS", "Owner Priority", "Preemptive"],
              ylabel="Tier9 Protection Ratio", log_scale=False, ylim=(0.0, None))
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Baseline (= 1)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_graphs(trial_df: pd.DataFrame, out: Path) -> None:
    paper_df = add_tier_tat_columns(trial_df)
    out.mkdir(parents=True, exist_ok=True)

    _plot_tier_group_band(paper_df, "tier_low_tat",
                          "Low Tier (Tier 1–3) Average TAT [s]", "(a)",
                          out / "low_tier_tat.png", methods=SHARING_METHODS)
    _plot_tier_group_band(paper_df, "tier_mid_tat",
                          "Mid Tier (Tier 4–6) Average TAT [s]", "(b)",
                          out / "mid_tier_tat.png", methods=SHARING_METHODS)
    _plot_tier_group_band(paper_df, "tier_high_tat",
                          "High Tier (Tier 7–9) Average TAT [s]", "(c)",
                          out / "high_tier_tat.png", methods=METHOD_ORDER)
    plot_protection_ratio(paper_df, out / "protection_ratio.png")

    for f in sorted(out.glob("*.png")):
        print(f"  Saved: {f.name}")


# ── fixed-ratio scenarios ──────────────────────────────────────────────
def run_fixed_scenario(name: str, user_ratios: list[float]) -> None:
    print(f"\n{'='*60}")
    print(f"Scenario: {name}  ({SCENARIO_LABELS[name]})")
    print(f"trial_count={TRIAL_COUNT}")
    print(f"{'='*60}")
    _, trial_df, _ = run_uniform_load_sweep(
        trial_count=TRIAL_COUNT,
        training_ratio=float(np.mean(user_ratios)),
        user_training_ratios=user_ratios,
        seed=SEED,
    )
    generate_graphs(trial_df, OUTPUT_BASE / name)


# ── random scenario (per-trial resampling) ─────────────────────────────
def run_random_scenario() -> None:
    print(f"\n{'='*60}")
    print(f"Scenario: random  ({SCENARIO_LABELS['random']})")
    print(f"trial_count={TRIAL_COUNT}  (ratios resampled each trial)")
    print(f"{'='*60}")

    rng = np.random.default_rng(SEED)
    tier_rates = config.GPU_PERFORMANCE_LEVELS.copy()
    all_rows: list[dict] = []

    for load in LOAD_POINTS:
        print(f"  [Load {load:.1f}] {TRIAL_COUNT} trials ...")
        for run_id in range(TRIAL_COUNT):
            trial_ratios = list(rng.uniform(0.0, 1.0, 18))
            rows, _ = run_single_trial(
                target_load=load,
                run_id=run_id,
                base_seed=SEED,
                user_training_ratios=trial_ratios,
                inference_mean=9580.0,
                inference_std=7000.0,
                training_mean=412180.0,
                training_std=600000.0,
                simulation_time=864000,
                tier_rates=tier_rates,
                acp_resident_gpu_count=0,
                acp_resident_gpu_rates=None,
                inf_overhead=0.2,
                train_overhead=0.2,
            )
            all_rows.extend(rows)

    trial_df = pd.DataFrame(all_rows)
    out = OUTPUT_BASE / "random"
    trial_df.to_csv(out / "trial_results.csv" if (out / "trial_results.csv").parent.exists()
                    else OUTPUT_BASE / "random_trial_results.csv", index=False)
    generate_graphs(trial_df, out)


# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    print(f"=== Heterogeneous Workload Scenarios  (trials={TRIAL_COUNT}) ===")
    for name, ratios in FIXED_SCENARIOS.items():
        run_fixed_scenario(name, ratios)
    run_random_scenario()
    print(f"\nAll scenarios done. Results in: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
