"""
高性能ユーザー保護の機械学習分析用データセットを生成する。

ランダム条件を多数生成し、4方式のシミュレーション結果を1行にまとめてCSVへ保存する。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from high_performance_protection_ml import build_dataset


def build_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset for high-performance-user protection analysis")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulation-time", type=int, default=86400)
    parser.add_argument("--output-dir", type=str, default="./outputs/high_performance_protection_ml")
    parser.add_argument(
        "--acp-rate-candidates",
        type=str,
        default="82.6,110.0,180.5,233.0,311.84",
        help="Comma-separated candidate processing rates for ACP resident GPUs",
    )
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    text = (value or "").strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    acp_rate_candidates = parse_float_list(args.acp_rate_candidates)

    output_dir = build_output_dir(args.output_dir)
    dataset = build_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        simulation_time=args.simulation_time,
        acp_rate_candidates=acp_rate_candidates,
    )

    csv_path = output_dir / "high_performance_protection_dataset.csv"
    dataset.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "dataset_summary.csv"
    summary = dataset.describe(include="all")
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print(f"DATASET_OUTPUT_DIR={output_dir.resolve()}")
    print(f"DATASET_CSV={csv_path.resolve()}")
    print(f"ROWS={len(dataset)}")
    print(f"COLUMNS={len(dataset.columns)}")


if __name__ == "__main__":
    main()
