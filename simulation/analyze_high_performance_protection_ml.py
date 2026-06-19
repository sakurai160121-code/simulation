"""
高性能ユーザー保護の機械学習分析。

データセットCSVを読み込み、
分類: 高性能ユーザー保護で最良の方式
回帰: 各方式の高性能ユーザー保護比率
を学習・評価する。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from high_performance_protection_ml import best_method_label_order, dataset_feature_columns


plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def build_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze high-performance-user protection with ML")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/high_performance_protection_ml_analysis")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("dataset is empty")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = df[dataset_feature_columns()].copy()
    target = df["best_method_for_high_tier"].copy()
    return features, target


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances: pd.Series, title: str, output_path: Path, top_n: int = 15) -> None:
    series = importances.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(series.index[::-1], series.values[::-1], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ratio_distributions(df: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "fcfs_high_tier_tat_ratio",
        "owner_priority_high_tier_tat_ratio",
        "preemptive_high_tier_tat_ratio",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([df[col].dropna().values for col in cols], tick_labels=["FCFS", "Owner Priority", "Preemptive"])
    ax.set_ylabel("high_tier_tat_ratio")
    ax.set_title("Distribution of high_tier_tat_ratio by method")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_label_frequency(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["best_method_for_high_tier"].value_counts().reindex(best_method_label_order(), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values, color="#2ca02c")
    ax.set_ylabel("Frequency")
    ax.set_title("Best method frequency for high-tier protection")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boundary_slice(clf: RandomForestClassifier, features: pd.DataFrame, output_path: Path) -> None:
    importances = pd.Series(clf.feature_importances_, index=features.columns).sort_values(ascending=False)
    top_two = importances.head(2).index.tolist()
    if len(top_two) < 2:
        return

    feature_a, feature_b = top_two
    medians = features.median(numeric_only=True)
    x = np.linspace(features[feature_a].min(), features[feature_a].max(), 80)
    y = np.linspace(features[feature_b].min(), features[feature_b].max(), 80)
    xx, yy = np.meshgrid(x, y)

    grid = pd.DataFrame([medians.values] * xx.size, columns=features.columns)
    grid[feature_a] = xx.ravel()
    grid[feature_b] = yy.ravel()
    pred = clf.predict(grid)

    mapping = {label: idx for idx, label in enumerate(best_method_label_order())}
    z = np.array([mapping.get(label, -1) for label in pred]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, z, levels=len(mapping), alpha=0.35, cmap="viridis")
    ax.set_xlabel(feature_a)
    ax.set_ylabel(feature_b)
    ax.set_title("Approximate selection boundary slice")
    fig.colorbar(contour, ax=ax, ticks=list(mapping.values()), label="Predicted best method index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args.output_dir)
    df = load_dataset(args.dataset)

    feature_columns = dataset_feature_columns()
    X = df[feature_columns].copy()
    y = df["best_method_for_high_tier"].copy()

    stratify_target = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify_target,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=args.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    labels = best_method_label_order()
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    classification_summary = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    (output_dir / "classification_summary.json").write_text(json.dumps(classification_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    feature_importance = pd.Series(clf.feature_importances_, index=feature_columns).sort_values(ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance_classifier.csv", encoding="utf-8-sig")
    plot_feature_importance(feature_importance, "Classifier feature importance", output_dir / "classifier_feature_importance.png")

    perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=args.seed, n_jobs=-1)
    perm_importance = pd.Series(perm.importances_mean, index=feature_columns).sort_values(ascending=False)
    perm_importance.to_csv(output_dir / "permutation_importance_classifier.csv", encoding="utf-8-sig")
    plot_feature_importance(perm_importance, "Classifier permutation importance", output_dir / "classifier_permutation_importance.png")

    plot_ratio_distributions(df, output_dir / "high_tier_ratio_distributions.png")
    plot_label_frequency(df, output_dir / "best_method_frequency.png")
    plot_boundary_slice(clf, X, output_dir / "selection_boundary_slice.png")

    regression_targets = [
        "fcfs_high_tier_tat_ratio",
        "owner_priority_high_tier_tat_ratio",
        "preemptive_high_tier_tat_ratio",
    ]
    regression_metrics: dict[str, dict[str, float]] = {}
    for target in regression_targets:
        y_reg = df[target].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_reg,
            test_size=0.2,
            random_state=args.seed,
        )
        reg = RandomForestRegressor(
            n_estimators=400,
            random_state=args.seed,
            n_jobs=-1,
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        regression_metrics[target] = {"rmse": rmse, "r2": r2}

        importance = pd.Series(reg.feature_importances_, index=feature_columns).sort_values(ascending=False)
        importance.to_csv(output_dir / f"feature_importance_{target}.csv", encoding="utf-8-sig")
        plot_feature_importance(importance, f"Regression feature importance: {target}", output_dir / f"{target}_feature_importance.png")

    (output_dir / "regression_metrics.json").write_text(json.dumps(regression_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report_summary = {
        "accuracy": float(accuracy),
        "best_method_frequency": df["best_method_for_high_tier"].value_counts().reindex(labels, fill_value=0).to_dict(),
        "top_classifier_features": feature_importance.head(10).to_dict(),
        "top_permutation_features": perm_importance.head(10).to_dict(),
        "regression_metrics": regression_metrics,
    }
    (output_dir / "analysis_report.json").write_text(json.dumps(report_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ANALYSIS_OUTPUT_DIR={output_dir.resolve()}")
    print(f"ACCURACY={accuracy:.4f}")
    print("CLASSIFICATION_LABELS=" + ",".join(labels))
    for target, metrics in regression_metrics.items():
        print(f"{target}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
