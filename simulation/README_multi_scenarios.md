# 複数シナリオ実行システム

## 概要
異なるconfig設定で複数のシナリオを自動実行し、シナリオごとに独立した出力ディレクトリに結果を保存します。

## 定義されているシナリオ

### scenario1: ベースライン（到着率0.005）
- 説明: 現在の設定をそのまま使用
- 到着率: 0.005（全ユーザー共通）
- エポック: 10

### scenario2: 高エポック（エポック200）
- 説明: エポック数を10→200に変更
- 到着率: 0.005（変更なし）
- エポック: 200

### scenario3: 性能別到着率
- 説明: GPU性能に応じて到着率を変化（高性能ほど高頻度）
- 到着率:
  - 低性能GPU（tier1-3）: 0.003-0.004
  - 中性能GPU（tier4-6）: 0.005-0.006
  - 高性能GPU（tier7-9）: 0.008-0.010
- エポック: 10

## 実行方法

```powershell
python run_multi_scenarios.py
```

## 出力構造

```
outputs/
├── scenario1/
│   ├── basic_scenarios/
│   ├── user_comparisons/
│   ├── iterative_results/
│   └── tables/
├── scenario2/
│   ├── basic_scenarios/
│   ├── user_comparisons/
│   ├── iterative_results/
│   └── tables/
└── scenario3/
    ├── basic_scenarios/
    ├── user_comparisons/
    ├── iterative_results/
    └── tables/
```

## 注意事項

- 実行中、config.pyが一時的に変更されますが、各シナリオ終了後に自動的に復元されます
- 元のconfig.pyはconfig_original.pyとしてバックアップされます
- 各シナリオは順次実行されます（並列実行ではありません）
- 全シナリオ完了後、バックアップファイルは自動削除されます
