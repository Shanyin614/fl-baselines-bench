#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
ROWS = []
DATASETS = {
    "UNSW-NB15": ROOT / "outputs/paper_baselines_unsw",
    "CICIDS2017": ROOT / "outputs/paper_baselines_cicids",
}
METHODS = ["fedavg", "ifca", "fesem", "cfl"]
METHOD_LABEL = {"fedavg": "FedAvg", "ifca": "IFCA", "fesem": "FeSEM", "cfl": "CFL"}
PREFERRED = [
    "dataset", "method", "round", "acc", "precision", "recall", "f1",
    "global_macro_f1", "client_avg_macro_f1", "tn", "fp", "fn", "tp",
    "k_pred", "ari", "nmi", "purity", "client_avg_acc", "micro_acc",
    "worst_client_acc", "acc_std",
]

for dataset, outdir in DATASETS.items():
    for method in METHODS:
        csv_path = outdir / f"{method}_seed42.csv"
        if not csv_path.exists():
            print(f"[missing] {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[empty] {csv_path}")
            continue
        row = df.iloc[-1].to_dict()
        row["dataset"] = dataset
        row["method"] = METHOD_LABEL[method]
        if method == "fedavg":
            for key in ["k_pred", "ari", "nmi", "purity"]:
                if key in row:
                    row[key] = None
        ROWS.append(row)

if not ROWS:
    raise SystemExit("No result CSVs found.")

summary = pd.DataFrame(ROWS)
cols = [c for c in PREFERRED if c in summary.columns] + [c for c in summary.columns if c not in PREFERRED]
summary = summary[cols]
out = ROOT / "outputs/nids_baselines_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False)
print(f"\nSaved summary -> {out}")

main_cols = [
    c for c in [
        "dataset", "method", "acc", "precision", "recall", "f1",
        "global_macro_f1", "client_avg_macro_f1", "k_pred", "ari", "nmi", "purity"
    ]
    if c in summary.columns
]
print(summary[main_cols].to_string(index=False))
