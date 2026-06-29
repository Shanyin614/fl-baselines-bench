import pandas as pd
from pathlib import Path

files = [
    "fedavg_fashionmnist_cares_seed42.csv",
    "fesem_k4_fashionmnist_cares_seed42.csv",
    "ifca_k4_nowarmup_fashionmnist_cares_seed42.csv",
]

base = Path("outputs/csv")

for name in files:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    print("\n===", name, "===")
    print("rows =", len(df))
    print("columns =", [c for c in ["round", "phase", "acc", "f1", "micro_acc", "global_macro_f1", "client_avg_acc"] if c in df.columns])

    if len(df) != 30:
        raise RuntimeError(f"{name}: expected 30 rows because FashionMNIST protocol has total_rounds=30 and eval_every=1, got {len(df)}")

    required = ["round", "acc", "f1", "micro_acc", "global_macro_f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}. Check BaseRunner.log_metrics acc/f1 aliases.")

    print(df[["round", "phase", "acc", "f1", "micro_acc", "global_macro_f1"]].tail(3).to_string(index=False))

print("\nAll three FashionMNIST baseline CSV files look OK.")
