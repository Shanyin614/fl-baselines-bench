from pathlib import Path
import re

p = Path("src/flbench/datasets/tabular.py")
s = p.read_text(encoding="utf-8")

if "drop_columns_cfg = cfg.dataset.get" in s:
    print("drop_columns support already exists")
else:
    pattern = r"(\n[ \t]*)for column in features_df\.columns:"
    m = re.search(pattern, s)
    if not m:
        raise SystemExit("Cannot find: for column in features_df.columns:")

    indent = m.group(1)

    insert = f'''{indent}drop_columns_cfg = cfg.dataset.get("drop_columns", [])
{indent}drop_columns_exact = {{str(col) for col in drop_columns_cfg}}
{indent}drop_columns_lower = {{str(col).lower() for col in drop_columns_cfg}}
{indent}auto_drop_contains = [str(x).lower() for x in cfg.dataset.get("auto_drop_if_contains", [])]

{indent}cols_to_drop = []
{indent}for col in list(features_df.columns):
{indent}    col_str = str(col)
{indent}    col_lower = col_str.lower()
{indent}    if (
{indent}        col_str in drop_columns_exact
{indent}        or col_lower in drop_columns_lower
{indent}        or any(token in col_lower for token in auto_drop_contains)
{indent}    ):
{indent}        cols_to_drop.append(col)

{indent}if cols_to_drop:
{indent}    features_df = features_df.drop(columns=cols_to_drop, errors="ignore")
{indent}    print(f"[dataset] dropped feature columns: {{cols_to_drop}}")

{indent}if features_df.shape[1] == 0:
{indent}    raise KeyError("No feature columns remain after applying drop_columns/auto_drop_if_contains")

'''

    s = s[:m.start()] + insert + s[m.start():]
    p.write_text(s, encoding="utf-8")
    print("patched", p)
