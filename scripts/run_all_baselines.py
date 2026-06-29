import subprocess
import sys
from pathlib import Path

import pandas as pd

root = Path.cwd()
protocols = [
    ('cicids2017', 'configs/protocols/cicids2017_cares_protocol.yaml'),
    ('unsw_nb15', 'configs/protocols/unsw_nb15_cares_protocol.yaml'),
]
methods = [
    ('fedavg', 'configs/methods/fedavg.yaml'),
    ('ifca', 'configs/methods/ifca.yaml'),
    ('fesem', 'configs/methods/fesem.yaml'),
    ('cfl', 'configs/methods/cfl.yaml'),
]

results = []
for dataset_name, protocol in protocols:
    for method_name, method_cfg in methods:
        output_name = f"{method_name}_{dataset_name}_seed42.csv"
        cmd = [
            sys.executable,
            'run.py',
            '--protocol', protocol,
            '--method-config', method_cfg,
            '--seed', '42',
            '--output-name', output_name,
        ]
        print(f'RUNNING {dataset_name} / {method_name}')
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise SystemExit(proc.returncode)

        out_path = root / 'outputs' / 'csv' / output_name
        if out_path.exists():
            df = pd.read_csv(out_path)
            last = df.iloc[-1]
            results.append({
                'dataset': dataset_name,
                'method': method_name,
                'acc': float(last.get('acc', float('nan'))),
                'f1': float(last.get('f1', float('nan'))),
                'client_avg_acc': float(last.get('client_avg_acc', float('nan'))),
                'micro_acc': float(last.get('micro_acc', float('nan'))),
                'global_macro_f1': float(last.get('global_macro_f1', float('nan'))),
                'runtime_sec': float(last.get('runtime_sec', float('nan'))),
            })

summary_path = root / 'outputs' / 'csv' / 'paper_summary.csv'
pd.DataFrame(results).to_csv(summary_path, index=False)
print('SUMMARY', summary_path)
print(pd.DataFrame(results).to_string(index=False))
