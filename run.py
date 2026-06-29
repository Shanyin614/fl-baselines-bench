#!/usr/bin/env python3
"""Root CLI for the baseline-only FL benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flbench.datasets.factory import build_clients
from flbench.methods.registry import build_runner
from flbench.models.factory import build_model_fn
from flbench.utils.config import ConfigNode, deep_update, load_yaml
from flbench.utils.seed import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline-only federated learning benchmark")
    p.add_argument("--protocol", type=str, required=True, help="Protocol YAML shared with CARES")
    p.add_argument("--method-config", type=str, required=True, help="Method YAML, e.g. configs/methods/fesem.yaml")
    p.add_argument("--method", type=str, default=None, help="Override method.name")
    p.add_argument("--k", type=int, default=None, help="Override method.num_clusters for fixed-K baselines")
    p.add_argument("--seed", type=int, default=None, help="Override runtime.seed")
    p.add_argument("--device", type=str, default=None, help="Override runtime.device: auto/cpu/cuda")
    p.add_argument("--output-dir", type=str, default=None, help="Override output.dir")
    p.add_argument("--output-name", type=str, default=None, help="Override output.name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_yaml(args.protocol)
    method_cfg = load_yaml(args.method_config)
    cfg_dict = deep_update(protocol, method_cfg)

    if args.method is not None:
        cfg_dict.setdefault("method", {})["name"] = args.method
    if args.k is not None:
        cfg_dict.setdefault("method", {})["num_clusters"] = args.k
    if args.seed is not None:
        cfg_dict.setdefault("runtime", {})["seed"] = args.seed
    if args.device is not None:
        cfg_dict.setdefault("runtime", {})["device"] = args.device
    if args.output_dir is not None:
        cfg_dict.setdefault("output", {})["dir"] = args.output_dir
    if args.output_name is not None:
        cfg_dict.setdefault("output", {})["name"] = args.output_name

    cfg = ConfigNode(cfg_dict)
    seed = int(cfg.runtime.get("seed", 42))
    set_seed(seed)
    device = get_device(str(cfg.runtime.get("device", "auto")))

    clients, dataset_info = build_clients(cfg, device=device)
    model_fn = build_model_fn(cfg, dataset_info=dataset_info)
    runner = build_runner(
        method_name=str(cfg.method.name),
        clients=clients,
        model_fn=model_fn,
        cfg=cfg,
        device=device,
        dataset_info=dataset_info,
    )
    summary = runner.run()

    print("\nFinal summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
