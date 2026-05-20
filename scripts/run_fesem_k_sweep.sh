#!/usr/bin/env bash
set -euo pipefail
for seed in 42 43 44; do
  for k in 2 3 4 5 6; do
    python run.py \
      --protocol configs/protocols/fashionmnist_cares_protocol.yaml \
      --method-config configs/methods/fesem.yaml \
      --k "$k" \
      --seed "$seed" \
      --output-name "fesem_k${k}_seed${seed}.csv"
  done
done
