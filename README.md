# fl-baselines-bench

A baseline-only PyTorch benchmark for comparing federated learning baselines against CARES under the same experimental protocol.

This repository intentionally does **not** include CARES. It provides baseline runners such as FedAvg and FeSEM, while keeping the protocol aligned with CARES-lite:

- same FashionMNIST label-group non-IID partition logic
- same SmallCNN architecture
- same training budget defaults: 30 total rounds, 5 warm-up rounds, client fraction 0.2, local epochs 1, SGD lr 0.02
- same train/val/test client split fields
- unified output CSV for later aggregation with CARES results

## Quick start

```bash
pip install -r requirements.txt
python run.py \
  --protocol configs/protocols/fashionmnist_cares_protocol.yaml \
  --method-config configs/methods/fesem.yaml \
  --k 4 \
  --seed 42
```

Run FedAvg:

```bash
python run.py \
  --protocol configs/protocols/fashionmnist_cares_protocol.yaml \
  --method-config configs/methods/fedavg.yaml \
  --seed 42
```

## FeSEM implementation in this benchmark

The implemented FeSEM baseline is a unified PyTorch re-implementation of fixed-K multi-center federated learning:

1. FedAvg warm-up for `train.warmup_rounds` rounds; these rounds are included in `train.total_rounds`.
2. Initialize `K` center models from the warm-up global model.
3. E-step: assign each client to the center model with the lowest local validation loss.
4. M-step: train selected clients from their assigned center model and aggregate updates within each cluster.
5. Repeat E-step according to `method.assignment_interval`.

The runner tracks `model_transmissions` to make multi-center overhead visible. This is an accounting metric, not a strict network simulator.

## Notes for fair comparison with CARES

The generated split file is reusable. Point CARES and this benchmark to the same `client_splits.json` if you want exact partition identity across projects.

For fixed-K baselines, run several K values, for example:

```bash
for k in 2 3 4 5 6; do
  python run.py --protocol configs/protocols/fashionmnist_cares_protocol.yaml \
    --method-config configs/methods/fesem.yaml --k $k --seed 42
done
```

Recommended first-stage baselines:

- FedAvg
- IFCA (to be added or imported from your existing implementation)
- FeSEM
- Oracle-Cluster FedAvg (to be added)
- CFL (to be added)
