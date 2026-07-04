#!/usr/bin/env bash
set -euo pipefail

# Run all NIDS baselines:
#   bash scripts/run_nids_baselines.sh
# Run a subset:
#   DATASETS="unsw" METHODS="ifca fesem" bash scripts/run_nids_baselines.sh

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

DATASETS="${DATASETS:-unsw cicids}"
METHODS="${METHODS:-fedavg ifca fesem cfl}"
SEED="${SEED:-42}"

for ds in $DATASETS; do
  case "$ds" in
    unsw)
      protocol="configs/protocols/unsw_nb15_cares_fair20.yaml"
      outdir="outputs/paper_baselines_unsw"
      ;;
    cicids)
      protocol="configs/protocols/cicids2017_cares_fair20.yaml"
      outdir="outputs/paper_baselines_cicids"
      ;;
    *)
      echo "Unknown dataset key: $ds" >&2
      exit 1
      ;;
  esac

  mkdir -p "$outdir"

  for method in $METHODS; do
    case "$method" in
      fedavg) method_config="configs/methods/fedavg.yaml" ;;
      ifca)   method_config="configs/methods/ifca_k2.yaml" ;;
      fesem)  method_config="configs/methods/fesem_k2.yaml" ;;
      cfl)    method_config="configs/methods/cfl_nids.yaml" ;;
      *)
        echo "Unknown method: $method" >&2
        exit 1
        ;;
    esac

    echo
    echo "============================================================"
    echo "Running dataset=$ds method=$method seed=$SEED"
    echo "============================================================"

    python run.py \
      --protocol "$protocol" \
      --method-config "$method_config" \
      --seed "$SEED" \
      --output-dir "$outdir" \
      --output-name "${method}_seed${SEED}.csv" \
      2>&1 | tee "$outdir/${method}_seed${SEED}.log"
  done
done

python scripts/collect_nids_results.py
