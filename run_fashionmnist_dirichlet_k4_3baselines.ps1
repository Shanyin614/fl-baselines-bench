# Run FashionMNIST-Dirichlet-K4 experiments: FedAvg, FeSEM-K4, IFCA-K4 no-warmup
# Usage from repo root:
#   powershell -ExecutionPolicy Bypass -File .\run_fashionmnist_dirichlet_k4_3baselines.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Device = "cpu"
$Seed = 42
$K = 4
$Protocol = "configs/protocols/fashionmnist_dirichlet_protocol.yaml"
$Tag = "fashionmnist_dirichlet_k4"

New-Item -ItemType Directory -Force outputs/csv | Out-Null
New-Item -ItemType Directory -Force configs/protocols | Out-Null
New-Item -ItemType Directory -Force configs/methods | Out-Null

# Write the Dirichlet-K4 protocol file.
# This is the second configuration: partition=dirichlet, num_true_clusters=4,
# inter alpha=0.1, intra alpha=10.0, 100 communication rounds.
@"
protocol:
  name: fashionmnist_dirichlet_protocol_v1

runtime:
  seed: 42
  device: auto

output:
  dir: outputs/csv
  name: results.csv
  save_round_metrics: true

dataset:
  name: fashionmnist
  data_root: data
  split_dir: splits/fashionmnist
  split_file: null

  num_classes: 10
  num_clients: 100

  partition: dirichlet
  num_true_clusters: 4
  dir_alpha_inter: 0.1
  dir_alpha_intra: 10.0

  # Only used if partition: label_group/manual; ignored by dirichlet partition.
  major_ratio: 0.85
  true_groups:
    - [0, 2, 6]
    - [1, 3]
    - [4, 8]
    - [5, 7, 9]

  train_samples_per_client: 400
  test_samples_per_client: 100
  val_ratio: 0.2
  num_workers: 0

model:
  name: small_cnn
  num_classes: 10

train:
  total_rounds: 100
  warmup_rounds: 5
  client_frac: 0.2
  local_epochs: 2
  batch_size: 64
  optimizer: sgd
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0
  proximal_mu: 0.0

eval:
  eval_every: 1
  assignment_split: val
  test_split: test
  metrics:
    - client_avg_acc
    - micro_acc
    - global_macro_f1
    - worst_client_acc
    - acc_std
    - ari
    - nmi
    - purity
"@ | Set-Content -Encoding utf8 $Protocol

$IfcaCfg = "configs/methods/ifca_fashionmnist_dirichlet_k4_nowarmup.yaml"

@"
method:
  name: ifca
  num_clusters: 4
  warmup_rounds: 0
  init: random_perturb
  center_perturb_sigma: 0.01
  assignment_split: val
  eval_assign_all: true
"@ | Set-Content -Encoding utf8 $IfcaCfg

$LogPath = "outputs/csv/run_${Tag}_3baselines_seed${Seed}.log"
Start-Transcript -Path $LogPath -Append | Out-Null

try {
  Write-Host "Protocol: $Protocol"
  Write-Host "Tag: $Tag"
  Write-Host "Seed: $Seed"
  Write-Host "Device: $Device"
  Write-Host "K: $K"

  Write-Host "Running FedAvg on FashionMNIST-Dirichlet-K4..."
  python run.py `
    --protocol $Protocol `
    --method-config configs/methods/fedavg.yaml `
    --seed $Seed `
    --device $Device `
    --output-name "fedavg_${Tag}_seed${Seed}.csv"

  Write-Host "Running FeSEM-K4 on FashionMNIST-Dirichlet-K4..."
  python run.py `
    --protocol $Protocol `
    --method-config configs/methods/fesem.yaml `
    --k $K `
    --seed $Seed `
    --device $Device `
    --output-name "fesem_k${K}_${Tag}_seed${Seed}.csv"

  Write-Host "Running IFCA-K4 no warmup on FashionMNIST-Dirichlet-K4..."
  python run.py `
    --protocol $Protocol `
    --method-config $IfcaCfg `
    --k $K `
    --seed $Seed `
    --device $Device `
    --output-name "ifca_k${K}_nowarmup_${Tag}_seed${Seed}.csv"

  Write-Host "Checking outputs..."
  python -c "import pandas as pd, pathlib; base=pathlib.Path('outputs/csv'); files=['fedavg_${Tag}_seed${Seed}.csv','fesem_k${K}_${Tag}_seed${Seed}.csv','ifca_k${K}_nowarmup_${Tag}_seed${Seed}.csv'];\nfor f in files:\n    p=base/f\n    df=pd.read_csv(p)\n    print('\\n' + f + ' rows=' + str(len(df)))\n    print(df.tail(3).to_string(index=False))"

  Write-Host "Done."
  Write-Host "Outputs saved under outputs/csv"
  Write-Host "Log saved to $LogPath"
}
finally {
  Stop-Transcript | Out-Null
}
