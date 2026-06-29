$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Device = "cpu"
$Seed = 42
$K = 4
$Protocol = "configs/protocols/fashionmnist_cares_protocol.yaml"
$Tag = "fashionmnist_labelgroup_k4"

New-Item -ItemType Directory -Force outputs/csv | Out-Null
New-Item -ItemType Directory -Force configs/methods | Out-Null

$LogPath = "outputs/csv/run_${Tag}_3baselines_seed${Seed}.log"
Start-Transcript -Path $LogPath -Append | Out-Null

try {
  $IfcaCfg = "configs/methods/ifca_fashionmnist_k4_nowarmup.yaml"

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

  Write-Host "Running FedAvg..."
  python run.py `
    --protocol $Protocol `
    --method-config configs/methods/fedavg.yaml `
    --seed $Seed `
    --device $Device `
    --output-name "fedavg_${Tag}_seed${Seed}.csv"

  Write-Host "Running FeSEM-K4..."
  python run.py `
    --protocol $Protocol `
    --method-config configs/methods/fesem.yaml `
    --k $K `
    --seed $Seed `
    --device $Device `
    --output-name "fesem_k${K}_${Tag}_seed${Seed}.csv"

  Write-Host "Running IFCA-K4 no warmup..."
  python run.py `
    --protocol $Protocol `
    --method-config $IfcaCfg `
    --k $K `
    --seed $Seed `
    --device $Device `
    --output-name "ifca_k${K}_nowarmup_${Tag}_seed${Seed}.csv"

  Write-Host "Checking outputs..."
  python -c "import pandas as pd, pathlib; files=['fedavg_${Tag}_seed${Seed}.csv','fesem_k${K}_${Tag}_seed${Seed}.csv','ifca_k${K}_nowarmup_${Tag}_seed${Seed}.csv']; base=pathlib.Path('outputs/csv'); [print('\n',f,'rows=',len(pd.read_csv(base/f)), pd.read_csv(base/f).tail(3)) for f in files]"

  Write-Host "Done."
  Write-Host "Log saved to $LogPath"
}
finally {
  Stop-Transcript | Out-Null
}
