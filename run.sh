#!/bin/sh
# Selected configuration of NeuroHubFormer per dataset (10-fold CV, seed 42).
# Usage: DATA_ROOT=/path/to/datasets GPU=0 sh run.sh
set -e
DATA_ROOT="${DATA_ROOT:-./datasets}"
GPU="${GPU:-0}"

run() {
  python main.py --data_root "$DATA_ROOT" --gpu "$GPU" --csv_path ./results/summary.csv "$@"
}

run --data_name abide_schaefer100     --lr 3e-4 --num_layers 2 --dropout 0.2 --interaction_dim 64  --shrinkage 0.02
run --data_name adni_schaefer100      --lr 5e-4 --num_layers 2 --dropout 0.2 --interaction_dim 128 --shrinkage 0.1
run --data_name hcpgender_schaefer100 --lr 7e-4 --num_layers 1 --dropout 0.3 --interaction_dim 64  --shrinkage 0.01
run --data_name hcpage_schaefer100    --lr 3e-4 --num_layers 1 --dropout 0.1 --interaction_dim 64  --shrinkage 0.01
