#!/bin/bash
# Ensure the package root is on PYTHONPATH so `import pdebench` works
# Compute project root (two levels up from this script: .../PDEBench-main)
# Resolve the script directory in a way that's compatible with /bin/sh
# If run with bash, BASH_SOURCE is available; otherwise fall back to $0.
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$SCRIPT_SOURCE")")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH

CUDA_VISIBLE_DEVICES='0' python3 "$SCRIPT_DIR/train_models_forward.py" +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False ++args.data_path='../data'
# CUDA_VISIBLE_DEVICES='0' python3 "$SCRIPT_DIR/train_models_forward.py" +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='Unet' ++args.if_training=False ++args.data_path='../data/'