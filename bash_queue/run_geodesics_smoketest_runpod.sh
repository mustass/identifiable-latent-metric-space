#!/bin/bash
source /path/to/venv/activate
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=.99

name="celeba_runpod_geodesics_sgd_smoketest"
echo $name
comnd="python3 scripts/inference_geodesics.py general.run_name=${name} inference=ensemble_geodesics_smoketest"
echo $comnd
eval $comnd
