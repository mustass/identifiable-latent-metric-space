#!/bin/bash
source /workspace/nnx/bin/activate
export CUDA_VISIBLE_DEVICES=0

name="celeba_runpod_geodesics_sgd_batch1"
echo $name
comnd="python3 scripts/inference_geodesics.py general.run_name=${name} inference=ensemble_geodesics"
echo $comnd
eval $comnd
