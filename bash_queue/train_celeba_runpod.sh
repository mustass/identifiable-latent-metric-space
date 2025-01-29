#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
seeds=(60 61 62 63 64)
for seed in "${seeds[@]}"
do
    name="celeba_pcr_seed_${seed}"
    echo $name
    comnd="python3 scripts/train.py general.run_name=${name} datamodule.batch_size=128 training.seed=${seed} training.checkpoint=/workspace/celeba_models/stas_runpod/${name} datamodule.dataset_root=/workspace/celeba_manual/celeba/" 
    echo $comnd
    eval $comnd
done
