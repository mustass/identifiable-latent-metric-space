#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(0 1 2 3 4)
for seed in "${seeds[@]}"
do
    name="celeba_seed_${seed}"
    echo $name
    comnd="python3 scripts/train.py general.run_name=${name} datamodule.batch_size=128 training.seed=${seed} training.checkpoint=/path/to/name/${name} datamodule.dataset_root=/path/to/data/celeba/" 
    echo $comnd
    eval $comnd
done
