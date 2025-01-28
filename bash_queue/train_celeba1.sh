#!/bin/bash
#BSUB -q gpua100
#BSUB -J celeba1
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu40gb]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/$celeba-%J.out
#BSUB -e logs/$celeba-%J.err
module load python3/3.10.15 cuda/12.6.3 cudnn/v9.6.0.74-prod-cuda-12.X 
source /work3/s210527/nnx//bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

seeds=(5 6 7 8 9)
for seed in "${seeds[@]}"
do
    name="celeba_pcr_seed_${seed}"
    echo $name
    comnd="python3 scripts/train.py general.run_name=${name} datamodule.batch_size=128 training.seed=${seed} training.checkpoint=/work3/s210527/celeba_models/stas/${name}" 
    echo $comnd
    eval $comnd
done