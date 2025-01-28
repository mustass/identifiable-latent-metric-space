#!/bin/bash
#BSUB -q gpua100
#BSUB -J geodesics_sgd_100
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/$celeba-%J.out
#BSUB -e logs/$celeba-%J.err
module load python3/3.10.15 cuda/12.6.3 cudnn/v9.6.0.74-prod-cuda-12.X 
source /work3/s210527/nnx//bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

name="celeba_hpc_geodesics_sgd_100_batch_2"
echo $name
comnd="python3 scripts/inference_geodesics.py general.run_name=${name} inference=ensemble_geodesics_batch_2"
echo $comnd
eval $comnd
