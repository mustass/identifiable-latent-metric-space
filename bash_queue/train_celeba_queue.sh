

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)  

for seed in "${seeds[@]}"
do
    name="celeba_hpc_seed_${seed}"
    echo $name
    comnd="python3 scripts/train.py general.run_name=${name} datamodule.batch_size=256 training.seed=${seed} datamodule.dataset_root=/work3/s210527/data/celeba_manual/celeba general.project_name=icml25_celeba_baseline training.num_epochs=1000" 

       cat > jobscripts/jobscript_training_${script_n}.sh << EOF
#!/bin/bash
#BSUB -q gpua100
#BSUB -J celeba
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu40gb]"
#BSUB -W 02:30
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/$celeba-%J.out
#BSUB -e logs/$celeba-%J.err
module load python3/3.10.15 cuda/12.6.3 cudnn/v9.6.0.74-prod-cuda-12.X 
source /work3/s210527/nnx//bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
$comnd
EOF
                echo "script_n: $script_n"
                bsub < ./jobscripts/jobscript_training_${script_n}.sh
                ((script_n++))
done