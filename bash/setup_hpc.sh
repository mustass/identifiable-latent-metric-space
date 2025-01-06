echo
module load python3/3.12.7 cuda/12.6.3 cudnn/v9.6.0.74-prod-cuda-12.X 
python3 -m pip install --upgrade pip
python3 -m venv /work3/s210527/venv_identifiable/
source /work3/s210527/venv_identifiable//bin/activate
python3 -m pip install -U "jax[cuda12]"
python3 -m pip install nvidia-dali-cuda120
python3 -m pip install numpy==1.26.0
python3 -m pip install -e .