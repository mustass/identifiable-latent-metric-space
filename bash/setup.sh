echo
python3 -m pip install --upgrade pip
python3 -m venv env
source env/bin/activate
python3 -m pip install -U "jax[cuda12]"
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install numpy==1.26.0
python3 -m pip install -e .