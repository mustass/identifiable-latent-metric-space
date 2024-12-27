#!/bin/bash
conda activate identifiable
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
jupyter_notebook_output=$(jupyter notebook --no-browser --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0')