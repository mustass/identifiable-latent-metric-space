
## Requirements

To install the required dependencies, run:
```bash
python3 -m pip install jax[cuda12] 
python3 -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install numpy==1.26.0
python3 -m pip install .
```

## Usage

### wandb
We use `wandb` for tracking metrics and plots while the model is training. Once running, the script will ask you to login using a wandb API key. After that, you will have to do no more. 

### Hydra configurations

We use [Hydra](https://hydra.cc/) for configuration management. The configuration files are located in the `configs` directory and are split up into separate sub-configs for the `datamodule`, `decoder`, `encoder`, `optimizer`, etc. configs. You can override any configuration parameter by passing it as a command-line argument.
### Training

 For example, to change the default settings and train, run:
```bash
python ./scripts/train.py datamodule.batch_size=256 datamodule.dataset_root="path/to/celeba/ model.params.num_decoders=8 model.params.z_dim=128 general.run_name=ultimate_model
```

To train a model with the base config, run:
```bash
python ./scripts/train.py
```

PRC (Perception) loss model:
```bash
python ./scripts/train.py datamodule.batch_size=64 datamodule.dataset_root="/data/celeba" model.params.num_decoders=8 model.params.z_dim=128 general.run_name=prc_test model=celeba_vae_prc loss=prc
```

### Evaluation (to come)

To evaluate the model (compute geodesics and create plots), run:
```bash
python ./scripts/inference_geodesics.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
