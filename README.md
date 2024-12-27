
## Requirements

To install the required dependencies, run:
```bash
python3 -m pip install jax[cuda12] 
python3 -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install .
```

## Usage

### Hydra configurations

We use [Hydra](https://hydra.cc/) for configuration management. The configuration files are located in the `configs` directory and are split up into separate sub-configs for the `datamodule`, `decoder`, `encoder`, `optimizer`, etc. configs. You can override any configuration parameter by passing it as a command-line argument. For example, to change the batch size, run:
```bash
python ./scripts/train.py datamodule.batch_size=64
```
### Training

To train a model with the basic config, run:
```bash
python ./scripts/train.py
```

### Evaluation

To evaluate the model (compute geodesics and create plots), run:
```bash
python ./scripts/inference_geodesics.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
