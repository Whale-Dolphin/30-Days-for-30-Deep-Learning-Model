# Configuration Guide

This directory contains example configuration files for the DL-Arch framework.

## Usage

After registering your models and datasets in their respective modules, you can run training with:

```bash
# Train a CNN model
python main.py --config configs/example_cnn.yaml --mode train

# Train a Transformer model  
python main.py --config configs/example_transformer.yaml --mode train

# Train an MLP model
python main.py --config configs/example_mlp.yaml --mode train

# Evaluate a model
python main.py --config configs/example_cnn.yaml --mode eval --resume outputs/cnn_experiment/best_model.pth

# Train and then evaluate
python main.py --config configs/example_cnn.yaml --mode both
```

## Configuration Structure

Each configuration file should contain:

### Model Section
```yaml
model:
  name: "your_registered_model_name"
  # model-specific parameters
```

### Data Section
```yaml
data:
  name: "your_registered_dataset_name"
  # dataset-specific parameters
  train:
    # training data parameters
  validation:
    # validation data parameters (optional)
```

### Training Section
```yaml
training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"  # adam, sgd, adamw
  scheduler: "step"  # step, cosine, plateau, or null
  loss: "cross_entropy"  # cross_entropy, mse, mae, bce, bce_with_logits
  # other training parameters
```

### Experiment Section
```yaml
experiment:
  seed: 42
  output_dir: "./outputs/experiment_name"
  device: "auto"  # auto, cpu, cuda
```

## Adding New Models and Datasets

1. Create your model class in `dl_arch/models/` and register it:
```python
from dl_arch import register_model

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # your implementation
```

2. Create your dataset class in `dl_arch/data/` and register it:
```python
from dl_arch import register_dataset

@register_dataset("my_dataset", dataset_type="map")
class MyDataset(BaseDataset):
    def __init__(self, config):
        super().__init__()
        # your implementation
```

3. Create a configuration file with your model and dataset names
4. Run training with `python main.py --config your_config.yaml --mode train`
