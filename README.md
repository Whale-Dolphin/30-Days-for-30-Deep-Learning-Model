# DL-Arch: Universal Deep Learning Architecture Framework

## Overview

A modular and extensible deep learning framework supporting CNN, Transformer, MLP, and other architectures with unified training and evaluation pipelines.

## Recent Optimizations (User Rules Compliance)

### 🔧 Code Documentation Standards
- ✅ Added detailed tensor operation comments with input/output shapes
- ✅ Included mathematical formulas using LaTeX notation in comments  
- ✅ Added purpose descriptions for all tensor operations
- ✅ All comments and docstrings are in English

### 📊 Logging Standards  
- ✅ Replaced Python's built-in logging with **loguru**
- ✅ Added structured logging with meaningful messages
- ✅ Implemented debug logging for tensor shapes and operations
- ✅ Added gradient norms and parameter monitoring

### ⚡ Deep Learning Framework Optimizations
- ✅ Enhanced PyTorch device detection: **CUDA > MPS > CPU** priority
- ✅ Added explicit device placement with Apple MPS support
- ✅ Implemented proper tensor dtype specifications
- ✅ Added tensor shape validation and assertions

### 📈 Training and Evaluation Monitoring
- ✅ Enhanced TensorBoard integration with:
  - Training/validation losses and metrics
  - Learning rate tracking
  - Gradient norm monitoring
  - Parameter histogram logging
  - Structured tag naming

### 🛡️ Error Handling
- ✅ Added proper error handling for tensor operations
- ✅ Implemented tensor shape and dtype validation
- ✅ Added assertions for critical tensor dimension assumptions
- ✅ Enhanced error logging with sufficient debugging context

### 📝 Code Style
- ✅ Added type hints for function parameters and return values
- ✅ Implemented meaningful variable names and docstrings
- ✅ Following PEP 8 style guidelines

## Architecture Support

### Models
- **SimpleCNN**: Convolutional Neural Network for image classification
- **SimpleMLP**: Multi-Layer Perceptron for tabular data
- **SimpleTransformer**: Transformer encoder for sequence classification

### Data Types
- **Image Data**: Via `dummy_image` dataset
- **Text Data**: Via `dummy_text` dataset  
- **Tabular Data**: Via `dummy_tabular` dataset
- **Streaming Data**: Support for large-scale datasets

## Quick Start

```bash
# Train MLP model
python main.py --config configs/example_mlp.yaml --mode train

# Train CNN model  
python main.py --config configs/example_cnn.yaml --mode train

# Train Transformer model
python main.py --config configs/example_transformer.yaml --mode train

# Evaluate trained model
python main.py --config configs/example_mlp.yaml --mode eval --resume outputs/mlp_experiment/best_model.pth

# Full training + evaluation
python main.py --config configs/example_transformer.yaml --mode both
```

## Device Support

The framework automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs) - First priority
2. **MPS** (Apple Metal) - Second priority  
3. **CPU** - Fallback option

## Logging and Monitoring

### Loguru Integration
- Structured logging with rotation
- Debug-level tensor shape tracking
- Error context preservation

### TensorBoard Features
- Real-time training metrics
- Gradient norm monitoring
- Parameter histogram visualization
- Learning rate tracking

## Key Features

- 🔄 **Modular Registry System**: Easy model and dataset registration
- 📊 **Comprehensive Metrics**: Built-in evaluation with multiple metrics
- 💾 **Checkpoint Management**: Automatic model saving and resuming
- 🎯 **Multi-Task Support**: Classification and regression tasks
- 🚀 **GPU Acceleration**: Automatic device detection and optimization
- 📈 **Visualization**: TensorBoard integration for training monitoring

## Project Structure

```
dl_arch/
├── data/           # Dataset classes and data loading utilities
├── models/         # Model architectures and base classes
├── training/       # Training loop and optimization
├── evaluation/     # Evaluation metrics and tools
└── utils/          # Utilities and helper functions

configs/            # YAML configuration files
outputs/            # Training outputs and checkpoints
```

## Configuration

All experiments are configured via YAML files with support for:
- Model hyperparameters
- Training settings
- Data configuration
- Device selection
- Logging preferences

## Contributing

This framework follows strict code quality standards:
- Type hints for all functions
- Comprehensive tensor operation documentation
- Structured logging throughout
- Proper error handling with context
