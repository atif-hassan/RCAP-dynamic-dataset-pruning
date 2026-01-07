# DataCull

**A lightweight, flexible PyTorch framework for data pruning during model training.**

DataCull provides modular, composable components for implementing and experimenting with data pruning algorithms. It enables efficient training by identifying and removing low-value samples from your dataset on-the-fly. 

DataCull comes with the **official implementation of the RCAP** (Robust Class-Aware Probabilistic) dynamic data pruning algorithm.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Available Methods](#available-methods)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Modular Design**: Clean abstractions for datasets, dataloaders, importance scoring, and logging. Decouples importance scoring and sampling logic, allowing you to mix and match the importance criteria and sampling strategies of different pruning methods.
- **Multiple Pruning Algorithms**: Built-in implementations of state-of-the-art data pruning methods
- **Dynamic and Static Pruning**: Support for both per-epoch (or per-n-epochs) re-sampling and one-time pruning
- **Per-Sample Tracking**: Automatically track metrics and importance scores for every sample across training epochs
- **PyTorch and PyTorch Lightning Compatible**: Drop-in replacements for PyTorch Dataset and DataLoader (no modification to existing workflows)
- **Flexible Importance Scoring**: Extensible framework for custom importance computation methods
- **Flexible Pruning**: Extensible framework for custom pruning logic

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- orjsonl
- tqdm

### From Source

Clone the repository and install:

```bash
pip install datacull
```

## Quick Start

### Basic Usage

Here's a minimal example using DataCull with a standard PyTorch dataset for dynamic data pruning:

```python
import torch
from torch.utils.data import DataLoader
from datacull import DCDataset, DCDataLoader, DCLogger, DCImportance

# 1. Wrap your existing dataset
dataset = DCDataset(your_pytorch_dataset)

# 2. Create a logger to track per-sample metrics
logger = DCLogger(trajectory_dir="./trajectory_directory/", save_every_k_epoch=1)

# 3. Create a dataloader that inherits DCDataLoader and implements the compute_subset function
dataloader = YourPruningDataLoader(
    dataset=dataset,
    pruning_rate=0.2,  # Remove 20% of samples
    batch_size=32,
    num_workers=4
)

# 4. During training, log metrics and resample
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y, idx = batch  # idx contains sample indices
        
        # Your training code here
        preds = model(x)
        
        # Log per-sample metrics (e.g., preds)
        logger.log_metric(epoch, idx, preds)
    
    # Compute importance scores
    # YourImportanceMethod must inherit the DCImportance class and implement the compute_importance function
    importance_computer = YourImportanceMethod(...)
    importance_scores = importance_computer.compute_importance()
    
    # Resample dataset based on importance
    dataloader.resample(importance_scores)
```

## Core Concepts

### DCDataset

A wrapper around PyTorch datasets that appends the sample index to each batch:

```python
from datacull import DCDataset

wrapped_dataset = DCDataset(your_dataset)
# Batch now returns: (*original_outputs, sample_index)
sample = wrapped_dataset[0]  # Returns (x, y, idx) instead of (x, y)
```

### DCDataLoader

A customizable DataLoader supporting dynamic sample pruning with importance scores:

- **Static Mode**: Prune once and reuse the same subset every epoch
- **Dynamic Mode**: Recompute the subset every epoch based on updated importance scores

```python
# Implement compute_subset() to define your pruning strategy
class MyPruner(DCDataLoader):
    def compute_subset(self, sample_importance):
        # Return indices of samples to keep
        return indices_to_keep
```

### DCLogger

Efficiently logs per-sample metrics across training epochs:

```python
logger = DCLogger(
    trajectory_dir="./trajectories/",
    save_every_k_epoch=1  # Save metrics every epoch
)

# During training
logger.log_metric(epoch, sample_indices, loss_values)
# Creates: ./trajectories/epoch{E}.jsonl
```

### DCImportance

Base class for computing importance scores from logged trajectories:

```python
importance = YourImportanceMethod(
    dataset=dataset,
    window_size=5,  # Look at 5 consecutive epochs
    logger_object=logger
)

scores = importance.compute_importance()  # Shape: (num_samples,)
```

## Available Methods

### AUM (Area Under the Margin)

**Class**: `AUMImportance` from `datacull.methods.CCS`

Identifies easy-to-learn samples by computing the margin between true class logits and max other class logits.

```python
from datacull.methods.CCS import AUMImportance

importance = AUMImportance(
    dataset=dataset,
    trajectory_length=num_epochs,
    logger_object=logger
)
scores = importance.compute_importance()
```

### CCS (Coverage-centric Coreset Selection)

**Class**: `CCSDataLoader` from `datacull.methods.CCS`

Uses AUM scores with stratified sampling to maintain dataset diversity.

### TDDS (Temporal Dual-Depth Scoring)

**Classes**: `TDDSImportance`, `TDDSDataLoader` from `datacull.methods.TDDS`

Leverages temporal stability of predictions across epochs.

```python
from datacull.methods.TDDS import TDDSImportance

importance = TDDSImportance(
    dataset=dataset,
    trajectory_length=num_epochs,
    window_size=5,
    decay=0.9,
    logger_object=logger
)
```

### MetriQ

**Class**: `MetriQDataLoader` from `datacull.methods.MetriQ`

Class-balanced pruning inversely proportional to per-class validation accuracy.

```python
from datacull.methods.MetriQ import MetriQDataLoader

# Requires validation accuracy per class
class_wise_acc = np.array([0.95, 0.80, 0.88])

dataloader = MetriQDataLoader(
    dataset=dataset,
    pruning_rate=0.3,
    class_wise_acc=class_wise_acc,
    batch_size=64
)
```

### RS2 (Repeated Random Sampling)

**Class**: `RS2DataLoader` from `datacull.methods.RS2`

Fast random sampling with optional stratification for class balance.

```python
from datacull.methods.RS2 import RS2DataLoader

dataloader = RS2DataLoader(
    dataset=dataset,
    pruning_rate=0.5,
    sampling_with_replacement=False,
    stratify=True,
    batch_size=64
)
```

### RCAP (Relative Class-aware Adaptive Pruning)

**Classes**: `RCAPImportance`, `RCAPDataLoader` from `datacull.methods.RCAP`

Dynamic class-aware probabilistic sampling using loss-based importance scores.

```python
from datacull.methods.RCAP import RCAPImportance, RCAPDataLoader

importance = RCAPImportance(
    dataset=dataset,
    logger_object=logger,
    beta=2.0,  # Temperature parameter
    clipping_threshold=np.log(num_classes)
)

dataloader = RCAPDataLoader(
    dataset=dataset,
    pruning_rate=0.3,
    batch_size=64
)
```

## API Reference

### Core Classes

#### DCDataset

```python
DCDataset(custom_dataset)
```

Wraps a PyTorch dataset to append sample indices.

**Parameters:**
- `custom_dataset` (Dataset): Any PyTorch-compliant dataset

**Returns:** Modified dataset where each sample includes the original index

#### DCDataLoader

```python
DCDataLoader(dataset, pruning_rate, static, **kwargs)
```

Base class for pruning-aware dataloaders.

**Parameters:**
- `dataset` (DCDataset): Wrapped dataset
- `pruning_rate` (float): Fraction of samples to keep (0.0-1.0)
- `static` (bool): If True, compute subset once; if False, every epoch
- `**kwargs`: Standard DataLoader arguments

**Methods:**
- `compute_subset(sample_importance)`: Implement to define pruning strategy
- `resample(sample_importance)`: Update loader with pruned subset

#### DCLogger

```python
DCLogger(trajectory_dir, save_every_k_epoch=1)
```

Logs per-sample metrics to disk.

**Parameters:**
- `trajectory_dir` (str): Directory to store trajectory JSONL files
- `save_every_k_epoch` (int): Save interval

**Methods:**
- `log_metric(epoch, sample_idx, metric)`: Log metrics for a batch

#### DCImportance

```python
DCImportance(dataset, window_size, logger_object, flush=False)
```

Base class for computing importance scores.

**Parameters:**
- `dataset` (DCDataset): The dataset
- `window_size` (int): Number of epochs to examine
- `logger_object` (DCLogger): Logger with trajectory data
- `flush` (bool): Delete trajectory files after reading

**Methods:**
- `compute_importance()`: Return importance scores (must be overridden)
- `extract_trajectory_segment(start_epoch)`: Load logged metrics

## Examples

### Complete Training Loop with CCS

```python
import torch
import numpy as np
from torch import nn
from datacull import DCDataset, DCLogger
from datacull.methods.CCS import CCSDataLoader, AUMImportance

# Setup
dataset = DCDataset(your_dataset)
logger = DCLogger(trajectory_dir="./trajectories/")
dataloader = CCSDataLoader(dataset, pruning_rate=0.2, batch_size=32)

model = YourModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(10):
    for batch in dataloader:
        x, y, idx = batch
        
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log logits for importance computation
        logger.log_metric(epoch, idx, logits.detach())
    
    # Compute importance and resample
    if epoch % 2 == 0:
        importance = AUMImportance(dataset, 10, logger)
        scores = importance.compute_importance()
        dataloader.resample(scores)
```

### Custom Pruning Strategy

```python
import numpy as np
from datacull import DCDataLoader

class RandomPruner(DCDataLoader):
    """Simple random pruning baseline"""
    
    def compute_subset(self, sample_importance):
        # Randomly select samples to keep
        indices = np.arange(self.total_num_samples)
        np.random.shuffle(indices)
        return indices[:self.required_num_samples].tolist()

# Use it
pruner = RandomPruner(dataset, pruning_rate=0.3, batch_size=64)
pruner.resample(None)
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use DataCull in your research, please cite it as:

```bibtex
@software{datacull2024,
  title = {DataCull: A Framework for Data Pruning and Curation},
  year = {2024},
}
```

## Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check existing documentation and examples
- Review the docstrings in source code for API details

---

**Happy pruning!** 🌱
