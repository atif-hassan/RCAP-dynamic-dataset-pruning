[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/datacull.svg)](https://pypi.org/project/datacull/)
[![Downloads](https://static.pepy.tech/badge/datacull)](https://pepy.tech/projects/datacull)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/datacull/commits/master)
# DataCull
DataCull is a **a lightweight, flexible PyTorch framework for data pruning during model training**. It provides modular, composable components for implementing and experimenting with data pruning algorithms. Since DataCull decouples importance scoring and sampling logic, it allows, **for the very first time**, mixing and matching the importance criteria and sampling strategies of different pruning methods.

DataCull comes with the [**official implementation of the RCAP** (Robust Class-Aware Probabilistic) dynamic data pruning algorithm](https://proceedings.mlr.press/v286/hassan25a.html).

It also includes the unofficial implementations of the following data pruning algorithms:
- [CCS (Coverage-centric Coreset Selection for High Pruning Rates)](https://openreview.net/forum?id=QwKvL6wC8Yi)
- [TDDS (Spanning Training Progress: Temporal Dual-Depth Scoring (TDDS) for Enhanced Dataset Pruning)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Spanning_Training_Progress_Temporal_Dual-Depth_Scoring_TDDS_for_Enhanced_Dataset_CVPR_2024_paper.pdf)
- [MetriQ (Robust Data Pruning: Uncovering and Overcoming Implicit Bias)](https://arxiv.org/html/2404.05579v1)
- [RS2 (Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning)](https://openreview.net/forum?id=JnRStoIuTe)


## Features

- **Modular Design**: Clean abstractions for datasets, dataloaders, importance scoring, and logging. Decouples importance scoring and sampling logic, allowing you to mix and match the importance criteria and sampling strategies of different pruning methods.
- **Multiple Pruning Algorithms**: Built-in implementations of state-of-the-art data pruning methods.
- **Dynamic and Static Pruning**: Support for both per-epoch (or per-n-epochs) re-sampling and one-time pruning.
- **Per-Sample Tracking**: Automatically track metrics and importance scores for every sample across training epochs.
- **PyTorch and PyTorch Lightning Compatible**: Drop-in replacements for PyTorch Dataset and DataLoader (no modification to existing workflows).
- **Flexible Importance Scoring**: Extensible framework for custom importance computation methods.
- **Flexible Pruning**: Extensible framework for custom pruning logic.

## How to install?
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
    batch_size=32
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

## Core Classes

### DCDataset

A wrapper around PyTorch datasets that appends the sample index to each batch:

```python
from datacull import DCDataset

wrapped_dataset = DCDataset(your_dataset)
# Batch now returns: (*original_outputs, sample_index)
```
---
### DCDataLoader

A customizable DataLoader supporting both dynamic and static sample pruning with importance scores.
```python
__init__(self, dataset: DCDataset, pruning_rate: float, static: bool, **kwargs)
```
- **datatset:** `DCDataset` Any Pytorch dataset wrapped with the DCDataset class.
- **prunting_rate:** `float (0,1)` The fraction of samples to remove.
- **static:** `bool` This variable decides whether to resample a new subset during training (dynamic mode) or resample only once (static mode).

```python
# This function needs to be implemented by the user when creating their own pruning algorithm
# It holds the pruning logic
# And, returns a list of indices (a subset) which determines which samples to keep
compute_subset(self, sample_importance: list)
```
- **sample_importance:** `list` A list containing an importance score corresponding to each sample in the dataset.

```python
# This function calls compute_subset
# It also determines whether to sample once (static) or more (dynamic)
resample(self, sample_importance: list)
```
- **sample_importance:** `list` A list containing an importance score corresponding to each sample in the dataset.

#### Example Usage
```python
# Implement compute_subset() to define your pruning strategy
class MyPruner(DCDataLoader):
    def compute_subset(self, sample_importance):
        # Write pruning logic using or not using sample_importance
        # Return indices of samples to keep
        return indices_to_keep

# create the data loader
my_data_loader = MyPruner(DCDataset(my_dataset), batch_size)
# Select a new subset
# Here, we assume that your pruning logic does not require importance scores
my_data_loader.resample(None)
```
---
### DCLogger

Efficiently logs per-sample metrics across training epochs.
```python
__init__(self, trajectory_dir: str, save_every_k_epoch: int=1)
```
- **trajectory_dir:** `string` The directory where a model's training metrics will be stored.
- **save_every_k_epoch:** `int (default=1)` Save metrics every k epochs.

```python
# This function needs to be called to save a given metric during training
log_metric(self, epoch: int, sample_idx: torch.Tensor, metric: torch.Tensor)
```
- **epoch:** `int` The current epoch number.
- **sample_idx:** `torch.Tensor` A batch of indices (provided automatically by the DCDataset class).
- **metric:** `torch.Tensor` A batch of metrics to log such as predictions or loss.

#### Example Usage
```python
logger = DCLogger(trajectory_dir="./trajectories/", save_every_k_epoch=2)

# During training
logger.log_metric(epoch, sample_indices, loss_values)
# Creates: ./trajectories/epoch{E}.jsonl
```
---
### DCImportance

Base class for computing importance scores from logged trajectories.
```python
__init__(self, dataset: DCDataset, window_size: int, logger_object: DCLogger, flush: bool = False)
```
- **dataset:** `DCDataset` A Pytorch dataset wrapped by the DCDataset class.
- **window_size:** `int` Determines the number of consecutive epochs to extract.
- **loggret_object:** `DCLogger` A DCLogger object to determine the logging directory and which epochs have been saved.
- **flush:** `bool (default False)` A boolean variable that determines whether to delete the metrics (trajectory segment) that have been currently read into memory, from disk (useful for dynamic methods)

```python
# Returns the segment `start_epoch:start_epoch + window_size` from the trajectory
extract_trajectory_segment(self, start_epoch: int)
```
- **start_epoch:** `int` Determines the point in the trajectory to extract the current segment from.

```python
# This function needs to be implemented by the user
# Returns a list containing the importance score for each sample.
compute_importance(self)
```

#### Example Usage
```python
# Create your sample importance class
class YourImportanceMethod(DCImportance):
    def compute_importance():
        for epoch in range(max_epochs - window_size + 1)
            segment = self.extract_trajectory_segment(epoch)
            # Write your sample importance logic here
        return sample_importance

# Create you sample importance object
importance_object = YourImportanceMethod(dataset=dataset, window_size=5, logger_object=logger)
importance_scores = importance_object.compute_importance()
```

## Available Methods

### AUM (Area Under the Margin)
Identifies easy-to-learn samples by computing the margin between true class logits and max other class logits.

```python
from datacull.methods.CCS import AUMImportance

importance = AUMImportance(dataset=dataset, trajectory_length=num_epochs, logger_object=logger)
scores = importance.compute_importance()
```

### CCS (Coverage-centric Coreset Selection)
Uses AUM scores with stratified sampling to maintain dataset diversity at high pruning rates. 
For a complete working example using AUM, [click here](https://github.com/atif-hassan/RCAP-dynamic-dataset-pruning/blob/main/examples/pytorch-lightning/ccs.ipynb)

```python
from datacull.methods.CCS import CCSDataLoader
train_dataloader = CCSDataLoader(dataset=train_set, pruning_rate=0.3, beta=0.1, num_strata=50, descending=False, batch_size=128, num_workers=1)
train_dataloader.resample(scores)
```

### TDDS (Temporal Dual-Depth Scoring)
Leverages temporal stability of predictions across epochs.
For a complete working example, [click here](https://github.com/atif-hassan/RCAP-dynamic-dataset-pruning/blob/main/examples/pytorch-lightning/tdds.ipynb)
```python
from datacull.methods.TDDS import TDDSImportance, TDDSDataLoader

importance_object = TDDSImportance(dataset=dataset, trajectory_length=num_epochs, window_size=5, decay=0.9, logger_object=logger)
scores = importance_object.compute_importance()
train_dataloader = TDDSDataLoader(dataset=train_set, pruning_rate=0.3, batch_size=128, num_workers=1)
train_dataloader.resample(scores)
```

### MetriQ
Class-balanced pruning, inversely proportional to per-class validation accuracy.
For a complete working example, [click here](https://github.com/atif-hassan/RCAP-dynamic-dataset-pruning/blob/main/examples/pytorch-lightning/metriq.ipynb)
```python
from datacull.methods.MetriQ import MetriQDataLoader

# Requires validation accuracy per class
class_wise_acc = np.array([0.95, 0.80, 0.88])

train_dataloader = MetriQDataLoader(dataset=dataset, pruning_rate=0.3, class_wise_acc=class_wise_acc, batch_size=64, num_workers=1)
train_dataloader.resample(None)
```

### RS2 (Repeated Random Sampling)
Fast random sampling with optional stratification for class balance.
For a complete working example, [click here](https://github.com/atif-hassan/RCAP-dynamic-dataset-pruning/blob/main/examples/pytorch-lightning/rs2.ipynb)
```python
from datacull.methods.RS2 import RS2DataLoader

dataloader = RS2DataLoader(dataset=dataset, pruning_rate=0.3, sampling_with_replacement=False, stratify=False, batch_size=64, num_workers=1)
train_dataloader.resample(None)
```

### RCAP (Relative Class-aware Adaptive Pruning)
Dynamic class-aware probabilistic sampling using loss-based importance scores.
For a complete working example, [click here](https://github.com/atif-hassan/RCAP-dynamic-dataset-pruning/blob/main/examples/pytorch-lightning/rcap.ipynb)
```python
from datacull.methods.RCAP import RCAPImportance, RCAPDataLoader

importance_object = RCAPImportance(dataset=dataset, logger_object=logger, beta=2.0, clipping_threshold=None)
train_dataloader = RCAPDataLoader(dataset=dataset, pruning_rate=0.3, batch_size=64, num_workers=1)
train_dataloader.resample(importance_object.compute_importance())
```
---
### An example of using separate importance and sampling techniques
```python
from datacull.methods.TDDS import TDDSImportance
from datacull.methods.CCS import CCSDataLoader

importance_object = TDDSImportance(dataset=dataset, trajectory_length=num_epochs, window_size=5, decay=0.9, logger_object=logger)
scores = importance_object.compute_importance()
train_dataloader = CCSDataLoader(dataset=train_set, pruning_rate=0.3, beta=0.1, num_strata=50, descending=False, batch_size=128, num_workers=1)
train_dataloader.resample(scores)
```
---

## Custom Pruning Strategy Example

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

## Future Ideas
- Pytorch specific exmaples
- Implement more data pruning algorithms

## Feature Requests
You can open an issue or contact me at <atif.hit.hassan@gmail.com>

## Citation

If you use DataCull in your research, please cite it as:

```bibtex
@inproceedings{hassanrcap,
  title={RCAP: Robust, Class-Aware, Probabilistic Dynamic Dataset Pruning},
  author={Hassan, Atif and Khare, Swanand and Paik, Jiaul H},
  booktitle={The 41st Conference on Uncertainty in Artificial Intelligence}
}
```
[Alternatively, use the following DBLP Bibtex link](https://dblp.org/rec/conf/uai/HassanKP25.html?view=bibtex)
