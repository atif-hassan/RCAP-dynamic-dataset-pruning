from ..data import DCDataLoader, DCDataset
from ..importance import DCImportance
from ..logger import DCLogger
import numpy as np
from tqdm import tqdm
import torch



class TDDSImportance(DCImportance):
    """
    TDDS (Temporal Dual-Depth Scoring) Importance.
    
    For each sliding window of logits:
        1. Convert logits → softmax probabilities.
        2. Compute temporal log-ratio between consecutive timesteps.
        3. Weight log-ratio by next timestep probability.
        4. Compute variance across window (per sample).
        5. Apply exponential moving average over epochs.

    TDDS is a temporal stability-based pruning score
    """

    def __init__(self, dataset: DCDataset, trajectory_length: int, window_size: int, decay: float, logger_object: DCLogger):
        """
        Parameters
        ----------
        trajectory_length : int
            Total number of saved trajectory epochs.
        window_size : int
            Window size for temporal smoothing.
        decay : float
            Exponential moving average decay factor.
        logger_object : DCLogger
            Loads per-epoch predictions.
        """
        self.trajectory_length = trajectory_length
        self.decay = decay
        # Shape of data is (window_size, num_samples, num_classes)
        self.logsoftmax = torch.nn.LogSoftmax(dim=2)
        super().__init__(dataset=dataset, window_size=window_size, logger_object=logger_object, flush=False)


    def compute_importance(self):
        """
        Compute TDDS importance for each sample using trajectory windows.

        Returns
        -------
        np.ndarray
            TDDS importance score for all samples.
        """
        for epoch in tqdm(range(self.trajectory_length - self.window_size + 1), desc="Computing TDDS importance"):
            # Load predictions of shape (window_size, num_samples, num_classes)
            preds = self.extract_trajectory_segment(epoch)
            preds = torch.tensor(preds, dtype=torch.float32)
            # Convert logits to probabilities using log-softmax
            log_probs = self.logsoftmax(preds)
            probs = torch.exp(log_probs)
            # Compute temporal log-ratio between consecutive steps
            # log(curr) - log(next) for all j in window
            # Vectorized via shifting
            log_curr = log_probs[:-1]
            log_next = log_probs[1:]
            log_ratio = log_curr - log_next
            # Weight log-ratio by next-step probability
            # Weighted = p(t+1) * ( log p(t) - log p(t+1) )
            weighted = probs[1:] * torch.abs(log_ratio)
            # Sum across classes → temporal loss per sample
            tdds = weighted.sum(dim=2)
            # Compute variance across time window (per sample)
            mean = tdds.mean(dim=0, keepdim=True)
            var = ((tdds - mean) ** 2).sum(dim=0)
            # Apply exponential moving average across windows
            if self.sample_importance is None:
                self.sample_importance = self.decay * var
            else:
                self.sample_importance = self.decay * var + (1 - self.decay) * self.sample_importance

        return self.sample_importance.numpy(force=True)





class TDDSDataLoader(DCDataLoader):
    """
    TDDSDataLoader implements the static TDDS (Temporal Dual-Depth Scoring)
    data pruning algorithm on top of DCDataLoader.

    TDDS selects the most important samples using a dual-depth scoring mechanism

    Parameters
    ----------
    dataset : DCDataset
        The wrapped dataset whose items are (x, y, idx).

    pruning_rate : float
        Fraction of total samples to keep after pruning.
    """

    def __init__(self, dataset: DCDataset, pruning_rate: float, **kwargs):
        super().__init__(dataset, pruning_rate, static=True, **kwargs)


    def compute_subset(self, sample_importance: np.ndarray):
        """
        Compute the TDDS subset

        Parameters
        ----------
        sample_importance : np.ndarray
            Importance scores for all training samples.

        Returns
        -------
        list
            List of selected sample indices after TDDS pruning.
        """
        indices = np.argsort(sample_importance)[self.total_num_samples-self.required_num_samples:]
        return indices.tolist()