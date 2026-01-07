import os
import orjsonl
from .logger import DCLogger
from .data import DCDataset




class DCImportance:
    """
    Base class for computing data pruning (DP) importance scores
    from per-epoch trajectory logs.

    Each trajectory file corresponds to one saved epoch and is stored as a
    `.jsonl` file. Each line in the file contains exactly one key–value pair:

        {"sample_index": metric_value}

    where `metric_value` may be:
        - a scalar (e.g., loss)
        - a vector (e.g., logits or probabilities)
        - any JSON-serializable structure

    The class extracts a sliding window of consecutive epochs and returns a
    nested structure of shape:

        (window_size, num_samples, *)

    Subclasses are responsible for converting this structure into tensors
    and computing per-sample importance scores.
    """

    def __init__(self, dataset: DCDataset, window_size: int, logger_object: DCLogger, flush: bool = False):
        """
        Parameters
        ----------
        dataset : DCDataset
            The dataset object to determine the number of samples.

        window_size : int
            Number of consecutive epochs to extract per trajectory window.

        logger_object : DCLogger
            Logger object that specifies trajectory directory and save frequency.

        flush : bool, optional
            If True, trajectory files are deleted after being read.
            Useful for dynamic pruning to reduce disk usage.
        """
        self.num_samples = dataset.__len__()
        self.window_size = window_size
        self.trajectory_dir = logger_object.trajectory_dir
        self.offset = logger_object.save_every_k_epoch
        self.flush = flush

        self.sample_importance = None
        # Validate trajectory directory once
        if not os.path.exists(self.trajectory_dir):
            raise ValueError(f"Trajectory directory does not exist: {self.trajectory_dir}")


    def extract_trajectory_segment(self, start_epoch: int):
        """
        Extract a trajectory segment consisting of `window_size` epochs.

        Epochs are sampled with stride equal to `save_every_k_epoch`, i.e.:

            start_epoch,
            start_epoch + offset,
            start_epoch + 2 * offset,
            ...

        Parameters
        ----------
        start_epoch : int
            Epoch index at which the window begins.

        Returns
        -------
        list
            Nested list of shape (num_extracted_epochs, num_samples, *),
            where each entry corresponds to the logged metric for a sample.

        Notes
        -----
        - Sample indices are assumed to range from [0, num_samples - 1].
        - Missing samples in a trajectory file will result in `None` entries.
        """
        segment = list()

        for epoch in range(start_epoch, start_epoch + self.window_size, self.offset):
            filename = os.path.join(self.trajectory_dir, f"epoch{epoch}.jsonl")

            if not os.path.exists(filename):
                raise FileNotFoundError(f"Expected trajectory file not found: {filename}")
            
            # Load JSONL file: list of {sample_index: metric}
            epoch_objects = orjsonl.load(filename)
            # Preallocate per-epoch container
            epoch_data = [None] * self.num_samples
            # Update the epoch_data array with the loaded values
            for obj in epoch_objects:
                # Each obj has exactly one key–value pair
                sample_index, value = next(iter(obj.items()))
                epoch_data[int(sample_index)] = value
            # Now update the segment with this epoch's data
            segment.append(epoch_data)

            # Optionally delete file after reading
            if self.flush:
                os.remove(filename)

        return segment


    def compute_importance(self):
        """
        Compute per-sample importance scores.

        Subclasses must implement this method.

        Returns
        -------
        array-like
            A 1D array or list of length `num_samples` containing
            the importance score for each sample.
        """
        raise NotImplementedError("Subclasses should implement this method.")
