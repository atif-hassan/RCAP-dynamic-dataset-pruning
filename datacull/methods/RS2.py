from ..data import DCDataLoader, DCDataset
import numpy as np




class RS2DataLoader(DCDataLoader):
    """
    RS2: Repeated Random Sampling for accelerating Time-To-Accuracy (TTA).

    Implements two modes:
        - sampling_with_replacement=True:
              Sample R% of the dataset every epoch 
              with the data in previous epoch available for next (with replacement).
        - sampling_with_replacement=False:
              Shuffle dataset once, take chunks sequentially until exhausted,
              then reshuffle. Ensures each sample is used equally over time.
        Optional:
        ---------
        - stratify = True:
            Sample each class proportionally to its class frequency 
            This works only for sampling with replacement.
    """
    
    def __init__(self, dataset: DCDataset, pruning_rate: float, sampling_with_replacement: bool=False, stratify: bool=False, **kwargs):
        self.sampling_with_replacement = sampling_with_replacement
        self.stratify = stratify
        # If stratified sampling is needed, compute class-level structure.
        if self.stratify:
            targets = np.asarray(dataset.dataset.targets)
            self.num_classes = len(np.unique(targets))
            self.class_wise_indices = [np.where(targets == c)[0] for c in range(self.num_classes)]
            self.class_wise_samples = np.asarray([len(idx) for idx in self.class_wise_indices])
            self.class_wise_ratio = self.class_wise_samples / np.sum(self.class_wise_samples)
        # Static=False → resample every epoch
        super().__init__(dataset, pruning_rate, static=False, **kwargs)
        # Pointer for chunking without replacement
        self.counter = 0
        # Precompute full index list for the "without replacement" scenario
        self.full_indices = np.arange(self.total_num_samples)
        self.required_num_samples
    
    
    def compute_subset(self, sample_importance: np.ndarray):
        """
        Compute a subset of samples according to the RS2 strategy.

        Parameters
        ----------
        sample_importance : np.ndarray
            Ignored. RS2 is purely random, not importance-based.

        Returns
        -------
        list
            Selected sample indices.
        """
        # Here, with replacement means sampling uniformly at random every time.
        # So previous epoch samples are available again.
        if self.sampling_with_replacement:
            # Allocate per-class budgets proportional to class frequencies
            if self.stratify:
                indices = list()
                for c in range(self.num_classes):
                    class_budget = int(self.required_num_samples * self.class_wise_ratio[c])
                    indices.extend(np.random.choice(self.class_wise_indices[c], size=class_budget, replace=False).tolist())
            else:
                indices = np.random.choice(range(self.total_num_samples), size=self.required_num_samples, replace=False).tolist()
        else:
            # If we have exhausted the shuffled array then reshuffle
            if self.counter + self.required_num_samples > self.total_num_samples:
                self.counter = 0
                np.random.shuffle(self.full_indices)
            # Slice the next chunk
            start = self.counter
            end = start + self.required_num_samples
            indices = self.full_indices[start:end].tolist()
            # Advance the pointer
            self.counter = end
        return indices