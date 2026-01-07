from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import numpy as np




class DCDataset(Dataset):
    """
    A lightweight wrapper around any PyTorch Dataset that ensures __getitem__
    returns the underlying sample together with its global dataset index.

    The wrapped dataset may return:
        (x, y)
        (x, y, meta)
        (anything...)

    This class adds the sample index as the final return argument:
        (*original_output, idx)

    This is useful for logging per-sample metrics or computing importance
    values that must be tracked across epochs.
    """
    
    def __init__(self, custom_dataset: Dataset):
        """
        Parameters
        ----------
        custom_dataset : Dataset
            Any PyTorch-compliant dataset implementing __getitem__ and __len__.
        """
        super().__init__()
        self.dataset = custom_dataset
    
    
    def __len__(self):
        """Return the total number of samples in the wrapped dataset."""
        return self.dataset.__len__()
    
    
    def __getitem__(self, idx):
        """
        Return a data sample with its corresponding index.

        Returns
        -------
        tuple
            (*original_dataset_output, idx)
        """
        return (*self.dataset.__getitem__(idx), idx)


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def update(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class DCDataLoader(DataLoader):
    """
    A customizable DataLoader that supports dynamic and static sample pruning.

    Unlike a standard DataLoader, this class allows the user to:
        - compute a subset of samples based on importance scores
        - use torch.utils.data.Subset to load only that subset
        - optionally perform the pruning only once (static=True)
          or re-sample every epoch (static=False)

    This class is meant to provide the infrastructure for data pruning
    algorithms. Subclasses must implement compute_subset().
    """
    
    def __init__(self, dataset: DCDataset, pruning_rate: float, static: bool, **kwargs):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset from which samples will be drawn. Typically a DCDataset.

        pruning_rate : float
            Fraction of samples to remove while pruning. Example: 0.2 → keep 80%.

        static : bool
            If True:
                Subset is computed once (first call to resample) and reused.
            If False:
                Subset is recomputed every time resample() is called.

        **kwargs :
            Additional arguments passed directly to DataLoader (batch_size, 
            shuffle, num_workers, pin_memory, etc.).
        """
        # Full dataset reference
        self.original_dataset = dataset
        # Static vs dynamic pruning
        self.static = static
        # Fraction of data to remove
        self.pruning_rate = pruning_rate
        # Cached subset (if static)
        self.subset_indices = None
        # Remove shuffle from kwargs to avoid conflicts with SubsetSampler
        kwargs.pop("shuffle", None)
        # Save user-specified DataLoader kwargs (needed for reinitialization)
        self._saved_init_kwargs = kwargs.copy()
        self.sampler = SubsetSampler(list(range(len(dataset.dataset))))
        # Initialize the DataLoader with the full dataset initially
        super().__init__(dataset, sampler=self.sampler, **kwargs)
        # Precompute total and required number of samples
        self.total_num_samples = len(dataset.dataset)
        self.required_num_samples = int(np.ceil(self.pruning_rate * self.total_num_samples))


    def compute_subset(self, sample_importance: list):
        """
        Compute and return the subset of sample indices to keep.

        Must be implemented by subclasses.

        Parameters
        ----------
        sample_importance : list or np.ndarray
            Importance score for every sample in the dataset.

        Returns
        -------
        list
            A list of integer sample indices to keep.
        """
        raise NotImplementedError("Subclasses should implement this method.")


    def resample(self, sample_importance: list):
        """
        Compute a new subset of samples based on sample importance and update
        the DataLoader to use only that subset.

        This method may be called at the start of each epoch.

        Behavior:
        ---------
        - If static=True and subset_indices already exists → do nothing.
        - Otherwise, compute_subset() is called to produce a new subset, and
          the DataLoader is reinitialized using torch.utils.data.Subset.

        Parameters
        ----------
        sample_importance : list or np.ndarray
            Importance scores used by compute_subset().
        """
        # Static mode: reuse the previously computed subset
        if self.static and self.subset_indices is not None:
            # Shuffle the indices
            np.random.shuffle(self.subset_indices)
            return
        # Compute new subset of indices
        self.subset_indices = self.compute_subset(sample_importance)
        # Shuffle the indices
        np.random.shuffle(self.subset_indices)
        # Update the sampler to use the new subset
        self.sampler.update(self.subset_indices)
