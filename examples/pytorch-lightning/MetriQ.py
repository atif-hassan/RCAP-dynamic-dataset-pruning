from ..data import DCDataLoader, DCDataset
import numpy as np


class MetriQDataLoader(DCDataLoader):
    """
    MetriQ Data Pruning Loader

    Allocates pruning quota per class inversely proportional to class-wise accuracy.
    Classes with lower validation accuracy receive a larger quota.
    """

    def __init__(self, dataset: DCDataset, pruning_rate: float, class_wise_acc: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        dataset : DCDataset
            Wrapped dataset (contains underlying dataset & sample indices).
        pruning_rate : float
            Fraction of total samples to keep.
        class_wise_acc : np.ndarray
            Validation accuracy per class (shape C).
        """
        self.class_wise_acc = class_wise_acc
        self.num_classes = len(class_wise_acc)

        # Precompute per-class sample lists for efficient sampling
        targets = np.asarray(dataset.dataset.targets)
        self.class_wise_indices = [np.where(targets == c)[0] for c in range(self.num_classes)]
        self.class_wise_samples = np.array([len(idx) for idx in self.class_wise_indices])

        super().__init__(dataset, pruning_rate, static=True, **kwargs)


    def compute_subset(self, sample_importance: np.ndarray):
        """
        Compute class-balanced pruning subset using the MetriQ rule.
        Parameters
        ----------
        sample_importance : np.ndarray
            Ignored. MetriQ is purely random, not importance-based.

        Returns
        -------
        list
            Selected sample indices.
        """

        # Compute normalization constant Z
        weights = (1 - self.class_wise_acc) * (self.class_wise_samples / self.total_num_samples)
        Z = np.sum(weights)
        # Compute class-wise density = retention fraction per class
        densities = (self.pruning_rate * (1 - self.class_wise_acc)) / Z
        # Fix densities > 1 (class cannot contribute > 100% of itself)
        # This step mirrors MetriQ's handling of saturated classes
        free_classes = np.ones(self.num_classes, dtype=bool)
        while True:
            # Find the classes that are over-saturated
            bad_classes = np.where(densities > 1)[0]
            if len(bad_classes) == 0:
                break
            # Set them to 1
            densities[bad_classes] = 1
            # And set these classes to not be free (meaning they cannot contribute)
            free_classes[bad_classes] = False
            # Find new normalization constant Z over free classes only
            remaining_budget = self.required_num_samples - np.sum(self.class_wise_samples[~free_classes])
            rem = np.where(free_classes)[0]
            Z = np.sum((1 - self.class_wise_acc[rem]) * self.class_wise_samples[rem])
            # Calculate new densities for free classes
            densities[rem] = (remaining_budget * (1 - self.class_wise_acc[rem])) / Z

        # Floating point allocation BEFORE rounding
        alloc_float = self.class_wise_samples * densities
        # Floor allocation
        alloc_floor = np.floor(alloc_float).astype(int)
        # Finally, Sample uniformly at random within each class
        selected_indices = list()
        for c in range(self.num_classes):
            n = alloc_floor[c]
            if n > 0:
                selected_indices.extend(np.random.choice(self.class_wise_indices[c], size=n, replace=False).tolist())
        return selected_indices