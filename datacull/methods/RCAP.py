from ..data import DCDataLoader, DCDataset
from ..importance import DCImportance
from ..logger import DCLogger
import numpy as np
from copy import deepcopy




class RCAPImportance(DCImportance):
    """
    RCAPImportance computes class-normalized, clipped, temperature-scaled
    importance scores from per-sample loss trajectories.

    Importance is computed independently within each class using a
    softmax over (clipped) losses.
    """
    def __init__(self, dataset: DCDataset, logger_object: DCLogger, beta: float, clipping_threshold: float=None):
        # Epoch counter to determine which epoch's loss to extract
        self.epoch = -1
        # Beta is the softmax temperature parameter
        # You can think of it as controlling how peaky the distribution is
        # For starters, you can use any of the following beta values {1/3, 1/2, 1, 2, 3}
        self.beta = beta
        # Find which samples belong to which class
        targets = np.asarray(dataset.dataset.targets)
        self.num_classes = len(np.unique(targets))
        self.class_wise_indices = [np.where(targets == c)[0] for c in range(self.num_classes)]
        self.class_wise_samples = np.array([len(idx) for idx in self.class_wise_indices])
        super().__init__(dataset=dataset, window_size=1, logger_object=logger_object, flush=True)
        # A clipping threshold
        if clipping_threshold is not None:
            self.clipping_threshold = clipping_threshold
        else:
            self.clipping_threshold = np.log(self.num_classes)
        # Initialize with uniform-loss baseline (\approx log C)
        self.sample_losses = np.ones(self.num_samples) * self.clipping_threshold
    
    
    def compute_importance(self):
        """
        Returns a per-sample importance distribution normalized within each class.
        """
        if self.epoch > -1:
            # Extract the loss values for the previous epoch
            loss_vals = np.asarray(self.extract_trajectory_segment(self.epoch))
        else:
            loss_vals = self.sample_losses
        # Update the epoch counter
        self.epoch += 1
        # Remove the first dimension (window size = 1)
        loss_vals = np.squeeze(loss_vals)
        # Update the entire sample loss array (keep previous if None)
        unmasked_indices = np.where(loss_vals != None)[0]
        self.sample_losses[unmasked_indices] = loss_vals[unmasked_indices]
        # Clip the loss values
        self.sample_losses = np.clip(self.sample_losses, 0, self.clipping_threshold)
        # Find the importance of each sample within its class
        sample_importance = np.zeros(self.num_samples)
        for c in range(self.num_classes):
            losses = self.sample_losses[self.class_wise_indices[c]]
            # Class-wise stable softmax
            z = losses / self.beta
            z = z - z.max()
            losses_importance = np.exp(z)
            losses_importance /= np.sum(losses_importance)
            sample_importance[self.class_wise_indices[c]] = losses_importance
        return sample_importance





class RCAPDataLoader(DCDataLoader):
    """
    RCAPDataLoader performs dynamic, class-aware probabilistic sampling
    using importance scores. You can swap in any importance scoring method.
    """
    def __init__(self, dataset: DCDataset, pruning_rate: float, **kwargs):
        # Find which samples belong to which class
        targets = np.asarray(dataset.dataset.targets)
        self.num_classes = len(np.unique(targets))
        self.class_wise_indices = [np.where(targets == c)[0] for c in range(self.num_classes)]
        self.class_wise_samples = np.array([len(idx) for idx in self.class_wise_indices])
        # This will hold the local copy of class-wise indices for sampling
        # This changes every epoch based on previous sampling
        self.local_class_wise_indices = deepcopy(self.class_wise_indices)
        super().__init__(dataset, pruning_rate, static=False, **kwargs)
    
    
    def compute_subset(self, sample_importance: np.ndarray):
        """
        Compute a dynamic RCAP subset for the current epoch.
        """
        if len(sample_importance) != self.total_num_samples:
            raise ValueError("Sample importance length mismatch.")
        # A local copy of required num samples to adjust for saturated classes
        required_num_samples = self.required_num_samples
        # First find the loss per class
        class_wise_loss = np.asarray([np.sum(sample_importance[self.local_class_wise_indices[c]]) for c in range(self.num_classes)])
        # Class priors
        class_fractions = self.class_wise_samples/np.sum(self.class_wise_samples)
        # Now find the fraction of samples to be selected per class
        class_wise_loss_avg = np.sqrt(class_wise_loss * class_fractions)
        class_wise_fraction = (class_wise_loss_avg/np.sum(class_wise_loss_avg)) * (self.required_num_samples/self.class_wise_samples)
        # Ensure that no class fraction exceeds 1
        while True:
            mask = np.where(class_wise_fraction>1)[0]
            if len(mask) > 0:
                class_wise_fraction[mask] = 1
                rem_mask = np.where(class_wise_fraction<1)[0]
                required_num_samples -= np.sum(self.class_wise_samples[mask])
                class_wise_fraction[rem_mask] = (class_wise_loss_avg[rem_mask]/np.sum(class_wise_loss_avg[rem_mask])) * (self.required_num_samples/self.class_wise_samples[rem_mask])
            else:
                break
        # Finally, find the number of samples to be selected per class
        class_wise_req_samples = np.ceil(class_wise_fraction * self.class_wise_samples)
        
        # Now sample from each class probabilistically
        indices = list()
        self.local_class_wise_indices = list()
        for c in range(self.num_classes):
            p = sample_importance[self.class_wise_indices[c]]
            # Ensure that the probabilities sum to 1
            if np.sum(p) != 1:
                p = p / np.sum(p)
            curr_class_indices = np.random.choice(self.class_wise_indices[c], size=int(class_wise_req_samples[c]), replace=False, p=p).tolist()
            indices.extend(curr_class_indices)
            self.local_class_wise_indices.append(curr_class_indices)
        
        return indices