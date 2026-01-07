from ..data import DCDataLoader, DCDataset
from ..importance import DCImportance
from ..logger import DCLogger
import numpy as np
from tqdm import tqdm
import torch





class AUMImportance(DCImportance):
    """
    Compute the Area Under the Margin (AUM) importance score for each training sample.

    AUM is defined as:
        AUM(i) = Mean_over_epochs[ logit_true(i) - max_other_logit(i) ]

    This is a static data pruning signal, meaning
    higher AUM → easier example → can be safely pruned first.
    """

    def __init__(self, dataset: DCDataset, trajectory_length: int, logger_object: DCLogger):
        """
        Parameters
        ----------
        trajectory_length : int
            Total number of epochs.

        window_size : int
            Size of prediction window to load from DCLogger.

        logger_object : DCLogger
            Object that stores per-epoch prediction JSONLs.

        targets : np.ndarray or list
            Ground-truth class labels for all samples.
        """
        self.trajectory_length = trajectory_length
        self.targets = np.asarray(dataset.dataset.targets)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        super().__init__(dataset=dataset, window_size=1, logger_object=logger_object, flush=False)


    def compute_importance(self):
        """
        Compute AUM scores by sliding through saved logits and
        accumulating margin = logit_true - max(logit_other).

        Returns
        -------
        np.ndarray
            AUM score for each sample.
        """
        for epoch in tqdm(range(self.trajectory_length - self.window_size + 1), desc="Computing AUM"):
            # Load prediction segment:
            predictions = self.extract_trajectory_segment(epoch)
            predictions = torch.tensor(predictions, dtype=torch.float32).squeeze()
            
            num_samples, num_classes = predictions.shape

            # Convert logits to softmax probabilities using log-softmax
            probs = torch.exp(self.logsoftmax(predictions))

            # Find the logit corresponding to the true class
            assigned_logit = probs[torch.arange(num_samples), self.targets]
            # Mask out the true class to get max over other classes
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[torch.arange(num_samples), self.targets] = True
            # fill true class with -inf so it is ignored
            masked = probs.masked_fill(mask, float('-inf'))
            # Find the largest other logit
            largest_other = torch.max(masked, dim=1).values
            # Finally calculate margin
            margin = assigned_logit - largest_other
            # Now, Accumulate
            if self.sample_importance is None:
                self.sample_importance = margin
            else:
                self.sample_importance += margin

        # Mean margin over the entire trajectory (AUM)
        return (self.sample_importance / self.trajectory_length).numpy(force=True)






class CCSDataLoader(DCDataLoader):
    """
    CCSDataLoader implements the static CCS (Coverage-centric coreset selection)
    data pruning algorithm on top of DCDataLoader.

    CCS removes the hardest β fraction of samples, stratifies the remaining
    samples by importance score, and allocates a pruning budget per stratum.
    Finally, it samples a fixed number of examples from each stratum to achieve
    pruning_rate * N total kept samples.
    """

    def __init__(self, dataset: DCDataset, pruning_rate: float, beta: float, num_strata: int, descending: bool=False, **kwargs):
        self.beta = beta
        assert beta <= pruning_rate, ValueError("CCS beta cannot be larger than pruning_rate.")
        self.num_strata = num_strata
        self.descending = descending
        super().__init__(dataset, pruning_rate, static=True, **kwargs)


    def compute_subset(self, sample_importance: np.ndarray):
        """
        Compute CCS subset using:
           1. Remove hardest beta fraction
           2. Stratify remaining samples by score
           3. Allocate budget and sample from each stratum
        """
        # Remove hardest beta fraction
        cutoff = int(np.floor(self.total_num_samples * self.beta))
        # Some importance scores give high importance to hard examples
        if self.descending:
            sorted_idx = np.argsort(sample_importance)[::-1]
        # while others give low importance to hard examples
        else:
            sorted_idx = np.argsort(sample_importance)
        # Keep only the easier examples
        work_indices = sorted_idx[cutoff:]
        work_scores = sample_importance[work_indices]

        # Stratify into num_strata bins
        min_s, max_s = work_scores.min(), work_scores.max()
        bin_edges = np.linspace(min_s, max_s, self.num_strata)
        strata_ids = np.digitize(work_scores, bin_edges) - 1

        # Create a bucket for each stratum
        B = [work_indices[strata_ids == s] for s in range(self.num_strata)]
        # Remove any empty bucket/stratum
        B = [arr for arr in B if len(arr) > 0]
        num_active = len(B)

        # Sanity Check
        if num_active == 0:
            return []
        
        # Now select from each stratum
        selected_samples = list()
        m = self.required_num_samples
        while True:
            # Select the stratum with the least number of examples
            B_min_index, B_min_size = min([[i, len(B[i])] for i in range(len(B))], key=lambda x: x[1])
            m_B = min([B_min_size, int(np.floor(m/len(B)))])
            # Select uniformly at random from this stratum
            np.random.shuffle(B[B_min_index])
            # Update coreset
            selected_samples.extend(B[B_min_index][:m_B])
            # Update stratum
            B.pop(B_min_index)
            # Update remaining samples to select
            m-= m_B
            
            if len(B) == 0 or m<= 0:
                return selected_samples
