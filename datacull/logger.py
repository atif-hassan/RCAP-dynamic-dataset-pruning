import torch
import orjsonl



class DCLogger:
    """
    Logger for saving per-sample metrics (e.g., predictions or losses)
    at each epoch of training.

    Each epoch is saved as a JSONL file:
        trajectory_dir/epoch{E}.jsonl

    Each line in the file is a JSON object with the form:
        {"sample_index": metric_tensor}

    This format allows fast append-only writes and efficient per-epoch loading.
    """
    
    def __init__(self, trajectory_dir: str, save_every_k_epoch: int=1):
        """
        Parameters
        ----------
        trajectory_dir : str
            Directory where per-epoch JSONL files will be written.
            Must include trailing '/' or '\\'.

        save_every_k_epoch : int, optional
            Save metrics only every k epochs (default: 1 — save every epoch).
        """
        self.trajectory_dir = trajectory_dir
        self.save_every_k_epoch = save_every_k_epoch    


    def log_metric(self, epoch: int, sample_idx: torch.Tensor, metric: torch.Tensor):
        """
        Log per-sample metric values for a single epoch.

        Parameters
        ----------
        epoch : int
            Epoch number to log.

        sample_idx : torch.Tensor
            Tensor of shape (N,) containing the dataset sample indices.

        metric : torch.Tensor
            Tensor of shape (N, ...) containing the corresponding metric
            (e.g., predicted vector of length 10, scalar loss, etc.).

        Notes
        -----
        - A new file is created for each epoch
        - Each metric is written as a single JSON object per line using orjsonl.append().
        """
        if epoch % self.save_every_k_epoch == 0:
            # Batch size consistency check (important to prevent mismatched writes)
            assert sample_idx.shape[0] == metric.shape[0], "Batch size is not same for sample indices and metric"
            # Convert idx and metric to numpy arrays for orjsonl
            sample_idx = sample_idx.tolist()
            metric = metric.tolist()
            # Build file path: trajectory_dir + 'epochX.jsonl'
            filename = self.trajectory_dir + "epoch"+str(epoch) + ".jsonl"
            # Write each sample metric as {"index": metric_row}
            for i, idx in enumerate(sample_idx):
                # Convert index to str (JSON keys must be string)
                orjsonl.append(filename, {str(idx): metric[i]})