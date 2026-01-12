import os
import time
from torch.utils.data import get_worker_info

class TimedDatasetMixin:
    """
    Mixin to track per-worker timing of __getitem__ calls.
    """
    def __init__(self, log_every=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timed_worker_stats = {}
        self._timed_log_every = log_every

    def __getitem__(self, idx):
        t0 = time.time()

        # Call the actual dataset __getitem__
        sample = super().__getitem__(idx)

        t1 = time.time()
        elapsed = t1 - t0

        # Track stats per worker
        worker = get_worker_info()
        wid = worker.id if worker else 0
        stats = self._timed_worker_stats.setdefault(wid, [])
        stats.append(elapsed)

        # Print periodic logging
        if len(stats) % self._timed_log_every == 0:
            avg = sum(stats[-self._timed_log_every:]) / self._timed_log_every
            print(f"[Worker {wid}] Avg time per sample (last {self._timed_log_every}): {avg*1000:.2f} ms")

        return sample

    def get_worker_stats(self):
        """Return dictionary of recorded worker times."""
        return self._timed_worker_stats