from abc import ABC, abstractmethod
from typing import Mapping
from pathlib import Path

from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):

    @abstractmethod
    def __init__(self, root: str, split = None, transform=None, target_transform=None, **kwargs):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @classmethod
    @abstractmethod
    def required_data(cls, config: Mapping) -> list[Path]:
        """Returns a list of paths to files or directories which are required for the dataset.
        This function allows to copy the required data to a temp dir
        (e.g. local to a compute node) before instantiating the dataset.
        """
        pass

class TemporalDatasetBase(Dataset, ABC):

    @abstractmethod
    def __init__(self, root: str, split = None, transform=None, target_transform=None, **kwargs):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __getitem__(self, index) -> list[tuple]:
        pass

    @abstractmethod
    def __len__(self):
        pass
