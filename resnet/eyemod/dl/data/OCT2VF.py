from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import ToDtype

from eyemod.dl.data.base_dataset import BaseDataset
from eyemod.definitions import Laterality, Split


class OCT2VF(BaseDataset):
    VALID_SUBSETS = ['davide']
    def __init__(self, root, split, transform=None, target_transform=None, target_name='mean_deviation', subset = None,**kwargs):
        super().__init__(root, split, transform, target_transform, **kwargs)

        self.target_name = target_name
        self.subset = subset
        self.overview_path = self.root / 'overview.csv'
        self.data = self._collect()

    def __getitem__(self, index):
        onh_slice = self._get_onh_slice(index)

        if self.transform:
            onh_slice = self.transform(onh_slice)

        ground_truth = self._get_target(index)
        
        if self.target_transform :
            ground_truth = self.target_transform(ground_truth)

        metadata = self._get_metadata(index)

        return onh_slice, ground_truth, metadata


    def __len__(self):
        return len(self.data)
    
    def _get_target(self, idx: int):
        if self.target_name == "mean_deviation":
            return self._get_mean_deviation(idx)
        elif self.target_name == "sensitivity_map":
            return self._get_visual_field(idx)
        else:
            raise ValueError(f"Invalid target name: {self.target_name}")

    def _get_onh_slice(self, idx: int):
        img_path = self.data.iloc[idx].onh_path
        img = read_image(img_path)
        img = ToDtype(torch.float32, scale=True)(img)
        return img

    def _get_mean_deviation(self, idx: int):
        vf_path = self.root / self.data.iloc[idx].vf_path
        sensitivity_values = _read_from_json(vf_path, 'sensitivity_values')
        normative_values = _read_from_json(vf_path, 'normative_values')
        mean_deviation = np.mean((normative_values - sensitivity_values))
        return torch.tensor([mean_deviation], dtype=torch.float32)      

    def _get_visual_field(self, idx: int):
        vf_path = self.root / self.data.iloc[idx].vf_path
        sensitivity_values = _read_from_json(vf_path, 'sensitivity_values')
        return torch.tensor(sensitivity_values, dtype=torch.float32)
        
    def _get_metadata(self, idx: int):
        row = self.data.iloc[idx]
         # pd.Series
        return row[['patient_id', 'laterality']].to_dict()

    def _collect(self) -> pd.DataFrame:
        if not isinstance(self.split, str):
            self.split = str(self.split)

        split_col = 'split'
        if self.subset is not None:
            assert self.subset in OCT2VF.VALID_SUBSETS,  f"Subset {self.subset} is not a valid subset."
            split_col = f'split_{self.subset}'

        df = pd.read_csv(self.overview_path)
        
        df = df.loc[df[split_col] == self.split]
        df = df.dropna(subset=split_col)

        df = df[df['onh_path'].notnull()]
        df['onh_path'] = self.root / df['onh_path']

        return df


class ONHOCT2VF(OCT2VF):
    """
    Version of the OCT2VF Dataset that returns circular ONH OCT scan as input.
    """
    def __init__(self, root: str, split: str, transform=None, target_transform=None, ground_truth_type: str = "sensitivity_map"):
        super().__init__(root, split, transform, target_transform, sampling_stride=1, ground_truth_type=ground_truth_type)
        self.overview = self._filter_overview()

    def _filter_overview(self):
        """Not all samples have an ONH image, filter out the samples that do not have an ONH image."""
        assert 'onh_path' in self.overview.columns, "Overview must have a column 'onh_path'"
        return self.overview[self.overview.onh_path.notnull()]

    def __getitem__(self, idx: int):
        
        onh_slice = self._get_onh_slice(idx)

        if self.transform:
            onh_slice = self.transform(onh_slice)

        ground_truth = self._get_ground_truth(idx)
        
        if self.target_transform :
            ground_truth = self.target_transform(ground_truth)

        metadata = self._get_metadata(idx)

        return onh_slice, ground_truth, metadata
    
    def _get_onh_slice(self, idx: int):
        img_path = self.root / self.overview.iloc[idx].onh_path
        img = read_image(img_path)
        img = ToDtype(torch.float32, scale=True)(img)
        return img

def _read_from_json(path: Path, value: str) -> np.ndarray:
    with open(path, 'r') as f:
        data = json.load(f)
    assert 'array' in data['data'][value], "Entry requires array sub-entry"
    return np.array(data['data'][value]['array'])
