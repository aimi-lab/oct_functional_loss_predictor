from __future__ import annotations 
from pathlib import Path
from typing import Iterable, Callable, Mapping, NamedTuple
from functools import partial


import pandas as pd
import numpy as np

import torch

from eyemod.visual_fields import VisualField
from eyemod.visual_fields.filter import Filter

from eyemod.data.glaucoma_stage import GlaucomaStage
from eyemod.dl.data.base_dataset import BaseDataset
from eyemod.dl.data.image2vf_utils import (
    _select_split,
    _get_single_image,
    _infer_tar_dir_name,
    _make_tar_paths_abs,
    _parse_vf_data,
)


class VisitIndex(NamedTuple):
    heyex_id_anon: str
    laterality: str
    acquisition_date: str

def _to_visit_indices(df: pd.DataFrame, columns: list[str] = None) -> list[VisitIndex]:
    """Convert DataFrame rows to list of VisitIndex namedtuples.
    
    Args:
        df: DataFrame with visit data
        columns: Column names to extract (default: INDEX_COLS)
        
    Returns:
        List of VisitIndex namedtuples
    """
    if columns is None:
        columns = Img2VfBase.INDEX_COLS
    
    # Fast conversion using to_numpy()
    return [VisitIndex(*row) for row in df[columns].to_numpy()]


class Img2VfBase(BaseDataset):
    INDEX_COLS = ["heyex_id_anon", "laterality", "acquisition_date"]
    VF_DATA_FILE = "data_visual_field.csv"

    def __init__(
        self,
        root,
        split: int | Iterable[int],
        transform=None,
        target_transform=None,
        targets="md",
        **kwargs,
    ):
        super().__init__(root, split, transform, target_transform, **kwargs)

        self.split = [split] if not isinstance(split, Iterable) else split
        self.targets = [targets] if isinstance(targets, str) else targets

        self.samples: list[tuple[VisitIndex]] = None
        self.img_data: list[pd.DataFrame] = None
        self.vf_data: dict[VisualField] = None
        self.target_fns: list[Callable] = None


    def __len__(self):
        return len(self.samples)

    def _setup_vf_data(
        self, filepath: Path, vf_filter: list[Filter] | Filter
    ) -> dict[VisualField]:

        assert self.samples is not None, "Missing samples, setup samples first."

        if isinstance(vf_filter, Filter):
            vf_filter = [vf_filter]

        df = pd.read_csv(filepath, index_col=self.INDEX_COLS)

        visits = self._get_unique_visits(self.samples)
        df = df.loc[visits]

        df = _parse_vf_data(df)
        vf_data = self._create_vf_objs(df, vf_filter)
        return vf_data

    @staticmethod
    def _create_vf_objs(df: pd.DataFrame, vf_filter: list[Filter]) -> dict[VisualField]:

        vf_data = {}
        for idx, row in df.iterrows():
            vf = VisualField.from_arrays(
                row["x"], row["y"], row["ph1"], row["nv"], laterality=idx[1]
            )
            if vf_filter is not None:
                vf = vf.filter(*vf_filter)
            vf_data[idx] = vf
        return vf_data

    def _setup_single_img_data(self, filepath: Path) -> pd.DataFrame:

        assert self.samples is not None, "Missing samples, setup samples first."

        tar_dir = filepath.parent / _infer_tar_dir_name(filepath.name)
        df = pd.read_csv(filepath, index_col=self.INDEX_COLS)
        df = _make_tar_paths_abs(tar_dir, df)

        # remove visits which are not used
        visits = self._get_unique_visits(self.samples)
        df = df.loc[visits]

        return df

    @staticmethod
    def _get_unique_visits(samples: list[tuple[VisitIndex]]) -> list[VisitIndex]:
        samples_flat = {item for s in samples for item in s}
        return list(samples_flat)

    def _get_targets(self, visits_indices: list[VisitIndex]) -> list[torch.Tensor]:
        return [fn(visits_indices) for fn in self.target_fns]

    def _get_target_functions(self, targets: str | list[str]):
        if isinstance(targets, str):
            targets = [targets]

        return [self._get_single_target_fn(t) for t in targets]

    def _get_single_target_fn(self, target: str) -> Callable:

        if "_" in target:
            target_type, timepoint = target.split("_")
            timepoint = int(timepoint)
        else:
            target_type = target
            timepoint = -1  # default to the last time point if nothing is passed

        fn = None
        if target_type == "md":
            fn = self._get_md
        elif target_type == "vfs":
            fn = self._get_vf_sensitivity
        elif target_type == "vfd":
            fn = self._get_vf_deviation
        elif target_type == "laterality":
            fn = self._get_laterality
        elif target_type == "stage":
            fn = self._get_stage
        else:
            raise ValueError(f"Unsupported target type: {target}")

        return partial(self._get_target_at_visit, visit_index=timepoint, getter_fn=fn)

    @staticmethod
    def _get_target_at_visit(
        visit_indices: VisitIndex,
        visit_index: int,
        getter_fn: Callable,
    ) -> torch.Tensor:
        return getter_fn(visit_indices[visit_index])

    def _get_vf(self, index: tuple) -> VisualField:
        return self.vf_data[index]

    def _get_md(self, index: tuple):
        vf = self._get_vf(index=index)
        md = vf.mean_deviation()
        return torch.tensor([md], dtype=torch.float32)

    def _get_vf_sensitivity(self, index: tuple):
        vf = self._get_vf(index=index)
        return torch.tensor(vf.data.sensitivity_values, dtype=torch.float32)

    def _get_vf_deviation(self, index: tuple):
        vf = self._get_vf(index=index)
        return torch.tensor(vf.data.deviation_values, dtype=torch.float32)

    def _get_vf_coordinates(self, index: tuple):
        vf = self._get_vf(index=index)
        return torch.tensor(vf.data.coordinates, dtype=torch.int16)

    def _get_laterality(self, index: tuple):
        laterality = index[1]
        if laterality == "L":
            return torch.tensor([1, 0], dtype=torch.float32)
        else:
            return torch.tensor([0, 1], dtype=torch.float32)

    def _get_stage(self, index: tuple):
        vf = self._get_vf(index=index)
        md = vf.mean_deviation()
        stage = GlaucomaStage.from_mean_deviation_octopus(md)
        stage = torch.tensor(stage.value, dtype=torch.long)
        stage_one_hot = torch.nn.functional.one_hot(stage, len(GlaucomaStage))
        return stage_one_hot.to(dtype=torch.float32)

    def _apply_target_transforms(
        self, targets: list[torch.Tensor]
    ) -> list[torch.Tensor]:

        for target_idx in range(len(targets)):
            targets[target_idx] = self._apply_transform_to_input(
                targets[target_idx],
                transform=self.target_transform,
                input_index=target_idx,
            )
        return targets

    def _get_images(
        self,
        visit_indices: tuple[VisitIndex],
        img_data: pd.DataFrame | list[pd.DataFrame],
    ) -> list[list[torch.Tensor]]:
        """
        Retrieve image data for specified visits across one or more imaging modalities.
        This method processes imaging data and extracts images corresponding to the given
        visit indices. It handles both single DataFrame and multiple DataFrame inputs,
        organizing the output by modality and visit."""

        if isinstance(img_data, pd.DataFrame):
            img_data = [img_data]

        img_modalities = []
        for data in img_data:
            img_visits = []
            for idx in visit_indices:
                img_visits.append(_get_single_image(idx, data))
            img_modalities.append(img_visits)
        return img_modalities

    def _get_image(self, index: tuple, modality_index=None):
        if modality_index:
            img_data = self.img_data[modality_index]
        else:
            img_data = self.img_data
        return _get_single_image(index=index, df=img_data)

    def _apply_image_transforms(
        self, images: list[list[torch.Tensor]]
    ) -> list[list[torch.Tensor]]:

        for modality_idx in range(len(images)):
            for visit_idx in range(len(images[modality_idx])):
                images[modality_idx][visit_idx] = self._apply_transform_to_input(
                    images[modality_idx][visit_idx],
                    transform=self.transform,
                    input_index=modality_idx,
                )
        return images

    @staticmethod
    def _apply_transform_to_input(
        input: torch.Tensor,
        transform: Callable | list[Callable],
        input_index: int = None,
    ) -> torch.Tensor:
        """Apply a transformation to the input tensor, supporting lists of transforms depending on input index."""
        if transform is None:
            return input
        if isinstance(transform, Iterable) and input_index is not None:
            return transform[input_index](input)
        elif isinstance(transform, Callable):
            return transform(input)
        else:
            raise ValueError(
                "Transform must be either a callable or an iterable of callables"
            )


class SingleTimepointMixin:
    """Mixin for datasets that setup samples from paired selection files."""

    def _setup_single_timepoint_samples(
        self, pairs_file: Path, second_only: bool = False
    ) -> list[tuple[VisitIndex]]:
        df_pairs = pd.read_csv(pairs_file)
        df_pairs = _select_split(df_pairs, self.split)

        df_first = df_pairs[["heyex_id_anon", "laterality", "date_first"]]
        df_first = df_first.rename(columns={"date_first": "acquisition_date"})

        df_second = df_pairs[["heyex_id_anon", "laterality", "date_second"]]
        df_second = df_second.rename(columns={"date_second": "acquisition_date"})

        if second_only:
            # zip the tuples to have structure list[tuple[VisitIndex]]
            return list(zip(_to_visit_indices(df_second)))
        else:
            df_samples = pd.concat([df_first, df_second], axis=0)
            df_samples = df_samples.drop_duplicates()
            # zip the tuples to have structure list[tuple[VisitIndex]]
            return list(zip(_to_visit_indices(df_samples)))

    def _get_single_timepoint_meta(self, idx: int, visit_indices: VisitIndex) -> dict:
        visit = visit_indices[0]
        meta = {"sample_id": idx}
        meta.update(visit._asdict())
        meta["md"] = self._get_md(visit)

        if any(["vf" in t for t in self.targets]):
            meta["coordinates"] = self._get_vf_coordinates(visit_indices[0])

        return meta


class MultiTimepointMixin:
    def _setup_paired_samples(self, pairs_file: Path) -> list[tuple[VisitIndex]]:
        df_pairs = pd.read_csv(pairs_file)
        df_pairs = _select_split(df_pairs, self.split)

        df_first = df_pairs[["heyex_id_anon", "laterality", "date_first"]]
        df_first = df_first.rename(columns={"date_first": "acquisition_date"})

        df_second = df_pairs[["heyex_id_anon", "laterality", "date_second"]]
        df_second = df_second.rename(columns={"date_second": "acquisition_date"})

        return list(zip(_to_visit_indices(df_first), _to_visit_indices(df_second)))

    def _setup_time_deltas(self, pairs_file: Path) -> torch.Tensor:
        df_pairs = pd.read_csv(pairs_file)
        df_pairs = _select_split(df_pairs, self.split)

        first = pd.to_datetime(df_pairs["date_first"])
        second = pd.to_datetime(df_pairs["date_second"])
        df_pairs["time_first"] = 0
        df_pairs["time_second"] = (second - first).dt.days

        time_deltas = torch.tensor(
            df_pairs[["time_first", "time_second"]].values, dtype=torch.float32
        )
        return time_deltas

    def _get_paired_timepoint_meta(
        self, index: int, visit_indices: tuple[VisitIndex]
    ) -> dict:
        meta = {"sample_id": index}
        meta["heyex_id_anon"] = visit_indices[0].heyex_id_anon
        meta["laterality"] = visit_indices[0].laterality
        meta["date_first"] = visit_indices[0].acquisition_date
        meta["date_second"] = visit_indices[1].acquisition_date

        if any(["vf" in t for t in self.targets]):
            meta["coordinates"] = self._get_vf_coordinates(visit_indices[0])

        return meta


class MultiModalMixin:

    def _setup_image_data(self, root: Path, datafiles: list[str]) -> list[pd.DataFrame]:
        return [self._setup_single_img_data(root / f) for f in datafiles]


class SingleTimepoint(Img2VfBase, SingleTimepointMixin):
    """
    Simple dataset based on the image2vf data.
    """

    def __init__(
        self,
        root,
        split,
        transform=None,
        target_transform=None,
        data_file: str = "data_onh_oct.csv",
        selection_file: str = "temporal/temporal_pairs_sequential.csv",
        second_only: bool = False,
        vf_filter: Filter | list[Filter] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, target_transform, **kwargs)

        self.samples = self._setup_single_timepoint_samples(
            self.root / selection_file, second_only
        )

        self.img_data = self._setup_single_img_data(self.root / data_file)
        self.vf_data = self._setup_vf_data(
            self.root / "data_visual_field.csv", vf_filter
        )

        self.target_fns = self._get_target_functions(self.targets)

    def __getitem__(self, index):

        visit_indices = self.samples[index]

        img = self._get_images(visit_indices, self.img_data)
        img = self._apply_image_transforms(img)

        img = img[0][0]
        img = img.expand(3, -1, -1)

        targets = self._get_targets(visit_indices)
        targets = self._apply_target_transforms(targets)

        targets = torch.concat(targets)

        meta = self._get_single_timepoint_meta(index, visit_indices)

        img_id = f"{meta['heyex_id_anon']}_{meta['laterality']}_{meta['acquisition_date']}"

        return {'images_thick': img, 'images_onh': img, 'values': targets, 'uuids': img_id,} 
        return sample
        return img, targets, meta

    @classmethod
    def required_data(cls, config: Mapping) -> list[Path]:
        root = Path(config["root"])

        vf_file = root / cls.VF_DATA_FILE
        selection_file = root / config.get(
            "selection_file", "temporal/temporal_pairs_sequential.csv"
        )
        data_file = root / config.get("data_file", "data_onh_oct.csv")
        tar_dir = root / _infer_tar_dir_name(data_file.name)

        return [vf_file, selection_file, data_file, tar_dir]


class PairedTimepoints(Img2VfBase, MultiTimepointMixin):

    def __init__(
        self,
        root,
        split: int | Iterable[int],
        transform=None,
        target_transform=None,
        pairs_file: str = "temporal/temporal_pairs_sequential.csv",
        data_file: str = "data_onh_oct.csv",
        vf_filter: Filter | list[Filter] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, target_transform, **kwargs)

        self.samples = self._setup_paired_samples(self.root / pairs_file)
        self.time_deltas = self._setup_time_deltas(self.root / pairs_file)

        self.img_data = self._setup_single_img_data(self.root / data_file)
        self.vf_data = self._setup_vf_data(
            self.root / "data_visual_field.csv", vf_filter
        )

        self.target_fns = self._get_target_functions(self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        visit_indices = self.samples[index]
        time_deltas = self.time_deltas[index]

        images = self._get_images(visit_indices, self.img_data)
        images = self._apply_image_transforms(images)

        assert len(images) == 1, "Expected to have only one modality"
        images = torch.stack(images[0], dim=0)

        targets = self._get_targets(visit_indices)
        targets = self._apply_target_transforms(targets)
        targets = torch.concat(targets)

        meta = self._get_paired_timepoint_meta(index, visit_indices)

        return images, time_deltas, targets, meta

    def _get_single_target_fn(self, target: str):
        """
        Adapted _get_target_fn to enable passing pair of indices to the getter function.
        And dealing with definition of target time point
        """
        if target == "diff":
            return self._get_md_diff
        else:
            return super()._get_single_target_fn(target)

    def _get_md_diff(self, pair_idx):
        return self._get_md(pair_idx[1]) - self._get_md(pair_idx[0])

    @classmethod
    def required_data(cls, config: Mapping) -> list[Path]:
        root = Path(config["root"])

        vf_file = root / cls.VF_DATA_FILE
        pairs_file = root / config.get(
            "pairs_file", "temporal/temporal_pairs_sequential.csv"
        )
        data_file = root / config.get("data_file", "data_onh_oct.csv")
        tar_dir = root / _infer_tar_dir_name(data_file.name)

        return [vf_file, pairs_file, data_file, tar_dir]


class MultiModal(Img2VfBase, SingleTimepointMixin, MultiModalMixin):

    def __init__(
        self,
        root,
        split: int | Iterable[int],
        transform=None,
        target_transform=None,
        data_files: list[str] = ["data_onh_oct.csv"],
        selection_file: str = "temporal/temporal_pairs_sequential.csv",
        second_only: bool = False,
        vf_filter: Filter | list[Filter] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, target_transform, **kwargs)

        self.samples = self._setup_single_timepoint_samples(
            self.root / selection_file, second_only
        )

        self.vf_data = self._setup_vf_data(self.root / self.VF_DATA_FILE, vf_filter)
        self.img_data = self._setup_image_data(self.root, data_files)

        self.target_fns = self._get_target_functions(self.targets)

    def __getitem__(self, index):

        visit_indices = self.samples[index]

        images = self._get_images(visit_indices, self.img_data)
        images = self._apply_image_transforms(images)

        images = [mod[0] for mod in images]

        targets = self._get_targets(visit_indices)
        targets = self._apply_target_transforms(targets)
        targets = torch.concat(targets)

        meta = self._get_single_timepoint_meta(index, visit_indices)

        return images, targets, meta

    @classmethod
    def required_data(cls, config: Mapping) -> list[Path]:
        root = Path(config["root"])

        vf_file = root / cls.VF_DATA_FILE
        selection_file = root / config.get(
            "selection_file", "temporal/temporal_pairs_sequential.csv"
        )
        data_files = [
            root / file for file in config.get("data_files", ["data_onh_oct.csv"])
        ]
        tar_dirs = [root / _infer_tar_dir_name(file.name) for file in data_files]

        return [vf_file, selection_file] + data_files + tar_dirs


class MultiModalTemporal(Img2VfBase, MultiTimepointMixin, MultiModalMixin):

    def __init__(
        self,
        root,
        split=None,
        transform=None,
        target_transform=None,
        pairs_file: str = "temporal/temporal_pairs_sequential.csv",
        data_files: list[str] = ["data_onh_oct.csv"],
        vf_filter: Filter | list[Filter] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, target_transform, **kwargs)

        self.samples = self._setup_paired_samples(self.root / pairs_file)
        self.time_deltas = self._setup_time_deltas(self.root / pairs_file)

        self.img_data = self._setup_image_data(self.root, data_files)
        self.vf_data = self._setup_vf_data(self.root / self.VF_DATA_FILE, vf_filter)

        self.target_fns = self._get_target_functions(self.targets)

    def __getitem__(self, index):

        visit_indices = self.samples[index]
        time_deltas = self.time_deltas[index]

        images = self._get_images(visit_indices, self.img_data)
        images = self._apply_image_transforms(images)

        assert len(images) == len(self.img_data)
        images = [torch.stack(img_modality, dim=0) for img_modality in images]

        targets = self._get_targets(visit_indices)
        targets = self._apply_target_transforms(targets)
        targets = torch.concat(targets)

        meta = self._get_paired_timepoint_meta(index, visit_indices)

        return images, time_deltas, targets, meta

    @classmethod
    def required_data(cls, config: Mapping) -> list[Path]:
        root = Path(config["root"])

        vf_file = root / cls.VF_DATA_FILE
        pairs_file = root / config.get(
            "pairs_file", "temporal/temporal_pairs_sequential.csv"
        )
        data_files = [
            root / file for file in config.get("data_files", ["data_onh_oct.csv"])
        ]
        tar_dirs = [root / _infer_tar_dir_name(file.name) for file in data_files]

        return [vf_file, pairs_file] + data_files + tar_dirs
