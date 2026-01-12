from pathlib import Path
from datetime import date
from typing import Iterable, Callable
import io

import pandas as pd
import numpy as np
import tifffile

import torch
from torch.nn.functional import pad
from torchvision.transforms.v2 import ToDtype


# Pair data functions

def _select_split(df: pd.DataFrame, split: int | list[int]) -> pd.DataFrame:
    assert 'split' in df, "Dataframe is missing the column 'split'"
    if not isinstance(split, Iterable):
        split = [split]    
    df = df[df['split'].isin(split)]
    df = df.drop(columns='split')
    return df

def _compute_relative_times(df_pairs: pd.DataFrame):
    """
    Compute the relative time of the two acquisitions in days.
    Consider the first as time zero.
    """
    first = pd.to_datetime(df_pairs['date_first'])
    second = pd.to_datetime(df_pairs['date_second'])
    df_pairs['time_first'] = 0
    df_pairs['time_second'] = (second - first).dt.days
    return df_pairs

def _get_samples_from_pairs(df_pairs: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    assert len(index_cols) == 3, "Expected 3 index columns"
    samples = df_pairs.melt(id_vars=index_cols[:2], value_name=index_cols[2])
    samples = samples[index_cols]
    samples = samples.drop_duplicates()
    return samples

def _get_pair(df_pairs: pd.DataFrame, index: int) -> tuple:

    pair = df_pairs.iloc[index]
    eye = list(pair[['heyex_id_anon', 'laterality']])

    first = eye + [pair['date_first']]
    second = eye + [pair['date_second']]

    return tuple(first), tuple(second)

def _get_image_times(df_pairs: pd.DataFrame, index: int):
    pair = df_pairs.iloc[index]
    times = pair.filter(regex='time').values.astype(np.float32)
    return torch.as_tensor(times)


# Sample data functions

def _remove_unused_samples(df_data: pd.DataFrame, df_samples: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    assert list(df_data.index.names) == index_cols, f"Index columns mismatch: {df_data.index.names} != {index_cols}"

    df_data = df_data.merge(df_samples, how='inner', left_index=True, right_on=index_cols)
    # dataframe looses index after merge
    df_data = df_data.set_index(index_cols)

    # Initially the data table should always hold more samples than the samples table,
    # Thus, intersecting it with the samples should result in the same number of entries.
    assert len(df_data) == len(df_samples), 'Something went wrong when selecting the data samples'
    return df_data

def _add_sample_id(samples: pd.DataFrame) -> pd.DataFrame:
    samples.insert(0, 'sample_id', np.arange(len(samples)))
    return samples


# Modality data functions

def _infer_tar_dir_name(data_filename: str) -> str:
    assert 'data_' in data_filename, 'Data filename does not contain "data_" prefix'
    base_name = data_filename.split('.')[0]
    dir_name = base_name.replace('data_', '')
    return dir_name

def _read_datafiles(root: Path, filepaths: list[str], index_cols: list[str]) -> list[pd.DataFrame]:
        
    assert isinstance(filepaths, Iterable), "Please provide a list of filepaths."
    root = Path(root)
    
    data = []
    for f in filepaths:
        df = pd.read_csv(root / f, index_col=index_cols)
        tar_dir = root / _infer_tar_dir_name(f)
        df = _make_tar_paths_abs(tar_dir, df)
        data.append(df)
    return data

def _parse_vf_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(_parse_vf_row, axis=1)

def _parse_vf_row(row):
    """
    Convert VF data in string format to numpy arrays.
    Some VF data is still in string format after loading it from a csv.

    Intended to be used with .apply() on a VF dataframe
    """
    row.x = np.asarray(eval(row.x))
    row.y = np.asarray(eval(row.y))
    row.ph1 = np.asarray(eval(row.ph1)).astype(float)
    if 'nv' in row:
        row.nv = np.asarray(eval(row.nv)).astype(float)
    return row

def _make_tar_paths_abs(tar_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    assert 'tar_file' in df, 'Missing tar_file column in Dataframe.'
    df['tar_file'] = tar_dir / df['tar_file']
    return df

def _get_rel_field_sensitivity_dB(index: tuple, df_vf: pd.DataFrame):
    vf = df_vf.loc[index]
    sensitivity = vf['ph1']
    normative = vf['nv']
    deviation = sensitivity - normative
    return torch.tensor(deviation, dtype=torch.float32)

def _get_md(index: tuple, df_vf: pd.DataFrame):
    d_dB = _get_rel_field_sensitivity_dB(index=index, df_vf=df_vf)
    return d_dB.mean().unsqueeze(0)

def _get_rel_field_sensitivity_lin(index: tuple, df_vf: pd.DataFrame):
    d_dB = _get_rel_field_sensitivity_dB(index=index, df_vf=df_vf)
    d_B = d_dB * 0.1
    d = torch.pow(torch.tensor([10]), d_B)
    return d

def _get_mean_rel_field_sensitivity(index: tuple, df_vf: pd.DataFrame):
    rel_sensitivity = _get_rel_field_sensitivity_lin(index=index, df_vf=df_vf)
    return rel_sensitivity.mean().unsqueeze(0)

def _get_md_antilog(index: tuple, df_vf: pd.DataFrame):
    mean_rel_sensitivity = _get_mean_rel_field_sensitivity(index=index, df_vf=df_vf)
    md = torch.log10(mean_rel_sensitivity) * 10
    return md

def _get_vf_sensitivity(index: tuple, df_vf: pd.DataFrame):
    vf = df_vf.loc[index]
    return torch.tensor(vf['ph1'], dtype=torch.float32)

def _get_vf_coordinates(index: tuple, df_vf: pd.DataFrame):
    vf = df_vf.loc[index]
    x = torch.tensor(vf['x'], dtype=torch.int16)
    y = torch.tensor(vf['y'], dtype=torch.int16)
    return torch.column_stack((x, y))

def _get_single_image(index: tuple, df: pd.DataFrame):
    img_data = df.loc[index]

    archive_path = img_data['tar_file']
    offset = img_data['offset']
    size = img_data['size']
    filename = str(img_data['filename'])

    buffer = _read_from_tar(archive_path, offset, size)

    try:
        if filename.endswith('.tif'):
            return _load_tiff(buffer)
        elif filename.endswith('.npy'):
            return _load_array(buffer)
        else:
            raise NotImplementedError(f'Loading of file {filename} from tar {archive_path} is not implemented.')
    finally:
        buffer.close()
    
def _read_from_tar(archive_path: Path, offset: int, size: int) -> io.BytesIO:
    with open(archive_path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    return io.BytesIO(data)

def _load_tiff(file: str | io.BytesIO) -> torch.Tensor:
    img = tifffile.imread(file)
    img = torch.from_numpy(img)
    img = ToDtype(torch.float32, scale=True)(img)
    # add channel dimension
    img = img.unsqueeze(0)
    # clone tensor to break reference to BytesIO buffer to prevent memory leak
    img = img.clone()
    return img

def _load_array(file: str | io.BytesIO) -> torch.Tensor:
    arr = np.load(file)
    arr = torch.from_numpy(arr)
    arr = ToDtype(torch.float32, scale=False)(arr)
    # clone tensor to break reference to BytesIO buffer to prevent memory leak
    arr = arr.clone()
    return arr

