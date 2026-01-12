
import numpy as np
import pandas as pd

def get_paired_data(df_pairs: pd.DataFrame, df_data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
     
    index_cols = ['heyex_id_anon', 'laterality', 'acquisition_date']
    df_data = df_data.set_index(index_cols)
    idx_first, idx_second = get_sample_indices_from_pair(df_pairs)

    df_first = df_data.loc[idx_first]
    df_second = df_data.loc[idx_second]

    return df_first, df_second

def concat_paired_data(df_first: pd.DataFrame, df_second: pd.DataFrame, columns: list[str] )-> pd.DataFrame:

    assert df_first.index.names == ['heyex_id_anon', 'laterality', 'acquisition_date']   
    assert df_second.index.names == ['heyex_id_anon', 'laterality', 'acquisition_date']

    # remove acquisition date from index
    df_first = df_first.reset_index(level=-1)
    df_second = df_second.reset_index(level=-1)

    return pd.concat(
    [
        df_first[columns].add_suffix("_first", axis=1),
        df_second[columns].add_suffix("_second", axis=1),
    ],
    axis=1,)

def get_sample_indices_from_pair(df_pairs: pd.DataFrame) -> tuple[list, list]:
    first = []
    second = []
    for i, row in df_pairs.iterrows():
        eye = list(row[['heyex_id_anon', 'laterality']])
        first.append(tuple(eye + [row['date_first']]))
        second.append(tuple(eye + [row['date_second']]))
    return first, second

def parse_vf_data(row):
    """
    Convert VF data in string format to numpy arrays.
    Some VF data is still in string format after loading it from a csv.

    Intended to be used with .apply() on a VF dataframe
    """
    row.x = np.asarray(eval(row.x))
    row.y = np.asarray(eval(row.y))
    row.ph1 = np.asarray(eval(row.ph1)).astype(float)
    row.nv = np.asarray(eval(row.nv)).astype(float)
    return row

def make_coord(row):
    """
    Create a coordinate numpy array from the x and y entries of a visual field.
    Intended to be used with .apply() on a VF dataframe
    """
    x = np.asarray(row['x'])[:, None]
    y = np.asarray(row['y'])[:, None]
    return np.concatenate((x, y), axis=-1)

def melt_pairs(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the pair data to a list of all cases contained in the pairs.
    """
    eye_identifier = ['heyex_id_anon', 'laterality']
    pair_cols = eye_identifier + ['date_first', 'date_second']
    assert all([c in df_pairs for c in pair_cols]), f"Missing one or more of the following columns in the input dataframe: {pair_cols}"

    index_cols = eye_identifier + ['acquisition_date']

    splits = None
    if 'split' in df_pairs:
        splits = df_pairs[eye_identifier + ['split']]
        df_pairs = df_pairs.drop(columns = ['split'])

    samples = df_pairs.melt(id_vars=eye_identifier, value_name=index_cols[2])
    samples = samples[index_cols]
    samples = samples.drop_duplicates()

    if splits is not None:
        samples = samples.merge(splits, how='left', on=eye_identifier)
        samples = samples.drop_duplicates()

    return samples


def compute_pairs_time_delta(df_pair: pd.DataFrame):
    assert 'date_first' in df_pair, 'Column date_first is missing.'
    assert 'date_second' in df_pair, 'Column date_second is missing'

    first = df_pair['date_first']
    second = df_pair['date_second']

    if not pd.api.types.is_datetime64_any_dtype(first):
        first = pd.to_datetime(first)

    if not pd.api.types.is_datetime64_any_dtype(second):
        second = pd.to_datetime(second)

    return second - first